#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "simd-mappings.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>

// =================================================================================================
// NEON优化的GEMM实现 (已重构为模板)
// =================================================================================================

#define MR 8
#define NR 4

// =================================================================================================
// 辅助函数和Kernel (大部分保持不变)
// =================================================================================================

// 声明所有需要的静态辅助函数
template<typename B_TYPE>
static void gemm_f32_ggml(const struct ggml_compute_params * params, int M, int N, int K,
                           const float* A, int lda, const B_TYPE* B_q, int ldb_q_unused,
                           float* C, int ldc);
static inline void quantize_block_q8_0(const float *x, float* y_d, int8_t* y_qs);
static void pack_A_q8_0_f32(int M, int K, const float* A, int lda, int8_t* A_qs_packed, float* A_d_packed);
static void gemm_q8_0_kernel_8x4_neon(int kc_size, int mr, const int8_t* A_qs_packed, const float* A_d_packed,
                                      const int8_t* B_qs_packed, const float* B_d_packed, float* C, int ldc,
                                      bool accumulate);

// B矩阵打包函数的模板声明
template<typename B_TYPE>
static void pack_B(int nc, int K, const B_TYPE* B_q, int ldb_q, int8_t* B_qs_packed, float* B_d_packed);


// 量化一个32个float的块为q8_0格式
static inline void quantize_block_q8_0(const float *x, float* y_d, int8_t* y_qs) {
    float32x4_t srcv [8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[4];

    for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + 4*j);
    for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);
    for (int j = 0; j < 4; j++) amaxv[j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
    for (int j = 0; j < 2; j++) amaxv[j] = vmaxq_f32(amaxv[2*j], amaxv[2*j+1]);
    const float amax = vmaxvq_f32(vmaxq_f32(amaxv[0], amaxv[1]));

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    *y_d = d;

    for (int j = 0; j < 8; j+=2) {
        const float32x4_t v0  = vmulq_n_f32(srcv[j], id);
        const float32x4_t v1  = vmulq_n_f32(srcv[j + 1], id);
        const int32x4_t   v0_i32 = vcvtnq_s32_f32(v0);
        const int32x4_t   v1_i32 = vcvtnq_s32_f32(v1);
        const int16x4_t   v0_i16 = vqmovn_s32(v0_i32);
        const int16x4_t   v1_i16 = vqmovn_s32(v1_i32);
        const int8x8_t    vi8  = vqmovn_s16(vcombine_s16(v0_i16, v1_i16));
        vst1_s8(y_qs + 4*j, vi8);
    }
}

// 打包矩阵A (F32) 的一个块，由单个线程调用
// 这个函数与B的类型无关，保持不变
void pack_A_q8_0_f32(
    int M, int K,
    const float* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0; // 假设所有类型处理的块大小都与QK8_0兼容

    int8_t A_qs_packed_buf[MR * QK8_0];

    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < K_BLOCKS; ++j) {
            for (int row = 0; row < MR; ++row) {
                if (i + row < M) {
                    quantize_block_q8_0(A + (i + row) * lda + j * QK8_0, &A_d_packed[j * MR + row], &A_qs_packed_buf[row * QK8_0]);
                }
            }
            for (int k = 0; k < QK8_0; k += 4) {
                for (int row = 0; row < MR; ++row) {
                    memcpy(A_qs_packed, &A_qs_packed_buf[row * QK8_0 + k], 4);
                    A_qs_packed += 4;
                }
            }
        }
        A_d_packed += MR * K_BLOCKS;
    }
}

// =================================================================================================
// B 矩阵打包模板特化
// =================================================================================================

// 模板特化 for block_q8_0
template<>
void pack_B<block_q8_0>(
    int nc, int K, const block_q8_0* B_q, int ldb_q,
    int8_t* B_qs_packed, float* B_d_packed)
{
    static_assert(NR == 4, "NR must be 4 for this implementation");
    const int K_BLOCKS = K / QK8_0;

    for (int j = 0; j < nc; j += NR) {
        // Pack deltas
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);
                }
            }
        }

        // Pack quants
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            int32x4x4_t b_buf0, b_buf1;
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    b_buf0.val[col] = vld1q_s32(reinterpret_cast<const int32_t*>((B_q + (j + col) * ldb_q + k_block)->qs + 0));
                    b_buf1.val[col] = vld1q_s32(reinterpret_cast<const int32_t*>((B_q + (j + col) * ldb_q + k_block)->qs + 16));
                }
            }
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed), b_buf0);
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed + 16 * NR), b_buf1);
            B_qs_packed += QK8_0 * NR;
        }
    }
}

// 模板特化 for block_q4_0
template<>
void pack_B<block_q4_0>(
    int nc, int K, const block_q4_0* B_q, int ldb_q,
    int8_t* B_qs_packed, float* B_d_packed)
{
    static_assert(NR == 4, "NR must be 4 for this implementation");
    // Q4_0和Q8_0有相同的块大小QK
    const int K_BLOCKS = K / QK4_0;

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(0x8);

    for (int j = 0; j < nc; j += NR) {
        // Pack deltas
        for (int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);
                }
            }
        }

        // Pack quants
        for (int k_block = 0; k_block < K_BLOCKS; ++k_block) {
           int32x4x4_t b_buf0, b_buf1;
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    const uint8x16_t v_u8 = vld1q_u8((B_q + (j + col) * ldb_q + k_block)->qs);

                    // 解包4-bit到8-bit并减去8
                    const int8x16_t v_i8_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v_u8, m4b)), s8b);
                    const int8x16_t v_i8_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v_u8, 4)), s8b);

                    b_buf0.val[col] = vreinterpretq_s32_s8(v_i8_l);
                    b_buf1.val[col] = vreinterpretq_s32_s8(v_i8_h);
                }
            }
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed), b_buf0);
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed + 16 * NR), b_buf1);
            B_qs_packed += QK4_0 * NR;
        }
    }
}


// 核心计算Kernel模板，针对不同的mr值进行特化以避免不必要的计算
template<int MR_val>
static void gemm_q8_0_kernel_8x4_neon_impl(
    int kc_size,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const float* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    static_assert(MR_val > 0 && MR_val <= MR, "MR_val must be in [1, 8]");

    const int KC_BLOCKS = kc_size / QK8_0;

    float32x4_t c_v[MR_val];
    if (accumulate) {
        for (int i = 0; i < MR_val; ++i) c_v[i] = vld1q_f32(C + i * ldc);
    } else {
        for (int i = 0; i < MR_val; ++i) c_v[i] = vdupq_n_f32(0.0f);
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const float* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR_val];
        for (int i = 0; i < MR_val; ++i) sum_v[i] = vdupq_n_s32(0);

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            const int8x16_t b_vec = vld1q_s8(b_ptr); b_ptr += 16;
            const int8x16_t a_vec_0 = vld1q_s8(a_ptr);

            // 由于MR_val是编译时常量，这些if判断会在编译时被优化掉
            if (MR_val >= 1) sum_v[0] = vdotq_laneq_s32(sum_v[0], b_vec, a_vec_0, 0);
            if (MR_val >= 2) sum_v[1] = vdotq_laneq_s32(sum_v[1], b_vec, a_vec_0, 1);
            if (MR_val >= 3) sum_v[2] = vdotq_laneq_s32(sum_v[2], b_vec, a_vec_0, 2);
            if (MR_val >= 4) sum_v[3] = vdotq_laneq_s32(sum_v[3], b_vec, a_vec_0, 3);

            if (MR_val > 4) {
                const int8x16_t a_vec_1 = vld1q_s8(a_ptr + 16);
                if (MR_val >= 5) sum_v[4] = vdotq_laneq_s32(sum_v[4], b_vec, a_vec_1, 0);
                if (MR_val >= 6) sum_v[5] = vdotq_laneq_s32(sum_v[5], b_vec, a_vec_1, 1);
                if (MR_val >= 7) sum_v[6] = vdotq_laneq_s32(sum_v[6], b_vec, a_vec_1, 2);
                if (MR_val >= 8) sum_v[7] = vdotq_laneq_s32(sum_v[7], b_vec, a_vec_1, 3);
            }
            a_ptr += 32;
        }

        const float32x4_t d_b_v = vld1q_f32(bd_ptr);
        bd_ptr += NR;

        for (int i = 0; i < MR_val; ++i) {
            c_v[i] = vmlaq_n_f32(c_v[i], vmulq_f32(vcvtq_f32_s32(sum_v[i]), d_b_v), ad_ptr[i]);
        }

        ad_ptr += MR;
    }

    for (int i = 0; i < MR_val; ++i) {
        vst1q_f32(C + i * ldc, c_v[i]);
    }
}

// 核心计算Kernel (分发器)，根据mr调用相应的模板实例
static void gemm_q8_0_kernel_8x4_neon(
    int kc_size, int mr,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const float* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    switch (mr) {
        case 1: gemm_q8_0_kernel_8x4_neon_impl<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_kernel_8x4_neon_impl<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_kernel_8x4_neon_impl<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_kernel_8x4_neon_impl<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 5: gemm_q8_0_kernel_8x4_neon_impl<5>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 6: gemm_q8_0_kernel_8x4_neon_impl<6>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 7: gemm_q8_0_kernel_8x4_neon_impl<7>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 8: gemm_q8_0_kernel_8x4_neon_impl<8>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        default:
            assert(false); // mr的值应该在[1, 8]的范围内
            break;
    }
}

// =================================================================================================
// 主计算函数 (模板化)
// =================================================================================================
template<typename B_TYPE>
static void gemm_f32_ggml(
    const struct ggml_compute_params * params,
    int M, int N, int K,
    const float* A, int lda,
    const B_TYPE* B_q, int ldb_q_unused,
    float* C, int ldc)
{
    (void)ldb_q_unused;
    assert(K % QK8_0 == 0); // 假设所有支持的类型块大小都一样

    const int K_BLOCKS = K / QK8_0;

    constexpr int MC = 32;
    constexpr int KC = 1024;
    constexpr int NC = 32;

    const int M_CEIL = (M + MR - 1) / MR * MR;

    // 从 wdata 分配内存
    size_t a_qs_packed_size = GGML_PAD(M_CEIL * K * sizeof(int8_t), 64);
    size_t a_d_packed_size  = GGML_PAD(M_CEIL * K_BLOCKS * sizeof(float), 64);
    assert(params->wsize >= a_qs_packed_size + a_d_packed_size);

    int8_t* A_qs_packed = (int8_t*)params->wdata;
    float* A_d_packed = (float*)((char*)params->wdata + a_qs_packed_size);

    const int ith = params->ith;
    const int nth = params->nth;

    // =================================================================
    // 阶段 1: 并行化 pack_A (F32 -> Q8_0) - 静态分割
    // =================================================================
    const int n_m_blocks = M_CEIL / MR;
    const int M_blocks_per_thread = (n_m_blocks + nth - 1) / nth;
    const int m_start_block = ith * M_blocks_per_thread;
    const int m_end_block = std::min((ith + 1) * M_blocks_per_thread, n_m_blocks);
    
    const int m_start = m_start_block * MR;
    const int m_end = std::min(m_end_block * MR, M);

    if (m_start < m_end) {
        pack_A_q8_0_f32(m_end - m_start, K, A + m_start * lda, lda, A_qs_packed + m_start * K, A_d_packed + m_start * K_BLOCKS);
    }
    ggml_barrier(params->threadpool);

    // =================================================================
    // 阶段 2: 并行化 GEMM 计算
    // =================================================================
    const int N_chunks = (N + NC - 1) / NC;
    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);

    int n_chunk_id = ith;
    while(n_chunk_id < N_chunks) {
        const int jc = n_chunk_id * NC;
        const int nc = std::min(NC, N - jc);

        // B的打包缓冲区是线程私有的（在栈上）
        int8_t B_qs_packed[KC * NC] __attribute__((aligned(64)));
        float B_d_packed[(KC / QK8_0) * NC] __attribute__((aligned(64)));

        for (int kc = 0; kc < K; kc += KC) {
            const int kc_size = std::min(KC, K - kc);
            const int kc_blocks = kc_size / QK8_0;
            const int k_block_offset = kc / QK8_0;
            
            pack_B<B_TYPE>(nc, kc_size, B_q + jc * K_BLOCKS + k_block_offset, K_BLOCKS, B_qs_packed, B_d_packed);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);

                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                         gemm_q8_0_kernel_8x4_neon(
                            kc_size, std::min(MR, mc - ir),
                            A_qs_packed + (ic + ir) * K + kc * MR,
                            A_d_packed + (ic + ir) * K_BLOCKS + k_block_offset * MR,
                            B_qs_packed + jr * kc_size,
                            B_d_packed + jr * kc_blocks,
                            C + (ic + ir) * ldc + (jc + jr), ldc,
                            kc != 0
                        );
                    }
                }
            }
        }
        n_chunk_id = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
    ggml_barrier(params->threadpool);
}
#endif // defined(__ARM_NEON)

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(params->nth > 0);
    assert(params->ith < params->nth);

    if (Ctype != GGML_TYPE_F32)
        return false;

    if (m % NR != 0 || n == 1)
        return false;

#if defined(__ARM_NEON)
    if (Btype == GGML_TYPE_F32) {
        // We can only handle cases where the second matrix is F32.
        // The first matrix (A) is the quantized one.
        switch (Atype) {
            case GGML_TYPE_Q8_0: {
                gemm_f32_ggml<block_q8_0>(params, n, m, k * QK8_0,
                                   (const float*)B, ldb,
                                   (const block_q8_0*)A, lda,
                                   (float*)C, ldc);
                return true;
            }
            case GGML_TYPE_Q4_0: {
                // Note: The K dimension for the function is the number of float elements.
                // For Q4_0 and Q8_0, this is the same.
                gemm_f32_ggml<block_q4_0>(params, n, m, k * QK4_0,
                                   (const float*)B, ldb,
                                   (const block_q4_0*)A, lda,
                                   (float*)C, ldc);
                return true;
            }
            default:
                return false;
        }
    }
#endif // __ARM_NEON

    return false;
}