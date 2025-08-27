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
// NEON优化的GEMM实现 (从 gemm_q8_0.cpp 移植并修改)
// =================================================================================================

#define MR 8
#define NR 4

// 声明所有需要的静态辅助函数
static void gemm_q8_0_f32_ggml(const struct ggml_compute_params * params, int M, int N, int K,
                               const float* A, int lda, const block_q8_0* B_q, int ldb_q_unused,
                               float* C, int ldc);
static inline void quantize_block_q8_0(const float *x, float* y_d, int8_t* y_qs);
static void pack_A_q8_0_f32(int M, int K, const float* A, int lda, int8_t* A_qs_packed, float* A_d_packed);
static void pack_B_q8_0(int nc, int K, const block_q8_0* B_q, int ldb_q, int8_t* B_qs_packed, float* B_d_packed);
static void gemm_q8_0_kernel_8x4_neon(int kc_size, int mr, const int8_t* A_qs_packed, const float* A_d_packed,
                                      const int8_t* B_qs_packed, const float* B_d_packed, float* C, int ldc,
                                      bool accumulate);

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
void pack_A_q8_0_f32(
    int M, int K,
    const float* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0;

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

// 打包矩阵B (Q8_0)
void pack_B_q8_0(
    int nc, int K, const block_q8_0* B_q, int ldb_q,
    int8_t* B_qs_packed, float* B_d_packed)
{
    static_assert(NR == 4, "NR must be 4 for this implementation");
    const int K_BLOCKS = K / QK8_0;
    const int K_div_4 = K / 4;

    for (int j = 0; j < nc; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);
                }
            }
        }

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

// 核心计算Kernel
static void gemm_q8_0_kernel_8x4_neon(
    int kc_size, int mr,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const float* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    const int KC_BLOCKS = kc_size / QK8_0;

    float32x4_t c_v[MR];
    if (accumulate) {
        for (int i = 0; i < mr; ++i) c_v[i] = vld1q_f32(C + i * ldc);
    } else {
        for (int i = 0; i < MR; ++i) c_v[i] = vdupq_n_f32(0.0f);
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const float* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR];
        for (int i = 0; i < MR; ++i) sum_v[i] = vdupq_n_s32(0);

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            int8x16_t a_vec_0 = vld1q_s8(a_ptr); a_ptr += 16;
            int8x16_t a_vec_1 = vld1q_s8(a_ptr); a_ptr += 16;
            int8x16_t b_vec = vld1q_s8(b_ptr); b_ptr += 16;

            sum_v[0] = vdotq_laneq_s32(sum_v[0], b_vec, a_vec_0, 0);
            sum_v[1] = vdotq_laneq_s32(sum_v[1], b_vec, a_vec_0, 1);
            sum_v[2] = vdotq_laneq_s32(sum_v[2], b_vec, a_vec_0, 2);
            sum_v[3] = vdotq_laneq_s32(sum_v[3], b_vec, a_vec_0, 3);

            sum_v[4] = vdotq_laneq_s32(sum_v[4], b_vec, a_vec_1, 0);
            sum_v[5] = vdotq_laneq_s32(sum_v[5], b_vec, a_vec_1, 1);
            sum_v[6] = vdotq_laneq_s32(sum_v[6], b_vec, a_vec_1, 2);
            sum_v[7] = vdotq_laneq_s32(sum_v[7], b_vec, a_vec_1, 3);
        }

        const float32x4_t d_b_v = vld1q_f32(bd_ptr);
        bd_ptr += NR;

        for (int i = 0; i < MR; ++i) {
            c_v[i] = vmlaq_n_f32(c_v[i], vmulq_f32(vcvtq_f32_s32(sum_v[i]), d_b_v), ad_ptr[i]);
        }

        ad_ptr += MR;
    }

    for (int i = 0; i < mr; ++i) {
        vst1q_f32(C + i * ldc, c_v[i]);
    }
}

// 使用ggml线程模型的主计算函数
static void gemm_q8_0_f32_ggml(
    const struct ggml_compute_params * params,
    int M, int N, int K,
    const float* A, int lda,
    const block_q8_0* B_q, int ldb_q_unused,
    float* C, int ldc)
{
    (void)ldb_q_unused;
    assert(K % QK8_0 == 0);

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
    const int n_m_blocks = (M + MR - 1) / MR;
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
            
            pack_B_q8_0(nc, kc_size, B_q + jc * K_BLOCKS + k_block_offset, K_BLOCKS, B_qs_packed, B_d_packed);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                if (mc <= 0) continue;

                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                         gemm_q8_0_kernel_8x4_neon(
                            kc_size, std::min(MR, mc - ir),
                            A_qs_packed + ic * K + k_block_offset * QK8_0 * MR,
                            A_d_packed + ic * K_BLOCKS + k_block_offset * MR,
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

    switch (Atype) {
#if defined(__ARM_NEON)
    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_F32)
            return false;
        if (n % MR != 0 && m % NR != 0) 
            return false;
        gemm_q8_0_f32_ggml(params, n, m, k * QK8_0,
                           (const float*)B, ldb,
                           (const block_q8_0*)A, lda,
                           (float*)C, ldc);
        return true;
    }
#endif
     default:
        return false;
    }

    return false; // Should not be reached if a case matches
}