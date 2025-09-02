#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "simd-mappings.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define MR 8
#define NR 4
#elif defined(__AVX2__)
#include <immintrin.h>
#define MR 8
#define NR 8
static inline __m256i to_int8_q4_0(const uint8_t * rsi)
{
    const __m256i off = _mm256_set1_epi8( 8 );
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_sub_epi8(_mm256_and_si256(lowMask, bytes), off);
}

static inline __m256i to_int8_mxfp4(const uint8_t * rsi)
{
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_mxfp4);
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    return _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(tmp, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(tmp, m4b)));
}
// Helper function to transpose an 8x8 matrix of 32-bit integers using AVX2
static inline void transpose_8x8_i32_avx2(__m256i row[8], __m256i col[8]) {
    __m256i first0 = _mm256_unpacklo_epi32(row[0], row[1]); // a0 b0 a1 b1 a2 b2 a3 b3
    __m256i first1 = _mm256_unpackhi_epi32(row[0], row[1]); // a4 b4 a5 b5 a6 b6 a7 b7 
    __m256i first2 = _mm256_unpacklo_epi32(row[2], row[3]); // c0 d0 c1 d1 c2 d2 c3 d3
    __m256i first3 = _mm256_unpackhi_epi32(row[2], row[3]); // c4 d4 c5 d5 c6 d6 c7 d7
    __m256i first4 = _mm256_unpacklo_epi32(row[4], row[5]); // e0 f0 e1 f1 e2 f2 e3 f3
    __m256i first5 = _mm256_unpackhi_epi32(row[4], row[5]); // e4 f4 e5 f5 e6 f6 e7 f7
    __m256i first6 = _mm256_unpacklo_epi32(row[6], row[7]); // g0 h0 g1 h1 g2 h2 g3 h3
    __m256i first7 = _mm256_unpackhi_epi32(row[6], row[7]); // g4 h4 g5 h5 g6 h6 g7 h7

    __m256i second0 = _mm256_unpacklo_epi64(first0, first2); // a0 b0 c0 d0 a1 b1 c1 d1
    __m256i second1 = _mm256_unpackhi_epi64(first0, first2); // a2 b2 c2 d2 a3 b3 c3 d3
    __m256i second2 = _mm256_unpacklo_epi64(first1, first3); // a4 b4 c4 d4 a5 b5 c5 d5
    __m256i second3 = _mm256_unpackhi_epi64(first1, first3); // a6 b6 c6 d6 a7 b7 c7 d7
    __m256i second4 = _mm256_unpacklo_epi64(first4, first6); // e0 f0 g0 h0 e1 f1 g1 h1
    __m256i second5 = _mm256_unpackhi_epi64(first4, first6); // e2 f2 g2 h2 e3 f3 g3 h3
    __m256i second6 = _mm256_unpacklo_epi64(first5, first7); // e4 f4 g4 h4 e5 f5 g5 h5
    __m256i second7 = _mm256_unpackhi_epi64(first5, first7); // e6 f6 g6 h6 e7 f7 g7 h7

    col[0] = _mm256_permute2x128_si256(second0, second4, 0x20);
    col[1] = _mm256_permute2x128_si256(second1, second5, 0x20);
    col[2] = _mm256_permute2x128_si256(second2, second6, 0x20);
    col[3] = _mm256_permute2x128_si256(second3, second7, 0x20);
    col[4] = _mm256_permute2x128_si256(second0, second4, 0x31);
    col[5] = _mm256_permute2x128_si256(second1, second5, 0x31);
    col[6] = _mm256_permute2x128_si256(second2, second6, 0x31);
    col[7] = _mm256_permute2x128_si256(second3, second7, 0x31);
}
#endif

template<typename B_TYPE>
static void gemm_f32_ggml(const struct ggml_compute_params * params, int M, int N, int K,
                           const float* A, int lda, const B_TYPE* B_q, int ldb_q_unused,
                           float* C, int ldc);
static inline void quantize_block_q8_0(const float *x, float* y_d, int8_t* y_qs);
static void pack_A_q8_0_f32(int M, int K, const float* A, int lda, int8_t* A_qs_packed, float* A_d_packed);
static void pack_A_q8_0_f32(int M, int K, const float** A, int offset, int8_t* A_qs_packed, float* A_d_packed);

// B矩阵打包函数的模板声明
template<typename B_TYPE>
static void pack_B(int nc, int K, const B_TYPE* B_q, int ldb_q, int8_t* B_qs_packed, float* B_d_packed);


// 量化一个32个float的块为q8_0格式
static inline void quantize_block_q8_0(const float *x, float* y_d, int8_t* y_qs) {
#if defined(__ARM_NEON)
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
#elif defined(__AVX2__)
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps( x );
    __m256 v1 = _mm256_loadu_ps( x + 8 );
    __m256 v2 = _mm256_loadu_ps( x + 16 );
    __m256 v3 = _mm256_loadu_ps( x + 24 );

    // Compute max(abs(e)) for the block
    const __m256 signBit = _mm256_set1_ps( -0.0f );
    __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

    __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
    max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
    const float maxScalar = _mm_cvtss_f32( max4 );

    // Quantize these floats
    const float d = maxScalar / 127.f;
    *y_d = d;
    const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps( id );

    // Apply the multiplier
    v0 = _mm256_mul_ps( v0, mul );
    v1 = _mm256_mul_ps( v1, mul );
    v2 = _mm256_mul_ps( v2, mul );
    v3 = _mm256_mul_ps( v3, mul );

    // Round to nearest integer
    v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
    v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
    v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
    v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

    // Convert floats to integers
    __m256i i0 = _mm256_cvtps_epi32( v0 );
    __m256i i1 = _mm256_cvtps_epi32( v1 );
    __m256i i2 = _mm256_cvtps_epi32( v2 );
    __m256i i3 = _mm256_cvtps_epi32( v3 );

    // Convert int32 to int16
    i0 = _mm256_packs_epi32( i0, i1 );
    i2 = _mm256_packs_epi32( i2, i3 );
    i0 = _mm256_packs_epi16( i0, i2 );
    const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
    i0 = _mm256_permutevar8x32_epi32( i0, perm );
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y_qs), i0);
#endif
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

void pack_A_q8_0_f32(
    int M, int K,
    const float** A, int offset,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0; // 假设所有类型处理的块大小都与QK8_0兼容

    int8_t A_qs_packed_buf[MR * QK8_0];

    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < K_BLOCKS; ++j) {
            for (int row = 0; row < MR; ++row) {
                if (i + row < M) {
                    quantize_block_q8_0(A[i + row] + offset + j * QK8_0, &A_d_packed[j * MR + row], &A_qs_packed_buf[row * QK8_0]);
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
#if defined(__ARM_NEON)
    static_assert(NR == 4, "NR must be 4 for this implementation");
    const int K_BLOCKS = K / QK8_0;

    for (int j = 0; j < nc; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            int32x4x4_t b_buf0, b_buf1;
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);

                    b_buf0.val[col] = vld1q_s32(reinterpret_cast<const int32_t*>((B_q + (j + col) * ldb_q + k_block)->qs + 0));
                    b_buf1.val[col] = vld1q_s32(reinterpret_cast<const int32_t*>((B_q + (j + col) * ldb_q + k_block)->qs + 16));
                }
            }
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed), b_buf0);
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed + 16 * NR), b_buf1);
            B_qs_packed += QK8_0 * NR;
        }
    }
#elif defined(__AVX2__)
    static_assert(NR == 8, "NR must be 8 for this implementation");
    const int K_BLOCKS = K / QK8_0;
    for (int j = 0; j < nc; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            __m256i row[8];
            __m256i col[8];
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);
                    row[col] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((B_q + (j + col) * ldb_q + k_block)->qs));
                }
            }

            transpose_8x8_i32_avx2(row, col);

            for (int i = 0; i < NR; ++i) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(B_qs_packed + i * QK8_0), col[i]);
            }
            B_qs_packed += QK8_0 * NR;
        }
    }
#else
    static_assert(false, "pack_B<block_q8_0> is only supported on ARM NEON or x86 AVX2");
#endif
}

// 模板特化 for block_q4_0
template<>
void pack_B<block_q4_0>(
    int nc, int K, const block_q4_0* B_q, int ldb_q,
    int8_t* B_qs_packed, float* B_d_packed)
{
#if defined(__ARM_NEON)
    static_assert(NR == 4, "NR must be 4 for this implementation");
    // Q4_0和Q8_0有相同的块大小QK
    const int K_BLOCKS = K / QK4_0;

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(0x8);

    for (int j = 0; j < nc; j += NR) {
        for (int k_block = 0; k_block < K_BLOCKS; ++k_block) {
           int32x4x4_t b_buf0, b_buf1;
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);

                    // 解包4-bit到8-bit并减去8
                    const uint8x16_t v_u8 = vld1q_u8((B_q + (j + col) * ldb_q + k_block)->qs);
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
#elif defined(__AVX2__)
    static_assert(NR == 8, "NR must be 8 for this implementation");
    const int K_BLOCKS = K / QK4_0;

    for (int j = 0; j < nc; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            __m256i row[8];
            __m256i col[8];
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_CPU_FP16_TO_FP32((B_q + (j + col) * ldb_q + k_block)->d);
                    row[col] = to_int8_q4_0((B_q + (j + col) * ldb_q + k_block)->qs);
                }
            }

            transpose_8x8_i32_avx2(row, col);

            for (int i = 0; i < NR; ++i) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(B_qs_packed + i * QK8_0), col[i]);
            }
            B_qs_packed += QK8_0 * NR;
        }
    }
#else
    static_assert(false, "pack_B<block_q4_0> is only supported on ARM NEON or x86 AVX2");
#endif
}

template<>
void pack_B<block_mxfp4>(
    int nc, int K, const block_mxfp4* B_q, int ldb_q,
    int8_t* B_qs_packed, float* B_d_packed)
{
#if defined(__ARM_NEON)
    static_assert(NR == 4, "NR must be 4 for this implementation");
    // Q4_0和Q8_0有相同的块大小QK
    const int K_BLOCKS = K / QK_MXFP4;

    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t values = vld1q_s8(kvalues_mxfp4);

    for (int j = 0; j < nc; j += NR) {
        for (int k_block = 0; k_block < K_BLOCKS; ++k_block) {
           int32x4x4_t b_buf0, b_buf1;
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_E8M0_TO_FP32_HALF((B_q + (j + col) * ldb_q + k_block)->e);

                    const uint8x16_t v_u8 = vld1q_u8((B_q + (j + col) * ldb_q + k_block)->qs);
                    const int8x16_t v_i8_l = ggml_vqtbl1q_s8(values, vandq_u8  (v_u8, m4b));
                    const int8x16_t v_i8_h = ggml_vqtbl1q_s8(values, vshrq_n_u8(v_u8, 4));

                    b_buf0.val[col] = vreinterpretq_s32_s8(v_i8_l);
                    b_buf1.val[col] = vreinterpretq_s32_s8(v_i8_h);
                }
            }
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed), b_buf0);
            vst4q_s32(reinterpret_cast<int32_t*>(B_qs_packed + 16 * NR), b_buf1);
            B_qs_packed += QK_MXFP4 * NR;
        }
    }
#elif defined(__AVX2__)
    static_assert(NR == 8, "NR must be 8 for this implementation");
    const int K_BLOCKS = K / QK_MXFP4;

    for (int j = 0; j < nc; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            __m256i row[8];
            __m256i col[8];
            for (int col = 0; col < NR; ++col) {
                if (j + col < nc) {
                    *B_d_packed++ = GGML_E8M0_TO_FP32_HALF((B_q + (j + col) * ldb_q + k_block)->e);
                    row[col] = to_int8_mxfp4((B_q + (j + col) * ldb_q + k_block)->qs);
                }
            }

            transpose_8x8_i32_avx2(row, col);

            for (int i = 0; i < NR; ++i) {
                _mm256_store_si256(reinterpret_cast<__m256i*>(B_qs_packed + i * QK8_0), col[i]);
            }
            B_qs_packed += QK8_0 * NR;
        }
    }
#else
    static_assert(false, "pack_B<block_q4_0> is only supported on ARM NEON or x86 AVX2");
#endif
}


// 核心计算Kernel模板，针对不同的mr值进行特化以避免不必要的计算
template<int MR_T>
static void gemm_q8_0_microkernel_impl(
    int kc_size,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const float* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    // Static assert ensures MR_T is valid at compile time
    static_assert(MR_T > 0 && MR_T <= MR, "MR_T must be within (0, MR]");

#if defined(__ARM_NEON)
    const int KC_BLOCKS = kc_size / QK8_0;

    // Arrays are declared with MR elements to simplify indexing and pointer arithmetic
    // while loops are adjusted to MR_T to avoid redundant work.
    float32x4_t c_v[MR_T];
    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vld1q_f32(C + i * ldc);
    } else {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vdupq_n_f32(0.0f);
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const float* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR_T];
        for (int i = 0; i < MR_T; ++i) sum_v[i] = vdupq_n_s32(0);

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            int8x16_t a_vec_0 = vld1q_s8(a_ptr);
            a_ptr += 16;
            int8x16_t a_vec_1;
            if constexpr (MR_T > 4) {
                a_vec_1 = vld1q_s8(a_ptr);
            }
            a_ptr += 16; // a_ptr always advances by 32 bytes (2 * 16 bytes) per k4_step

            int8x16_t b_vec = vld1q_s8(b_ptr);
            b_ptr += 16;

            // Use if constexpr to conditionally compile dot product operations
            if constexpr (MR_T > 0) sum_v[0] = vdotq_laneq_s32(sum_v[0], b_vec, a_vec_0, 0);
            if constexpr (MR_T > 1) sum_v[1] = vdotq_laneq_s32(sum_v[1], b_vec, a_vec_0, 1);
            if constexpr (MR_T > 2) sum_v[2] = vdotq_laneq_s32(sum_v[2], b_vec, a_vec_0, 2);
            if constexpr (MR_T > 3) sum_v[3] = vdotq_laneq_s32(sum_v[3], b_vec, a_vec_0, 3);
            if constexpr (MR_T > 4) sum_v[4] = vdotq_laneq_s32(sum_v[4], b_vec, a_vec_1, 0);
            if constexpr (MR_T > 5) sum_v[5] = vdotq_laneq_s32(sum_v[5], b_vec, a_vec_1, 1);
            if constexpr (MR_T > 6) sum_v[6] = vdotq_laneq_s32(sum_v[6], b_vec, a_vec_1, 2);
            if constexpr (MR_T > 7) sum_v[7] = vdotq_laneq_s32(sum_v[7], b_vec, a_vec_1, 3);
        }

        const float32x4_t d_b_v = vld1q_f32(bd_ptr);
        bd_ptr += NR;

        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = vmlaq_n_f32(c_v[i], vmulq_f32(vcvtq_f32_s32(sum_v[i]), d_b_v), ad_ptr[i]);
        }

        ad_ptr += MR;
    }

    for (int i = 0; i < MR_T; ++i) {
        vst1q_f32(C + i * ldc, c_v[i]);
    }
#elif defined(__AVX2__)
    const int KC_BLOCKS = kc_size / QK8_0;

    __m256 c_v[MR_T];
    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_loadu_ps(C + i * ldc);
        }
    } else {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_setzero_ps();
        }
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const float* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        __m256i sum[MR];
        for (int i = 0; i < MR_T; ++i) sum[i] = _mm256_setzero_si256();

#if defined(__AVXVNNI__)
        __m256i sum_a_vec = _mm256_setzero_si256();

        for (int k = 0; k < QK8_0; k += 4) {
            __m256i b_vec = _mm256_load_si256((__m256i const*)b_ptr);
            b_vec = _mm256_sub_epi8(b_vec, _mm256_set1_epi8(-128)); // b_vec + 128
            b_ptr += NR * 4;
            __m256i a_vec = _mm256_load_si256((__m256i const*)a_ptr);
            sum_a_vec = _mm256_dpbusd_avx_epi32(sum_a_vec, _mm256_set1_epi8(1), a_vec);

            for (int i = 0; i < MR_T; ++i) {
                __m256i a_vec = _mm256_set1_epi32(*((const int*)(a_ptr + i * 4)));
                sum[i] = _mm256_dpbusd_avx_epi32(sum[i], b_vec, a_vec);
            }
            a_ptr += MR * 4; // Move to next set of A data
        }

        __m256 bd_vec = _mm256_loadu_ps(bd_ptr);
        bd_ptr += NR;

        union { float f[8]; __m256 v; } sum_a_vec_f;
        sum_a_vec_f.v = _mm256_cvtepi32_ps(sum_a_vec);
        for (int i = 0; i < MR_T; ++i) {
            __m256 sum_f = _mm256_fmadd_ps(_mm256_set1_ps(sum_a_vec_f.f[i]), _mm256_set1_ps(-128.0f), _mm256_cvtepi32_ps(sum[i]));
            c_v[i] = _mm256_fmadd_ps(_mm256_mul_ps(sum_f, _mm256_broadcast_ss(&ad_ptr[i])) , bd_vec, c_v[i]);
        }
#else
        for (int k = 0; k < QK8_0; k += 4) {
            __m256i b_vec = _mm256_load_si256((__m256i const*)b_ptr);
            b_ptr += NR * 4;

            __m256i b_vec_abs = _mm256_sign_epi8(b_vec, b_vec); // abs
            // Process each row of A
            for (int i = 0; i < MR_T; ++i) {
                __m256i a_vec = _mm256_set1_epi32(*((int*)(a_ptr + i * 4)));
                __m256i a_vec_sign_b = _mm256_sign_epi8(a_vec, b_vec);
                sum[i] += _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(b_vec_abs, a_vec_sign_b));
            }
            a_ptr += MR * 4; // Move to next set of A data
        }

        __m256 bd_vec = _mm256_loadu_ps(bd_ptr);
        bd_ptr += NR;

        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sum[i]), _mm256_broadcast_ss(&ad_ptr[i])) , bd_vec, c_v[i]);
        }
#endif

        ad_ptr += MR;
    }

    for (int i = 0; i < MR_T; ++i) { // Loop only up to MR_T
        _mm256_storeu_ps(C + i * ldc, c_v[i]);
    }
#else
    static_assert(false, "gemm_q8_0_microkernel_impl is only supported on ARM NEON or x86 AVX2");
#endif
}
// 核心计算Kernel (分发器)，根据mr调用相应的模板实例
static void gemm_q8_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const float* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    switch (mr) {
        case 1: gemm_q8_0_microkernel_impl<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_microkernel_impl<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_microkernel_impl<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_microkernel_impl<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 5: gemm_q8_0_microkernel_impl<5>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 6: gemm_q8_0_microkernel_impl<6>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 7: gemm_q8_0_microkernel_impl<7>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 8: gemm_q8_0_microkernel_impl<8>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
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
    const B_TYPE* B_q,
    float* C, int ldc)
{
    assert(K % QK8_0 == 0); // 假设所有支持的类型块大小都一样

    const int K_BLOCKS = K / QK8_0;

    constexpr int MC = 32;
    constexpr int KC = 1024;
    constexpr int NC = 32;

    const int M_CEIL = (M + MR - 1) / MR * MR;

    // 从 wdata 分配内存
    size_t a_qs_packed_size = GGML_PAD(M_CEIL * K * sizeof(int8_t), 64);
    size_t a_d_packed_size  = GGML_PAD(M_CEIL * K_BLOCKS * sizeof(float), 64);
    if (params->wsize < a_qs_packed_size + a_d_packed_size) {
        fprintf(stderr,"ggml_sgemm_q8_0_f32: wsize = %zu too small, need %zu\n", params->wsize, a_qs_packed_size + a_d_packed_size);
        return;
    }

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
                         gemm_q8_0_microkernel(
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

template<typename B_TYPE, int M_MAX>
static void gemm_f32_ggml_indirect(
    int M, int N, int K,
    const float** A,
    const B_TYPE* B_q,
    float** C)
{
    assert(K % QK8_0 == 0); // 假设所有支持的类型块大小都一样

    const int K_BLOCKS = K / QK8_0;

    constexpr int MC = 32;
    constexpr int KC = 1024;
    constexpr int NC = 32;

    int8_t A_qs_packed[M_MAX * KC] __attribute__((aligned(64)));
    float A_d_packed[M_MAX * KC / QK8_0] __attribute__((aligned(64)));

    int8_t B_qs_packed[KC * NC] __attribute__((aligned(64)));
    float B_d_packed[(KC / QK8_0) * NC] __attribute__((aligned(64)));

    for (int kc = 0; kc < K; kc += KC) {
        const int kc_size = std::min(KC, K - kc);
        const int kc_blocks = kc_size / QK8_0;
        const int k_block_offset = kc / QK8_0;
        pack_A_q8_0_f32(M, kc_size, A, kc, A_qs_packed, A_d_packed);

        for (int jc = 0; jc < N; jc += NC) {
            const int nc = std::min(NC, N - jc);
            pack_B<B_TYPE>(nc, kc_size, B_q + jc * K_BLOCKS + k_block_offset, K_BLOCKS, B_qs_packed, B_d_packed);

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                float C_buf[MR * NR] __attribute__((aligned(64)));
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < mc; ir += MR) {
                         gemm_q8_0_microkernel(
                            kc_size, std::min(MR, mc - ir),
                            A_qs_packed + (ic + ir) * kc_size,
                            A_d_packed + (ic + ir) * kc_blocks,
                            B_qs_packed + jr * kc_size,
                            B_d_packed + jr * kc_blocks,
                            C_buf, NR,
                            false
                        );
                        for (int i = 0; i < std::min(MR, mc - ir); ++i) {
                            for (int j = 0; j < std::min(NR, nc - jr); ++j) {
                                if (kc == 0) {
                                    C[ic + ir + i][jc + jr + j] = C_buf[i * NR + j];
                                } else {
                                    C[ic + ir + i][jc + jr + j] += C_buf[i * NR + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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

    if (m % NR != 0)
        return false;

    if (Btype == GGML_TYPE_F32) {
        // We can only handle cases where the second matrix is F32.
        // The first matrix (A) is the quantized one.
        switch (Atype) {
            case GGML_TYPE_Q8_0: {
                gemm_f32_ggml<block_q8_0>(params, n, m, k,
                                   (const float*)B, ldb,
                                   (const block_q8_0*)A,
                                   (float*)C, ldc);
                return true;
            }
            case GGML_TYPE_Q4_0: {
                // Note: The K dimension for the function is the number of float elements.
                // For Q4_0 and Q8_0, this is the same.
                gemm_f32_ggml<block_q4_0>(params, n, m, k,
                                   (const float*)B, ldb,
                                   (const block_q4_0*)A,
                                   (float*)C, ldc);
                return true;
            }
            case GGML_TYPE_MXFP4: {
                // Note: The K dimension for the function is the number of float elements.
                // For Q4_0 and Q8_0, this is the same.
                gemm_f32_ggml<block_mxfp4>(params, n, m, k,
                                   (const float*)B, ldb,
                                   (const block_mxfp4*)A,
                                   (float*)C, ldc);
                return true;
            }
            default:
                return false;
        }
    }

    (void) lda;

    return false;
}

struct MoETask {
    int expert_id;
    int start;
    int batchsize;
};

static inline void preprocess_ids(int n_expert, int n_expert_used, int n_tokens,
                    const int32_t* ids, int64_t ld_ids,
                    int* token_ids, int* expert_start, int* expert_counts) {
    memset(expert_counts, 0, n_expert * sizeof(int));
    memset(expert_start, 0, (n_expert + 1) * sizeof(int));

    // 计数每个 expert 的 token 数量
    for (int t = 0; t < n_tokens; ++t) {
        for (int e = 0; e < n_expert_used; ++e) {
            int expert_id = ids[t * ld_ids + e];
            if (expert_id >= 0 && expert_id < n_expert) {
                expert_counts[expert_id]++;
            }
        }
    }

    // 构建前缀和
    for (int i = 0; i < n_expert; ++i) {
        expert_start[i + 1] = expert_start[i] + expert_counts[i];
    }

    memset(expert_counts, 0, n_expert * sizeof(int));

    // 构建 token_ids
    for (int t = 0; t < n_tokens; ++t) {
        for (int e = 0; e < n_expert_used; ++e) {
            int expert_id = ids[t * ld_ids + e];
            if (expert_id >= 0 && expert_id < n_expert) {
                int pos = expert_start[expert_id] + expert_counts[expert_id]++;
                token_ids[pos] = t;
            }
        }
    }
}

static inline int build_tasks(int n_expert, const int* expert_start,
                int max_batch, MoETask* tasks) {
    int task_count = 0;
    for (int e = 0; e < n_expert; ++e) {
        int start = expert_start[e];
        int count = expert_start[e + 1] - expert_start[e];
        for (int i = 0; i < count; i += max_batch) {
            int batch_size = std::min(max_batch, count - i);
            tasks[task_count++] = {e, start + i, batch_size};
        }
    }
    return task_count;
}


/*
    c = ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    b   -> [cols, n_expert_used, n_tokens]
    ids -> [n_expert_used, n_tokens] (i32)
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_expert_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/

bool llamafile_sgemm_id(const struct ggml_compute_params * params, int cols, int rows, int n_expert, int n_expert_used, int n_tokens,
                        const void *as, const float *b, int64_t ldb, const int32_t*ids, int64_t ld_ids, float* c, int64_t ldc, int Atype, bool broadcastb)
{

    // printf("llamafile_sgemm_id: cols=%d, rows=%d, n_expert=%d, n_expert_used=%d, n_tokens=%d, Atype=%d\n broadcastb=%d\n",
    //        cols, rows, n_expert, n_expert_used, n_tokens, Atype, broadcastb);

    if (n_expert_used * n_tokens < n_expert) {
        return false;
    }

    if (!((Atype == GGML_TYPE_Q8_0) || Atype == GGML_TYPE_Q4_0 || Atype == GGML_TYPE_MXFP4)) {
        return false;
    }

    constexpr int MAX_BATCH = 64;
    
    const int ith = params->ith;
    const int nth = params->nth;

    // 共享内存结构
    const size_t shared_mem_size = params->wsize;
    char* shared_mem = (char*)params->wdata;

    // 分配共享内存
    MoETask* tasks = (MoETask*)shared_mem;
    const int max_task_num = n_expert + (n_tokens * n_expert_used + MAX_BATCH - 1) / MAX_BATCH;
    int* token_ids = (int*)(tasks + max_task_num);
    int* task_count_ptr = token_ids + n_expert_used * n_tokens;

    // 各部分内存大小计算
    size_t tasks_size = max_task_num * sizeof(MoETask);
    size_t token_ids_size = n_expert_used * n_tokens * sizeof(int);
    size_t task_count_size = 1 * sizeof(int);

    // 总空间
    size_t total_shared_mem_size = tasks_size + token_ids_size + task_count_size;
    if (total_shared_mem_size > shared_mem_size) {
        return -1;
    }

    if (ith == 0) {
        int expert_counts[n_expert];
        int expert_start[n_expert + 1];
        preprocess_ids(n_expert, n_expert_used, n_tokens, ids, ld_ids, token_ids, expert_start, expert_counts);
        int task_count = build_tasks(n_expert, expert_start, MAX_BATCH, tasks);
        *task_count_ptr = task_count;
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);

    int task_count = *task_count_ptr;
    int task_id = ith;
    while (task_id < task_count) {
        const MoETask& task = tasks[task_id];
        int expert_id = task.expert_id;
        int batch_size = task.batchsize;

        float* c_ptrs[MAX_BATCH];
        const float* b_ptrs[MAX_BATCH];

        for (int i = 0; i < batch_size; ++i) {
            int token_id = token_ids[task.start + i];
            int expert_idx = -1;
            for (int e = 0; e < n_expert_used; ++e) {
                if (ids[token_id * ld_ids + e] == expert_id) {
                    expert_idx = e;
                    break;
                }
            }
            assert(expert_idx != -1);

            if (broadcastb) {
                b_ptrs[i] = &b[token_id * ldb];
            } else {
                b_ptrs[i] = &b[(token_id * n_expert_used + expert_idx) * ldb];
            }
            c_ptrs[i] = &c[token_id * (ldc * n_expert_used) + expert_idx * ldc];
        }

        switch (Atype) {
            case GGML_TYPE_Q8_0:
                gemm_f32_ggml_indirect<block_q8_0, MAX_BATCH>(
                    batch_size, rows, cols,
                    (const float**)b_ptrs,
                    (const block_q8_0*)as + expert_id * rows * (cols / QK8_0),
                    (float**)c_ptrs
                );
                break;
            case GGML_TYPE_Q4_0:
                gemm_f32_ggml_indirect<block_q4_0, MAX_BATCH>(
                    batch_size, rows, cols,
                    (const float**)b_ptrs,
                    (const block_q4_0*)as + expert_id * rows * (cols / QK4_0),
                    (float**)c_ptrs
                );
                break;
            case GGML_TYPE_MXFP4:
                gemm_f32_ggml_indirect<block_mxfp4, MAX_BATCH>(
                    batch_size, rows, cols,
                    (const float**)b_ptrs,
                    (const block_mxfp4*)as + expert_id * rows * (cols / QK_MXFP4),
                    (float**)c_ptrs
                );
                break;
            default:
                assert(false);
        }
        task_id = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
    ggml_barrier(params->threadpool);
    return true;
}
