#define GGML_COMMON_IMPL_CPP
#include "tiny-blas.h"
#include "simd-mappings.h"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <omp.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define MR 8
#define NR 4
static inline void transpose_4x4(int32x4x4_t &b_buf) {
    // 第一步：对行进行交错操作
    int32x4x2_t tmp0 = vtrnq_s32(b_buf.val[0], b_buf.val[1]); // 处理第0行和第1行
    int32x4x2_t tmp1 = vtrnq_s32(b_buf.val[2], b_buf.val[3]); // 处理第2行和第3行
    
    // 第二步：重新排列结果
    // tmp0.val[0] = [a00, a10, a02, a12]
    // tmp0.val[1] = [a01, a11, a03, a13]
    // tmp1.val[0] = [a20, a30, a22, a32]
    // tmp1.val[1] = [a21, a31, a23, a33]
    
    // 提取并重新组合
    int32x4_t row0 = vcombine_s32(vget_low_s32(tmp0.val[0]), vget_low_s32(tmp1.val[0]));
    int32x4_t row1 = vcombine_s32(vget_low_s32(tmp0.val[1]), vget_low_s32(tmp1.val[1]));
    int32x4_t row2 = vcombine_s32(vget_high_s32(tmp0.val[0]), vget_high_s32(tmp1.val[0]));
    int32x4_t row3 = vcombine_s32(vget_high_s32(tmp0.val[1]), vget_high_s32(tmp1.val[1]));
    
    b_buf.val[0] = row0;
    b_buf.val[1] = row1;
    b_buf.val[2] = row2;
    b_buf.val[3] = row3;
}

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
    const __m128i tmp_values = _mm_loadu_si128( (const __m128i*)kvalues_mxfp4);
    const __m256i values128 = _mm256_set_m128i(tmp_values, tmp_values);
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i m4b  = _mm256_set1_epi8(0x0f);
    return _mm256_shuffle_epi8(values128, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp), m4b));
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

void pack_B_q8_0(
    int N, int K, const block_q8_0* B_q, int ldb_q,
    int8_t* B_qs_packed, ggml_half* B_d_packed)
{
   const int K_BLOCKS = K / QK8_0;
    for (int j = 0; j < N; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    *B_d_packed++ = (B_q + (j + col) * ldb_q + k_block)->d;
                }
            }
        }

        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int k_rem = 0; k_rem < QK8_0; k_rem += 4) {
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        memcpy(B_qs_packed, (B_q + (j + col) * ldb_q + k_block)->qs + k_rem, 4);
                    }
                    B_qs_packed += 4;
                }
            }
        }
    }
}

void pack_B_q8_0_reverse(
    int N, int K, block_q8_0* B_q, int ldb_q,
    const int8_t* B_qs_packed, const ggml_half* B_d_packed)
{
   const int K_BLOCKS = K / QK8_0;
    for (int j = 0; j < N; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    (B_q + (j + col) * ldb_q + k_block)->d = *B_d_packed++;
                }
            }
        }

        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int k_rem = 0; k_rem < QK8_0; k_rem += 4) {
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        memcpy((B_q + (j + col) * ldb_q + k_block)->qs + k_rem, B_qs_packed, 4);
                    }
                    B_qs_packed += 4;
                }
            }
        }
    }
}

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
static void pack_A_q8_0_f32(
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

static void pack_A_q8_0_f32(
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

// B矩阵打包函数的模板声明
template<typename B_TYPE>
static void pack_B(int nc, int K, const B_TYPE* B_q, int ldb_q, int8_t* B_qs_packed, float* B_d_packed);

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

const size_t ALIGNMENT = 64; // 64 bytes for AVX-512 or general good practice

// --- 线程局部缓冲区管理 (简单的内存池) ---
class ThreadLocalBufferArena {
private:
    size_t current_offset;
    float* base_ptr;
    size_t total_arena_size;

public:
    ThreadLocalBufferArena(size_t total_arena_size) : current_offset(0), base_ptr(nullptr), total_arena_size(total_arena_size) {
        base_ptr = (float*)aligned_alloc(ALIGNMENT, total_arena_size);
        if (!base_ptr) {
            exit(1);
        }
    }

    ~ThreadLocalBufferArena() {
        if (base_ptr) {
            free(base_ptr);
        }
    }

    // Allocate aligned memory from the arena
    void* allocate(size_t size) {
        size_t aligned_size = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        if (current_offset + aligned_size > total_arena_size) {
            std::cerr << "Thread " << omp_get_thread_num() << ": Arena out of memory! Requested " 
                      << aligned_size << ", available " << (total_arena_size - current_offset) << std::endl;
            exit(1);
        }
        float* ptr = (float*)((char*)base_ptr + current_offset);
        current_offset += aligned_size;
        return ptr;
    }

    // Reset the arena for reuse (don't actually free, just reset offset)
    void reset() {
        current_offset = 0;
    }
};

static thread_local ThreadLocalBufferArena tls_arena(256 * 1024);

// 核心计算Kernel模板，针对不同的mr值进行特化以避免不必要的计算

// Template specialized microkernel implementation
template <int MR_T, typename B_SCALE_TYPE>
static void gemm_q8_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed,
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
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

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

        float32x4_t d_b_v;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            d_b_v = vld1q_f32(bd_ptr);
        } else {
            d_b_v = vcvt_f32_f16(vld1_f16((const __fp16 *) bd_ptr));
        }
        bd_ptr += NR;

        float32x4_t d_a_v0, d_a_v1;
        d_a_v0 = vld1q_f32(ad_ptr);
        if constexpr (MR_T > 4) d_a_v1 = vld1q_f32(ad_ptr + 4);
        ad_ptr += MR;

        if constexpr (MR_T > 0) c_v[0] = vmlaq_laneq_f32(c_v[0], vmulq_f32(vcvtq_f32_s32(sum_v[0]), d_b_v), d_a_v0, 0);
        if constexpr (MR_T > 1) c_v[1] = vmlaq_laneq_f32(c_v[1], vmulq_f32(vcvtq_f32_s32(sum_v[1]), d_b_v), d_a_v0, 1);
        if constexpr (MR_T > 2) c_v[2] = vmlaq_laneq_f32(c_v[2], vmulq_f32(vcvtq_f32_s32(sum_v[2]), d_b_v), d_a_v0, 2);
        if constexpr (MR_T > 3) c_v[3] = vmlaq_laneq_f32(c_v[3], vmulq_f32(vcvtq_f32_s32(sum_v[3]), d_b_v), d_a_v0, 3);

        if constexpr (MR_T > 4) c_v[4] = vmlaq_laneq_f32(c_v[4], vmulq_f32(vcvtq_f32_s32(sum_v[4]), d_b_v), d_a_v1, 0);
        if constexpr (MR_T > 5) c_v[5] = vmlaq_laneq_f32(c_v[5], vmulq_f32(vcvtq_f32_s32(sum_v[5]), d_b_v), d_a_v1, 1);
        if constexpr (MR_T > 6) c_v[6] = vmlaq_laneq_f32(c_v[6], vmulq_f32(vcvtq_f32_s32(sum_v[6]), d_b_v), d_a_v1, 2);
        if constexpr (MR_T > 7) c_v[7] = vmlaq_laneq_f32(c_v[7], vmulq_f32(vcvtq_f32_s32(sum_v[7]), d_b_v), d_a_v1, 3);
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
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

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

        __m256 bd_vec;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            bd_vec = _mm256_loadu_ps(bd_ptr);
        } else {
            bd_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*) bd_ptr));
        }
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
    // Error case if neither NEON nor AVX2 is defined
    std::cerr << "Error: gemm_q8_0_microkernel_specialized requires either ARM NEON or AVX2." << std::endl;
    exit(1);
#endif
}

template<typename B_SCALE_TYPE>
static void gemm_q8_0_microkernel(
    int kc_size, int mr, // 'mr' is the runtime value
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    // Ensure mr is within the valid range for the defined MR (max micro-rows)
    assert(mr > 0 && mr <= MR && "mr value out of bounds for microkernel dispatch");

    // Dispatch to the specialized template function
    switch (mr) {
        case 1: gemm_q8_0_microkernel_specialized<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_microkernel_specialized<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_microkernel_specialized<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_microkernel_specialized<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
#if (defined(__AVX2__) && MR == 8) || (defined(__ARM_NEON) && MR == 8) // MR is 8 for both architectures in the provided code
        case 5: gemm_q8_0_microkernel_specialized<5>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 6: gemm_q8_0_microkernel_specialized<6>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 7: gemm_q8_0_microkernel_specialized<7>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 8: gemm_q8_0_microkernel_specialized<8>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
#endif
        default:
            // Fallback for unsupported mr values (should ideally be caught by assert)
            std::cerr << "Error: Unsupported mr value (" << mr << ") for gemm_q8_0_microkernel dispatch." << std::endl;
            exit(1);
    }
}


// =================================================================================================
// 主计算函数 (模板化)
// =================================================================================================
template<typename B_TYPE>
static void gemm_f32_ggml(
    int M, int N, int K,
    const float* A, int lda,
    const B_TYPE* B_q,
    float* C, int ldc, void* wdata)
{
    assert(K % QK8_0 == 0); // 假设所有支持的类型块大小都一样

    const int K_BLOCKS = K / QK8_0;

    constexpr int MC = 32;
    constexpr int KC = 1024;
    constexpr int NC = 32;

    const int M_CEIL = (M + MR - 1) / MR * MR;

    // 从 wdata 分配内存
    size_t a_qs_packed_size = GGML_PAD(M_CEIL * K * sizeof(int8_t), 64);
    //size_t a_d_packed_size  = GGML_PAD(M_CEIL * K_BLOCKS * sizeof(float), 64);

    int8_t* A_qs_packed = (int8_t*)wdata;
    float* A_d_packed = (float*)((char*) wdata + a_qs_packed_size);


    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < M; i += MR) {
            for (int j = 0; j < K_BLOCKS; ++j) {
                int8_t a_qs_buf[MR * QK8_0] __attribute__((aligned(64)));
                int M_rem = std::min(MR, M - i);
                float* A_d_packed_ptr = A_d_packed + i * K_BLOCKS;
                int8_t* A_qs_packed_ptr = A_qs_packed + i * K;
                float* current_A_d_ptr = A_d_packed_ptr + j * MR;
                for (int row = 0; row < M_rem; ++row) {
                    quantize_block_q8_0(A + (i + row) * lda + j * QK8_0, &current_A_d_ptr[row], &a_qs_buf[row * QK8_0]);
                }
                int8_t* current_qs_ptr = A_qs_packed_ptr + j * QK8_0 * MR;
                for (int k = 0; k < QK8_0; k += 4) {
                    for (int row = 0; row < MR; ++row) {
                        memcpy(current_qs_ptr, &a_qs_buf[row * QK8_0 + k], 4);
                        current_qs_ptr += 4;
                    }
                }
            }
        }

        #pragma omp barrier

        int8_t* B_qs_packed = (int8_t*) tls_arena.allocate(NC * KC * sizeof(int8_t));
        float* B_d_packed = (float*) tls_arena.allocate( KC * NC / QK8_0 * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            const int nc = std::min(NC, N - jc);

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
        }
        tls_arena.reset();
    }
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

    int8_t* A_qs_packed = (int8_t*) tls_arena.allocate(M_MAX * KC * sizeof(int8_t));
    float* A_d_packed = (float*) tls_arena.allocate(M_MAX * KC / QK8_0 * sizeof(float));

    int8_t* B_qs_packed = (int8_t*) tls_arena.allocate(NC * KC * sizeof(int8_t));
    float* B_d_packed = (float*) tls_arena.allocate( KC * NC / QK8_0 * sizeof(float));

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
    tls_arena.reset();
}

bool tiny_sgemm(int64_t m, int64_t n, int64_t k,
                const void *A, int64_t lda, const float* B, int64_t ldb,
                float *C, int64_t ldc, void* wdata, int Atype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
 
    if (m % NR != 0)
        return false;

    switch (Atype) {
        case GGML_TYPE_Q8_0: {
            gemm_f32_ggml<block_q8_0>(n, m, k,
                                B, ldb,
                                (const block_q8_0*)A,
                                C, ldc, wdata);
            return true;
        }
        case GGML_TYPE_Q4_0: {
            gemm_f32_ggml<block_q4_0>(n, m, k,
                                B, ldb,
                                (const block_q4_0*)A,
                                C, ldc, wdata);
            return true;
        }
        case GGML_TYPE_MXFP4: {
            gemm_f32_ggml<block_mxfp4>(n, m, k,
                                B, ldb,
                                (const block_mxfp4*)A,
                                C, ldc, wdata);
            return true;
        }
        default:
            return false;
    }
    (void) lda;

    return false;
}

static inline void preprocess_ids(int n_expert, int n_expert_used, int n_tokens,
                    const int32_t* ids, int64_t ld_ids,
                    TokenInfo* token_ids, int* expert_start, int* expert_counts) {
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
                token_ids[pos] = {t, e};
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

bool tiny_moe_sgemm(int cols, int rows, int n_expert, int n_expert_used, int n_tokens,
                    const void *as, const float *b, int64_t ldb, const int32_t*ids, int64_t ld_ids, float* c, int64_t ldc, void* wdata, int Atype, bool broadcastb)
{

    // printf("tiny_sgemm_id: cols=%d, rows=%d, n_expert=%d, n_expert_used=%d, n_tokens=%d, Atype=%d\n broadcastb=%d\n",
    //        cols, rows, n_expert, n_expert_used, n_tokens, Atype, broadcastb);

    constexpr int MAX_BATCH = 32;

    // 共享内存结构
    char* shared_mem = (char*)wdata;

    // 分配共享内存
    MoETask* tasks = (MoETask*)shared_mem;
    const int max_task_num = n_expert + (n_tokens * n_expert_used + MAX_BATCH - 1) / MAX_BATCH;
    TokenInfo* token_ids = (TokenInfo*)(tasks + max_task_num);

    std::vector<int> expert_counts(n_expert, 0);
    std::vector<int> expert_start(n_expert + 1, 0);
    preprocess_ids(n_expert, n_expert_used, n_tokens, ids, ld_ids, token_ids, expert_start.data(), expert_counts.data());
    int task_count = build_tasks(n_expert, expert_start.data(), MAX_BATCH, tasks);

    #pragma omp parallel for schedule(dynamic)
    for (int task_id = 0; task_id < task_count; task_id++) {
        const MoETask& task = tasks[task_id];
        int expert_id = task.expert_id;
        int batch_size = task.batchsize;

        float* c_ptrs[MAX_BATCH];
        const float* b_ptrs[MAX_BATCH];

        for (int i = 0; i < batch_size; ++i) {
            int token_id = token_ids[task.start + i].token_id;
            int expert_idx = token_ids[task.start + i].expert_idx; 
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
    }
    return true;
}

void gemm_q8_0_gotoblas_omp_packed(
    int M, int N, int K,
    const float* A,
    const int8_t* B_qs_packed,
    const ggml_half* B_d_packed_f16,
    float* C, void* wdata)
{
    assert(K % QK8_0 == 0);
    const int K_BLOCKS = K / QK8_0;

    const int M_CEIL = (M + MR - 1) / MR * MR;

    // 从 wdata 分配内存
    size_t a_qs_packed_size = GGML_PAD(M_CEIL * K * sizeof(int8_t), 64);
    //size_t a_d_packed_size  = GGML_PAD(M_CEIL * K_BLOCKS * sizeof(float), 64);

    int8_t* A_qs_packed = (int8_t*)wdata;
    float* A_d_packed = (float*)((char*) wdata + a_qs_packed_size);

    // =================================================================
    // 常规路径: 当 M > MR 时 (原始代码逻辑)
    // =================================================================
    assert(N % NR == 0); // 常规路径假定N是NR的倍数

    const int MC = 32;
    const int KC = 1024;
    const int NC = 32;

    #pragma omp parallel
    {
        // 阶段 1: 并行化 pack_A
        #pragma omp for collapse(2)
        for (int i = 0; i < M; i += MR) {
            for (int j = 0; j < K_BLOCKS; ++j) {
                int8_t a_qs_buf[MR * QK8_0] __attribute__((aligned(64)));
                int M_rem = std::min(MR, M - i);
                float* A_d_packed_ptr = A_d_packed + i * K_BLOCKS;
                int8_t* A_qs_packed_ptr = A_qs_packed + i * K;
                float* current_A_d_ptr = A_d_packed_ptr + j * MR;
                for (int row = 0; row < M_rem; ++row) {
                    quantize_block_q8_0(A + (i + row) * K + j * QK8_0, &current_A_d_ptr[row], &a_qs_buf[row * QK8_0]);
                }
                int8_t* current_qs_ptr = A_qs_packed_ptr + j * QK8_0 * MR;
                for (int k = 0; k < QK8_0; k += 4) {
                    for (int row = 0; row < MR; ++row) {
                        memcpy(current_qs_ptr, &a_qs_buf[row * QK8_0 + k], 4);
                        current_qs_ptr += 4;
                    }
                }
            }
        }

        // 阶段 2: 并行化 GEMM 计算

        #pragma omp for schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            const int nc = std::min(NC, N - jc);

            for (int kc = 0; kc < K; kc += KC) {
                const int kc_size = std::min(KC, K - kc);
                const int k_block_offset = kc / QK8_0;

                for (int ic = 0; ic < M; ic += MC) {
                    const int mc = std::min(MC, M - ic);

                    for (int jr = 0; jr < nc; jr += NR) {
                        for (int ir = 0; ir < mc; ir += MR) {
                            gemm_q8_0_microkernel(
                                kc_size, std::min(MR, mc - ir),
                                A_qs_packed + (ic + ir) * K + kc * MR,
                                A_d_packed + (ic + ir) * K_BLOCKS + k_block_offset * MR,
                                B_qs_packed + (jc + jr) * K + kc * NR,
                                B_d_packed_f16 + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                                C + (ic + ir) * N + (jc + jr), N,
                                kc != 0
                            );
                        }
                    }
                }
            }
        }
    }
 }