#pragma once
#include <stdint.h>
#include <stdbool.h>

#include "ggml-common.h"
#include "ggml-quants.h"
#ifdef __cplusplus
extern "C" {
#endif

bool tiny_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const float* B, int64_t ldb, float *C, int64_t ldc, void* wdata, int Atype);

struct MoETask {
    int expert_id;
    int start;
    int batchsize;
};

struct TokenInfo {
    int token_id;
    int expert_idx;
};

bool tiny_moe_sgemm(int cols, int rows, int n_expert, int n_expert_used, int n_tokens,
                        const void *as, const float *b, int64_t ldb, const int32_t*ids, int64_t ld_ids, float* c, int64_t ldc, void* wdata, int Atype, bool broadcastb);

enum repack : uint8_t { None, Q8_0, Q4_0};

void pack_B_q8_0(int N, int K, const block_q8_0* B_q, int ldb_q, int8_t* B_qs_packed, ggml_half* B_d_packed);

void pack_B_q8_0_reverse(int N, int K, block_q8_0* B_q, int ldb_q, const int8_t* B_qs_packed, const ggml_half* B_d_packed);

void gemm_q8_0_gotoblas_omp_packed(int M, int N, int K, const float* A, const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16, float* C, void* wdata);

#ifdef __cplusplus
}
#endif