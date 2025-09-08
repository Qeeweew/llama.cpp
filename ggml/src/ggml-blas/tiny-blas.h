#pragma once
#include <stdint.h>
#include <stdbool.h>

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

#ifdef __cplusplus
}
#endif