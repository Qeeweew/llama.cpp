#pragma once
#include <stdint.h>
#include <stdbool.h>

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);

bool llamafile_sgemm_id(const struct ggml_compute_params * params, int cols, int rows, int n_expert, int n_expert_used, int n_tokens,
                        const void *as, const float *b, int64_t ldb, const int32_t*ids, int64_t ld_ids, float* c, int64_t ldc, int Atype, bool broadcastb);

#ifdef __cplusplus
}
#endif
