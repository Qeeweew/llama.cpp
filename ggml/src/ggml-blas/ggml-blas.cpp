#include "ggml-impl.h"
#include "ggml-blas.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "omp.h"
#include "tiny-blas.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <future>
#include <vector>
#include <cstring>

struct ggml_backend_blas_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
};

static void ggml_backend_blas_mul_mat(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    constexpr int MAX_ALIGN_N = 16;
    const int64_t ne_plane      = ((ne01 + MAX_ALIGN_N - 1) / MAX_ALIGN_N * MAX_ALIGN_N * ne00) / 32;
    const size_t  desired_wsize = ne_plane*(sizeof(float) + 32 * sizeof(int8_t));

    if (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(new char[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    void * wdata = ctx->work_data.get();

    omp_set_num_threads(ctx->n_threads);

    if (*((enum repack*)(src0->extra)) == Q8_0) {
        for (int64_t i13 = 0; i13 < ne13; i13++) {
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                const int8_t* B_qs_packed = (const int8_t *) ((const char *)src0->data + i12/r2*nb02 + i13/r3*nb03);
                const ggml_half* B_d_packed = (const ggml_half*) &B_qs_packed[ne00 * ne01];
                const char* src1_ptr = (const char*) src1->data + i12*nb12 + i13*nb13;
                gemm_q8_0_gotoblas_omp_packed(ne11, ne01, ne00, 
                    (const float*) src1_ptr, B_qs_packed, B_d_packed, 
                    reinterpret_cast<float*>((char *)dst->data + i12*nb2 + i13*nb3), wdata);
            }
        }
        return;
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
            for (int64_t i12 = 0; i12 < ne12; i12++) {
                tiny_sgemm(ne01, ne11, ne00,
                           (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                           nb01/ggml_type_size(src0->type),
                           reinterpret_cast<const float *>((const char *)src1->data + i12*nb12 + i13*nb13),
                           nb11/ggml_type_size(src1->type),
                           reinterpret_cast<float*>((char *)dst->data + i12*nb2 + i13*nb3),
                           nb1/ggml_type_size(dst->type),
                           wdata,src0->type);
        }
    }
}

static void ggml_backend_blas_mul_mat_id(ggml_backend_blas_context * ctx, struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS


    const enum ggml_type type = src0->type;

    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_expert_used = ids->ne[0]; // n_expert_used
    const int n_expert = ne02;            // n_expert
    const int n_tokens = ne12;
    constexpr int MAX_BATCH = 32;

    const int max_task_num = n_expert + (n_tokens * n_expert_used + MAX_BATCH - 1) / MAX_BATCH;
    size_t tasks_size = max_task_num * sizeof(MoETask);
    size_t token_ids_size = n_expert_used * n_tokens * sizeof(TokenInfo);

    // // 总空间
    size_t desired_wsize = tasks_size + token_ids_size;

    if (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(new char[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    void * wdata = ctx->work_data.get();

    // printf("ne10 = %lld, ne11 = %lld, ne12 = %lld, ne13 = %lld\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    // printf("nb10 = %lld, nb11 = %lld, nb12 = %lld, nb13 = %lld\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);

    if (src1->type == GGML_TYPE_F32) {
        tiny_moe_sgemm(
            src0->ne[0], src0->ne[1], n_expert, n_expert_used, n_tokens,
            src0->data, (const float*) src1->data, (int64_t) src1->nb[1] / 4, (const int32_t*) ids->data, (int64_t) ids->nb[1] / 4, (float*) dst->data, dst->nb[1] / 4,
            wdata,
            type, src1->ne[1] == 1);
    }

}

// backend interface

static const char * ggml_backend_blas_get_name(ggml_backend_t backend) {
    return "BLAS";

    GGML_UNUSED(backend);
}

static void ggml_backend_blas_free(ggml_backend_t backend) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_blas_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_blas_mul_mat(ctx, node);
                break;
            case GGML_OP_MUL_MAT_ID:
                ggml_backend_blas_mul_mat_id(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_blas_get_name,
    /* .free                    = */ ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_blas_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_blas_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_blas_init(void) {
    ggml_backend_blas_context * ctx = new ggml_backend_blas_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_blas_guid());
}

void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));

    ggml_backend_blas_context * ctx = (ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_blas_device_get_name(ggml_backend_dev_t dev) {
    return "BLAS";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_blas_device_get_description(ggml_backend_dev_t dev) {
    return "BLAS";

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_blas_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_blas_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_blas_device_get_name(dev);
    props->description = ggml_backend_blas_device_get_description(dev);
    props->type        = ggml_backend_blas_device_get_type(dev);
    ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_blas_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_blas_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}



static void * ggml_backend_blas_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_blas_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_blas_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_blas_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);

    static enum repack r_none = None;
    static enum repack r_q8_0 = Q8_0;

    if (offset == 0 && tensor->type == GGML_TYPE_Q8_0 && tensor->ne[2] == 1 && tensor->ne[3] == 1 && tensor->ne[0] % 8 == 0) {
        const int K = tensor->ne[0];
         const int N = tensor->ne[1];
        int8_t* B_qs_packed = (int8_t*) tensor->data;
        ggml_half* B_d_packed = (ggml_half*) &B_qs_packed[N * K];

        // printf("repack %d %d\n", N, K);
        pack_B_q8_0(N, K, (const block_q8_0 *)data, tensor->nb[1] / sizeof(block_q8_0), B_qs_packed, B_d_packed);
        tensor->extra = (void*) &r_q8_0;
    } else {

        memcpy((char *)tensor->data + offset, data, size);
        tensor->extra = (void*) &r_none;
    }


    GGML_UNUSED(buffer);
}

static void ggml_backend_blas_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    enum repack *r = (enum repack*)tensor->extra;
    if (r && *r == Q8_0) {
        const int K = tensor->ne[0];
         const int N = tensor->ne[1];
        int8_t* B_qs_packed = (int8_t*) tensor->data;
        ggml_half* B_d_packed = (ggml_half*) &B_qs_packed[N * K];
        // printf("reversed %d %d\n", N, K);
        pack_B_q8_0_reverse(N, K, (block_q8_0 *)data, tensor->nb[1] / sizeof(block_q8_0), B_qs_packed, B_d_packed);
    } else {
        memcpy(data, (const char *)tensor->data + offset, size);
    }
    GGML_UNUSED(buffer);
}

static bool ggml_backend_blas_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_blas_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_blas_buffer_i = {
    /* .free_buffer     = */ ggml_backend_blas_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_blas_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_blas_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_blas_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_blas_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_blas_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_blas_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_blas_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_blas_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_blas_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_blas_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_blas_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_blas_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_blas_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_blas_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "BLAS";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_blas_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_blas_buffer_i, data, size);
}

static size_t ggml_backend_blas_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_blas_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_blas_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_blas_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_blas_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_blas_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_blas_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_blas_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_blas_buffer_type;
}

static const char * ggml_backend_blas_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "BLAS_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_blas_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_blas_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_blas_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_blas_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_blas_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_blas_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_blas_buffer_type;
}

static ggml_backend_buffer_t ggml_backend_blas_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_blas_buffer_from_ptr_type(), ggml_backend_blas_buffer_from_ptr_i, ptr, size);
}

static ggml_backend_buffer_t ggml_backend_blas_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_blas_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static ggml_backend_buffer_type_t ggml_backend_blas_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_blas_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const int64_t ne1 = op->ne[1];

            const int64_t min_batch = 1;

            return ggml_is_contiguous(src0) &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne1 >= min_batch) &&
                   (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_MXFP4);
        }

        case GGML_OP_MUL_MAT_ID:
        {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            return (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_MXFP4) && src1->type == GGML_TYPE_F32;
        }

        default:
            return false;

    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_blas_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_blas_device_i = {
    /* .get_name             = */ ggml_backend_blas_device_get_name,
    /* .get_description      = */ ggml_backend_blas_device_get_description,
    /* .get_memory           = */ ggml_backend_blas_device_get_memory,
    /* .get_type             = */ ggml_backend_blas_device_get_type,
    /* .get_props            = */ ggml_backend_blas_device_get_props,
    /* .init_backend         = */ ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_blas_reg_get_name(ggml_backend_reg_t reg) {
    return "BLAS";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_blas_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_blas_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_blas_device = {
        /* .iface   = */ ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_blas_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_blas_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_blas_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_blas_reg_i = {
    /* .get_name         = */ ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ ggml_backend_blas_reg_get_device_count,
    /* .get_device       = */ ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ ggml_backend_blas_get_proc_address,
};

ggml_backend_reg_t ggml_backend_blas_reg(void) {
    static struct ggml_backend_reg ggml_backend_blas_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_blas_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
