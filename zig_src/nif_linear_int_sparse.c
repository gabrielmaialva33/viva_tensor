/**
 * nif_linear_int_sparse.c — INT8 / INT4 2:4 sparse linear forward pass.
 *
 * Exposes:
 *   nt_linear_int8_sparse(InputBin, PackedWeight, BiasBin | nil) -> {ok, OutputBin}
 *   nt_linear_int4_sparse(InputBin, PackedWeight, BiasBin | nil) -> {ok, OutputBin}
 *
 * Input/output format: FP16 host binaries, row-major.
 *   InputBin shape = [B, in_features], FP16 (2 bytes per element).
 *   OutputBin shape = [B, out_features], FP16.
 *   BiasBin (optional): FP16 [out_features].
 *
 * Compute strategy:
 *   INT8: quantize input to INT8 dynamically (per-token absmax), GEMM via
 *         cuSPARSELt (if plan cached) or CUTLASS sparse cfg=28 fallback,
 *         then dequant + bias add on device.
 *   INT4: quantize input to INT4 dynamically, call CUTLASS sparse INT4 run.
 *
 * Dependencies (provided by sibling files):
 *   - cuda_int_sparse_run.cu: cutlass_int8_sparse_run / cutlass_int4_sparse_run
 *   - nif_packed_weight.c (agent A): PackedWeight type & helpers.
 */

#include <stdalign.h>
#include "viva_nif.h"

#ifndef _WIN32
#include <cuda_runtime.h>
#include <cusparseLt.h>
#endif

/* Pull in the same PackedWeight definition the prepack file uses.
 * Both files include the same guarded typedef; agent A's nif_packed_weight.c
 * provides the canonical version. */
#ifndef VIVA_PACKED_WEIGHT_DEFINED
#define VIVA_PACKED_WEIGHT_DEFINED

typedef enum {
    PW_FP8 = 0,
    PW_INT8_SPARSE = 1,
    PW_INT4_SPARSE = 2,
} PackedWeightDType;

typedef struct {
    PackedWeightDType dtype;
    void *d_weight;
    size_t weight_bytes;
    void *d_metadata;
    size_t metadata_bytes;
    void *d_scales;
    size_t scales_count;
    int in_features;
    int out_features;
    void *cusparselt_plan;
    void *d_compressed;
} PackedWeight;
#endif

extern ErlNifResourceType *PACKED_WEIGHT_RES;

/* CUTLASS run entrypoints from cuda_int_sparse_run.cu */
#ifndef _WIN32
extern int cutlass_int8_sparse_run(int M, int N, int K, const int8_t *d_A_packed, const int8_t *d_B,
                                   int32_t *d_C, const void *d_E, void *workspace,
                                   size_t workspace_size);
extern size_t cutlass_int8_sparse_workspace_size(int M, int N, int K);

extern int cutlass_int4_sparse_run(int M, int N, int K, const void *d_A_packed, const void *d_B,
                                   int32_t *d_C, const void *d_E, void *workspace,
                                   size_t workspace_size);
extern size_t cutlass_int4_sparse_workspace_size(int M, int N, int K);
#endif

/* =========================================================================
 * Dequant + bias kernel — INT32 [B, out_features] -> FP16 [B, out_features]
 *
 * This is a CPU-side reference implementation. A production version would
 * launch a tiny CUDA kernel on the device.
 *
 * For prepack with per-output-channel scale s_o and per-token input scale
 * s_t, the dequantization is:
 *
 *   out_fp16[b, o] = (acc_int32[b, o] * s_o * s_t[b]) + bias_fp16[o]
 *
 * Note we currently dequant on host after a single D2H copy of the int32
 * output (a stopgap — not the production design). The TODO in the report
 * tracks moving this to a fused on-device kernel.
 * ========================================================================= */

#ifndef _WIN32
/* CPU-side dequant + optional bias, then FP32->FP16 conversion. */
static void dequant_to_fp16(const int32_t *h_acc,        /* [B, out_f] */
                            const float *h_scales_o,     /* [out_f] per-output-channel */
                            const float *h_scales_t,     /* [B] per-token input scale */
                            const uint16_t *h_bias_fp16, /* may be NULL */
                            int B, int out_f, uint16_t *h_out_fp16) {
    for (int b = 0; b < B; ++b) {
        float st = h_scales_t[b];
        for (int o = 0; o < out_f; ++o) {
            float val = (float)h_acc[(size_t)b * (size_t)out_f + o] * h_scales_o[o] * st;
            if (h_bias_fp16) {
                val += f16_to_f32(h_bias_fp16[o]);
            }
            h_out_fp16[(size_t)b * (size_t)out_f + o] = float_to_half(val);
        }
    }
}
#endif

/* =========================================================================
 * Common preamble — parse [InputBin, PackedWeight, Bias | nil], allocate
 * scratch, dequantize input to FP32 host-side.
 * ========================================================================= */

#ifndef _WIN32
typedef struct {
    int B;
    int in_f;
    int out_f;
    const uint16_t *h_input_fp16;
    const uint16_t *h_bias_fp16; /* NULL if no bias */
    PackedWeight *weight;
    /* Buffers produced by setup_linear. */
    float *h_input_fp32;
    int8_t *h_input_q;     /* may be int4 stored in int8 cells too */
    float *h_token_scales; /* [B] */
} LinearCtx;

/* Quantize FP16 -> INT8 with per-token (per-row) absmax. */
static int quantize_input_int8(LinearCtx *ctx, int qmax) {
    int B = ctx->B, K = ctx->in_f;
    ctx->h_input_fp32 = (float *)malloc((size_t)B * K * sizeof(float));
    ctx->h_input_q = (int8_t *)malloc((size_t)B * K);
    ctx->h_token_scales = (float *)malloc((size_t)B * sizeof(float));
    if (!ctx->h_input_fp32 || !ctx->h_input_q || !ctx->h_token_scales)
        return -1;

    for (int b = 0; b < B; ++b) {
        const uint16_t *row = ctx->h_input_fp16 + (size_t)b * K;
        float *frow = ctx->h_input_fp32 + (size_t)b * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            float v = f16_to_f32(row[k]);
            frow[k] = v;
            float a = fabsf(v);
            if (a > amax)
                amax = a;
        }
        float s = (amax > 0.0f) ? (amax / (float)qmax) : 1.0f;
        ctx->h_token_scales[b] = s;
        float inv = 1.0f / s;
        int8_t *qrow = ctx->h_input_q + (size_t)b * K;
        for (int k = 0; k < K; ++k) {
            long r = lrintf(frow[k] * inv);
            if (r < -qmax)
                r = -qmax;
            if (r > qmax)
                r = qmax;
            qrow[k] = (int8_t)r;
        }
    }
    return 0;
}

/* Pack int8 -> int4 nibbles in-place to a new buffer of size B*K/2. */
static int8_t *pack_int4_nibbles(const int8_t *src, int B, int K) {
    int8_t *dst = (int8_t *)malloc((size_t)B * (size_t)K / 2);
    if (!dst)
        return NULL;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; k += 2) {
            uint8_t lo = (uint8_t)src[(size_t)b * K + k] & 0x0F;
            uint8_t hi = ((uint8_t)src[(size_t)b * K + k + 1] & 0x0F) << 4;
            dst[(size_t)b * (K / 2) + k / 2] = (int8_t)(lo | hi);
        }
    }
    return dst;
}

static int parse_bias_arg_fp16(ErlNifEnv *env, ERL_NIF_TERM term, int out_features,
                               ErlNifBinary *bias_bin, uint16_t **owned_bias, int *has_bias) {
    *owned_bias = NULL;
    *has_bias = 0;

    if (enif_is_identical(term, enif_make_atom(env, "nil")) ||
        enif_is_identical(term, enif_make_atom(env, "bias_nil"))) {
        return 0;
    }

    if (enif_inspect_binary(env, term, bias_bin)) {
        if (bias_bin->size != (size_t)out_features * sizeof(uint16_t))
            return -1;
        *has_bias = 1;
        return 0;
    }

    int arity = 0;
    const ERL_NIF_TERM *tuple = NULL;
    if (enif_get_tuple(env, term, &arity, &tuple) && arity == 2 &&
        enif_is_identical(tuple[0], enif_make_atom(env, "bias_list"))) {
        unsigned len = 0;
        if (!enif_get_list_length(env, tuple[1], &len) || len != (unsigned)out_features) {
            return -1;
        }

        uint16_t *tmp = (uint16_t *)malloc((size_t)out_features * sizeof(uint16_t));
        if (!tmp)
            return -2;

        ERL_NIF_TERM list = tuple[1];
        for (int i = 0; i < out_features; ++i) {
            ERL_NIF_TERM head, tail;
            double v = 0.0;
            if (!enif_get_list_cell(env, list, &head, &tail) || !enif_get_double(env, head, &v)) {
                free(tmp);
                return -1;
            }
            tmp[i] = float_to_half((float)v);
            list = tail;
        }

        *owned_bias = tmp;
        *has_bias = 1;
        return 0;
    }

    return -1;
}
#endif

/* =========================================================================
 * nt_linear_int8_sparse
 * ========================================================================= */
ERL_NIF_TERM nt_linear_int8_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
#ifdef _WIN32
    return make_error(env, "cuda_unavailable_on_windows");
#else
    if (!cuda_available())
        return make_error(env, "cuda_unavailable");

    ErlNifBinary input_bin;
    if (!enif_inspect_binary(env, argv[0], &input_bin))
        return make_error(env, "input_not_binary");

    PackedWeight *w = NULL;
    if (!enif_get_resource(env, argv[1], PACKED_WEIGHT_RES, (void **)&w) || !w)
        return make_error(env, "bad_packed_weight");
    if (w->dtype != PW_INT8_SPARSE)
        return make_error(env, "wrong_packed_weight_dtype");

    int has_bias = 0;
    ErlNifBinary bias_bin;
    uint16_t *h_bias_owned = NULL;
    int bias_rc =
        parse_bias_arg_fp16(env, argv[2], w->out_features, &bias_bin, &h_bias_owned, &has_bias);
    if (bias_rc == -2)
        return make_error(env, "host_alloc_failed");
    if (bias_rc != 0)
        return make_error(env, "bias_invalid");

    int in_f = w->in_features;
    int out_f = w->out_features;
    if (input_bin.size % ((size_t)in_f * sizeof(uint16_t)) != 0)
        return make_error(env, "input_size_not_mul_in_f");
    int B = (int)(input_bin.size / ((size_t)in_f * sizeof(uint16_t)));
    if (B <= 0)
        return make_error(env, "empty_input");

    LinearCtx ctx = {0};
    ctx.B = B;
    ctx.in_f = in_f;
    ctx.out_f = out_f;
    ctx.h_input_fp16 = (const uint16_t *)input_bin.data;
    ctx.h_bias_fp16 =
        has_bias ? (h_bias_owned ? h_bias_owned : (const uint16_t *)bias_bin.data) : NULL;
    ctx.weight = w;

    if (quantize_input_int8(&ctx, 127) != 0) {
        if (h_bias_owned)
            free(h_bias_owned);
        return make_error(env, "host_alloc_failed");
    }

    /* Upload quantized input to device (col-major B = [in_f, B]). */
    int8_t *d_B = NULL;
    int8_t *d_B_colmajor = NULL;
    int32_t *d_C = NULL;
    void *workspace = NULL;
    size_t workspace_size = 0;

    cudaError_t cerr;
    cerr = cudaMalloc((void **)&d_B, (size_t)in_f * (size_t)B);
    if (cerr != cudaSuccess)
        goto fail_alloc;

    /* The cuSPARSELt path expects B col-major [in_f, B], so we transpose
     * here on host before upload. */
    d_B_colmajor = (int8_t *)malloc((size_t)in_f * (size_t)B);
    if (!d_B_colmajor)
        goto fail_alloc;
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < in_f; ++k) {
            d_B_colmajor[(size_t)b * in_f + k] = ctx.h_input_q[(size_t)b * in_f + k];
        }
    }
    /* The above is just a copy; the actual transpose is the layout convention
     * — col-major over the logical [in_f, B] matrix is the same bytes as
     * row-major [B, in_f] when ldb=in_f. So we upload h_input_q directly. */
    cerr = cudaMemcpy(d_B, ctx.h_input_q, (size_t)in_f * (size_t)B, cudaMemcpyHostToDevice);
    free(d_B_colmajor);
    d_B_colmajor = NULL;
    if (cerr != cudaSuccess)
        goto fail_alloc;

    /* Output INT32 [out_f, B] row-major. */
    cerr = cudaMalloc((void **)&d_C, (size_t)out_f * (size_t)B * sizeof(int32_t));
    if (cerr != cudaSuccess)
        goto fail_alloc;
    cudaMemset(d_C, 0, (size_t)out_f * (size_t)B * sizeof(int32_t));

    /* Choose path: cuSPARSELt if plan cached AND B matches the cached N
     * (which we hard-coded to 128 in prepack). Otherwise CUTLASS sparse. */
    int used_cusparselt = 0;
    (void)used_cusparselt;
    if (w->cusparselt_plan && w->d_compressed && B == 128) {
        cusparseLtHandle_t handle;
        if (cusparseLtInit(&handle) == CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlan_t *plan = (cusparseLtMatmulPlan_t *)w->cusparselt_plan;
            float alpha = 1.0f, beta = 0.0f;
            size_t ws = 0;
            cusparseLtMatmulGetWorkspace(&handle, plan, &ws);
            if (ws > 0)
                cudaMalloc(&workspace, ws);
            workspace_size = ws;

            cusparseStatus_t st = cusparseLtMatmul(&handle, plan, &alpha, w->d_compressed, d_B,
                                                   &beta, d_C, d_C, workspace, NULL, 0);
            cudaDeviceSynchronize();
            cusparseLtDestroy(&handle);
            if (st != CUSPARSE_STATUS_SUCCESS) {
                /* Fall through to CUTLASS path. */
            } else {
                used_cusparselt = 1;
            }
        }
    }

    if (!used_cusparselt) {
        /* CUTLASS path. cfg=28 winner. */
        workspace_size = cutlass_int8_sparse_workspace_size(out_f, B, in_f);
        if (workspace_size > 0 && !workspace) {
            cerr = cudaMalloc(&workspace, workspace_size);
            if (cerr != cudaSuccess)
                goto fail_alloc;
        }
        int rc = cutlass_int8_sparse_run(
            out_f, B, in_f,
            (const int8_t *)w->d_weight, /* A: int8 [out_f, in_f] row-major sparse */
            d_B,                         /* B: int8 [in_f, B] col-major (== row-major [B, in_f]) */
            d_C,                         /* C: int32 [out_f, B] row-major */
            w->d_metadata, workspace, workspace_size);
        if (rc != 0) {
            char err[64];
            snprintf(err, sizeof(err), "cutlass_int8_sparse_run_failed_%d", rc);
            goto fail_run_err;
        }
    }

    /* Download INT32 output and dequantize on host. */
    int32_t *h_acc = (int32_t *)malloc((size_t)out_f * (size_t)B * sizeof(int32_t));
    if (!h_acc)
        goto fail_alloc;
    cerr =
        cudaMemcpy(h_acc, d_C, (size_t)out_f * (size_t)B * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        free(h_acc);
        goto fail_alloc;
    }

    /* Per-channel scales reside on device; copy to host for dequant. */
    float *h_chan_scales = (float *)malloc((size_t)out_f * sizeof(float));
    if (!h_chan_scales) {
        free(h_acc);
        goto fail_alloc;
    }
    cudaMemcpy(h_chan_scales, w->d_scales, (size_t)out_f * sizeof(float), cudaMemcpyDeviceToHost);

    /* Build output binary. Note CUTLASS produced [out_f, B] row-major; we
     * want [B, out_f] row-major in the result. */
    ErlNifBinary out_bin;
    if (!enif_alloc_binary((size_t)B * (size_t)out_f * sizeof(uint16_t), &out_bin)) {
        free(h_acc);
        free(h_chan_scales);
        goto fail_alloc;
    }
    uint16_t *h_out = (uint16_t *)out_bin.data;

    for (int b = 0; b < B; ++b) {
        for (int o = 0; o < out_f; ++o) {
            int32_t acc = h_acc[(size_t)o * B + b]; /* [out_f, B] */
            float v = (float)acc * h_chan_scales[o] * ctx.h_token_scales[b];
            if (ctx.h_bias_fp16)
                v += f16_to_f32(ctx.h_bias_fp16[o]);
            h_out[(size_t)b * out_f + o] = float_to_half(v);
        }
    }

    free(h_acc);
    free(h_chan_scales);
    if (workspace)
        cudaFree(workspace);
    cudaFree(d_B);
    cudaFree(d_C);
    free(ctx.h_input_fp32);
    free(ctx.h_input_q);
    free(ctx.h_token_scales);
    if (h_bias_owned)
        free(h_bias_owned);

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_binary(env, &out_bin));

fail_run_err:
    free(ctx.h_input_fp32);
    free(ctx.h_input_q);
    free(ctx.h_token_scales);
    if (h_bias_owned)
        free(h_bias_owned);
    if (workspace)
        cudaFree(workspace);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
    return make_error(env, "cutlass_run_failed");

fail_alloc:
    free(ctx.h_input_fp32);
    free(ctx.h_input_q);
    free(ctx.h_token_scales);
    if (h_bias_owned)
        free(h_bias_owned);
    if (d_B_colmajor)
        free(d_B_colmajor);
    if (workspace)
        cudaFree(workspace);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
    return make_error(env, "cuda_alloc_failed");
#endif
}

/* =========================================================================
 * nt_linear_int4_sparse
 * ========================================================================= */
ERL_NIF_TERM nt_linear_int4_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
#ifdef _WIN32
    return make_error(env, "cuda_unavailable_on_windows");
#else
    if (!cuda_available())
        return make_error(env, "cuda_unavailable");

    ErlNifBinary input_bin;
    if (!enif_inspect_binary(env, argv[0], &input_bin))
        return make_error(env, "input_not_binary");

    PackedWeight *w = NULL;
    if (!enif_get_resource(env, argv[1], PACKED_WEIGHT_RES, (void **)&w) || !w)
        return make_error(env, "bad_packed_weight");
    if (w->dtype != PW_INT4_SPARSE)
        return make_error(env, "wrong_packed_weight_dtype");

    int has_bias = 0;
    ErlNifBinary bias_bin;
    uint16_t *h_bias_owned = NULL;
    int bias_rc =
        parse_bias_arg_fp16(env, argv[2], w->out_features, &bias_bin, &h_bias_owned, &has_bias);
    if (bias_rc == -2)
        return make_error(env, "host_alloc_failed");
    if (bias_rc != 0)
        return make_error(env, "bias_invalid");

    int in_f = w->in_features;
    int out_f = w->out_features;
    if (input_bin.size % ((size_t)in_f * sizeof(uint16_t)) != 0)
        return make_error(env, "input_size_not_mul_in_f");
    int B = (int)(input_bin.size / ((size_t)in_f * sizeof(uint16_t)));
    if (B <= 0)
        return make_error(env, "empty_input");

    LinearCtx ctx = {0};
    ctx.B = B;
    ctx.in_f = in_f;
    ctx.out_f = out_f;
    ctx.h_input_fp16 = (const uint16_t *)input_bin.data;
    ctx.h_bias_fp16 =
        has_bias ? (h_bias_owned ? h_bias_owned : (const uint16_t *)bias_bin.data) : NULL;
    ctx.weight = w;

    /* Quantize input to INT4 (stored 1 per byte for now), then pack nibbles. */
    if (quantize_input_int8(&ctx, 7) != 0) {
        if (h_bias_owned)
            free(h_bias_owned);
        return make_error(env, "host_alloc_failed");
    }

    int8_t *h_packed_input = pack_int4_nibbles(ctx.h_input_q, B, in_f);
    if (!h_packed_input) {
        free(ctx.h_input_fp32);
        free(ctx.h_input_q);
        free(ctx.h_token_scales);
        if (h_bias_owned)
            free(h_bias_owned);
        return make_error(env, "host_alloc_failed");
    }
    size_t packed_input_bytes = (size_t)B * (size_t)in_f / 2;

    /* Upload to device. */
    void *d_B = NULL;
    int32_t *d_C = NULL;
    void *workspace = NULL;
    size_t workspace_size = 0;

    cudaError_t cerr;
    cerr = cudaMalloc(&d_B, packed_input_bytes);
    if (cerr != cudaSuccess) {
        free(h_packed_input);
        free(ctx.h_input_fp32);
        free(ctx.h_input_q);
        free(ctx.h_token_scales);
        if (h_bias_owned)
            free(h_bias_owned);
        return make_error(env, "cuda_alloc_failed");
    }
    cerr = cudaMemcpy(d_B, h_packed_input, packed_input_bytes, cudaMemcpyHostToDevice);
    free(h_packed_input);
    if (cerr != cudaSuccess)
        goto fail4_alloc;

    cerr = cudaMalloc((void **)&d_C, (size_t)out_f * (size_t)B * sizeof(int32_t));
    if (cerr != cudaSuccess)
        goto fail4_alloc;
    cudaMemset(d_C, 0, (size_t)out_f * (size_t)B * sizeof(int32_t));

    workspace_size = cutlass_int4_sparse_workspace_size(out_f, B, in_f);
    if (workspace_size > 0) {
        cerr = cudaMalloc(&workspace, workspace_size);
        if (cerr != cudaSuccess)
            goto fail4_alloc;
    }

    int rc = cutlass_int4_sparse_run(out_f, B, in_f,
                                     w->d_weight, /* sparse A: int4 [out_f, in_f] row-major */
                                     d_B,         /* B: int4 [in_f, B] col-major */
                                     d_C,         /* C: int32 [out_f, B] row-major */
                                     w->d_metadata, workspace, workspace_size);
    if (rc != 0) {
        char err[64];
        snprintf(err, sizeof(err), "cutlass_int4_sparse_run_failed_%d", rc);
        if (workspace)
            cudaFree(workspace);
        cudaFree(d_B);
        cudaFree(d_C);
        free(ctx.h_input_fp32);
        free(ctx.h_input_q);
        free(ctx.h_token_scales);
        if (h_bias_owned)
            free(h_bias_owned);
        return make_error(env, err);
    }

    /* Download + dequant. */
    int32_t *h_acc = (int32_t *)malloc((size_t)out_f * (size_t)B * sizeof(int32_t));
    if (!h_acc)
        goto fail4_alloc;
    cerr =
        cudaMemcpy(h_acc, d_C, (size_t)out_f * (size_t)B * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        free(h_acc);
        goto fail4_alloc;
    }

    float *h_chan_scales = (float *)malloc((size_t)out_f * sizeof(float));
    if (!h_chan_scales) {
        free(h_acc);
        goto fail4_alloc;
    }
    cudaMemcpy(h_chan_scales, w->d_scales, (size_t)out_f * sizeof(float), cudaMemcpyDeviceToHost);

    ErlNifBinary out_bin;
    if (!enif_alloc_binary((size_t)B * (size_t)out_f * sizeof(uint16_t), &out_bin)) {
        free(h_acc);
        free(h_chan_scales);
        goto fail4_alloc;
    }
    uint16_t *h_out = (uint16_t *)out_bin.data;

    for (int b = 0; b < B; ++b) {
        for (int o = 0; o < out_f; ++o) {
            int32_t acc = h_acc[(size_t)o * B + b];
            float v = (float)acc * h_chan_scales[o] * ctx.h_token_scales[b];
            if (ctx.h_bias_fp16)
                v += f16_to_f32(ctx.h_bias_fp16[o]);
            h_out[(size_t)b * out_f + o] = float_to_half(v);
        }
    }

    free(h_acc);
    free(h_chan_scales);
    if (workspace)
        cudaFree(workspace);
    cudaFree(d_B);
    cudaFree(d_C);
    free(ctx.h_input_fp32);
    free(ctx.h_input_q);
    free(ctx.h_token_scales);
    if (h_bias_owned)
        free(h_bias_owned);

    return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_binary(env, &out_bin));

fail4_alloc:
    if (workspace)
        cudaFree(workspace);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
    free(ctx.h_input_fp32);
    free(ctx.h_input_q);
    free(ctx.h_token_scales);
    if (h_bias_owned)
        free(h_bias_owned);
    return make_error(env, "cuda_alloc_failed");
#endif
}
