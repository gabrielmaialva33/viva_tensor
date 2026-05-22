/*
 * Copyright (c) 2026 viva_tensor contributors
 *
 * MIT License
 *
 * Built on top of Marlin (Apache 2.0, IST-DASLab/marlin).
 */

#include "viva_nif.h"
extern "C" {
#include "viva_marlin_pack.h"
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

extern int marlin_cuda(const void *A, const void *B, void *C, void *s, int prob_m, int prob_n,
                       int prob_k, void *workspace, int groupsize, int dev, cudaStream_t stream,
                       int thread_k, int thread_n, int sms, int max_par);

typedef struct {
    int K, N, groupsize, groups;
    void *d_B;
    void *d_s;
    void *d_workspace;
    size_t b_bytes;
    size_t s_bytes;
    size_t ws_bytes;
} MarlinPackedResource;

extern "C" ErlNifResourceType *marlin_packed_resource_type = NULL;

extern "C" void marlin_packed_resource_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    MarlinPackedResource *r = (MarlinPackedResource *)obj;
    if (!r)
        return;
    if (r->d_B) {
        cudaFree(r->d_B);
        r->d_B = NULL;
    }
    if (r->d_s) {
        cudaFree(r->d_s);
        r->d_s = NULL;
    }
    if (r->d_workspace) {
        cudaFree(r->d_workspace);
        r->d_workspace = NULL;
    }
}

static ERL_NIF_TERM marlin_error_code(ErlNifEnv *env, const char *reason, int code) {
    return enif_make_tuple3(env, enif_make_atom(env, "error"), enif_make_atom(env, reason),
                            enif_make_int(env, code));
}

static void fill_random_fp16(uint16_t *dst, size_t n) {
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        dst[i] = (uint16_t)(0x3c00u | ((x >> 10) & 0x03ffu));
    }
}

static void fill_random_bytes(uint8_t *dst, size_t n) {
    uint32_t x = 0x9e3779b9u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        dst[i] = (uint8_t)(x >> 24);
    }
}

static void fill_fp16_one(uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = 0x3c00u;
}

static int cuda_fail(cudaError_t err, int code, const char *what) {
    if (err == cudaSuccess)
        return 0;
    fprintf(stderr, "marlin bench cuda error at %s: %s\n", what, cudaGetErrorString(err));
    return code;
}

extern "C" int viva_marlin_w4a16_mm(int M, int N, int K, const void *d_A_fp16,
                                    const void *d_B_int4_marlin, void *d_C_fp16,
                                    const void *d_scales_fp16, void *d_workspace, int groupsize) {
    return marlin_cuda(d_A_fp16, d_B_int4_marlin, d_C_fp16, (void *)d_scales_fp16, M, N, K,
                       d_workspace, groupsize, 0, 0, -1, -1, -1, 16);
}

extern "C" int viva_marlin_w4a16_bench(int M, int N, int K, int groupsize, int iters) {
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0)
        return -1;
    if (groupsize != -1 && (groupsize <= 0 || K % groupsize != 0))
        return -2;

    const int max_par = 16;
    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_bytes = ((size_t)K * (size_t)N) / 2;
    const size_t c_elems = (size_t)M * (size_t)N;
    const size_t scale_rows = (groupsize == -1) ? 1u : (size_t)K / groupsize;
    const size_t scale_elems = scale_rows * (size_t)N;
    const size_t workspace_elems = ((size_t)N / 128u) * (size_t)max_par;

    uint16_t *h_A = (uint16_t *)malloc(a_elems * sizeof(uint16_t));
    uint8_t *h_B = (uint8_t *)malloc(b_bytes);
    uint16_t *h_scales = (uint16_t *)malloc(scale_elems * sizeof(uint16_t));
    if (!h_A || !h_B || !h_scales) {
        free(h_A);
        free(h_B);
        free(h_scales);
        return -3;
    }

    fill_random_fp16(h_A, a_elems);
    fill_random_bytes(h_B, b_bytes);
    fill_fp16_one(h_scales, scale_elems);

    void *d_A = NULL;
    void *d_B = NULL;
    void *d_C = NULL;
    void *d_scales = NULL;
    void *d_workspace = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    float elapsed_ms = 0.0f;
    int ret = 0;

#define CHECK_CUDA(expr, code)                  \
    do {                                        \
        ret = cuda_fail((expr), (code), #expr); \
        if (ret != 0)                           \
            goto cleanup;                       \
    } while (0)

    CHECK_CUDA(cudaMalloc(&d_A, a_elems * sizeof(uint16_t)), -10);
    CHECK_CUDA(cudaMalloc(&d_B, b_bytes), -11);
    CHECK_CUDA(cudaMalloc(&d_C, c_elems * sizeof(uint16_t)), -12);
    CHECK_CUDA(cudaMalloc(&d_scales, scale_elems * sizeof(uint16_t)), -13);
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_elems * sizeof(int)), -14);
    CHECK_CUDA(cudaMemcpy(d_A, h_A, a_elems * sizeof(uint16_t), cudaMemcpyHostToDevice), -15);
    CHECK_CUDA(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice), -16);
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales, scale_elems * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               -17);
    CHECK_CUDA(cudaMemset(d_C, 0, c_elems * sizeof(uint16_t)), -18);
    CHECK_CUDA(cudaMemset(d_workspace, 0, workspace_elems * sizeof(int)), -19);
    CHECK_CUDA(cudaEventCreate(&start), -20);
    CHECK_CUDA(cudaEventCreate(&stop), -21);

    for (int i = 0; i < 5; i++) {
        ret = viva_marlin_w4a16_mm(M, N, K, d_A, d_B, d_C, d_scales, d_workspace, groupsize);
        if (ret != 0)
            goto cleanup;
    }
    CHECK_CUDA(cudaDeviceSynchronize(), -22);

    CHECK_CUDA(cudaEventRecord(start, 0), -23);
    for (int i = 0; i < iters; i++) {
        ret = viva_marlin_w4a16_mm(M, N, K, d_A, d_B, d_C, d_scales, d_workspace, groupsize);
        if (ret != 0)
            goto cleanup;
    }
    CHECK_CUDA(cudaEventRecord(stop, 0), -24);
    CHECK_CUDA(cudaEventSynchronize(stop), -25);
    CHECK_CUDA(cudaGetLastError(), -26);

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop), -27);

    {
        double ms_avg = (double)elapsed_ms / (double)iters;
        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops = flops / (ms_avg * 1.0e9);
        printf("M=%d N=%d K=%d g=%d ms_avg=%.4f tflops=%.1f\n", M, N, K, groupsize, ms_avg, tflops);
        fflush(stdout);
    }

cleanup:
    if (start)
        cudaEventDestroy(start);
    if (stop)
        cudaEventDestroy(stop);
    if (d_workspace)
        cudaFree(d_workspace);
    if (d_scales)
        cudaFree(d_scales);
    if (d_C)
        cudaFree(d_C);
    if (d_B)
        cudaFree(d_B);
    if (d_A)
        cudaFree(d_A);
    free(h_scales);
    free(h_B);
    free(h_A);
    return ret;

#undef CHECK_CUDA
}

extern "C" ERL_NIF_TERM viva_marlin_w4a16_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
    (void)argc;
    int m, n, k, groupsize, iters;
    if (!enif_get_int(env, argv[0], &m) || !enif_get_int(env, argv[1], &n) ||
        !enif_get_int(env, argv[2], &k) || !enif_get_int(env, argv[3], &groupsize) ||
        !enif_get_int(env, argv[4], &iters))
        return enif_make_atom(env, "invalid_args");

    int result = viva_marlin_w4a16_bench(m, n, k, groupsize, iters);
    if (result != 0) {
        char errbuf[64];
        snprintf(errbuf, sizeof(errbuf), "marlin_w4a16_failed_%d", result);
        return enif_make_atom(env, errbuf);
    }
    return enif_make_int(env, 0);
}

extern "C" ERL_NIF_TERM viva_marlin_w4a16_prepack_nif(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
    if (argc != 5)
        return marlin_error_code(env, "bad_arity", -1);

    ErlNifBinary w_bin;
    ErlNifBinary s_bin;
    int K = 0, N = 0, groupsize = 0;
    if (!enif_inspect_binary(env, argv[0], &w_bin))
        return marlin_error_code(env, "invalid_weight_binary", -2);
    if (!enif_inspect_binary(env, argv[1], &s_bin))
        return marlin_error_code(env, "invalid_scale_binary", -3);
    if (!enif_get_int(env, argv[2], &K) || !enif_get_int(env, argv[3], &N) ||
        !enif_get_int(env, argv[4], &groupsize))
        return marlin_error_code(env, "invalid_dimensions", -4);
    if (K <= 0 || N <= 0)
        return marlin_error_code(env, "invalid_dimensions", -5);
    if (groupsize != -1 && (groupsize <= 0 || K % groupsize != 0))
        return marlin_error_code(env, "invalid_groupsize", -6);
    if (K % 16 != 0 || N % 128 != 0)
        return marlin_error_code(env, "invalid_marlin_shape", -7);

    int groups = (groupsize == -1) ? 1 : (K / groupsize);
    size_t w_bytes = (size_t)K * (size_t)N * sizeof(uint16_t);
    size_t s_bytes = (size_t)groups * (size_t)N * sizeof(uint16_t);
    size_t b_bytes = ((size_t)K * (size_t)N) / 2u;
    size_t ws_bytes = ((size_t)N / 128u) * 16u * sizeof(int);

    if (w_bin.size != w_bytes)
        return marlin_error_code(env, "weight_size_mismatch", -8);
    if (s_bin.size != s_bytes)
        return marlin_error_code(env, "scale_size_mismatch", -9);

    uint32_t *out_B = (uint32_t *)malloc(b_bytes);
    uint16_t *out_s = (uint16_t *)malloc(s_bytes);
    if (!out_B || !out_s) {
        free(out_B);
        free(out_s);
        return marlin_error_code(env, "host_alloc_failed", -10);
    }

    int pack_rc = viva_marlin_pack((const uint16_t *)w_bin.data, (const uint16_t *)s_bin.data, K, N,
                                   groupsize, out_B, out_s);
    if (pack_rc != 0) {
        free(out_B);
        free(out_s);
        return marlin_error_code(env, "pack_failed", pack_rc);
    }

    void *d_B = NULL;
    void *d_s = NULL;
    void *d_workspace = NULL;
    MarlinPackedResource *resource = NULL;
    ERL_NIF_TERM term;
    cudaError_t err = cudaMalloc(&d_B, b_bytes);
    if (err != cudaSuccess)
        goto cuda_fail;
    err = cudaMalloc(&d_s, s_bytes);
    if (err != cudaSuccess)
        goto cuda_fail;
    err = cudaMalloc(&d_workspace, ws_bytes);
    if (err != cudaSuccess)
        goto cuda_fail;
    err = cudaMemcpy(d_B, out_B, b_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto cuda_fail;
    err = cudaMemcpy(d_s, out_s, s_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto cuda_fail;
    err = cudaMemset(d_workspace, 0, ws_bytes);
    if (err != cudaSuccess)
        goto cuda_fail;

    free(out_B);
    free(out_s);

    resource = (MarlinPackedResource *)enif_alloc_resource(marlin_packed_resource_type,
                                                           sizeof(MarlinPackedResource));
    if (!resource) {
        cudaFree(d_workspace);
        cudaFree(d_s);
        cudaFree(d_B);
        return marlin_error_code(env, "resource_alloc_failed", -20);
    }

    resource->K = K;
    resource->N = N;
    resource->groupsize = groupsize;
    resource->groups = groups;
    resource->d_B = d_B;
    resource->d_s = d_s;
    resource->d_workspace = d_workspace;
    resource->b_bytes = b_bytes;
    resource->s_bytes = s_bytes;
    resource->ws_bytes = ws_bytes;

    term = enif_make_resource(env, resource);
    enif_release_resource(resource);
    return term;

cuda_fail:
    int cuda_code = (int)err;
    if (d_workspace)
        cudaFree(d_workspace);
    if (d_s)
        cudaFree(d_s);
    if (d_B)
        cudaFree(d_B);
    free(out_B);
    free(out_s);
    return marlin_error_code(env, "cuda_failed", cuda_code);
}

extern "C" ERL_NIF_TERM viva_marlin_w4a16_get_b_bytes_nif(ErlNifEnv *env, int argc,
                                                          const ERL_NIF_TERM argv[]) {
    if (argc != 1)
        return marlin_error_code(env, "bad_arity", -1);

    MarlinPackedResource *resource = NULL;
    if (!enif_get_resource(env, argv[0], marlin_packed_resource_type, (void **)&resource))
        return marlin_error_code(env, "invalid_resource", -2);

    ErlNifBinary bin;
    if (!enif_alloc_binary(resource->b_bytes, &bin))
        return marlin_error_code(env, "binary_alloc_failed", -3);

    cudaError_t err = cudaMemcpy(bin.data, resource->d_B, resource->b_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        enif_release_binary(&bin);
        return marlin_error_code(env, "cuda_failed", (int)err);
    }

    return enif_make_binary(env, &bin);
}
