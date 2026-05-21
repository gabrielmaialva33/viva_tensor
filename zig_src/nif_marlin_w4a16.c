/*
 * Copyright (c) 2026 viva_tensor contributors
 *
 * MIT License
 *
 * Built on top of Marlin (Apache 2.0, IST-DASLab/marlin).
 */

#include "viva_nif.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

extern int marlin_cuda(const void *A, const void *B, void *C, void *s,
                       int prob_m, int prob_n, int prob_k, void *workspace,
                       int groupsize, int dev, cudaStream_t stream,
                       int thread_k, int thread_n, int sms, int max_par);

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
  fprintf(stderr, "marlin bench cuda error at %s: %s\n", what,
          cudaGetErrorString(err));
  return code;
}

extern "C" int viva_marlin_w4a16_mm(int M, int N, int K,
                                    const void *d_A_fp16,
                                    const void *d_B_int4_marlin,
                                    void *d_C_fp16,
                                    const void *d_scales_fp16,
                                    void *d_workspace, int groupsize) {
  return marlin_cuda(d_A_fp16, d_B_int4_marlin, d_C_fp16,
                     (void *)d_scales_fp16, M, N, K, d_workspace, groupsize, 0,
                     0, -1, -1, -1, 16);
}

extern "C" int viva_marlin_w4a16_bench(int M, int N, int K, int groupsize,
                                       int iters) {
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

#define CHECK_CUDA(expr, code)                                                 \
  do {                                                                         \
    ret = cuda_fail((expr), (code), #expr);                                    \
    if (ret != 0)                                                              \
      goto cleanup;                                                            \
  } while (0)

  CHECK_CUDA(cudaMalloc(&d_A, a_elems * sizeof(uint16_t)), -10);
  CHECK_CUDA(cudaMalloc(&d_B, b_bytes), -11);
  CHECK_CUDA(cudaMalloc(&d_C, c_elems * sizeof(uint16_t)), -12);
  CHECK_CUDA(cudaMalloc(&d_scales, scale_elems * sizeof(uint16_t)), -13);
  CHECK_CUDA(cudaMalloc(&d_workspace, workspace_elems * sizeof(int)), -14);
  CHECK_CUDA(cudaMemcpy(d_A, h_A, a_elems * sizeof(uint16_t),
                        cudaMemcpyHostToDevice),
             -15);
  CHECK_CUDA(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice), -16);
  CHECK_CUDA(cudaMemcpy(d_scales, h_scales, scale_elems * sizeof(uint16_t),
                        cudaMemcpyHostToDevice),
             -17);
  CHECK_CUDA(cudaMemset(d_C, 0, c_elems * sizeof(uint16_t)), -18);
  CHECK_CUDA(cudaMemset(d_workspace, 0, workspace_elems * sizeof(int)), -19);
  CHECK_CUDA(cudaEventCreate(&start), -20);
  CHECK_CUDA(cudaEventCreate(&stop), -21);

  for (int i = 0; i < 5; i++) {
    ret = viva_marlin_w4a16_mm(M, N, K, d_A, d_B, d_C, d_scales, d_workspace,
                               groupsize);
    if (ret != 0)
      goto cleanup;
  }
  CHECK_CUDA(cudaDeviceSynchronize(), -22);

  CHECK_CUDA(cudaEventRecord(start, 0), -23);
  for (int i = 0; i < iters; i++) {
    ret = viva_marlin_w4a16_mm(M, N, K, d_A, d_B, d_C, d_scales, d_workspace,
                               groupsize);
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
    printf("M=%d N=%d K=%d g=%d ms_avg=%.4f tflops=%.1f\n", M, N, K,
           groupsize, ms_avg, tflops);
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

extern "C" ERL_NIF_TERM viva_marlin_w4a16_bench_nif(
    ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, groupsize, iters;
  if (!enif_get_int(env, argv[0], &m) || !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &groupsize) ||
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
