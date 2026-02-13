/**
 * nif_cpu_ops.c - CPU NIF operations for viva_tensor
 *
 * Extracted from nif_entry.c (lines 438-1526). Contains all CPU-side NIF
 * functions for tensor math:
 *
 *   - Element-wise Operations: nt_add, nt_sub, nt_mul, nt_scale, nt_negate
 *   - Reductions: nt_dot, nt_sum, nt_max, nt_min
 *   - Matrix Operations: nt_matmul_blas (MKL/OpenBLAS), nt_matmul_inplace,
 *     nt_matmul_cuda (CUDA+CPU fallback), nt_matmul_cuda_fp32,
 *     nt_matmul_int8_tc, nt_matmul_fp16_tc, nt_matmul_int8_lt, nt_transpose
 *   - Activation Functions: nt_relu, nt_sigmoid, nt_exp_nif, nt_log_nif
 *   - In-Place Mutation NIFs: nt_add_mut, nt_scale_mut, nt_negate_mut, nt_relu_mut
 *   - Fused Kernels: nt_saturn_blend, nt_fused_linear_relu_nif
 *   - Helpers: float_to_half (IEEE 754 FP16 conversion)
 */

#include "viva_nif.h"

/* =========================================================================
 * NIF Resource API — Element-wise Operations (resource → resource)
 * ========================================================================= */

/** nt_add(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_add(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_add(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_sub(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_sub(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_sub(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_mul(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_mul(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_mul(a->data, b->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_scale(Ref, Scalar) -> {ok, RefC} */
ERL_NIF_TERM nt_scale(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");
  int ok;
  double scalar = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_scalar");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_scale(a->data, scalar, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_negate(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_negate(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_negate(a->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * NIF Resource API — Reductions (resource → scalar)
 * ========================================================================= */

/** nt_dot(RefA, RefB) -> {ok, Float} */
ERL_NIF_TERM nt_dot(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  double result = vt_simd_dot(a->data, b->data, a->size);
  return make_ok(env, enif_make_double(env, result));
}

/** nt_sum(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_sum(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double result = vt_simd_sum(a->data, a->size);
  return make_ok(env, enif_make_double(env, result));
}

/** nt_max(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_max(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double mx = vt_simd_max(a->data, a->size);
  return make_ok(env, enif_make_double(env, mx));
}

/** nt_min(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_min(ErlNifEnv *env, int argc,
                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double mn = vt_simd_min(a->data, a->size);
  return make_ok(env, enif_make_double(env, mn));
}

/* =========================================================================
 * NIF Resource API — Matrix Operations
 * ========================================================================= */

/** nt_matmul(RefA, RefB, M, N, K) -> {ok, RefC}
 *  Now uses BLAS directly (MKL/OpenBLAS) - Zig GEMM removed for simplicity.
 *  This is just an alias for nt_matmul_blas.
 */
ERL_NIF_TERM nt_matmul_blas(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]);  /* Forward declaration */
#define nt_matmul nt_matmul_blas  /* Alias */

/** nt_matmul_blas(RefA, RefB, M, N, K) -> {ok, RefC}
 *  DGEMM via MKL (Windows) or best available BLAS (Linux, runtime-detected).
 */
ERL_NIF_TERM nt_matmul_blas(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  /* C = alpha * A @ B + beta * C
   * cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
   * Row-major: lda=k, ldb=n, ldc=n
   */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  /* MKL direct-linked DGEMM */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)m, (int)n, (int)k,
              1.0, a->data, (int)k,
              b->data, (int)n,
              0.0, c->data, (int)n);
#else
  /* Fallback: use dynamically loaded backend */
  if (g_dgemm) {
    blas_dgemm((int)m, (int)n, (int)k,
               1.0, a->data, (int)k,
               b->data, (int)n,
               0.0, c->data, (int)n);
  } else {
    /* No BLAS available - return error */
    free(c->data);
    free(c);
    return make_error(env, "no_blas_backend");
  }
#endif

  return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_inplace(RefA, RefB, RefC, M, N, K) -> ok
 *  Zero-allocation matmul: writes result into existing C tensor.
 *  Eliminates malloc + page-fault overhead (~8ms for large matrices).
 */
ERL_NIF_TERM nt_matmul_inplace(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  NativeTensor *c = get_tensor(env, argv[2]);
  if (!a || !b || !c)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[3], &m_int) ||
      !enif_get_int(env, argv[4], &n_int) ||
      !enif_get_int(env, argv[5], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n) || c->size != (int)(m * n))
    return make_error(env, "size_mismatch");

#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)m, (int)n, (int)k,
              1.0, a->data, (int)k,
              b->data, (int)n,
              0.0, c->data, (int)n);
#else
  if (g_dgemm) {
    blas_dgemm((int)m, (int)n, (int)k,
               1.0, a->data, (int)k,
               b->data, (int)n,
               0.0, c->data, (int)n);
  } else {
    return make_error(env, "no_blas_backend");
  }
#endif

  return enif_make_atom(env, "ok");
}

/* All CUDA externs are in viva_nif.h (included above) */

/** nt_matmul_cuda(RefA, RefB, M, N, K) -> {ok, RefC}
 *  cuBLAS DGEMM on GPU, falls back to BLAS if CUDA not available.
 */
ERL_NIF_TERM nt_matmul_cuda(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

#ifndef _WIN32
  /* Try CUDA/cuBLAS first */
  if (cuda_available()) {
    int result = cuda_dgemm(m_int, n_int, k_int,
                            1.0, a->data, k_int,
                            b->data, n_int,
                            0.0, c->data, n_int);
    if (result == 0) {
      return make_ok(env, make_tensor_term(env, c));
    }
    /* CUDA failed, fall through to CPU */
    fprintf(stderr, "[viva_tensor] CUDA fallback to CPU (error %d)\n", result);
  }
#endif

  /* Fallback to CPU BLAS */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)m, (int)n, (int)k,
              1.0, a->data, (int)k,
              b->data, (int)n,
              0.0, c->data, (int)n);
#else
  if (g_dgemm) {
    blas_dgemm((int)m, (int)n, (int)k,
               1.0, a->data, (int)k,
               b->data, (int)n,
               0.0, c->data, (int)n);
  } else {
    free(c->data);
    free(c);
    return make_error(env, "no_blas_backend");
  }
#endif

  return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_cuda_fp32(RefA, RefB, M, N, K) -> {ok, RefC}
 *  cuBLAS SGEMM (FP32 TF32 Tensor Cores). Auto-converts double <-> float.
 */
#ifndef _WIN32
ERL_NIF_TERM nt_matmul_cuda_fp32(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  if (!cuda_available())
    return make_error(env, "cuda_not_available");

  /* Allocate float buffers for conversion */
  size_t size_a = m * k;
  size_t size_b = k * n;
  size_t size_c = m * n;

  float *a_f32 = (float *)malloc(size_a * sizeof(float));
  float *b_f32 = (float *)malloc(size_b * sizeof(float));
  float *c_f32 = (float *)malloc(size_c * sizeof(float));

  if (!a_f32 || !b_f32 || !c_f32) {
    free(a_f32); free(b_f32); free(c_f32);
    return make_error(env, "out_of_memory");
  }

  /* Convert double -> float (vectorizable, fast) */
  for (size_t i = 0; i < size_a; i++) a_f32[i] = (float)a->data[i];
  for (size_t i = 0; i < size_b; i++) b_f32[i] = (float)b->data[i];

  /* cuBLAS SGEMM (TF32 Tensor Cores) */
  int result = cuda_sgemm(m_int, n_int, k_int,
                          1.0f, a_f32, k_int,
                          b_f32, n_int,
                          0.0f, c_f32, n_int);

  if (result != 0) {
    free(a_f32); free(b_f32); free(c_f32);
    return make_error(env, "cuda_sgemm_failed");
  }

  /* Allocate output tensor */
  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c) {
    free(a_f32); free(b_f32); free(c_f32);
    return make_error(env, "out_of_memory");
  }

  /* Convert float -> double */
  for (size_t i = 0; i < size_c; i++) c->data[i] = (double)c_f32[i];

  free(a_f32); free(b_f32); free(c_f32);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_int8_tc(RefA, RefB, M, N, K) -> {ok, RefC}
 *  INT8 IMMA Tensor Cores via cublasGemmEx. Auto-quantizes f64 input.
 */
ERL_NIF_TERM nt_matmul_int8_tc(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  if (!cuda_int8_available())
    return make_error(env, "int8_tensor_cores_not_available");

  size_t size_a = m * k;
  size_t size_b = k * n;
  size_t size_c = m * n;

  /* Quantize A and B to INT8 */
  int8_t *a_i8 = (int8_t *)malloc(size_a);
  int8_t *b_i8 = (int8_t *)malloc(size_b);
  int32_t *c_i32 = (int32_t *)malloc(size_c * sizeof(int32_t));

  if (!a_i8 || !b_i8 || !c_i32) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "out_of_memory");
  }

  /* Find absmax for quantization */
  double a_max = 0.0, b_max = 0.0;
  for (size_t i = 0; i < size_a; i++) {
    double v = fabs(a->data[i]);
    if (v > a_max) a_max = v;
  }
  for (size_t i = 0; i < size_b; i++) {
    double v = fabs(b->data[i]);
    if (v > b_max) b_max = v;
  }

  /* Quantize to INT8 range [-127, 127] */
  double a_scale = (a_max > 0) ? 127.0 / a_max : 1.0;
  double b_scale = (b_max > 0) ? 127.0 / b_max : 1.0;

  for (size_t i = 0; i < size_a; i++) {
    double scaled = a->data[i] * a_scale;
    a_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
  }
  for (size_t i = 0; i < size_b; i++) {
    double scaled = b->data[i] * b_scale;
    b_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
  }

  /* cuBLAS INT8 GEMM (Tensor Cores) */
  int result = cuda_igemm(m_int, n_int, k_int,
                          1, a_i8, k_int,
                          b_i8, n_int,
                          0, c_i32, n_int);

  if (result != 0) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "cuda_int8_gemm_failed");
  }

  /* Allocate output tensor and dequantize */
  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "out_of_memory");
  }

  /* Dequantize: C_f64 = C_i32 / (a_scale * b_scale) */
  double dequant_scale = 1.0 / (a_scale * b_scale);
  for (size_t i = 0; i < size_c; i++) {
    c->data[i] = (double)c_i32[i] * dequant_scale;
  }

  free(a_i8); free(b_i8); free(c_i32);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_int8_tc_available() -> true | false
 *  Check if INT8 Tensor Cores are available (RTX 20xx+)
 */
ERL_NIF_TERM nt_int8_tc_available(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
  (void)argc; (void)argv;
  if (cuda_int8_available()) {
    return enif_make_atom(env, "true");
  } else {
    return enif_make_atom(env, "false");
  }
}

/* FP16 helper: convert float to half (IEEE 754) */
uint16_t float_to_half(float f) {
  uint32_t x = *(uint32_t*)&f;
  uint32_t sign = (x >> 31) & 0x1;
  uint32_t exp = (x >> 23) & 0xFF;
  uint32_t mant = x & 0x7FFFFF;

  uint16_t h;
  if (exp == 0) {
    h = (sign << 15);  /* Zero or denormal -> zero */
  } else if (exp == 0xFF) {
    h = (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);  /* Inf/NaN */
  } else {
    int new_exp = (int)exp - 127 + 15;
    if (new_exp >= 31) {
      h = (sign << 15) | 0x7C00;  /* Overflow -> Inf */
    } else if (new_exp <= 0) {
      h = (sign << 15);  /* Underflow -> Zero */
    } else {
      h = (sign << 15) | (new_exp << 10) | (mant >> 13);
    }
  }
  return h;
}


/** nt_matmul_fp16_tc(RefA, RefB, M, N, K) -> {ok, RefC}
 *  FP16 Tensor Cores via cublasGemmEx. Auto-converts f64 <-> FP16.
 */
ERL_NIF_TERM nt_matmul_fp16_tc(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  if (!cuda_fp16_available())
    return make_error(env, "fp16_tensor_cores_not_available");

  size_t size_a = m * k;
  size_t size_b = k * n;
  size_t size_c = m * n;

  /* Convert A and B to FP16 */
  uint16_t *a_fp16 = (uint16_t *)malloc(size_a * sizeof(uint16_t));
  uint16_t *b_fp16 = (uint16_t *)malloc(size_b * sizeof(uint16_t));
  float *c_fp32 = (float *)malloc(size_c * sizeof(float));

  if (!a_fp16 || !b_fp16 || !c_fp32) {
    free(a_fp16); free(b_fp16); free(c_fp32);
    return make_error(env, "out_of_memory");
  }

  /* Convert f64 -> FP16 */
  for (size_t i = 0; i < size_a; i++) {
    a_fp16[i] = float_to_half((float)a->data[i]);
  }
  for (size_t i = 0; i < size_b; i++) {
    b_fp16[i] = float_to_half((float)b->data[i]);
  }

  /* cuBLAS FP16 GEMM (Tensor Cores) */
  int result = cuda_hgemm(m_int, n_int, k_int,
                          1.0f, a_fp16, k_int,
                          b_fp16, n_int,
                          0.0f, c_fp32, n_int);

  if (result != 0) {
    free(a_fp16); free(b_fp16); free(c_fp32);
    return make_error(env, "cuda_fp16_gemm_failed");
  }

  /* Allocate output tensor and convert FP32 -> f64 */
  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c) {
    free(a_fp16); free(b_fp16); free(c_fp32);
    return make_error(env, "out_of_memory");
  }

  for (size_t i = 0; i < size_c; i++) {
    c->data[i] = (double)c_fp32[i];
  }

  free(a_fp16); free(b_fp16); free(c_fp32);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_fp16_tc_available() -> true | false */
ERL_NIF_TERM nt_fp16_tc_available(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
  (void)argc; (void)argv;
  if (cuda_fp16_available()) {
    return enif_make_atom(env, "true");
  } else {
    return enif_make_atom(env, "false");
  }
}

/** nt_matmul_int8_lt(RefA, RefB, M, N, K) -> {ok, RefC}
 *  cublasLt INT8 IMMA Tensor Cores (vs cublasGemmEx which uses DP4A).
 */
ERL_NIF_TERM nt_matmul_int8_lt(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");

  int m_int, n_int, k_int;
  if (!enif_get_int(env, argv[2], &m_int) ||
      !enif_get_int(env, argv[3], &n_int) ||
      !enif_get_int(env, argv[4], &k_int))
    return make_error(env, "invalid_dimensions");

  size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
  if (a->size != (int)(m * k) || b->size != (int)(k * n))
    return make_error(env, "size_mismatch");

  if (!cuda_int8_lt_available())
    return make_error(env, "int8_lt_tensor_cores_not_available");

  size_t size_a = m * k;
  size_t size_b = k * n;
  size_t size_c = m * n;

  /* Quantize A and B to INT8 */
  int8_t *a_i8 = (int8_t *)malloc(size_a);
  int8_t *b_i8 = (int8_t *)malloc(size_b);
  int32_t *c_i32 = (int32_t *)malloc(size_c * sizeof(int32_t));

  if (!a_i8 || !b_i8 || !c_i32) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "out_of_memory");
  }

  /* Find absmax for quantization */
  double a_max = 0.0, b_max = 0.0;
  for (size_t i = 0; i < size_a; i++) {
    double v = fabs(a->data[i]);
    if (v > a_max) a_max = v;
  }
  for (size_t i = 0; i < size_b; i++) {
    double v = fabs(b->data[i]);
    if (v > b_max) b_max = v;
  }

  /* Quantize to INT8 range [-127, 127] */
  double a_scale = (a_max > 0) ? 127.0 / a_max : 1.0;
  double b_scale = (b_max > 0) ? 127.0 / b_max : 1.0;

  for (size_t i = 0; i < size_a; i++) {
    double scaled = a->data[i] * a_scale;
    a_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
  }
  for (size_t i = 0; i < size_b; i++) {
    double scaled = b->data[i] * b_scale;
    b_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
  }

  /* cublasLt INT8 IMMA Tensor Cores */
  int result = cuda_igemm_lt(m_int, n_int, k_int,
                             1.0f, a_i8, k_int,
                             b_i8, n_int,
                             0.0f, c_i32, n_int);

  if (result != 0) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "cuda_int8_lt_gemm_failed");
  }

  /* Allocate output tensor and dequantize */
  int out_shape[2] = {m_int, n_int};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c) {
    free(a_i8); free(b_i8); free(c_i32);
    return make_error(env, "out_of_memory");
  }

  /* Dequantize: C_f64 = C_i32 / (a_scale * b_scale) */
  double dequant_scale = 1.0 / (a_scale * b_scale);
  for (size_t i = 0; i < size_c; i++) {
    c->data[i] = (double)c_i32[i] * dequant_scale;
  }

  free(a_i8); free(b_i8); free(c_i32);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_int8_lt_available() -> true | false
 *  Check if cublasLt INT8 IMMA Tensor Cores are available
 */
ERL_NIF_TERM nt_int8_lt_available(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
  (void)argc; (void)argv;
  if (cuda_int8_lt_available()) {
    return enif_make_atom(env, "true");
  } else {
    return enif_make_atom(env, "false");
  }
}
#endif

/** nt_transpose(Ref) -> {ok, RefC}  (creates contiguous transposed copy) */
ERL_NIF_TERM nt_transpose(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a || a->ndim != 2)
    return make_error(env, "invalid_tensor");

  int rows = a->shape[0], cols = a->shape[1];
  int out_shape[2] = {cols, rows};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      c->data[j * rows + i] = a->data[i * cols + j];

  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * NIF Resource API — Activation Functions
 * ========================================================================= */

/** nt_relu(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_relu(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_relu(a->data, c->data, a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_sigmoid(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_sigmoid(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_sigmoid(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_exp(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_exp_nif(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_exp(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_log(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_log_nif(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_simd_log(a->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * In-Place Mutation NIFs
 * "Quebrar a imutabilidade dentro do Zig para economizar RAM"
 * ========================================================================= */

/** nt_add_mut(RefA, RefB) -> ok. Modifies A in-place: A += B */
ERL_NIF_TERM nt_add_mut(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  vt_simd_add_mut(a->data, b->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_scale_mut(RefA, Scalar) -> ok. Modifies A in-place: A *= scalar */
ERL_NIF_TERM nt_scale_mut(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  double scalar;
  if (!enif_get_double(env, argv[1], &scalar))
    return make_error(env, "invalid_scalar");

  vt_simd_scale_mut(a->data, scalar, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_negate_mut(RefA) -> ok. Modifies A in-place: A = -A */
ERL_NIF_TERM nt_negate_mut(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  vt_simd_negate_mut(a->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/** nt_relu_mut(RefA) -> ok. Modifies A in-place: A = max(0, A) */
ERL_NIF_TERM nt_relu_mut(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  vt_simd_relu_mut(a->data, (size_t)a->size);
  return enif_make_atom(env, "ok");
}

/* =========================================================================
 * Retro / Fused Kernels
 * ========================================================================= */

/** nt_saturn_blend(Texture, Shade, Bias) -> {ok, RefC}
 * VDP1-inspired: result = texture + (shade - bias). Pure addition, no mul. */
ERL_NIF_TERM nt_saturn_blend(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *texture = get_tensor(env, argv[0]);
  NativeTensor *shade = get_tensor(env, argv[1]);
  if (!texture || !shade)
    return make_error(env, "invalid_tensor");
  if (texture->size != shade->size)
    return make_error(env, "size_mismatch");

  double bias;
  if (!enif_get_double(env, argv[2], &bias))
    return make_error(env, "invalid_bias");

  NativeTensor *c = alloc_tensor_uninit(texture->ndim, texture->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_saturn_blend(texture->data, shade->data, bias, c->data,
                  (size_t)texture->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_fused_linear_relu(A, B, Bias, M, N, K) -> {ok, RefC}
 * Fused: C = max(0, A@B + bias). Uses BLAS for matmul + Zig SIMD for bias+relu.
 */
ERL_NIF_TERM nt_fused_linear_relu_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  NativeTensor *bias = get_tensor(env, argv[2]);
  if (!a || !b || !bias)
    return make_error(env, "invalid_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) || !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dims");

  if (a->size != m * k || b->size != k * n || bias->size != n)
    return make_error(env, "shape_mismatch");

  int out_shape[2] = {m, n};
  NativeTensor *c = alloc_tensor_uninit(2, out_shape);
  if (!c)
    return make_error(env, "out_of_memory");

  /* Step 1: C = A @ B via BLAS */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, 1.0, a->data, k, b->data, n, 0.0, c->data, n);
#else
  if (g_dgemm) {
    blas_dgemm(m, n, k, 1.0, a->data, k, b->data, n, 0.0, c->data, n);
  } else {
    free(c->data);
    free(c);
    return make_error(env, "no_blas_backend");
  }
#endif

  /* Step 2: C[i,j] += bias[j] for each row, then ReLU in-place */
  for (int i = 0; i < m; i++) {
    vt_simd_add(c->data + i * n, bias->data, c->data + i * n, (size_t)n);
  }
  vt_simd_relu_mut(c->data, (size_t)(m * n));

  return make_ok(env, make_tensor_term(env, c));
}
