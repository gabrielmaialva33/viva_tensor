/** nif_cuda_fp16.c - CudaTensor16 (FP16) NIFs. Tensor Core HGEMM, fused ops, benchmarks. */

#include "viva_nif.h"

ErlNifResourceType *CUDA_TENSOR16_RESOURCE = NULL;

void cuda_tensor16_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  CudaTensor16 *t = (CudaTensor16 *)obj;
  if (t->d_data) cuda_tensor_free(t->d_data);
  if (t->d_data_t) cuda_tensor_free(t->d_data_t);
  if (t->d_acc) cuda_tensor_free(t->d_acc);
  if (t->shape) free(t->shape);
}

CudaTensor16 *alloc_cuda_tensor16(int ndim, const int *shape) {
  CudaTensor16 *t = (CudaTensor16 *)enif_alloc_resource(CUDA_TENSOR16_RESOURCE, sizeof(CudaTensor16));
  if (!t) return NULL;

  t->ndim = ndim;
  t->size = 1;
  for (int i = 0; i < ndim; i++) t->size *= shape[i];

  t->shape = (int *)malloc(ndim * sizeof(int));
  if (!t->shape) {
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  /* Allocate FP16 on GPU */
  t->d_data = (uint16_t *)cuda_tensor_alloc_fp16((size_t)t->size);
  if (!t->d_data) {
    free(t->shape);
    enif_release_resource(t);
    return NULL;
  }

  t->d_data_t = NULL; /* Allocated on demand for TN path */
  t->d_acc = NULL;     /* Allocated on demand for output */
  return t;
}

CudaTensor16 *get_cuda_tensor16(ErlNifEnv *env, ERL_NIF_TERM term) {
  CudaTensor16 *t;
  if (!enif_get_resource(env, term, CUDA_TENSOR16_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM make_cuda_tensor16_term(ErlNifEnv *env, CudaTensor16 *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

/* FP16 to float32 conversion (float_to_half is in nif_cpu_ops.c via viva_nif.h) */
float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      uint32_t result = sign;
      return *(float*)&result;  /* Zero */
    }
    /* Denormalized */
    exp = 1;
    while (!(mant & 0x400)) {
      mant <<= 1;
      exp--;
    }
    mant &= 0x3FF;
    exp = exp - 1 + 127 - 15;
  } else if (exp == 31) {
    uint32_t result = sign | 0x7F800000 | (mant << 13);
    return *(float*)&result;  /* Inf or NaN */
  } else {
    exp = exp - 15 + 127;
  }

  uint32_t result = sign | (exp << 23) | (mant << 13);
  return *(float*)&result;
}

/** ct16_from_list(Data, Shape) -> {ok, CudaTensor16Ref}
 *  Create FP16 tensor on GPU. Converts f64 -> FP16 during upload.
 */
ERL_NIF_TERM ct16_from_list(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cuda_fp16_available())
    return make_error(env, "fp16_not_available");

  /* Parse shape */
  unsigned shape_len;
  if (!enif_get_list_length(env, argv[1], &shape_len) || shape_len == 0)
    return make_error(env, "invalid_shape");

  int *shape = (int *)malloc(shape_len * sizeof(int));
  if (!shape) return make_error(env, "out_of_memory");

  ERL_NIF_TERM shape_head, shape_tail = argv[1];
  int expected_size = 1;
  for (unsigned i = 0; i < shape_len; i++) {
    if (!enif_get_list_cell(env, shape_tail, &shape_head, &shape_tail)) {
      free(shape);
      return make_error(env, "invalid_shape");
    }
    int dim;
    if (!enif_get_int(env, shape_head, &dim) || dim <= 0) {
      free(shape);
      return make_error(env, "invalid_shape");
    }
    shape[i] = dim;
    expected_size *= dim;
  }

  /* Parse data list and convert to FP16 */
  unsigned data_len;
  if (!enif_get_list_length(env, argv[0], &data_len) || (int)data_len != expected_size) {
    free(shape);
    return make_error(env, "data_shape_mismatch");
  }

  uint16_t *h_data = (uint16_t *)malloc(expected_size * sizeof(uint16_t));
  if (!h_data) {
    free(shape);
    return make_error(env, "out_of_memory");
  }

  ERL_NIF_TERM data_head, data_tail = argv[0];
  for (int i = 0; i < expected_size; i++) {
    if (!enif_get_list_cell(env, data_tail, &data_head, &data_tail)) {
      free(h_data);
      free(shape);
      return make_error(env, "invalid_data");
    }
    double val;
    if (!enif_get_double(env, data_head, &val)) {
      int ival;
      if (!enif_get_int(env, data_head, &ival)) {
        free(h_data);
        free(shape);
        return make_error(env, "invalid_data");
      }
      val = (double)ival;
    }
    h_data[i] = float_to_half((float)val);
  }

  /* Allocate CudaTensor16 and upload */
  CudaTensor16 *t = alloc_cuda_tensor16((int)shape_len, shape);
  free(shape);
  if (!t) {
    free(h_data);
    return make_error(env, "cuda_alloc_failed");
  }

  if (cuda_tensor_upload_fp16(t->d_data, h_data, (size_t)expected_size) != 0) {
    free(h_data);
    enif_release_resource(t);
    return make_error(env, "cuda_upload_failed");
  }

  /* Pre-transpose B for TN cublasLt path (FP16 Tensor Cores) */
  if (t->ndim == 2) {
    int rows = t->shape[0];
    int cols = t->shape[1];
    uint16_t *h_transposed = (uint16_t *)malloc(expected_size * sizeof(uint16_t));
    if (h_transposed) {
      for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
          h_transposed[c * rows + r] = h_data[r * cols + c];

      t->d_data_t = (uint16_t *)cuda_tensor_alloc_fp16((size_t)expected_size);
      if (t->d_data_t)
        cuda_tensor_upload_fp16(t->d_data_t, h_transposed, (size_t)expected_size);
      free(h_transposed);
    }
  }

  free(h_data);
  return make_ok(env, make_cuda_tensor16_term(env, t));
}

/** ct16_to_list(CudaTensor16Ref) -> {ok, List}
 *  Download FP16 from GPU, convert to f64.
 */
ERL_NIF_TERM ct16_to_list(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *t = get_cuda_tensor16(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_tensor16");

  uint16_t *h_data = (uint16_t *)malloc(t->size * sizeof(uint16_t));
  if (!h_data) return make_error(env, "out_of_memory");

  /* Download FP16 from GPU (device->host) */
  if (cuda_tensor_download_fp16(h_data, t->d_data, (size_t)t->size) != 0) {
    free(h_data);
    return make_error(env, "cuda_download_failed");
  }

  ERL_NIF_TERM *terms = (ERL_NIF_TERM *)malloc(t->size * sizeof(ERL_NIF_TERM));
  if (!terms) {
    free(h_data);
    return make_error(env, "out_of_memory");
  }

  for (int i = 0; i < t->size; i++) {
    terms[i] = enif_make_double(env, (double)f16_to_f32(h_data[i]));
  }

  ERL_NIF_TERM list = enif_make_list_from_array(env, terms, t->size);
  free(terms);
  free(h_data);
  return make_ok(env, list);
}

/** ct16_shape(CudaTensor16Ref) -> {ok, Shape} */
ERL_NIF_TERM ct16_shape(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *t = get_cuda_tensor16(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_tensor16");

  ERL_NIF_TERM *dims = (ERL_NIF_TERM *)malloc(t->ndim * sizeof(ERL_NIF_TERM));
  if (!dims) return make_error(env, "out_of_memory");

  for (int i = 0; i < t->ndim; i++) {
    dims[i] = enif_make_int(env, t->shape[i]);
  }

  ERL_NIF_TERM list = enif_make_list_from_array(env, dims, t->ndim);
  free(dims);
  return make_ok(env, list);
}

/** ct16_matmul(RefA, RefB, M, N, K) -> {ok, RefC_FP32}
 *  FP16 HGEMM with Tensor Cores!
 *  Input: FP16 tensors on GPU
 *  Output: FP32 CudaTensor (for accuracy)
 *  FP16 Tensor Core HGEMM on pre-uploaded GPU data.
 */
ERL_NIF_TERM ct16_matmul(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return make_error(env, "invalid_dimensions");

  if (a->size != m * k || b->size != k * n)
    return make_error(env, "size_mismatch");

  /* Output is FP32 CudaTensor for accuracy */
  int out_shape[2] = {m, n};
  CudaTensor *c = alloc_cuda_tensor(2, out_shape);
  if (!c) return make_error(env, "cuda_alloc_failed");

  /* HGEMM: FP16 input, FP32 output with Tensor Cores! */
  int result = cuda_hgemm_gpu(m, n, k,
                               1.0f, a->d_data, k,
                               b->d_data, n,
                               0.0f, c->d_data, n);

  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_hgemm_failed");
  }

  return make_ok(env, make_cuda_tensor_term(env, c));
}

/** ct16_matmul_inplace(RefA, RefB, RefC, M, N, K) -> ok
 *  Pure FP16 in-place matmul: FP16 input -> FP16 output -> CUBLAS_COMPUTE_16F
 *  Zero allocation, half the bandwidth. Maximum Tensor Core throughput!
 */
ERL_NIF_TERM ct16_matmul_inplace_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dimensions");

  if (a->size != m * k || b->size != k * n || c->size != m * n)
    return make_error(env, "size_mismatch");

  /* Pure FP16 async: cublasGemmEx COMPUTE_16F (no sync per call) */
  int result = cuda_hgemm_gpu_pure16_async(m, n, k, a->d_data, k, b->d_data, n, c->d_data, n);

  if (result != 0)
    return make_error(env, "cuda_hgemm_pure16_failed");

  return enif_make_atom(env, "ok");
}

/** ct16_matmul_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  FP16 in-place matmul looped in C. Eliminates ALL Erlang overhead.
 *  For benchmarking: measures pure GPU kernel throughput.
 */
ERL_NIF_TERM ct16_matmul_bench_nif(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  /* Pure async cublasGemmEx loop -- no descriptor overhead */
  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_gpu_pure16_async(m, n, k, a->d_data, k, b->d_data, n, c->d_data, n);
    if (result != 0)
      return make_error(env, "cuda_hgemm_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_lt_32f_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  cublasLt FP16 GEMM with COMPUTE_32F_FAST_16F (no epilogue, baseline).
 */
ERL_NIF_TERM ct16_matmul_lt_32f_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_lt_32f(m, n, k, a->d_data, b->d_data, c->d_data);
    if (result != 0)
      return make_error(env, "cuda_lt_32f_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_fused_relu(RefA, RefB, RefC, M, N, K) -> ok
 *  FP16 fused GEMM+ReLU: C = ReLU(A @ B). Activation is FREE!
 */
ERL_NIF_TERM ct16_matmul_fused_relu_nif(ErlNifEnv *env, int argc,
                                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_args");

  int result = cuda_hgemm_fused_relu(m, n, k, a->d_data, b->d_data, c->d_data);
  if (result != 0)
    return make_error(env, "cuda_fused_relu_failed");

  return enif_make_atom(env, "ok");
}

/** ct16_matmul_fused_gelu(RefA, RefB, RefC, M, N, K) -> ok
 *  FP16 fused GEMM+GELU: C = GELU(A @ B). Activation is FREE!
 */
ERL_NIF_TERM ct16_matmul_fused_gelu_nif(ErlNifEnv *env, int argc,
                                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_args");

  int result = cuda_hgemm_fused_gelu(m, n, k, a->d_data, b->d_data, c->d_data);
  if (result != 0)
    return make_error(env, "cuda_fused_gelu_failed");

  return enif_make_atom(env, "ok");
}

/** ct16_matmul_fused_relu_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  FP16 fused GEMM+ReLU bench loop in C. Zero Erlang overhead.
 */
ERL_NIF_TERM ct16_matmul_fused_relu_bench_nif(ErlNifEnv *env, int argc,
                                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_fused_relu(m, n, k, a->d_data, b->d_data, c->d_data);
    if (result != 0)
      return make_error(env, "cuda_fused_relu_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_fused_gelu_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  FP16 fused GEMM+GELU bench loop in C. Zero Erlang overhead.
 */
ERL_NIF_TERM ct16_matmul_fused_gelu_bench_nif(ErlNifEnv *env, int argc,
                                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_fused_gelu(m, n, k, a->d_data, b->d_data, c->d_data);
    if (result != 0)
      return make_error(env, "cuda_fused_gelu_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_batched_bench(M, N, K, BatchCount, Iters) -> ok
 *  FP16 batched GEMM bench. Allocates contiguous GPU buffers internally.
 *  Measures aggregate throughput for batch*M*N*K FLOPs.
 */
ERL_NIF_TERM ct16_matmul_batched_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, batch_count, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &batch_count) ||
      !enif_get_int(env, argv[4], &iters))
    return make_error(env, "invalid_args");

  /* Allocate contiguous GPU buffers for all batches */
  extern uint16_t* cuda_tensor_alloc_fp16(size_t);
  size_t sizeA = (size_t)batch_count * m * k;
  size_t sizeB = (size_t)batch_count * k * n;
  size_t sizeC = (size_t)batch_count * m * n;

  uint16_t *d_A = cuda_tensor_alloc_fp16(sizeA);
  uint16_t *d_B = cuda_tensor_alloc_fp16(sizeB);
  uint16_t *d_C = cuda_tensor_alloc_fp16(sizeC);

  if (!d_A || !d_B || !d_C) {
    extern void cuda_tensor_free(void*);
    if (d_A) cuda_tensor_free(d_A);
    if (d_B) cuda_tensor_free(d_B);
    if (d_C) cuda_tensor_free(d_C);
    return make_error(env, "gpu_alloc_failed");
  }

  /* Warmup (1 iter) */
  cuda_hgemm_batched(m, n, k, batch_count, d_A, d_B, d_C);
  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  /* Bench loop */
  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_batched(m, n, k, batch_count, d_A, d_B, d_C);
    if (result != 0) {
      extern void cuda_tensor_free(void*);
      cuda_tensor_free(d_A); cuda_tensor_free(d_B); cuda_tensor_free(d_C);
      return make_error(env, "cuda_batched_bench_failed");
    }
  }

  extern void cuda_tensor_free(void*);
  cuda_tensor_free(d_A);
  cuda_tensor_free(d_B);
  cuda_tensor_free(d_C);

  return enif_make_atom(env, "ok");
}

/** fp8_matmul_lt_tn_bench(M, N, K, Iters) -> ok
 *  FP8 E4M3 GEMM bench via cublasLt TN layout.
 *  Allocates FP8 GPU buffers internally, pre-transposes B, runs C-loop.
 *  Allocates FP8 GPU buffers, pre-transposes B, runs C-loop bench.
 */
ERL_NIF_TERM fp8_matmul_lt_tn_bench_nif(ErlNifEnv *env, int argc,
                                                  const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters))
    return make_error(env, "invalid_args");

  /* Allocate FP8 (uint8_t) GPU buffers for A[M*K] and B_T[N*K], FP16 for C[M*N] */
  extern uint8_t* cuda_tensor_alloc_fp8(size_t);
  extern uint16_t* cuda_tensor_alloc_fp16(size_t);
  extern void cuda_tensor_free(void*);

  size_t sizeA  = (size_t)m * k;
  size_t sizeBT = (size_t)n * k;   /* B transposed: N*K */
  size_t sizeC  = (size_t)m * n;

  uint8_t  *d_A  = cuda_tensor_alloc_fp8(sizeA);
  uint8_t  *d_BT = cuda_tensor_alloc_fp8(sizeBT);
  uint16_t *d_C  = cuda_tensor_alloc_fp16(sizeC);

  if (!d_A || !d_BT || !d_C) {
    if (d_A)  cuda_tensor_free(d_A);
    if (d_BT) cuda_tensor_free(d_BT);
    if (d_C)  cuda_tensor_free(d_C);
    return make_error(env, "gpu_alloc_failed");
  }

  /* Warmup (1 iter) */
  cuda_fp8gemm_lt_gpu_tn(m, n, k, d_A, d_BT, d_C);
  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  /* Bench loop -- no sync between iters for max throughput */
  for (int i = 0; i < iters; i++) {
    int result = cuda_fp8gemm_lt_gpu_tn(m, n, k, d_A, d_BT, d_C);
    if (result != 0) {
      cuda_tensor_free(d_A); cuda_tensor_free(d_BT); cuda_tensor_free(d_C);
      return make_error(env, "cuda_fp8_tn_bench_failed");
    }
  }

  cuda_tensor_free(d_A);
  cuda_tensor_free(d_BT);
  cuda_tensor_free(d_C);

  return enif_make_atom(env, "ok");
}

/** cutlass_fp8_f16acc_bench_nif(M, N, K, Iters) -> ok
 *  CUTLASS FP8 E4M3 GEMM with FP16 accumulator (bypasses GeForce FP8+FP32 half-rate).
 *  A[M,K] row-major, B[K,N] col-major, C[M,N] row-major (all FP8 in, FP16 out)
 */
ERL_NIF_TERM cutlass_fp8_f16acc_bench_nif(ErlNifEnv *env, int argc,
                                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters))
    return make_error(env, "invalid_args");

  extern uint8_t* cuda_tensor_alloc_fp8(size_t);
  extern uint16_t* cuda_tensor_alloc_fp16(size_t);
  extern void cuda_tensor_free(void*);

  /* A[M,K] row-major (FP8), B[K,N] col-major (FP8), C[M,N] row-major (FP16) */
  size_t sizeA = (size_t)m * k;
  size_t sizeB = (size_t)k * n;   /* col-major: same total elements */
  size_t sizeC = (size_t)m * n;

  uint8_t  *d_A = cuda_tensor_alloc_fp8(sizeA);
  uint8_t  *d_B = cuda_tensor_alloc_fp8(sizeB);
  uint16_t *d_C = cuda_tensor_alloc_fp16(sizeC);

  if (!d_A || !d_B || !d_C) {
    if (d_A) cuda_tensor_free(d_A);
    if (d_B) cuda_tensor_free(d_B);
    if (d_C) cuda_tensor_free(d_C);
    return make_error(env, "gpu_alloc_failed");
  }

  /* Warmup */
  cutlass_fp8_gemm_f16acc(m, n, k, d_A, d_B, d_C);
  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  /* Bench loop -- no sync between iters for max throughput */
  for (int i = 0; i < iters; i++) {
    int result = cutlass_fp8_gemm_f16acc(m, n, k, d_A, d_B, d_C);
    if (result != 0) {
      cuda_tensor_free(d_A); cuda_tensor_free(d_B); cuda_tensor_free(d_C);
      char errbuf[64];
      snprintf(errbuf, sizeof(errbuf), "cutlass_fp8_f16acc_failed_%d", result);
      return make_error(env, errbuf);
    }
  }

  cuda_tensor_free(d_A);
  cuda_tensor_free(d_B);
  cuda_tensor_free(d_C);

  return enif_make_atom(env, "ok");
}

/** cutlass_fp8_f32acc_bench_nif(M, N, K, Iters) -> ok
 *  CUTLASS FP8 E4M3 GEMM with FP32 accumulator (same rate as cuBLASLt).
 */
ERL_NIF_TERM cutlass_fp8_f32acc_bench_nif(ErlNifEnv *env, int argc,
                                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters))
    return make_error(env, "invalid_args");

  extern uint8_t* cuda_tensor_alloc_fp8(size_t);
  extern uint16_t* cuda_tensor_alloc_fp16(size_t);
  extern void cuda_tensor_free(void*);

  size_t sizeA = (size_t)m * k;
  size_t sizeB = (size_t)k * n;
  size_t sizeC = (size_t)m * n;

  uint8_t  *d_A = cuda_tensor_alloc_fp8(sizeA);
  uint8_t  *d_B = cuda_tensor_alloc_fp8(sizeB);
  uint16_t *d_C = cuda_tensor_alloc_fp16(sizeC);

  if (!d_A || !d_B || !d_C) {
    if (d_A) cuda_tensor_free(d_A);
    if (d_B) cuda_tensor_free(d_B);
    if (d_C) cuda_tensor_free(d_C);
    return make_error(env, "gpu_alloc_failed");
  }

  cutlass_fp8_gemm_f32acc(m, n, k, d_A, d_B, d_C);
  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  for (int i = 0; i < iters; i++) {
    int result = cutlass_fp8_gemm_f32acc(m, n, k, d_A, d_B, d_C);
    if (result != 0) {
      cuda_tensor_free(d_A); cuda_tensor_free(d_B); cuda_tensor_free(d_C);
      char errbuf[64];
      snprintf(errbuf, sizeof(errbuf), "cutlass_fp8_f32acc_failed_%d", result);
      return make_error(env, errbuf);
    }
  }

  cuda_tensor_free(d_A);
  cuda_tensor_free(d_B);
  cuda_tensor_free(d_C);

  return enif_make_atom(env, "ok");
}

/** cutlass_int8_sparse_bench_nif(M, N, K, Iters, Config) -> ok
 *  CUTLASS INT8 2:4 Structured Sparse GEMM benchmark.
 *  Config: 0=128x256x128, 1=256x128x128, 2=128x128x256, 3=128x128x128
 */
ERL_NIF_TERM cutlass_int8_sparse_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters, config;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters) ||
      !enif_get_int(env, argv[4], &config))
    return make_error(env, "invalid_args");

  int result;
  switch (config) {
    case 0: result = cutlass_int8_sparse_gemm_bench(m, n, k, iters); break;
    case 1: result = cutlass_int8_sparse_gemm_bench_b(m, n, k, iters); break;
    case 2: result = cutlass_int8_sparse_gemm_bench_c(m, n, k, iters); break;
    case 3: result = cutlass_int8_sparse_gemm_bench_d(m, n, k, iters); break;
    default: return make_error(env, "invalid_config");
  }

  if (result != 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "cutlass_int8_sparse_failed_%d", result);
    return make_error(env, errbuf);
  }

  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  return enif_make_atom(env, "ok");
}

/** cutlass_int8_sparse_bench_ex_nif(M, N, K, Iters, Config, SplitK) -> ok
 *  Extended CUTLASS INT8 2:4 Sparse GEMM with split-K support.
 *  Config: 0=128x128x128/3stg, 1=128x128x256/2stg, 2=128x256x128/2stg,
 *          3=256x128x128/2stg, 4=64x128x128/3stg, 5=128x128x128/4stg
 *  SplitK: number of K-dimension partitions (1,2,4,8,16)
 */
ERL_NIF_TERM cutlass_int8_sparse_bench_ex_nif(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters, config, split_k;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters) ||
      !enif_get_int(env, argv[4], &config) ||
      !enif_get_int(env, argv[5], &split_k))
    return make_error(env, "invalid_args");

  int result = cutlass_int8_sparse_gemm_bench_ex(m, n, k, iters, config, split_k);

  if (result != 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "cutlass_int8_sparse_ex_failed_%d", result);
    return make_error(env, errbuf);
  }

  extern void cuda_explicit_sync(void);
  cuda_explicit_sync();

  return enif_make_atom(env, "ok");
}

/** cusparselt_int8_sparse_bench_nif(M, N, K, Iters, Mode) -> {ok, ElapsedUs}
 *  cuSPARSELt INT8 2:4 Sparse GEMM -- returns kernel-only time in microseconds.
 *  Mode: 0=auto (unused, cuSPARSELt auto-tunes via MatmulSearch).
 */
ERL_NIF_TERM cusparselt_int8_sparse_bench_nif(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters, mode;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters) ||
      !enif_get_int(env, argv[4], &mode))
    return make_error(env, "invalid_args");

  int result = cusparselt_int8_sparse_bench(m, n, k, iters, mode);

  if (result < 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "cusparselt_int8_failed_%d", result);
    return make_error(env, errbuf);
  }

  /* result > 0 = elapsed microseconds (kernel-only from CUDA events) */
  return enif_make_tuple2(env,
    enif_make_atom(env, "ok"),
    enif_make_int(env, result));
}

/** cusparselt_fp8_sparse_bench_nif(M, N, K, Iters) -> {ok, ElapsedUs}
 *  cuSPARSELt FP8 E4M3 2:4 Sparse GEMM -- returns kernel-only time in microseconds.
 */
ERL_NIF_TERM cusparselt_fp8_sparse_bench_nif(ErlNifEnv *env, int argc,
                                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters))
    return make_error(env, "invalid_args");

  int result = cusparselt_fp8_sparse_bench(m, n, k, iters);

  if (result < 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "cusparselt_fp8_failed_%d", result);
    return make_error(env, errbuf);
  }

  return enif_make_tuple2(env,
    enif_make_atom(env, "ok"),
    enif_make_int(env, result));
}

/** cusparselt_fp16_sparse_bench_nif(M, N, K, Iters) -> {ok, ElapsedUs}
 *  cuSPARSELt FP16 2:4 Sparse GEMM -- returns kernel-only time in microseconds.
 */
ERL_NIF_TERM cusparselt_fp16_sparse_bench_nif(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters))
    return make_error(env, "invalid_args");

  int result = cusparselt_fp16_sparse_bench(m, n, k, iters);

  if (result < 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "cusparselt_fp16_failed_%d", result);
    return make_error(env, errbuf);
  }

  return enif_make_tuple2(env,
    enif_make_atom(env, "ok"),
    enif_make_int(env, result));
}

/** cutlass_int4_sparse_bench_nif(M, N, K, Iters, Config, SplitK) -> {ok, ElapsedUs}
 *  CUTLASS INT4 2:4 Sparse GEMM benchmark.
 *  Returns kernel-only elapsed time in microseconds.
 */
ERL_NIF_TERM cutlass_int4_sparse_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  int m, n, k, iters, config, split_k;
  if (!enif_get_int(env, argv[0], &m) ||
      !enif_get_int(env, argv[1], &n) ||
      !enif_get_int(env, argv[2], &k) ||
      !enif_get_int(env, argv[3], &iters) ||
      !enif_get_int(env, argv[4], &config) ||
      !enif_get_int(env, argv[5], &split_k))
    return make_error(env, "invalid_args");

  int result = cutlass_int4_sparse_gemm_bench(m, n, k, iters, config, split_k);

  if (result < 0) {
    char errbuf[64];
    snprintf(errbuf, sizeof(errbuf), "int4_sparse_failed_%d", result);
    return make_error(env, errbuf);
  }

  return enif_make_tuple2(env,
    enif_make_atom(env, "ok"),
    enif_make_int(env, result));
}

/** ct16_matmul_fused_relu_tn_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  FP16 fused GEMM+ReLU bench with TN layout (B pre-transposed).
 */
ERL_NIF_TERM ct16_matmul_fused_relu_tn_bench_nif(ErlNifEnv *env, int argc,
                                                          const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");
  if (!b->d_data_t) return make_error(env, "b_not_transposed");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_fused_relu_tn(m, n, k, a->d_data, b->d_data_t, c->d_data);
    if (result != 0)
      return make_error(env, "cuda_fused_relu_tn_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_fused_gelu_tn_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  FP16 fused GEMM+GELU bench with TN layout (B pre-transposed).
 */
ERL_NIF_TERM ct16_matmul_fused_gelu_tn_bench_nif(ErlNifEnv *env, int argc,
                                                          const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor16");
  if (!b->d_data_t) return make_error(env, "b_not_transposed");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  for (int i = 0; i < iters; i++) {
    int result = cuda_hgemm_fused_gelu_tn(m, n, k, a->d_data, b->d_data_t, c->d_data);
    if (result != 0)
      return make_error(env, "cuda_fused_gelu_tn_bench_failed");
  }
  return enif_make_atom(env, "ok");
}

/** ct16_available() -> true | false */
ERL_NIF_TERM ct16_available(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  return cuda_fp16_available() ? enif_make_atom(env, "true")
                                : enif_make_atom(env, "false");
}

/** cuda_sync() -> ok
 *  Explicit GPU sync - call when you need results
 */
ERL_NIF_TERM nif_cuda_sync(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  cuda_explicit_sync();
  return enif_make_atom(env, "ok");
}

/** ct16_matmul_async(RefA, RefB, M, N, K) -> {ok, RefC}
 *  FP16 HGEMM async (no sync) - Tensor Cores without sync overhead
 *  FP16 HGEMM async — Tensor Cores without sync overhead.
 */
ERL_NIF_TERM ct16_matmul_async(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor16 *a = get_cuda_tensor16(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_tensor16");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return make_error(env, "invalid_dimensions");

  if (a->size != m * k || b->size != k * n)
    return make_error(env, "size_mismatch");

  int out_shape[2] = {m, n};
  CudaTensor *c = alloc_cuda_tensor(2, out_shape);
  if (!c) return make_error(env, "cuda_alloc_failed");

  /* HGEMM async - no sync! */
  int result = cuda_hgemm_gpu_async(m, n, k,
                                     1.0f, a->d_data, k,
                                     b->d_data, n,
                                     0.0f, c->d_data, n);

  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_hgemm_async_failed");
  }

  return make_ok(env, make_cuda_tensor_term(env, c));
}
