/** nif_cuda_int8.c - CudaInt8Tensor NIFs. INT8 IMMA Tensor Cores via cublasLt TN. */

#include "viva_nif.h"

ErlNifResourceType *CUDA_INT8_TENSOR_RESOURCE = NULL;

void cuda_int8_tensor_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  CudaInt8Tensor *t = (CudaInt8Tensor *)obj;
  if (t->d_data) cuda_tensor_free(t->d_data);
  if (t->d_data_t) cuda_tensor_free(t->d_data_t);
  if (t->d_acc) cuda_tensor_free(t->d_acc);
  if (t->shape) free(t->shape);
}

CudaInt8Tensor *alloc_cuda_int8_tensor(int ndim, const int *shape) {
  CudaInt8Tensor *t = (CudaInt8Tensor *)enif_alloc_resource(CUDA_INT8_TENSOR_RESOURCE, sizeof(CudaInt8Tensor));
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

  /* Allocate INT8 on GPU */
  t->d_data = cuda_tensor_alloc_int8((size_t)t->size);
  if (!t->d_data) {
    free(t->shape);
    enif_release_resource(t);
    return NULL;
  }

  t->d_data_t = NULL;  /* Transposed copy, allocated in ct_int8_from_list for 2D */
  t->d_acc = NULL;  /* Allocated on demand for output */
  return t;
}

CudaInt8Tensor *get_cuda_int8_tensor(ErlNifEnv *env, ERL_NIF_TERM term) {
  CudaInt8Tensor *t;
  if (!enif_get_resource(env, term, CUDA_INT8_TENSOR_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM make_cuda_int8_tensor_term(ErlNifEnv *env, CudaInt8Tensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

/** ct_int8_available() -> true | false
 *  Check if INT8 Tensor Cores (cublasLt IMMA) are available.
 */
ERL_NIF_TERM ct_int8_available(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  return cuda_int8_lt_available() ? enif_make_atom(env, "true")
                                   : enif_make_atom(env, "false");
}

/** ct_int8_from_list(Data, Shape) -> {ok, CudaInt8TensorRef}
 *  Create CudaInt8Tensor from list, quantize and upload to GPU ONCE.
 *  Data should be floats in range [-1.0, 1.0], quantized to INT8 [-127, 127].
 */
ERL_NIF_TERM ct_int8_from_list(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cuda_int8_lt_available())
    return make_error(env, "int8_tensor_cores_not_available");

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

  /* Parse data list */
  unsigned data_len;
  if (!enif_get_list_length(env, argv[0], &data_len) || (int)data_len != expected_size) {
    free(shape);
    return make_error(env, "data_shape_mismatch");
  }

  /* Convert to float array, then find absmax for quantization */
  float *f_data = (float *)malloc(expected_size * sizeof(float));
  if (!f_data) {
    free(shape);
    return make_error(env, "out_of_memory");
  }

  float absmax = 0.0f;
  ERL_NIF_TERM data_head, data_tail = argv[0];
  for (int i = 0; i < expected_size; i++) {
    if (!enif_get_list_cell(env, data_tail, &data_head, &data_tail)) {
      free(f_data);
      free(shape);
      return make_error(env, "invalid_data");
    }
    double val;
    if (!enif_get_double(env, data_head, &val)) {
      int ival;
      if (!enif_get_int(env, data_head, &ival)) {
        free(f_data);
        free(shape);
        return make_error(env, "invalid_data");
      }
      val = (double)ival;
    }
    f_data[i] = (float)val;
    float abs_val = f_data[i] < 0 ? -f_data[i] : f_data[i];
    if (abs_val > absmax) absmax = abs_val;
  }

  /* Quantize to INT8 */
  int8_t *h_data = (int8_t *)malloc(expected_size * sizeof(int8_t));
  if (!h_data) {
    free(f_data);
    free(shape);
    return make_error(env, "out_of_memory");
  }

  float scale = (absmax > 0.0f) ? 127.0f / absmax : 1.0f;
  for (int i = 0; i < expected_size; i++) {
    float scaled = f_data[i] * scale;
    if (scaled > 127.0f) scaled = 127.0f;
    if (scaled < -127.0f) scaled = -127.0f;
    h_data[i] = (int8_t)scaled;
  }
  free(f_data);

  /* Allocate CudaInt8Tensor and upload */
  CudaInt8Tensor *t = alloc_cuda_int8_tensor((int)shape_len, shape);
  free(shape);
  if (!t) {
    free(h_data);
    return make_error(env, "cuda_alloc_failed");
  }

  if (cuda_tensor_upload_int8(t->d_data, h_data, (size_t)expected_size) != 0) {
    free(h_data);
    enif_release_resource(t);
    return make_error(env, "cuda_upload_failed");
  }

  /* For 2D tensors, create transposed copy for TN IMMA Tensor Cores.
   * Original: row-major [rows][cols] -> Transposed: row-major [cols][rows]
   * Pre-transpose B for TN layout (IMMA requires TN for full throughput). */
  if (shape_len == 2) {
    int rows = t->shape[0], cols = t->shape[1];
    int8_t *h_transposed = (int8_t *)malloc(expected_size * sizeof(int8_t));
    if (h_transposed) {
      for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
          h_transposed[c * rows + r] = h_data[r * cols + c];

      t->d_data_t = cuda_tensor_alloc_int8((size_t)expected_size);
      if (t->d_data_t) {
        cuda_tensor_upload_int8(t->d_data_t, h_transposed, (size_t)expected_size);
      }
      free(h_transposed);
    }
  }

  free(h_data);
  return make_ok(env, make_cuda_int8_tensor_term(env, t));
}

/** ct_int8_to_list(CudaInt8TensorRef) -> {ok, List}
 *  Download INT32 accumulator from GPU and convert to Erlang list.
 */
ERL_NIF_TERM ct_int8_to_list(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;

  CudaInt8Tensor *t = get_cuda_int8_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_int8_tensor");

  if (!t->d_acc) return make_error(env, "no_accumulator_data");

  /* Download INT32 accumulator */
  int32_t *h_data = (int32_t *)malloc(t->size * sizeof(int32_t));
  if (!h_data) return make_error(env, "out_of_memory");

  if (cuda_tensor_download_int32(h_data, t->d_acc, (size_t)t->size) != 0) {
    free(h_data);
    return make_error(env, "cuda_download_failed");
  }

  /* Convert to Erlang list of integers */
  ERL_NIF_TERM *terms = (ERL_NIF_TERM *)malloc(t->size * sizeof(ERL_NIF_TERM));
  if (!terms) {
    free(h_data);
    return make_error(env, "out_of_memory");
  }

  for (int i = 0; i < t->size; i++) {
    terms[i] = enif_make_int(env, h_data[i]);
  }

  ERL_NIF_TERM list = enif_make_list_from_array(env, terms, t->size);
  free(h_data);
  free(terms);

  return make_ok(env, list);
}

/** ct_int8_shape(CudaInt8TensorRef) -> {ok, Shape} */
ERL_NIF_TERM ct_int8_shape(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;

  CudaInt8Tensor *t = get_cuda_int8_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_int8_tensor");

  ERL_NIF_TERM *dims = (ERL_NIF_TERM *)malloc(t->ndim * sizeof(ERL_NIF_TERM));
  if (!dims) return make_error(env, "out_of_memory");

  for (int i = 0; i < t->ndim; i++) {
    dims[i] = enif_make_int(env, t->shape[i]);
  }

  ERL_NIF_TERM shape_list = enif_make_list_from_array(env, dims, t->ndim);
  free(dims);

  return make_ok(env, shape_list);
}

/** ct_int8_matmul(RefA, RefB, M, N, K) -> {ok, RefC}
 *  INT8 GEMM with Tensor Cores on GPU - sync version.
 *  A [M*K] * B [K*N] = C [M*N]
 *  Output is INT32 accumulator.
 */
ERL_NIF_TERM ct_int8_matmul(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;

  CudaInt8Tensor *a = get_cuda_int8_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_int8_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k)) {
    return make_error(env, "invalid_dimensions");
  }

  if (a->size != m * k || b->size != k * n) {
    return make_error(env, "dimension_mismatch");
  }

  /* Allocate output tensor */
  int out_shape[2] = {m, n};
  CudaInt8Tensor *c = alloc_cuda_int8_tensor(2, out_shape);
  if (!c) return make_error(env, "cuda_alloc_failed");

  /* Allocate INT32 accumulator for output */
  c->d_acc = cuda_tensor_alloc_int32((size_t)(m * n));
  if (!c->d_acc) {
    enif_release_resource(c);
    return make_error(env, "cuda_alloc_failed");
  }

  /* IMMA Tensor Core matmul — TN path, falls back to NN if dims unaligned. */
  int result;
  if (b->d_data_t) {
    result = cuda_igemm_lt_gpu_tn(m, n, k, a->d_data, b->d_data_t, c->d_acc);
    if (result != 0)
      result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  } else {
    result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  }
  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_igemm_lt_gpu_failed");
  }

  return make_ok(env, make_cuda_int8_tensor_term(env, c));
}

/** ct_int8_matmul_async(RefA, RefB, M, N, K) -> {ok, RefC}
 *  INT8 GEMM with Tensor Cores on GPU - ASYNC version (NO SYNC!)
 *  For pipeline benchmarking - call cuda_sync/0 when done.
 *  INT8 IMMA async — for pipeline benchmarking.
 */
ERL_NIF_TERM ct_int8_matmul_async(ErlNifEnv *env, int argc,
                                          const ERL_NIF_TERM argv[]) {
  (void)argc;

  CudaInt8Tensor *a = get_cuda_int8_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_int8_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k)) {
    return make_error(env, "invalid_dimensions");
  }

  if (a->size != m * k || b->size != k * n) {
    return make_error(env, "dimension_mismatch");
  }

  /* Allocate output tensor */
  int out_shape[2] = {m, n};
  CudaInt8Tensor *c = alloc_cuda_int8_tensor(2, out_shape);
  if (!c) return make_error(env, "cuda_alloc_failed");

  /* Allocate INT32 accumulator for output */
  c->d_acc = cuda_tensor_alloc_int32((size_t)(m * n));
  if (!c->d_acc) {
    enif_release_resource(c);
    return make_error(env, "cuda_alloc_failed");
  }

  /* Execute IMMA Tensor Core matmul - TN path, NO SYNC!
   * Falls back to NN if TN heuristic fails (non-aligned dims). */
  int result;
  if (b->d_data_t) {
    result = cuda_igemm_lt_gpu_tn(m, n, k, a->d_data, b->d_data_t, c->d_acc);
    if (result != 0)
      result = cuda_igemm_lt_gpu_async(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  } else {
    result = cuda_igemm_lt_gpu_async(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  }
  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_igemm_lt_gpu_async_failed");
  }

  return make_ok(env, make_cuda_int8_tensor_term(env, c));
}

/** ct_int8_matmul_inplace(RefA, RefB, RefC, M, N, K) -> ok
 *  INT8 in-place matmul. RefC must have d_acc already allocated.
 *  Zero allocation per call!
 */
ERL_NIF_TERM ct_int8_matmul_inplace_nif(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaInt8Tensor *a = get_cuda_int8_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  CudaInt8Tensor *c = get_cuda_int8_tensor(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_int8_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dimensions");

  if (a->size != m * k || b->size != k * n || c->size != m * n)
    return make_error(env, "size_mismatch");

  /* Ensure C has accumulator allocated */
  if (!c->d_acc) {
    c->d_acc = cuda_tensor_alloc_int32((size_t)(m * n));
    if (!c->d_acc) return make_error(env, "cuda_alloc_failed");
  }

  /* TN path for IMMA Tensor Cores, falls back to NN if dims unaligned. */
  int result;
  if (b->d_data_t) {
    result = cuda_igemm_lt_gpu_tn(m, n, k, a->d_data, b->d_data_t, c->d_acc);
    if (result != 0)
      result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  } else {
    result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
  }
  if (result != 0)
    return make_error(env, "cuda_igemm_lt_inplace_failed");

  return enif_make_atom(env, "ok");
}

/** ct_int8_matmul_bench(RefA, RefB, RefC, M, N, K, Iters) -> ok
 *  INT8 in-place matmul looped in C. Eliminates ALL Erlang overhead.
 */
ERL_NIF_TERM ct_int8_matmul_bench_nif(ErlNifEnv *env, int argc,
                                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaInt8Tensor *a = get_cuda_int8_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  CudaInt8Tensor *c = get_cuda_int8_tensor(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_int8_tensor");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  if (!c->d_acc) {
    c->d_acc = cuda_tensor_alloc_int32((size_t)(m * n));
    if (!c->d_acc) return make_error(env, "cuda_alloc_failed");
  }

  for (int i = 0; i < iters; i++) {
    int result;
    if (b->d_data_t) {
      result = cuda_igemm_lt_gpu_tn(m, n, k, a->d_data, b->d_data_t, c->d_acc);
      if (result != 0)
        result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
    } else {
      result = cuda_igemm_lt_gpu(m, n, k, a->d_data, k, b->d_data, n, c->d_acc, n);
    }
    if (result != 0)
      return make_error(env, "cuda_igemm_bench_failed");
  }
  return enif_make_atom(env, "ok");
}
