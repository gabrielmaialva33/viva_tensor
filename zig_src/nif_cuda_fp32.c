/** nif_cuda_fp32.c - CudaTensor (FP32) NIFs. Persistent GPU memory, SGEMM via cuBLAS. */

#include "viva_nif.h"

ErlNifResourceType *CUDA_TENSOR_RESOURCE = NULL;

void cuda_tensor_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  CudaTensor *t = (CudaTensor *)obj;
  if (t->d_data) cuda_tensor_free(t->d_data);
  if (t->shape) free(t->shape);
}

CudaTensor *alloc_cuda_tensor(int ndim, const int *shape) {
  CudaTensor *t = (CudaTensor *)enif_alloc_resource(CUDA_TENSOR_RESOURCE, sizeof(CudaTensor));
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

  t->d_data = cuda_tensor_alloc((size_t)t->size);
  if (!t->d_data) {
    free(t->shape);
    enif_release_resource(t);
    return NULL;
  }

  return t;
}

CudaTensor *get_cuda_tensor(ErlNifEnv *env, ERL_NIF_TERM term) {
  CudaTensor *t;
  if (!enif_get_resource(env, term, CUDA_TENSOR_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM make_cuda_tensor_term(ErlNifEnv *env, CudaTensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

/** ct_from_list(Data, Shape) -> {ok, CudaTensorRef}
 *  Create CudaTensor from list, upload to GPU ONCE.
 *  Subsequent operations stay on GPU - no transfer overhead!
 */
ERL_NIF_TERM ct_from_list(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cuda_available())
    return make_error(env, "cuda_not_available");

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

  /* Convert to float array (FP32 on GPU) */
  float *h_data = (float *)malloc(expected_size * sizeof(float));
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
    h_data[i] = (float)val;
  }

  /* Allocate CudaTensor and upload */
  CudaTensor *t = alloc_cuda_tensor((int)shape_len, shape);
  free(shape);
  if (!t) {
    free(h_data);
    return make_error(env, "cuda_alloc_failed");
  }

  if (cuda_tensor_upload(t->d_data, h_data, (size_t)expected_size) != 0) {
    free(h_data);
    enif_release_resource(t);
    return make_error(env, "cuda_upload_failed");
  }

  free(h_data);
  return make_ok(env, make_cuda_tensor_term(env, t));
}

/** ct_to_list(CudaTensorRef) -> {ok, List}
 *  Download from GPU to CPU. Only call when you need the data!
 */
ERL_NIF_TERM ct_to_list(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor *t = get_cuda_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_tensor");

  float *h_data = (float *)malloc(t->size * sizeof(float));
  if (!h_data) return make_error(env, "out_of_memory");

  if (cuda_tensor_download(h_data, t->d_data, (size_t)t->size) != 0) {
    free(h_data);
    return make_error(env, "cuda_download_failed");
  }

  ERL_NIF_TERM *terms = (ERL_NIF_TERM *)malloc(t->size * sizeof(ERL_NIF_TERM));
  if (!terms) {
    free(h_data);
    return make_error(env, "out_of_memory");
  }

  for (int i = 0; i < t->size; i++) {
    terms[i] = enif_make_double(env, (double)h_data[i]);
  }

  ERL_NIF_TERM list = enif_make_list_from_array(env, terms, t->size);
  free(terms);
  free(h_data);
  return make_ok(env, list);
}

/** ct_shape(CudaTensorRef) -> {ok, Shape} */
ERL_NIF_TERM ct_shape(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor *t = get_cuda_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_cuda_tensor");

  ERL_NIF_TERM *dims = (ERL_NIF_TERM *)malloc(t->ndim * sizeof(ERL_NIF_TERM));
  if (!dims) return make_error(env, "out_of_memory");

  for (int i = 0; i < t->ndim; i++) {
    dims[i] = enif_make_int(env, t->shape[i]);
  }

  ERL_NIF_TERM list = enif_make_list_from_array(env, dims, t->ndim);
  free(dims);
  return make_ok(env, list);
}

/** ct_matmul(RefA, RefB, M, N, K) -> {ok, RefC}
 *  SGEMM with data ALREADY on GPU - NO PCIe transfer!
 *  SGEMM on pre-uploaded GPU tensors (no PCIe overhead).
 */
ERL_NIF_TERM ct_matmul(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor *a = get_cuda_tensor(env, argv[0]);
  CudaTensor *b = get_cuda_tensor(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_tensor");

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

  int result = cuda_sgemm_gpu(m, n, k,
                               1.0f, a->d_data, k,
                               b->d_data, n,
                               0.0f, c->d_data, n);

  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_sgemm_failed");
  }

  return make_ok(env, make_cuda_tensor_term(env, c));
}

/** ct_matmul_inplace(RefA, RefB, RefC, M, N, K) -> ok
 *  FP32 in-place matmul on GPU. Zero allocation!
 */
ERL_NIF_TERM ct_matmul_inplace_nif(ErlNifEnv *env, int argc,
                                           const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor *a = get_cuda_tensor(env, argv[0]);
  CudaTensor *b = get_cuda_tensor(env, argv[1]);
  CudaTensor *c = get_cuda_tensor(env, argv[2]);
  if (!a || !b || !c) return make_error(env, "invalid_cuda_tensor");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dimensions");

  if (a->size != m * k || b->size != k * n || c->size != m * n)
    return make_error(env, "size_mismatch");

  int result = cuda_sgemm_gpu_inplace(m, n, k,
                                       1.0f, a->d_data, k,
                                       b->d_data, n,
                                       0.0f, c->d_data, n);

  if (result != 0)
    return make_error(env, "cuda_sgemm_inplace_failed");

  return enif_make_atom(env, "ok");
}

/** ct_matmul_async(RefA, RefB, M, N, K) -> {ok, RefC}
 *  FP32 SGEMM async (no sync) - for pipelined workloads
 */
ERL_NIF_TERM ct_matmul_async(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  CudaTensor *a = get_cuda_tensor(env, argv[0]);
  CudaTensor *b = get_cuda_tensor(env, argv[1]);
  if (!a || !b) return make_error(env, "invalid_cuda_tensor");

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

  /* SGEMM async - no sync! */
  int result = cuda_sgemm_gpu_async(m, n, k,
                                     1.0f, a->d_data, k,
                                     b->d_data, n,
                                     0.0f, c->d_data, n);

  if (result != 0) {
    enif_release_resource(c);
    return make_error(env, "cuda_sgemm_async_failed");
  }

  return make_ok(env, make_cuda_tensor_term(env, c));
}
