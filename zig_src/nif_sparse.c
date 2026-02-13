/**
 * nif_sparse.c - SparseTensor NIFs (2:4 structured sparsity via cuSPARSELt)
 * Supports FP16 and INT8 sparse tensor creation, matmul, and benchmarking.
 */

#include "viva_nif.h"

ErlNifResourceType *SPARSE_TENSOR_RESOURCE = NULL;

void sparse_tensor_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  SparseTensorInternal *t = (SparseTensorInternal *)obj;
  sparse_tensor_free(t);
}

ERL_NIF_TERM make_sparse_tensor_term(ErlNifEnv *env, SparseTensorInternal *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

SparseTensorInternal *get_sparse_tensor(ErlNifEnv *env, ERL_NIF_TERM term) {
  SparseTensorInternal *t;
  if (!enif_get_resource(env, term, SPARSE_TENSOR_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

/** sparse_available() -> true | false */
ERL_NIF_TERM sparse_available(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  return cusparselt_available() ? enif_make_atom(env, "true")
                                 : enif_make_atom(env, "false");
}

/** sparse_from_ct16(CudaTensor16Ref) -> {ok, SparseTensorRef} | {error, Reason}
 *  Create a 2:4 sparse tensor from a CudaTensor16 (FP16 on GPU).
 *  Prunes to 2:4 pattern and compresses to ~50% size.
 *  Dimensions must be multiples of 16 for FP16.
 */
ERL_NIF_TERM sparse_from_ct16(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  CudaTensor16 *ct16 = get_cuda_tensor16(env, argv[0]);
  if (!ct16) return make_error(env, "invalid_cuda_tensor16");

  if (ct16->ndim != 2) return make_error(env, "must_be_2d_matrix");

  int64_t rows = ct16->shape[0];
  int64_t cols = ct16->shape[1];

  if (rows % 16 != 0 || cols % 16 != 0)
    return make_error(env, "dimensions_must_be_multiples_of_16");

  SparseTensorInternal *sparse = (SparseTensorInternal *)enif_alloc_resource(
      SPARSE_TENSOR_RESOURCE, sizeof(SparseTensorInternal));
  if (!sparse) return make_error(env, "alloc_failed");

  int result = sparse_tensor_create_fp16(ct16->d_data, rows, cols, sparse);
  if (result != 0) {
    enif_release_resource(sparse);
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_create_failed_%d", result);
    return make_error(env, err_msg);
  }

  return make_ok(env, make_sparse_tensor_term(env, sparse));
}

/** sparse_shape(SparseTensorRef) -> {ok, [Rows, Cols]} */
ERL_NIF_TERM sparse_shape(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  (void)argc;
  SparseTensorInternal *t = get_sparse_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_sparse_tensor");

  ERL_NIF_TERM dims[2] = {
    enif_make_int64(env, t->rows),
    enif_make_int64(env, t->cols)
  };
  return make_ok(env, enif_make_list_from_array(env, dims, 2));
}

/** sparse_compression_ratio(SparseTensorRef) -> {ok, Float} */
ERL_NIF_TERM sparse_compression_ratio(ErlNifEnv *env, int argc,
                                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  SparseTensorInternal *t = get_sparse_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_sparse_tensor");

  size_t dense_size = (size_t)t->rows * (size_t)t->cols * sizeof(uint16_t);
  double ratio = (double)dense_size / (double)t->compressed_size;
  return make_ok(env, enif_make_double(env, ratio));
}

/* === FP16 Sparse Matmul === */

/** sparse_matmul(SparseTensor, CudaTensor16_B, M, N, K) -> {ok, CudaTensor16Ref}
 *  C = A_sparse @ B_dense (FP16, allocating output)
 */
ERL_NIF_TERM sparse_matmul_nif(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  if (!sparse || !b) return make_error(env, "invalid_input");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return make_error(env, "invalid_dimensions");

  if (sparse->rows != m || sparse->cols != k)
    return make_error(env, "sparse_dimension_mismatch");
  if (b->size != k * n)
    return make_error(env, "b_size_mismatch");
  if (n % 16 != 0)
    return make_error(env, "n_must_be_multiple_of_16");

  int out_shape[2] = {m, n};
  CudaTensor16 *c = alloc_cuda_tensor16(2, out_shape);
  if (!c) return make_error(env, "alloc_failed");

  int result = sparse_matmul_fp16(sparse, b->d_data, c->d_data, n, 1.0f, 0.0f);
  if (result != 0) {
    enif_release_resource(c);
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_matmul_failed_%d", result);
    return make_error(env, err_msg);
  }

  return make_ok(env, make_cuda_tensor16_term(env, c));
}

/** sparse_matmul_inplace(SparseTensor, B, C, M, N, K) -> ok
 *  In-place sparse FP16 GEMM (zero allocation)
 */
ERL_NIF_TERM sparse_matmul_inplace_nif(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!sparse || !b || !c) return make_error(env, "invalid_input");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dimensions");

  if (n % 16 != 0)
    return make_error(env, "n_must_be_multiple_of_16");

  int result = sparse_matmul_fp16(sparse, b->d_data, c->d_data, n, 1.0f, 0.0f);
  if (result != 0) {
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_matmul_failed_%d", result);
    return make_error(env, err_msg);
  }

  return enif_make_atom(env, "ok");
}

/** sparse_matmul_bench(SparseTensor, B, C, M, N, K, Iters) -> ok */
ERL_NIF_TERM sparse_matmul_bench_nif(ErlNifEnv *env, int argc,
                                              const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!sparse || !b || !c) return make_error(env, "invalid_input");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  int result = sparse_matmul_fp16_bench(sparse, b->d_data, c->d_data, n, iters);
  if (result != 0) {
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_bench_failed_%d", result);
    return make_error(env, err_msg);
  }

  return enif_make_atom(env, "ok");
}

/** sparse_matmul_bench_tn(SparseTensor, B, C, M, N, K, Iters) -> ok
 * TN layout variant */
ERL_NIF_TERM sparse_matmul_bench_tn_nif(ErlNifEnv *env, int argc,
                                                 const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaTensor16 *b = get_cuda_tensor16(env, argv[1]);
  CudaTensor16 *c = get_cuda_tensor16(env, argv[2]);
  if (!sparse || !b || !c) return make_error(env, "invalid_input");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  int result = sparse_matmul_fp16_bench_tn(sparse, b->d_data, c->d_data, n, iters);
  if (result != 0) {
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_bench_tn_failed_%d", result);
    return make_error(env, err_msg);
  }

  return enif_make_atom(env, "ok");
}

/* === INT8 Sparse (2:4 via cuSPARSELt) === */

/** sparse_from_int8(CudaInt8TensorRef) -> {ok, SparseTensorRef} | {error, Reason}
 *  Create a 2:4 sparse tensor from INT8 GPU data.
 *  Dimensions must be multiples of 16.
 */
ERL_NIF_TERM sparse_from_int8(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  CudaInt8Tensor *ct = get_cuda_int8_tensor(env, argv[0]);
  if (!ct) return make_error(env, "invalid_cuda_int8_tensor");

  if (ct->ndim != 2) return make_error(env, "must_be_2d_matrix");

  int64_t rows = ct->shape[0];
  int64_t cols = ct->shape[1];

  if (rows % 16 != 0 || cols % 16 != 0)
    return make_error(env, "dimensions_must_be_multiples_of_16");

  SparseTensorInternal *sparse = (SparseTensorInternal *)enif_alloc_resource(
      SPARSE_TENSOR_RESOURCE, sizeof(SparseTensorInternal));
  if (!sparse) return make_error(env, "alloc_failed");

  int result = sparse_tensor_create_int8(ct->d_data, rows, cols, sparse);
  if (result != 0) {
    enif_release_resource(sparse);
    if (result == -100)
      return make_error(env, "int8_sparse_not_supported");
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_create_int8_failed_%d", result);
    return make_error(env, err_msg);
  }

  return make_ok(env, make_sparse_tensor_term(env, sparse));
}

/** sparse_matmul_int8(SparseTensor, CudaInt8Tensor_B, M, N, K) -> {ok, CudaInt8TensorRef}
 *  C = A_sparse @ B_dense (INT8, allocating output)
 */
ERL_NIF_TERM sparse_matmul_int8_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  if (!sparse || !b) return make_error(env, "invalid_input");

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return make_error(env, "invalid_dimensions");

  if (sparse->rows != m || sparse->cols != k)
    return make_error(env, "sparse_dimension_mismatch");
  if (b->size != k * n)
    return make_error(env, "b_size_mismatch");
  if (n % 16 != 0)
    return make_error(env, "n_must_be_multiple_of_16");

  int out_shape[2] = {m, n};
  CudaInt8Tensor *c = alloc_cuda_int8_tensor(2, out_shape);
  if (!c) return make_error(env, "alloc_failed");

  int result = sparse_matmul_int8(sparse, b->d_data, c->d_data, n, 1.0f, 0.0f);
  if (result != 0) {
    enif_release_resource(c);
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_matmul_int8_failed_%d", result);
    return make_error(env, err_msg);
  }

  return make_ok(env, make_cuda_int8_tensor_term(env, c));
}

/** sparse_matmul_int8_inplace(SparseTensor, B, C, M, N, K) -> ok */
ERL_NIF_TERM sparse_matmul_int8_inplace_nif(ErlNifEnv *env, int argc,
                                                      const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  CudaInt8Tensor *c = get_cuda_int8_tensor(env, argv[2]);
  if (!sparse || !b || !c) return make_error(env, "invalid_input");

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return make_error(env, "invalid_dimensions");

  if (n % 16 != 0)
    return make_error(env, "n_must_be_multiple_of_16");

  int result = sparse_matmul_int8(sparse, b->d_data, c->d_data, n, 1.0f, 0.0f);
  if (result != 0) {
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_matmul_int8_failed_%d", result);
    return make_error(env, err_msg);
  }

  return enif_make_atom(env, "ok");
}

/** sparse_matmul_int8_bench(SparseTensor, B, C, M, N, K, Iters) -> ok */
ERL_NIF_TERM sparse_matmul_int8_bench_nif(ErlNifEnv *env, int argc,
                                                    const ERL_NIF_TERM argv[]) {
  (void)argc;

  if (!cusparselt_available())
    return make_error(env, "cusparselt_not_available");

  SparseTensorInternal *sparse = get_sparse_tensor(env, argv[0]);
  CudaInt8Tensor *b = get_cuda_int8_tensor(env, argv[1]);
  CudaInt8Tensor *c = get_cuda_int8_tensor(env, argv[2]);
  if (!sparse || !b || !c) return make_error(env, "invalid_input");

  int m, n, k, iters;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &iters))
    return make_error(env, "invalid_args");

  int result = sparse_matmul_int8_bench(sparse, b->d_data, c->d_data, n, iters);
  if (result != 0) {
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sparse_int8_bench_failed_%d", result);
    return make_error(env, err_msg);
  }

  return enif_make_atom(env, "ok");
}
