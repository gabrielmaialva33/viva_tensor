/**
 * nif_quant.c - Quantization NIFs (INT8 and NF4)
 * Fused dequant+matmul, resource-based quantized tensors, accessors.
 */

#include "viva_nif.h"

/* === Fused Quantized Matmul === */

/** nt_matmul_int8(A, B_quant_list, B_scale, M, N, K) -> {ok, C}
 *  A: NativeTensor, B: INT8 quantized list with scale.
 *  Dequant happens on-the-fly during accumulation.
 */
ERL_NIF_TERM nt_matmul_int8(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  if (argc != 6) return enif_make_badarg(env);

  NativeTensor *a;
  if (!enif_get_resource(env, argv[0], TENSOR_RESOURCE, (void **)&a))
    return enif_make_badarg(env);

  double b_scale;
  if (!enif_get_double(env, argv[2], &b_scale))
    return enif_make_badarg(env);

  int m, n, k;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k))
    return enif_make_badarg(env);

  if (a->size != m * k)
    return make_error(env, "a_size_mismatch");

  unsigned b_len;
  if (!enif_get_list_length(env, argv[1], &b_len) || (int)b_len != k * n)
    return make_error(env, "b_size_mismatch");

  int8_t *b_quant = malloc(b_len);
  if (!b_quant) return make_error(env, "alloc_failed");

  ERL_NIF_TERM list = argv[1];
  ERL_NIF_TERM head;
  for (unsigned i = 0; i < b_len; i++) {
    if (!enif_get_list_cell(env, list, &head, &list)) {
      free(b_quant);
      return make_error(env, "list_parse");
    }
    int val;
    if (!enif_get_int(env, head, &val)) {
      free(b_quant);
      return make_error(env, "not_int");
    }
    b_quant[i] = (int8_t)(val < -127 ? -127 : (val > 127 ? 127 : val));
  }

  NativeTensor *c = enif_alloc_resource(TENSOR_RESOURCE, sizeof(NativeTensor));
  if (!c) { free(b_quant); return make_error(env, "alloc_failed"); }

  c->data = aligned_tensor_alloc(m * n * sizeof(double));
  if (!c->data) {
    free(b_quant);
    enif_release_resource(c);
    return make_error(env, "alloc_failed");
  }

  c->shape = malloc(2 * sizeof(int));
  c->strides = malloc(2 * sizeof(int));
  if (!c->shape || !c->strides) {
    aligned_tensor_free(c->data);
    if (c->shape) free(c->shape);
    if (c->strides) free(c->strides);
    free(b_quant);
    enif_release_resource(c);
    return make_error(env, "alloc_failed");
  }

  c->shape[0] = m;
  c->shape[1] = n;
  c->strides[0] = n;
  c->strides[1] = 1;
  c->ndim = 2;
  c->size = m * n;
  c->owns_data = 1;

  vt_matmul_int8(a->data, b_quant, b_scale, m, n, k, c->data);

  free(b_quant);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** nt_quantize_int8(Tensor) -> {ok, {QuantList, Scale}} */
ERL_NIF_TERM nt_quantize_int8(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  if (argc != 1) return enif_make_badarg(env);

  NativeTensor *t;
  if (!enif_get_resource(env, argv[0], TENSOR_RESOURCE, (void **)&t))
    return enif_make_badarg(env);

  int8_t *quant = malloc(t->size);
  if (!quant) return make_error(env, "alloc_failed");

  double scale = vt_quantize_int8(t->data, quant, t->size);

  ERL_NIF_TERM list = enif_make_list(env, 0);
  for (int i = t->size - 1; i >= 0; i--) {
    list = enif_make_list_cell(env, enif_make_int(env, quant[i]), list);
  }

  free(quant);

  ERL_NIF_TERM tuple = enif_make_tuple2(env, list, enif_make_double(env, scale));
  return make_ok(env, tuple);
}

/** nt_matmul_nf4(A, B_indices, B_scales, M, N, K, BlockSize) -> {ok, C}
 *  NF4 fused dequant+matmul. 8x compression.
 */
ERL_NIF_TERM nt_matmul_nf4(ErlNifEnv *env, int argc,
                                  const ERL_NIF_TERM argv[]) {
  if (argc != 7) return enif_make_badarg(env);

  NativeTensor *a;
  if (!enif_get_resource(env, argv[0], TENSOR_RESOURCE, (void **)&a))
    return enif_make_badarg(env);

  int m, n, k, block_size;
  if (!enif_get_int(env, argv[3], &m) ||
      !enif_get_int(env, argv[4], &n) ||
      !enif_get_int(env, argv[5], &k) ||
      !enif_get_int(env, argv[6], &block_size))
    return enif_make_badarg(env);

  if (block_size <= 0) block_size = 64;

  if (a->size != m * k)
    return make_error(env, "a_size_mismatch");

  unsigned idx_len;
  if (!enif_get_list_length(env, argv[1], &idx_len))
    return make_error(env, "indices_not_list");

  size_t expected_bytes = ((size_t)k * n + 1) / 2;
  if (idx_len != expected_bytes)
    return make_error(env, "indices_size_mismatch");

  uint8_t *b_indices = malloc(idx_len);
  if (!b_indices) return make_error(env, "alloc_failed");

  ERL_NIF_TERM list = argv[1];
  ERL_NIF_TERM head;
  for (unsigned i = 0; i < idx_len; i++) {
    if (!enif_get_list_cell(env, list, &head, &list)) {
      free(b_indices);
      return make_error(env, "list_parse");
    }
    int val;
    if (!enif_get_int(env, head, &val)) {
      free(b_indices);
      return make_error(env, "not_int");
    }
    b_indices[i] = (uint8_t)(val & 0xFF);
  }

  unsigned scales_len;
  if (!enif_get_list_length(env, argv[2], &scales_len)) {
    free(b_indices);
    return make_error(env, "scales_not_list");
  }

  int num_blocks = (k + block_size - 1) / block_size;
  if ((int)scales_len != num_blocks * n) {
    free(b_indices);
    return make_error(env, "scales_size_mismatch");
  }

  double *b_scales = malloc(scales_len * sizeof(double));
  if (!b_scales) {
    free(b_indices);
    return make_error(env, "alloc_failed");
  }

  list = argv[2];
  for (unsigned i = 0; i < scales_len; i++) {
    if (!enif_get_list_cell(env, list, &head, &list)) {
      free(b_indices);
      free(b_scales);
      return make_error(env, "list_parse");
    }
    double val;
    if (!enif_get_double(env, head, &val)) {
      free(b_indices);
      free(b_scales);
      return make_error(env, "not_float");
    }
    b_scales[i] = val;
  }

  NativeTensor *c = enif_alloc_resource(TENSOR_RESOURCE, sizeof(NativeTensor));
  if (!c) {
    free(b_indices);
    free(b_scales);
    return make_error(env, "alloc_failed");
  }

  c->data = aligned_tensor_alloc(m * n * sizeof(double));
  if (!c->data) {
    free(b_indices);
    free(b_scales);
    enif_release_resource(c);
    return make_error(env, "alloc_failed");
  }

  c->shape = malloc(2 * sizeof(int));
  c->strides = malloc(2 * sizeof(int));
  if (!c->shape || !c->strides) {
    aligned_tensor_free(c->data);
    if (c->shape) free(c->shape);
    if (c->strides) free(c->strides);
    free(b_indices);
    free(b_scales);
    enif_release_resource(c);
    return make_error(env, "alloc_failed");
  }

  c->shape[0] = m;
  c->shape[1] = n;
  c->strides[0] = n;
  c->strides[1] = 1;
  c->ndim = 2;
  c->size = m * n;
  c->owns_data = 1;

  vt_matmul_nf4(a->data, b_indices, b_scales, m, n, k, block_size, c->data);

  free(b_indices);
  free(b_scales);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/* === Resource-Based Quantized Tensors === */

/** nt_to_qint8(Tensor) -> {ok, QuantInt8Tensor} */
ERL_NIF_TERM nt_to_qint8(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 1) return enif_make_badarg(env);

  NativeTensor *t;
  if (!enif_get_resource(env, argv[0], TENSOR_RESOURCE, (void **)&t))
    return enif_make_badarg(env);

  if (t->ndim != 2)
    return make_error(env, "must_be_2d");

  QuantInt8Tensor *q = (QuantInt8Tensor *)enif_alloc_resource(
      QINT8_RESOURCE, sizeof(QuantInt8Tensor));
  if (!q) return make_error(env, "alloc_failed");

  q->data = (int8_t *)malloc(t->size);
  q->shape = (int *)malloc(t->ndim * sizeof(int));
  if (!q->data || !q->shape) {
    if (q->data) free(q->data);
    if (q->shape) free(q->shape);
    enif_release_resource(q);
    return make_error(env, "alloc_failed");
  }

  q->scale = vt_quantize_int8(t->data, q->data, t->size);
  q->ndim = t->ndim;
  q->size = t->size;
  memcpy(q->shape, t->shape, t->ndim * sizeof(int));

  return make_ok(env, make_qint8_term(env, q));
}

/** nt_matmul_qint8(A, B_qint8, M, N, K) -> {ok, C}
 *  Dequant B + MKL DGEMM.
 */
ERL_NIF_TERM nt_matmul_qint8(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  if (argc != 5) return enif_make_badarg(env);

  NativeTensor *a = get_tensor(env, argv[0]);
  QuantInt8Tensor *b = get_qint8(env, argv[1]);
  if (!a || !b)
    return enif_make_badarg(env);

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return enif_make_badarg(env);

  if (a->size != m * k)
    return make_error(env, "a_size_mismatch");
  if (b->size != k * n)
    return make_error(env, "b_size_mismatch");

  int shape[2] = {m, n};
  NativeTensor *c = alloc_tensor_uninit(2, shape);
  if (!c) return make_error(env, "alloc_failed");

  double *b_dequant = (double *)malloc(k * n * sizeof(double));
  if (!b_dequant) {
    free(c);
    return make_error(env, "alloc_failed");
  }

  double scale = b->scale;
  for (int i = 0; i < k * n; i++) {
    b_dequant[i] = (double)b->data[i] * scale;
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k,
              1.0, a->data, k,
              b_dequant, n,
              0.0, c->data, n);

  free(b_dequant);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_to_qnf4(Tensor, BlockSize) -> {ok, QuantNF4Tensor} */
ERL_NIF_TERM nt_to_qnf4(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  if (argc != 2) return enif_make_badarg(env);

  NativeTensor *t;
  if (!enif_get_resource(env, argv[0], TENSOR_RESOURCE, (void **)&t))
    return enif_make_badarg(env);

  int block_size;
  if (!enif_get_int(env, argv[1], &block_size) || block_size <= 0)
    block_size = 64;

  if (t->ndim != 2)
    return make_error(env, "must_be_2d");

  QuantNF4Tensor *q = (QuantNF4Tensor *)enif_alloc_resource(
      QNF4_RESOURCE, sizeof(QuantNF4Tensor));
  if (!q) return make_error(env, "alloc_failed");

  q->block_size = block_size;
  q->size = t->size;
  q->ndim = t->ndim;
  q->num_blocks = (t->size + block_size - 1) / block_size;
  q->packed_size = (t->size + 1) / 2;

  q->indices = (uint8_t *)malloc(q->packed_size);
  q->scales = (double *)malloc(q->num_blocks * sizeof(double));
  q->shape = (int *)malloc(t->ndim * sizeof(int));

  if (!q->indices || !q->scales || !q->shape) {
    if (q->indices) free(q->indices);
    if (q->scales) free(q->scales);
    if (q->shape) free(q->shape);
    enif_release_resource(q);
    return make_error(env, "alloc_failed");
  }

  memcpy(q->shape, t->shape, t->ndim * sizeof(int));

  vt_quantize_nf4(t->data, q->indices, q->scales, t->size, block_size);

  return make_ok(env, make_qnf4_term(env, q));
}

/* NF4 quantization levels (QLoRA paper) */
static const double NF4_LEVELS[16] = {
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
};

/** nt_matmul_qnf4(A, B_qnf4, M, N, K) -> {ok, C}
 *  Dequant NF4 B + MKL DGEMM.
 */
ERL_NIF_TERM nt_matmul_qnf4(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  if (argc != 5) return enif_make_badarg(env);

  NativeTensor *a = get_tensor(env, argv[0]);
  QuantNF4Tensor *b = get_qnf4(env, argv[1]);
  if (!a || !b)
    return enif_make_badarg(env);

  int m, n, k;
  if (!enif_get_int(env, argv[2], &m) ||
      !enif_get_int(env, argv[3], &n) ||
      !enif_get_int(env, argv[4], &k))
    return enif_make_badarg(env);

  if (a->size != m * k)
    return make_error(env, "a_size_mismatch");
  if (b->size != k * n)
    return make_error(env, "b_size_mismatch");

  int shape[2] = {m, n};
  NativeTensor *c = alloc_tensor_uninit(2, shape);
  if (!c) return make_error(env, "alloc_failed");

  double *b_dequant = (double *)malloc(k * n * sizeof(double));
  if (!b_dequant) {
    free(c);
    return make_error(env, "alloc_failed");
  }

  int block_size = b->block_size;
  int num_blocks_k = (k + block_size - 1) / block_size;

  for (int row = 0; row < k; row++) {
    int block_row = row / block_size;
    for (int col = 0; col < n; col++) {
      int linear_idx = row * n + col;
      int byte_idx = linear_idx / 2;
      int is_high = (linear_idx % 2);

      uint8_t packed = b->indices[byte_idx];
      int nf4_idx = is_high ? (packed >> 4) : (packed & 0x0F);

      double scale = b->scales[block_row * n + col];

      b_dequant[linear_idx] = NF4_LEVELS[nf4_idx] * scale;
    }
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k,
              1.0, a->data, k,
              b_dequant, n,
              0.0, c->data, n);

  free(b_dequant);
  return make_ok(env, make_tensor_term(env, c));
}

/** qint8_scale(QuantInt8Tensor) -> {ok, Scale} */
ERL_NIF_TERM qint8_scale(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 1) return enif_make_badarg(env);
  QuantInt8Tensor *q = get_qint8(env, argv[0]);
  if (!q) return enif_make_badarg(env);
  return make_ok(env, enif_make_double(env, q->scale));
}

/** qint8_shape(QuantInt8Tensor) -> {ok, [Rows, Cols]} */
ERL_NIF_TERM qint8_shape(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 1) return enif_make_badarg(env);
  QuantInt8Tensor *q = get_qint8(env, argv[0]);
  if (!q) return enif_make_badarg(env);

  ERL_NIF_TERM shape[2] = {
    enif_make_int(env, q->shape[0]),
    enif_make_int(env, q->shape[1])
  };
  return make_ok(env, enif_make_list_from_array(env, shape, 2));
}

/** qnf4_info(QuantNF4Tensor) -> {ok, #{block_size, num_blocks, compression}} */
ERL_NIF_TERM qnf4_info(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  if (argc != 1) return enif_make_badarg(env);
  QuantNF4Tensor *q = get_qnf4(env, argv[0]);
  if (!q) return enif_make_badarg(env);

  double compression = (double)(q->size * sizeof(double)) /
                       (q->packed_size + q->num_blocks * sizeof(double));

  ERL_NIF_TERM keys[] = {
    enif_make_atom(env, "block_size"),
    enif_make_atom(env, "num_blocks"),
    enif_make_atom(env, "compression")
  };
  ERL_NIF_TERM vals[] = {
    enif_make_int(env, q->block_size),
    enif_make_int(env, q->num_blocks),
    enif_make_double(env, compression)
  };
  ERL_NIF_TERM map;
  enif_make_map_from_arrays(env, keys, vals, 3, &map);
  return make_ok(env, map);
}
