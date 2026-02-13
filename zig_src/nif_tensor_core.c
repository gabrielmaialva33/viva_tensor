/**
 * nif_tensor_core.c - NativeTensor resource types, lifecycle, helpers, constructors, and accessors
 *
 * Extracted from nif_entry.c to improve modularity of the NIF codebase.
 * Contains:
 *   - NativeTensor resource type and lifecycle (alloc_tensor, alloc_tensor_uninit, get_tensor, make_tensor_term)
 *   - QuantInt8Tensor resource type and lifecycle (4x compression)
 *   - QuantNF4Tensor resource type and lifecycle (8x compression)
 *   - Helper functions (parse_shape, list_to_doubles, doubles_to_list, make_ok, make_ok_nil, make_error, get_number)
 *   - NIF constructors (nt_zeros, nt_ones, nt_fill, nt_from_list)
 *   - NIF accessors (nt_to_list, nt_shape, nt_size)
 */

#include "viva_nif.h"

/* =========================================================================
 * NativeTensor - Resource type and lifecycle (struct in viva_nif.h)
 * ========================================================================= */

ErlNifResourceType *TENSOR_RESOURCE = NULL;

void tensor_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  NativeTensor *t = (NativeTensor *)obj;
  if (t->owns_data && t->data)
    aligned_tensor_free(t->data);
  if (t->shape)
    free(t->shape);
  if (t->strides)
    free(t->strides);
}

/** Allocate a new NativeTensor resource with given shape. Data is zeroed. */
NativeTensor *alloc_tensor(int ndim, const int *shape) {
  NativeTensor *t = (NativeTensor *)enif_alloc_resource(TENSOR_RESOURCE,
                                                        sizeof(NativeTensor));
  if (!t)
    return NULL;

  t->ndim = ndim;
  t->owns_data = 1;

  /* Compute size and strides (row-major) */
  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];

  t->shape = (int *)malloc(ndim * sizeof(int));
  t->strides = (int *)malloc(ndim * sizeof(int));
  if (!t->shape || !t->strides) {
    if (t->shape)
      free(t->shape);
    if (t->strides)
      free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    t->data = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  /* Row-major strides */
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    t->strides[i] = stride;
    stride *= shape[i];
  }

  /* Allocate 64-byte aligned zeroed data */
  t->data = (double *)aligned_tensor_alloc(t->size * sizeof(double));
  if (!t->data) {
    free(t->shape);
    free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memset(t->data, 0, t->size * sizeof(double));

  return t;
}

/** Allocate NativeTensor with uninitialized data (use when overwriting all). */
NativeTensor *alloc_tensor_uninit(int ndim, const int *shape) {
  NativeTensor *t = (NativeTensor *)enif_alloc_resource(TENSOR_RESOURCE,
                                                        sizeof(NativeTensor));
  if (!t)
    return NULL;

  t->ndim = ndim;
  t->owns_data = 1;

  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];

  t->shape = (int *)malloc(ndim * sizeof(int));
  t->strides = (int *)malloc(ndim * sizeof(int));
  if (!t->shape || !t->strides) {
    if (t->shape)
      free(t->shape);
    if (t->strides)
      free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    t->data = NULL;
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    t->strides[i] = stride;
    stride *= shape[i];
  }

  t->data = (double *)aligned_tensor_alloc(t->size * sizeof(double));
  if (!t->data) {
    free(t->shape);
    free(t->strides);
    t->shape = NULL;
    t->strides = NULL;
    enif_release_resource(t);
    return NULL;
  }

  return t;
}

/** Get NativeTensor from an Erlang resource term */
NativeTensor *get_tensor(ErlNifEnv *env, ERL_NIF_TERM term) {
  NativeTensor *t;
  if (!enif_get_resource(env, term, TENSOR_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

/** Wrap a NativeTensor as an Erlang term (transfers ownership to GC) */
ERL_NIF_TERM make_tensor_term(ErlNifEnv *env, NativeTensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t); /* GC now owns it */
  return term;
}

/* =========================================================================
 * QuantInt8Tensor - INT8 quantized tensor resource (4x compression)
 * Zero overhead: quantize ONCE, matmul MANY times without list conversion!
 * ========================================================================= */

/* QuantInt8Tensor struct defined in viva_nif.h */

ErlNifResourceType *QINT8_RESOURCE = NULL;

void qint8_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  QuantInt8Tensor *t = (QuantInt8Tensor *)obj;
  if (t->data) free(t->data);
  if (t->shape) free(t->shape);
}

QuantInt8Tensor *get_qint8(ErlNifEnv *env, ERL_NIF_TERM term) {
  QuantInt8Tensor *t;
  if (!enif_get_resource(env, term, QINT8_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM make_qint8_term(ErlNifEnv *env, QuantInt8Tensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

/* =========================================================================
 * QuantNF4Tensor - NF4 quantized tensor resource (8x compression)
 * Blockwise quantization with per-block scales for QLoRA-style compression.
 * ========================================================================= */

/* QuantNF4Tensor struct defined in viva_nif.h */

ErlNifResourceType *QNF4_RESOURCE = NULL;

void qnf4_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  QuantNF4Tensor *t = (QuantNF4Tensor *)obj;
  if (t->indices) free(t->indices);
  if (t->scales) free(t->scales);
  if (t->shape) free(t->shape);
}

QuantNF4Tensor *get_qnf4(ErlNifEnv *env, ERL_NIF_TERM term) {
  QuantNF4Tensor *t;
  if (!enif_get_resource(env, term, QNF4_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM make_qnf4_term(ErlNifEnv *env, QuantNF4Tensor *t) {
  ERL_NIF_TERM term = enif_make_resource(env, t);
  enif_release_resource(t);
  return term;
}

/** Parse shape from Erlang list of ints */
int parse_shape(ErlNifEnv *env, ERL_NIF_TERM list, int *out_shape,
                int *out_ndim) {
  unsigned len;
  if (!enif_get_list_length(env, list, &len) || len == 0 || len > 8)
    return 0;
  *out_ndim = (int)len;

  ERL_NIF_TERM head, tail = list;
  int i = 0;
  while (enif_get_list_cell(env, tail, &head, &tail)) {
    int val;
    if (!enif_get_int(env, head, &val) || val <= 0)
      return 0;
    out_shape[i++] = val;
  }
  return 1;
}

/* =========================================================================
 * Helpers (legacy list-based API)
 * ========================================================================= */

double *list_to_doubles(ErlNifEnv *env, ERL_NIF_TERM list,
                        unsigned *out_len) {
  unsigned length;
  if (!enif_get_list_length(env, list, &length))
    return NULL;
  double *arr = (double *)malloc(length * sizeof(double));
  if (!arr)
    return NULL;

  ERL_NIF_TERM head, tail = list;
  unsigned i = 0;
  while (enif_get_list_cell(env, tail, &head, &tail)) {
    double val;
    if (enif_get_double(env, head, &val)) {
      arr[i++] = val;
    } else {
      int ival;
      long lval;
      if (enif_get_int(env, head, &ival))
        arr[i++] = (double)ival;
      else if (enif_get_long(env, head, &lval))
        arr[i++] = (double)lval;
      else {
        free(arr);
        return NULL;
      }
    }
  }
  *out_len = length;
  return arr;
}

ERL_NIF_TERM doubles_to_list(ErlNifEnv *env, const double *arr,
                             unsigned len) {
  ERL_NIF_TERM result = enif_make_list(env, 0);
  for (unsigned i = len; i > 0;) {
    i--;
    result = enif_make_list_cell(env, enif_make_double(env, arr[i]), result);
  }
  return result;
}

ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM value) {
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), value);
}

ERL_NIF_TERM make_ok_nil(ErlNifEnv *env) {
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_atom(env, "nil"));
}

ERL_NIF_TERM make_error(ErlNifEnv *env, const char *reason) {
  return enif_make_tuple2(env, enif_make_atom(env, "error"),
                          enif_make_atom(env, reason));
}

double get_number(ErlNifEnv *env, ERL_NIF_TERM term, int *ok) {
  double val;
  if (enif_get_double(env, term, &val)) {
    *ok = 1;
    return val;
  }
  int ival;
  if (enif_get_int(env, term, &ival)) {
    *ok = 1;
    return (double)ival;
  }
  long lval;
  if (enif_get_long(env, term, &lval)) {
    *ok = 1;
    return (double)lval;
  }
  *ok = 0;
  return 0.0;
}

/* =========================================================================
 * NIF Resource API — Constructors
 * ========================================================================= */

/** nt_zeros(Shape) -> {ok, Ref} */
ERL_NIF_TERM nt_zeros(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");

  NativeTensor *t = alloc_tensor(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  /* data already zeroed by calloc */

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_ones(Shape) -> {ok, Ref} */
ERL_NIF_TERM nt_ones(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  for (int i = 0; i < t->size; i++)
    t->data[i] = 1.0;

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_fill(Shape, Value) -> {ok, Ref} */
ERL_NIF_TERM nt_fill(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[0], shape, &ndim))
    return make_error(env, "invalid_shape");
  int ok;
  double val = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_value");

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t)
    return make_error(env, "out_of_memory");
  for (int i = 0; i < t->size; i++)
    t->data[i] = val;

  return make_ok(env, make_tensor_term(env, t));
}

/** nt_from_list(Data, Shape) -> {ok, Ref} */
ERL_NIF_TERM nt_from_list(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  int shape[8], ndim;
  if (!parse_shape(env, argv[1], shape, &ndim))
    return make_error(env, "invalid_shape");

  unsigned data_len;
  double *data = list_to_doubles(env, argv[0], &data_len);
  if (!data)
    return make_error(env, "invalid_data");

  /* Validate size matches shape */
  int expected_size = 1;
  for (int i = 0; i < ndim; i++)
    expected_size *= shape[i];
  if ((int)data_len != expected_size) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  NativeTensor *t = alloc_tensor_uninit(ndim, shape);
  if (!t) {
    free(data);
    return make_error(env, "out_of_memory");
  }
  memcpy(t->data, data, data_len * sizeof(double));
  free(data);

  return make_ok(env, make_tensor_term(env, t));
}

/* =========================================================================
 * NIF Resource API — Accessors
 * ========================================================================= */

/** nt_to_list(Ref) -> {ok, List} */
ERL_NIF_TERM nt_to_list(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");
  return make_ok(env, doubles_to_list(env, t->data, t->size));
}

/** nt_shape(Ref) -> {ok, ShapeList} */
ERL_NIF_TERM nt_shape(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");

  ERL_NIF_TERM shape_list = enif_make_list(env, 0);
  for (int i = t->ndim - 1; i >= 0; i--)
    shape_list =
        enif_make_list_cell(env, enif_make_int(env, t->shape[i]), shape_list);
  return make_ok(env, shape_list);
}

/** nt_size(Ref) -> {ok, Int} */
ERL_NIF_TERM nt_size(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t)
    return make_error(env, "invalid_tensor");
  return make_ok(env, enif_make_int(env, t->size));
}
