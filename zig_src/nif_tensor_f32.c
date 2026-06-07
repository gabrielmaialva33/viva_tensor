/**
 * nif_tensor_f32.c - First-class single-precision (FP32) NativeTensorF32
 *
 * Storage is `float` (half the memory of NativeTensor's double). The matmul
 * path calls SGEMM directly on the stored floats — no per-call double<->float
 * conversion, so it wins even on small matrices unlike nt_matmul_sgemm.
 *
 * Backends, in order of the build configuration:
 *   - Direct CBLAS: MKL (Intel/AMD, Win/Linux) or Apple Accelerate (macOS/ARM)
 *   - Dynamic CBLAS: OpenBLAS / ARM PL via blas_sgemm (Linux generic)
 *
 * NIFs: ntf_zeros, ntf_fill, ntf_from_list, ntf_to_list, ntf_shape, ntf_size,
 *       ntf_matmul, ntf_from_f64, ntf_to_f64.
 */

#include "viva_nif.h"

/* ========================================================================= */
/* Resource type + lifecycle                                                 */
/* ========================================================================= */

ErlNifResourceType *TENSOR_F32_RESOURCE = NULL;

void tensor_f32_destructor(ErlNifEnv *env, void *obj) {
    (void)env;
    NativeTensorF32 *t = (NativeTensorF32 *)obj;
    if (t->owns_data && t->data)
        aligned_tensor_free(t->data);
    if (t->owner)
        enif_release_resource(t->owner);
    if (t->shape)
        free(t->shape);
    if (t->strides)
        free(t->strides);
}

static NativeTensorF32 *alloc_f32_common(int ndim, const int *shape, int zero) {
    NativeTensorF32 *t =
        (NativeTensorF32 *)enif_alloc_resource(TENSOR_F32_RESOURCE, sizeof(NativeTensorF32));
    if (!t)
        return NULL;

    t->ndim = ndim;
    t->owns_data = 1;
    t->owner = NULL;
    t->offset = 0;

    t->size = 1;
    for (int i = 0; i < ndim; i++)
        t->size *= shape[i];

    t->shape = (int *)malloc(ndim * sizeof(int));
    t->strides = (int *)malloc(ndim * sizeof(int));
    if (!t->shape || !t->strides) {
        free(t->shape);
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

    t->data = (float *)aligned_tensor_alloc(t->size * sizeof(float));
    if (!t->data) {
        free(t->shape);
        free(t->strides);
        t->shape = NULL;
        t->strides = NULL;
        enif_release_resource(t);
        return NULL;
    }
    if (zero)
        memset(t->data, 0, t->size * sizeof(float));

    return t;
}

NativeTensorF32 *alloc_tensor_f32(int ndim, const int *shape) {
    return alloc_f32_common(ndim, shape, 1);
}

NativeTensorF32 *alloc_tensor_f32_uninit(int ndim, const int *shape) {
    return alloc_f32_common(ndim, shape, 0);
}

NativeTensorF32 *get_tensor_f32(ErlNifEnv *env, ERL_NIF_TERM term) {
    NativeTensorF32 *t;
    if (!enif_get_resource(env, term, TENSOR_F32_RESOURCE, (void **)&t))
        return NULL;
    return t;
}

ERL_NIF_TERM make_tensor_f32_term(ErlNifEnv *env, NativeTensorF32 *t) {
    ERL_NIF_TERM term = enif_make_resource(env, t);
    enif_release_resource(t); /* GC now owns it */
    return term;
}

/* ========================================================================= */
/* Helpers (shape parse / list <-> float)                                    */
/* ========================================================================= */

/* parse_shape, make_ok, make_error declared in viva_nif.h (nif_tensor_core.c) */

static int fill_floats_from_list(ErlNifEnv *env, ERL_NIF_TERM list, float *out, int expected) {
    ERL_NIF_TERM head, tail = list;
    int i = 0;
    double val;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        if (i >= expected)
            return 0;
        if (enif_get_double(env, head, &val)) {
            out[i] = (float)val;
        } else {
            int ival;
            if (!enif_get_int(env, head, &ival))
                return 0;
            out[i] = (float)ival;
        }
        i++;
    }
    return i == expected;
}

static ERL_NIF_TERM floats_to_list(ErlNifEnv *env, const float *arr, int len) {
    ERL_NIF_TERM result = enif_make_list(env, 0);
    for (int i = len - 1; i >= 0; i--)
        result = enif_make_list_cell(env, enif_make_double(env, (double)arr[i]), result);
    return result;
}

/* ========================================================================= */
/* Constructors                                                              */
/* ========================================================================= */

/** ntf_zeros(Shape) -> {ok, RefF32} */
ERL_NIF_TERM ntf_zeros(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[8], ndim;
    if (!parse_shape(env, argv[0], shape, &ndim))
        return make_error(env, "invalid_shape");
    NativeTensorF32 *t = alloc_tensor_f32(ndim, shape);
    if (!t)
        return make_error(env, "out_of_memory");
    return make_ok(env, make_tensor_f32_term(env, t));
}

/** ntf_fill(Shape, Value) -> {ok, RefF32} */
ERL_NIF_TERM ntf_fill(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[8], ndim;
    if (!parse_shape(env, argv[0], shape, &ndim))
        return make_error(env, "invalid_shape");
    double dval;
    if (!enif_get_double(env, argv[1], &dval)) {
        int ival;
        if (!enif_get_int(env, argv[1], &ival))
            return make_error(env, "invalid_value");
        dval = (double)ival;
    }
    NativeTensorF32 *t = alloc_tensor_f32_uninit(ndim, shape);
    if (!t)
        return make_error(env, "out_of_memory");
    float fv = (float)dval;
    for (int i = 0; i < t->size; i++)
        t->data[i] = fv;
    return make_ok(env, make_tensor_f32_term(env, t));
}

/** ntf_from_list(DataList, Shape) -> {ok, RefF32} */
ERL_NIF_TERM ntf_from_list(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    int shape[8], ndim;
    if (!parse_shape(env, argv[1], shape, &ndim))
        return make_error(env, "invalid_shape");
    NativeTensorF32 *t = alloc_tensor_f32_uninit(ndim, shape);
    if (!t)
        return make_error(env, "out_of_memory");
    if (!fill_floats_from_list(env, argv[0], t->data, t->size)) {
        enif_release_resource(t);
        return make_error(env, "data_shape_mismatch");
    }
    return make_ok(env, make_tensor_f32_term(env, t));
}

/* ========================================================================= */
/* Accessors                                                                 */
/* ========================================================================= */

/** ntf_to_list(RefF32) -> {ok, [Float]} */
ERL_NIF_TERM ntf_to_list(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensorF32 *t = get_tensor_f32(env, argv[0]);
    if (!t)
        return make_error(env, "invalid_tensor");
    return make_ok(env, floats_to_list(env, t->data + t->offset, t->size));
}

/** ntf_shape(RefF32) -> {ok, [Int]} */
ERL_NIF_TERM ntf_shape(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensorF32 *t = get_tensor_f32(env, argv[0]);
    if (!t)
        return make_error(env, "invalid_tensor");
    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (int i = t->ndim - 1; i >= 0; i--)
        list = enif_make_list_cell(env, enif_make_int(env, t->shape[i]), list);
    return make_ok(env, list);
}

/** ntf_size(RefF32) -> {ok, Int} */
ERL_NIF_TERM ntf_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensorF32 *t = get_tensor_f32(env, argv[0]);
    if (!t)
        return make_error(env, "invalid_tensor");
    return make_ok(env, enif_make_int(env, t->size));
}

/* ========================================================================= */
/* Matmul (native SGEMM, no conversion)                                      */
/* ========================================================================= */

/** ntf_matmul(RefA, RefB, M, N, K) -> {ok, RefC}
 *  C[m,n] = A[m,k] @ B[k,n] in pure FP32 via SGEMM.
 */
ERL_NIF_TERM ntf_matmul(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensorF32 *a = get_tensor_f32(env, argv[0]);
    NativeTensorF32 *b = get_tensor_f32(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
        !enif_get_int(env, argv[4], &k_int))
        return make_error(env, "invalid_dimensions");

    size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
    if (a->size != (int)(m * k) || b->size != (int)(k * n))
        return make_error(env, "size_mismatch");

    int out_shape[2] = {m_int, n_int};
    NativeTensorF32 *c = alloc_tensor_f32_uninit(2, out_shape);
    if (!c)
        return make_error(env, "out_of_memory");

    const float *ad = a->data + a->offset;
    const float *bd = b->data + b->offset;

#if defined(_WIN32) || defined(USE_MKL_DIRECT) || defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0f, ad, (int)k,
                bd, (int)n, 0.0f, c->data, (int)n);
#else
    if (g_sgemm) {
        blas_sgemm((int)m, (int)n, (int)k, 1.0f, ad, (int)k, bd, (int)n, 0.0f, c->data, (int)n);
    } else {
        enif_release_resource(c);
        return make_error(env, "no_sgemm_backend");
    }
#endif

    return make_ok(env, make_tensor_f32_term(env, c));
}

/* ========================================================================= */
/* Conversions FP64 <-> FP32                                                 */
/* ========================================================================= */

/** ntf_from_f64(NativeTensorRef) -> {ok, RefF32}  (down-convert) */
ERL_NIF_TERM ntf_from_f64(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *src = get_tensor(env, argv[0]);
    if (!src)
        return make_error(env, "invalid_tensor");
    NativeTensorF32 *t = alloc_tensor_f32_uninit(src->ndim, src->shape);
    if (!t)
        return make_error(env, "out_of_memory");
    const double *s = src->data + src->offset;
    for (int i = 0; i < t->size; i++)
        t->data[i] = (float)s[i];
    return make_ok(env, make_tensor_f32_term(env, t));
}

/** ntf_to_f64(RefF32) -> {ok, NativeTensorRef}  (up-convert) */
ERL_NIF_TERM ntf_to_f64(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensorF32 *src = get_tensor_f32(env, argv[0]);
    if (!src)
        return make_error(env, "invalid_tensor");
    NativeTensor *t = alloc_tensor_uninit(src->ndim, src->shape);
    if (!t)
        return make_error(env, "out_of_memory");
    const float *s = src->data + src->offset;
    for (int i = 0; i < t->size; i++)
        t->data[i] = (double)s[i];
    return make_ok(env, make_tensor_term(env, t));
}
