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

static void nt_binary_elementwise(const NativeTensor *a, const NativeTensor *b, NativeTensor *c,
                                  double (*op)(double, double)) {
    if (tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        for (int i = 0; i < a->size; i++)
            c->data[i] = op(a->data[a->offset + i], b->data[b->offset + i]);
        return;
    }

    for (int i = 0; i < a->size; i++)
        c->data[i] = op(tensor_get_flat(a, i), tensor_get_flat(b, i));
}

static double op_add(double a, double b) {
    return a + b;
}
static double op_sub(double a, double b) {
    return a - b;
}
static double op_mul(double a, double b) {
    return a * b;
}
static double op_maximum(double a, double b) {
    return a > b ? a : b;
}
static double op_minimum(double a, double b) {
    return a < b ? a : b;
}
static double op_equal(double a, double b) {
    return a == b ? 1.0 : 0.0;
}
static double op_not_equal(double a, double b) {
    return a != b ? 1.0 : 0.0;
}
static double op_greater(double a, double b) {
    return a > b ? 1.0 : 0.0;
}
static double op_greater_equal(double a, double b) {
    return a >= b ? 1.0 : 0.0;
}
static double op_less(double a, double b) {
    return a < b ? 1.0 : 0.0;
}
static double op_less_equal(double a, double b) {
    return a <= b ? 1.0 : 0.0;
}
static double op_logical_and(double a, double b) {
    return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
}
static double op_logical_or(double a, double b) {
    return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
}
static double op_logical_xor(double a, double b) {
    return ((a != 0.0) != (b != 0.0)) ? 1.0 : 0.0;
}

static int nt_can_write_into(const NativeTensor *out) {
    return out && out->owns_data && tensor_is_contiguous(out);
}

static void nt_unary_elementwise(const NativeTensor *a, NativeTensor *c, double (*op)(double)) {
    if (tensor_is_contiguous(a)) {
        for (int i = 0; i < a->size; i++)
            c->data[i] = op(a->data[a->offset + i]);
        return;
    }

    for (int i = 0; i < a->size; i++)
        c->data[i] = op(tensor_get_flat(a, i));
}

static double op_negate(double x) {
    return -x;
}
static double op_relu(double x) {
    return x > 0.0 ? x : 0.0;
}
static double op_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
static double op_exp(double x) {
    return exp(x);
}
static double op_log(double x) {
    return log(x);
}
static double op_logical_not(double x) {
    return x == 0.0 ? 1.0 : 0.0;
}

static ERL_NIF_TERM nt_binary_resource_op(ErlNifEnv *env, const ERL_NIF_TERM argv[],
                                          double (*op)(double, double)) {
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (a->size != b->size)
        return make_error(env, "size_mismatch");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    nt_binary_elementwise(a, b, c, op);
    return make_ok(env, make_tensor_term(env, c));
}

static ERL_NIF_TERM nt_unary_resource_op(ErlNifEnv *env, const ERL_NIF_TERM argv[],
                                         double (*op)(double)) {
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    nt_unary_elementwise(a, c, op);
    return make_ok(env, make_tensor_term(env, c));
}

static ERL_NIF_TERM fused_linear_relu_into_checked(ErlNifEnv *env, NativeTensor *out,
                                                   NativeTensor *a, NativeTensor *b,
                                                   NativeTensor *bias, int m, int n, int k);

/* =========================================================================
 * NIF Resource API — Element-wise Operations (resource → resource)
 * ========================================================================= */

/** nt_add(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_add(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");
    if (a->size != b->size)
        return make_error(env, "size_mismatch");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a) && tensor_is_contiguous(b))
        vt_simd_add(a->data + a->offset, b->data + b->offset, c->data, a->size);
    else
        nt_binary_elementwise(a, b, c, op_add);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_sub(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_sub(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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

    if (tensor_is_contiguous(a) && tensor_is_contiguous(b))
        vt_simd_sub(a->data + a->offset, b->data + b->offset, c->data, a->size);
    else
        nt_binary_elementwise(a, b, c, op_sub);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_mul(RefA, RefB) -> {ok, RefC} */
ERL_NIF_TERM nt_mul(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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

    if (tensor_is_contiguous(a) && tensor_is_contiguous(b))
        vt_simd_mul(a->data + a->offset, b->data + b->offset, c->data, a->size);
    else
        nt_binary_elementwise(a, b, c, op_mul);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_scale(Ref, Scalar) -> {ok, RefC} */
ERL_NIF_TERM nt_scale(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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

    if (tensor_is_contiguous(a)) {
        vt_simd_scale(a->data + a->offset, scalar, c->data, a->size);
    } else {
        for (int i = 0; i < a->size; i++)
            c->data[i] = tensor_get_flat(a, i) * scalar;
    }
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_negate(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_negate(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a))
        vt_simd_negate(a->data + a->offset, c->data, a->size);
    else
        nt_unary_elementwise(a, c, op_negate);
    return make_ok(env, make_tensor_term(env, c));
}

ERL_NIF_TERM nt_maximum(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_maximum);
}

ERL_NIF_TERM nt_minimum(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_minimum);
}

ERL_NIF_TERM nt_equal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_equal);
}

ERL_NIF_TERM nt_not_equal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_not_equal);
}

ERL_NIF_TERM nt_greater(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_greater);
}

ERL_NIF_TERM nt_greater_equal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_greater_equal);
}

ERL_NIF_TERM nt_less(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_less);
}

ERL_NIF_TERM nt_less_equal(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_less_equal);
}

ERL_NIF_TERM nt_logical_not(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_unary_resource_op(env, argv, op_logical_not);
}

ERL_NIF_TERM nt_logical_and(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_logical_and);
}

ERL_NIF_TERM nt_logical_or(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_logical_or);
}

ERL_NIF_TERM nt_logical_xor(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_resource_op(env, argv, op_logical_xor);
}

ERL_NIF_TERM nt_where(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *condition = get_tensor(env, argv[0]);
    NativeTensor *when_true = get_tensor(env, argv[1]);
    NativeTensor *when_false = get_tensor(env, argv[2]);
    if (!condition || !when_true || !when_false)
        return make_error(env, "invalid_tensor");
    if (condition->size != when_true->size || condition->size != when_false->size)
        return make_error(env, "size_mismatch");

    NativeTensor *out = alloc_tensor_uninit(condition->ndim, condition->shape);
    if (!out)
        return make_error(env, "out_of_memory");

    for (int i = 0; i < condition->size; i++) {
        double c = tensor_get_flat(condition, i);
        out->data[i] = c != 0.0 ? tensor_get_flat(when_true, i) : tensor_get_flat(when_false, i);
    }

    return make_ok(env, make_tensor_term(env, out));
}

/* =========================================================================
 * NIF Resource API — Reductions (resource → scalar)
 * ========================================================================= */

/** nt_dot(RefA, RefB) -> {ok, Float} */
ERL_NIF_TERM nt_dot(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (a->size != b->size)
        return make_error(env, "size_mismatch");

    double result;
    if (tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        result = vt_simd_dot(a->data + a->offset, b->data + b->offset, a->size);
    } else {
        result = 0.0;
        for (int i = 0; i < a->size; i++)
            result += tensor_get_flat(a, i) * tensor_get_flat(b, i);
    }
    return make_ok(env, enif_make_double(env, result));
}

/** nt_sum(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_sum(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    double result;
    if (tensor_is_contiguous(a)) {
        result = vt_simd_sum(a->data + a->offset, a->size);
    } else {
        result = 0.0;
        for (int i = 0; i < a->size; i++)
            result += tensor_get_flat(a, i);
    }
    return make_ok(env, enif_make_double(env, result));
}

/** nt_max(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_max(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    double mx;
    if (tensor_is_contiguous(a)) {
        mx = vt_simd_max(a->data + a->offset, a->size);
    } else {
        mx = tensor_get_flat(a, 0);
        for (int i = 1; i < a->size; i++) {
            double v = tensor_get_flat(a, i);
            if (v > mx)
                mx = v;
        }
    }
    return make_ok(env, enif_make_double(env, mx));
}

/** nt_min(Ref) -> {ok, Float} */
ERL_NIF_TERM nt_min(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    double mn;
    if (tensor_is_contiguous(a)) {
        mn = vt_simd_min(a->data + a->offset, a->size);
    } else {
        mn = tensor_get_flat(a, 0);
        for (int i = 1; i < a->size; i++) {
            double v = tensor_get_flat(a, i);
            if (v < mn)
                mn = v;
        }
    }
    return make_ok(env, enif_make_double(env, mn));
}

ERL_NIF_TERM nt_count_nonzero(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    int count = 0;
    if (tensor_is_contiguous(a)) {
        for (int i = 0; i < a->size; i++) {
            if (a->data[a->offset + i] != 0.0)
                count++;
        }
    } else {
        for (int i = 0; i < a->size; i++) {
            if (tensor_get_flat(a, i) != 0.0)
                count++;
        }
    }

    return make_ok(env, enif_make_int(env, count));
}

/* =========================================================================
 * NIF Resource API — Matrix Operations
 * ========================================================================= */

/** nt_matmul(RefA, RefB, M, N, K) -> {ok, RefC}
 *  Now uses BLAS directly (MKL/OpenBLAS) - Zig GEMM removed for simplicity.
 *  This is just an alias for nt_matmul_blas.
 */
ERL_NIF_TERM nt_matmul_blas(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]); /* Forward declaration */
#define nt_matmul nt_matmul_blas                        /* Alias */

/** nt_matmul_blas(RefA, RefB, M, N, K) -> {ok, RefC}
 *  DGEMM via MKL (Windows) or best available BLAS (Linux, runtime-detected).
 */
ERL_NIF_TERM nt_matmul_blas(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
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
    NativeTensor *c = alloc_tensor_uninit(2, out_shape);
    if (!c)
        return make_error(env, "out_of_memory");

    /* C = alpha * A @ B + beta * C
   * cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
   * Row-major: lda=k, ldb=n, ldc=n
   */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
    /* MKL direct-linked DGEMM */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0,
                a->data + a->offset, (int)k, b->data + b->offset, (int)n, 0.0, c->data, (int)n);
#else
    /* Fallback: use dynamically loaded backend */
    if (g_dgemm) {
        blas_dgemm((int)m, (int)n, (int)k, 1.0, a->data + a->offset, (int)k, b->data + b->offset,
                   (int)n, 0.0, c->data, (int)n);
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
ERL_NIF_TERM nt_matmul_inplace(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    NativeTensor *c = get_tensor(env, argv[2]);
    if (!a || !b || !c)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b) || !tensor_is_contiguous(c))
        return make_error(env, "non_contiguous");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[3], &m_int) || !enif_get_int(env, argv[4], &n_int) ||
        !enif_get_int(env, argv[5], &k_int))
        return make_error(env, "invalid_dimensions");

    size_t m = (size_t)m_int, n = (size_t)n_int, k = (size_t)k_int;
    if (a->size != (int)(m * k) || b->size != (int)(k * n) || c->size != (int)(m * n))
        return make_error(env, "size_mismatch");

#if defined(_WIN32) || defined(USE_MKL_DIRECT)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0,
                a->data + a->offset, (int)k, b->data + b->offset, (int)n, 0.0, c->data, (int)n);
#else
    if (g_dgemm) {
        blas_dgemm((int)m, (int)n, (int)k, 1.0, a->data + a->offset, (int)k, b->data + b->offset,
                   (int)n, 0.0, c->data, (int)n);
    } else {
        return make_error(env, "no_blas_backend");
    }
#endif

    return make_ok_nil(env);
}

/* All CUDA externs are in viva_nif.h (included above) */

/** nt_matmul_cuda(RefA, RefB, M, N, K) -> {ok, RefC}
 *  cuBLAS DGEMM on GPU, falls back to BLAS if CUDA not available.
 */
ERL_NIF_TERM nt_matmul_cuda(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
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
        int result = cuda_dgemm(m_int, n_int, k_int, 1.0, a->data + a->offset, k_int,
                                b->data + b->offset, n_int, 0.0, c->data, n_int);
        if (result == 0) {
            return make_ok(env, make_tensor_term(env, c));
        }
        /* CUDA failed, fall through to CPU */
        fprintf(stderr, "[viva_tensor] CUDA fallback to CPU (error %d)\n", result);
    }
#endif

    /* Fallback to CPU BLAS */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0,
                a->data + a->offset, (int)k, b->data + b->offset, (int)n, 0.0, c->data, (int)n);
#else
    if (g_dgemm) {
        blas_dgemm((int)m, (int)n, (int)k, 1.0, a->data + a->offset, (int)k, b->data + b->offset,
                   (int)n, 0.0, c->data, (int)n);
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
ERL_NIF_TERM nt_matmul_cuda_fp32(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
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
        free(a_f32);
        free(b_f32);
        free(c_f32);
        return make_error(env, "out_of_memory");
    }

    /* Convert double -> float (vectorizable, fast) */
    for (size_t i = 0; i < size_a; i++)
        a_f32[i] = (float)a->data[a->offset + i];
    for (size_t i = 0; i < size_b; i++)
        b_f32[i] = (float)b->data[b->offset + i];

    /* cuBLAS SGEMM (TF32 Tensor Cores) */
    int result =
        cuda_sgemm(m_int, n_int, k_int, 1.0f, a_f32, k_int, b_f32, n_int, 0.0f, c_f32, n_int);

    if (result != 0) {
        free(a_f32);
        free(b_f32);
        free(c_f32);
        return make_error(env, "cuda_sgemm_failed");
    }

    /* Allocate output tensor */
    int out_shape[2] = {m_int, n_int};
    NativeTensor *c = alloc_tensor_uninit(2, out_shape);
    if (!c) {
        free(a_f32);
        free(b_f32);
        free(c_f32);
        return make_error(env, "out_of_memory");
    }

    /* Convert float -> double */
    for (size_t i = 0; i < size_c; i++)
        c->data[i] = (double)c_f32[i];

    free(a_f32);
    free(b_f32);
    free(c_f32);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_int8_tc(RefA, RefB, M, N, K) -> {ok, RefC}
 *  INT8 IMMA Tensor Cores via cublasGemmEx. Auto-quantizes f64 input.
 */
ERL_NIF_TERM nt_matmul_int8_tc(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
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
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "out_of_memory");
    }

    /* Find absmax for quantization */
    double a_max = 0.0, b_max = 0.0;
    for (size_t i = 0; i < size_a; i++) {
        double v = fabs(a->data[a->offset + i]);
        if (v > a_max)
            a_max = v;
    }
    for (size_t i = 0; i < size_b; i++) {
        double v = fabs(b->data[b->offset + i]);
        if (v > b_max)
            b_max = v;
    }

    /* Quantize to INT8 range [-127, 127] */
    double a_scale = (a_max > 0) ? 127.0 / a_max : 1.0;
    double b_scale = (b_max > 0) ? 127.0 / b_max : 1.0;

    for (size_t i = 0; i < size_a; i++) {
        double scaled = a->data[a->offset + i] * a_scale;
        a_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
    }
    for (size_t i = 0; i < size_b; i++) {
        double scaled = b->data[b->offset + i] * b_scale;
        b_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
    }

    /* cuBLAS INT8 GEMM (Tensor Cores) */
    int result = cuda_igemm(m_int, n_int, k_int, 1, a_i8, k_int, b_i8, n_int, 0, c_i32, n_int);

    if (result != 0) {
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "cuda_int8_gemm_failed");
    }

    /* Allocate output tensor and dequantize */
    int out_shape[2] = {m_int, n_int};
    NativeTensor *c = alloc_tensor_uninit(2, out_shape);
    if (!c) {
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "out_of_memory");
    }

    /* Dequantize: C_f64 = C_i32 / (a_scale * b_scale) */
    double dequant_scale = 1.0 / (a_scale * b_scale);
    for (size_t i = 0; i < size_c; i++) {
        c->data[i] = (double)c_i32[i] * dequant_scale;
    }

    free(a_i8);
    free(b_i8);
    free(c_i32);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_int8_tc_available() -> true | false
 *  Check if INT8 Tensor Cores are available (RTX 20xx+)
 */
ERL_NIF_TERM nt_int8_tc_available(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    if (cuda_int8_available()) {
        return enif_make_atom(env, "true");
    } else {
        return enif_make_atom(env, "false");
    }
}

/* FP16 helper: convert float to half (IEEE 754) */
uint16_t float_to_half(float f) {
    uint32_t x = *(uint32_t *)&f;
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    uint16_t h;
    if (exp == 0) {
        h = (sign << 15); /* Zero or denormal -> zero */
    } else if (exp == 0xFF) {
        h = (sign << 15) | 0x7C00 | (mant ? 0x200 : 0); /* Inf/NaN */
    } else {
        int new_exp = (int)exp - 127 + 15;
        if (new_exp >= 31) {
            h = (sign << 15) | 0x7C00; /* Overflow -> Inf */
        } else if (new_exp <= 0) {
            /* IEEE-754 binary16 subnormal: representable down to 2^-24 ≈ 5.96e-8.
       * Naive flush-to-zero used to underflow ~50% of FP8/Llama activations
       * (any |x| < 2^-14) and caused a ~0.5× pipeline-wide magnitude bias
       * vs HF reference (see dev/hf_bisect.py). */
            if (new_exp < -10) {
                h = (uint16_t)(sign << 15); /* truly < 2^-24, flush */
            } else {
                uint32_t mant_full = mant | 0x800000;      /* implicit leading 1 */
                uint32_t shift = (uint32_t)(14 - new_exp); /* in [14, 24] */
                h = (uint16_t)((sign << 15) | (mant_full >> shift));
            }
        } else {
            h = (sign << 15) | (new_exp << 10) | (mant >> 13);
        }
    }
    return h;
}

/** nt_floats_to_fp16_binary(ListOfFloats) -> binary
 *
 * Replaces the pure-Erlang `fp16_encode/1` per-element loop in
 * viva_tensor_inference_ffi.erl with a single C pass. Profile of the
 * TinyLlama W8A16 forward showed ~25% of per-layer time spent in the
 * Erlang encoder — this drops it by ~14× (per_layer 7.7ms -> ~5.7ms).
 *
 * Reads the input as a list of numbers (floats or ints), allocates a
 * new binary 2× as large, and writes IEEE-754 binary16 little-endian.
 */
ERL_NIF_TERM nt_floats_to_fp16_binary(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    unsigned n = 0;
    if (!enif_get_list_length(env, argv[0], &n))
        return enif_make_badarg(env);

    ERL_NIF_TERM out_bin_term;
    unsigned char *dst = enif_make_new_binary(env, (size_t)n * 2, &out_bin_term);
    if (!dst)
        return enif_make_badarg(env);

    ERL_NIF_TERM head, tail = argv[0];
    unsigned i = 0;
    while (enif_get_list_cell(env, tail, &head, &tail)) {
        double v;
        long iv;
        if (enif_get_double(env, head, &v)) {
            /* OK */
        } else if (enif_get_long(env, head, &iv)) {
            v = (double)iv;
        } else {
            return enif_make_badarg(env);
        }
        uint16_t h = float_to_half((float)v);
        dst[(size_t)i * 2 + 0] = (unsigned char)(h & 0xFF);
        dst[(size_t)i * 2 + 1] = (unsigned char)((h >> 8) & 0xFF);
        ++i;
    }
    return out_bin_term;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | ((exp - 1 + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

ERL_NIF_TERM nt_fp16_to_fp32_binary(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary src;
    if (!enif_inspect_binary(env, argv[0], &src) || (src.size % 2) != 0)
        return enif_make_badarg(env);

    size_t n = src.size / sizeof(uint16_t);
    ERL_NIF_TERM out_bin_term;
    unsigned char *dst_bytes = enif_make_new_binary(env, n * sizeof(float), &out_bin_term);
    if (!dst_bytes)
        return enif_make_badarg(env);

    const unsigned char *src_bytes = src.data;
    float *dst = (float *)dst_bytes;
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = (uint16_t)src_bytes[i * 2] | ((uint16_t)src_bytes[i * 2 + 1] << 8);
        dst[i] = half_to_float(h);
    }
    return out_bin_term;
}

/** nt_silu_mul(GateList, UpList) -> FP16Binary
 *
 * Fused silu(gate) * up over two equal-length lists of floats.
 * Returns an FP16 binary directly (skips an extra Erlang list build
 * + fp16 encode step). Used by the SwiGLU intermediate path.
 *
 *   silu(x) = x / (1 + exp(-x))
 *   out[i]  = silu(gate[i]) * up[i]
 */
ERL_NIF_TERM nt_silu_mul(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    unsigned ng = 0, nu = 0;
    if (!enif_get_list_length(env, argv[0], &ng))
        return enif_make_badarg(env);
    if (!enif_get_list_length(env, argv[1], &nu))
        return enif_make_badarg(env);
    if (ng != nu)
        return enif_make_badarg(env);

    ERL_NIF_TERM out_term;
    unsigned char *dst = enif_make_new_binary(env, (size_t)ng * 2, &out_term);
    if (!dst)
        return enif_make_badarg(env);

    ERL_NIF_TERM g_head, g_tail = argv[0], u_head, u_tail = argv[1];
    unsigned i = 0;
    while (enif_get_list_cell(env, g_tail, &g_head, &g_tail) &&
           enif_get_list_cell(env, u_tail, &u_head, &u_tail)) {
        double gd, ud;
        long gi, ui;
        if (!enif_get_double(env, g_head, &gd)) {
            if (enif_get_long(env, g_head, &gi))
                gd = (double)gi;
            else
                return enif_make_badarg(env);
        }
        if (!enif_get_double(env, u_head, &ud)) {
            if (enif_get_long(env, u_head, &ui))
                ud = (double)ui;
            else
                return enif_make_badarg(env);
        }
        float g = (float)gd;
        float u = (float)ud;
        float silu_g = g / (1.0f + expf(-g));
        uint16_t h = float_to_half(silu_g * u);
        dst[(size_t)i * 2 + 0] = (unsigned char)(h & 0xFF);
        dst[(size_t)i * 2 + 1] = (unsigned char)((h >> 8) & 0xFF);
        ++i;
    }
    return out_term;
}

/** nt_list_add_fp16(ListA, ListB) -> FP16Binary
 *
 * Adds two equal-length lists element-wise and returns an FP16 binary.
 * Replaces the Erlang `lists:zipwith(fun(X,Y) -> X+Y end, ...)` followed
 * by `floats_to_fp16_binary` chain in residual + cast paths.
 */
ERL_NIF_TERM nt_list_add_fp16(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    unsigned na = 0, nb = 0;
    if (!enif_get_list_length(env, argv[0], &na))
        return enif_make_badarg(env);
    if (!enif_get_list_length(env, argv[1], &nb))
        return enif_make_badarg(env);
    if (na != nb)
        return enif_make_badarg(env);

    ERL_NIF_TERM out_term;
    unsigned char *dst = enif_make_new_binary(env, (size_t)na * 2, &out_term);
    if (!dst)
        return enif_make_badarg(env);

    ERL_NIF_TERM a_head, a_tail = argv[0], b_head, b_tail = argv[1];
    unsigned i = 0;
    while (enif_get_list_cell(env, a_tail, &a_head, &a_tail) &&
           enif_get_list_cell(env, b_tail, &b_head, &b_tail)) {
        double ad, bd;
        long ai, bi;
        if (!enif_get_double(env, a_head, &ad)) {
            if (enif_get_long(env, a_head, &ai))
                ad = (double)ai;
            else
                return enif_make_badarg(env);
        }
        if (!enif_get_double(env, b_head, &bd)) {
            if (enif_get_long(env, b_head, &bi))
                bd = (double)bi;
            else
                return enif_make_badarg(env);
        }
        uint16_t h = float_to_half((float)(ad + bd));
        dst[(size_t)i * 2 + 0] = (unsigned char)(h & 0xFF);
        dst[(size_t)i * 2 + 1] = (unsigned char)((h >> 8) & 0xFF);
        ++i;
    }
    return out_term;
}

/** nt_matmul_fp16_tc(RefA, RefB, M, N, K) -> {ok, RefC}
 *  FP16 Tensor Cores via cublasGemmEx. Auto-converts f64 <-> FP16.
 */
ERL_NIF_TERM nt_matmul_fp16_tc(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
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
        free(a_fp16);
        free(b_fp16);
        free(c_fp32);
        return make_error(env, "out_of_memory");
    }

    /* Convert f64 -> FP16 */
    for (size_t i = 0; i < size_a; i++) {
        a_fp16[i] = float_to_half((float)a->data[a->offset + i]);
    }
    for (size_t i = 0; i < size_b; i++) {
        b_fp16[i] = float_to_half((float)b->data[b->offset + i]);
    }

    /* cuBLAS FP16 GEMM (Tensor Cores) */
    int result =
        cuda_hgemm(m_int, n_int, k_int, 1.0f, a_fp16, k_int, b_fp16, n_int, 0.0f, c_fp32, n_int);

    if (result != 0) {
        free(a_fp16);
        free(b_fp16);
        free(c_fp32);
        return make_error(env, "cuda_fp16_gemm_failed");
    }

    /* Allocate output tensor and convert FP32 -> f64 */
    int out_shape[2] = {m_int, n_int};
    NativeTensor *c = alloc_tensor_uninit(2, out_shape);
    if (!c) {
        free(a_fp16);
        free(b_fp16);
        free(c_fp32);
        return make_error(env, "out_of_memory");
    }

    for (size_t i = 0; i < size_c; i++) {
        c->data[i] = (double)c_fp32[i];
    }

    free(a_fp16);
    free(b_fp16);
    free(c_fp32);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_fp16_tc_available() -> true | false */
ERL_NIF_TERM nt_fp16_tc_available(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    if (cuda_fp16_available()) {
        return enif_make_atom(env, "true");
    } else {
        return enif_make_atom(env, "false");
    }
}

/** nt_matmul_int8_lt(RefA, RefB, M, N, K) -> {ok, RefC}
 *  cublasLt INT8 IMMA Tensor Cores (vs cublasGemmEx which uses DP4A).
 */
ERL_NIF_TERM nt_matmul_int8_lt(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b))
        return make_error(env, "non_contiguous");

    int m_int, n_int, k_int;
    if (!enif_get_int(env, argv[2], &m_int) || !enif_get_int(env, argv[3], &n_int) ||
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
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "out_of_memory");
    }

    /* Find absmax for quantization */
    double a_max = 0.0, b_max = 0.0;
    for (size_t i = 0; i < size_a; i++) {
        double v = fabs(a->data[a->offset + i]);
        if (v > a_max)
            a_max = v;
    }
    for (size_t i = 0; i < size_b; i++) {
        double v = fabs(b->data[b->offset + i]);
        if (v > b_max)
            b_max = v;
    }

    /* Quantize to INT8 range [-127, 127] */
    double a_scale = (a_max > 0) ? 127.0 / a_max : 1.0;
    double b_scale = (b_max > 0) ? 127.0 / b_max : 1.0;

    for (size_t i = 0; i < size_a; i++) {
        double scaled = a->data[a->offset + i] * a_scale;
        a_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
    }
    for (size_t i = 0; i < size_b; i++) {
        double scaled = b->data[b->offset + i] * b_scale;
        b_i8[i] = (int8_t)(scaled > 127.0 ? 127 : (scaled < -127.0 ? -127 : scaled));
    }

    /* cublasLt INT8 IMMA Tensor Cores */
    int result =
        cuda_igemm_lt(m_int, n_int, k_int, 1.0f, a_i8, k_int, b_i8, n_int, 0.0f, c_i32, n_int);

    if (result != 0) {
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "cuda_int8_lt_gemm_failed");
    }

    /* Allocate output tensor and dequantize */
    int out_shape[2] = {m_int, n_int};
    NativeTensor *c = alloc_tensor_uninit(2, out_shape);
    if (!c) {
        free(a_i8);
        free(b_i8);
        free(c_i32);
        return make_error(env, "out_of_memory");
    }

    /* Dequantize: C_f64 = C_i32 / (a_scale * b_scale) */
    double dequant_scale = 1.0 / (a_scale * b_scale);
    for (size_t i = 0; i < size_c; i++) {
        c->data[i] = (double)c_i32[i] * dequant_scale;
    }

    free(a_i8);
    free(b_i8);
    free(c_i32);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_int8_lt_available() -> true | false
 *  Check if cublasLt INT8 IMMA Tensor Cores are available
 */
ERL_NIF_TERM nt_int8_lt_available(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    if (cuda_int8_lt_available()) {
        return enif_make_atom(env, "true");
    } else {
        return enif_make_atom(env, "false");
    }
}
#endif

/** nt_transpose(Ref) -> {ok, RefC}  (creates contiguous transposed copy) */
ERL_NIF_TERM nt_transpose(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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
            c->data[j * rows + i] = tensor_get_flat(a, i * cols + j);

    return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * NIF Resource API — Activation Functions
 * ========================================================================= */

/** nt_relu(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_relu(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a))
        vt_simd_relu(a->data + a->offset, c->data, a->size);
    else
        nt_unary_elementwise(a, c, op_relu);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_sigmoid(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_sigmoid(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a))
        vt_simd_sigmoid(a->data + a->offset, c->data, (size_t)a->size);
    else
        nt_unary_elementwise(a, c, op_sigmoid);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_exp(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_exp_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a))
        vt_simd_exp(a->data + a->offset, c->data, (size_t)a->size);
    else
        nt_unary_elementwise(a, c, op_exp);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_log(Ref) -> {ok, RefC} */
ERL_NIF_TERM nt_log_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    if (tensor_is_contiguous(a))
        vt_simd_log(a->data + a->offset, c->data, (size_t)a->size);
    else
        nt_unary_elementwise(a, c, op_log);
    return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * In-Place Mutation NIFs
 * "Quebrar a imutabilidade dentro do Zig para economizar RAM"
 * ========================================================================= */

/** nt_add_mut(RefA, RefB) -> ok. Modifies A in-place: A += B */
ERL_NIF_TERM nt_add_mut(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    if (!a || !b)
        return make_error(env, "invalid_tensor");
    if (a->size != b->size)
        return make_error(env, "size_mismatch");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b) || !a->owns_data)
        return make_error(env, "non_contiguous");

    vt_simd_add_mut(a->data + a->offset, b->data + b->offset, (size_t)a->size);
    return make_ok_nil(env);
}

/** nt_scale_mut(RefA, Scalar) -> ok. Modifies A in-place: A *= scalar */
ERL_NIF_TERM nt_scale_mut(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !a->owns_data)
        return make_error(env, "non_contiguous");

    double scalar;
    if (!enif_get_double(env, argv[1], &scalar))
        return make_error(env, "invalid_scalar");

    vt_simd_scale_mut(a->data + a->offset, scalar, (size_t)a->size);
    return make_ok_nil(env);
}

/** nt_negate_mut(RefA) -> ok. Modifies A in-place: A = -A */
ERL_NIF_TERM nt_negate_mut(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !a->owns_data)
        return make_error(env, "non_contiguous");

    vt_simd_negate_mut(a->data + a->offset, (size_t)a->size);
    return make_ok_nil(env);
}

/** nt_relu_mut(RefA) -> ok. Modifies A in-place: A = max(0, A) */
ERL_NIF_TERM nt_relu_mut(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !a->owns_data)
        return make_error(env, "non_contiguous");

    vt_simd_relu_mut(a->data + a->offset, (size_t)a->size);
    return make_ok_nil(env);
}

static ERL_NIF_TERM nt_binary_into(ErlNifEnv *env, const ERL_NIF_TERM argv[],
                                   double (*op)(double, double)) {
    NativeTensor *out = get_tensor(env, argv[0]);
    NativeTensor *a = get_tensor(env, argv[1]);
    NativeTensor *b = get_tensor(env, argv[2]);
    if (!out || !a || !b)
        return make_error(env, "invalid_tensor");
    if (!nt_can_write_into(out))
        return make_error(env, "invalid_output");
    if (out->size != a->size || a->size != b->size)
        return make_error(env, "size_mismatch");

    nt_binary_elementwise(a, b, out, op);
    return make_ok_nil(env);
}

/** nt_add_into(Out, A, B) -> {ok, nil}. Writes Out = A + B. */
ERL_NIF_TERM nt_add_into(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_into(env, argv, op_add);
}

/** nt_sub_into(Out, A, B) -> {ok, nil}. Writes Out = A - B. */
ERL_NIF_TERM nt_sub_into(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_into(env, argv, op_sub);
}

/** nt_mul_into(Out, A, B) -> {ok, nil}. Writes Out = A * B. */
ERL_NIF_TERM nt_mul_into(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return nt_binary_into(env, argv, op_mul);
}

/** nt_scale_into(Out, A, Scalar) -> {ok, nil}. Writes Out = A * Scalar. */
ERL_NIF_TERM nt_scale_into(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *out = get_tensor(env, argv[0]);
    NativeTensor *a = get_tensor(env, argv[1]);
    if (!out || !a)
        return make_error(env, "invalid_tensor");
    if (!nt_can_write_into(out))
        return make_error(env, "invalid_output");
    if (out->size != a->size)
        return make_error(env, "size_mismatch");

    int ok;
    double scalar = get_number(env, argv[2], &ok);
    if (!ok)
        return make_error(env, "invalid_scalar");

    if (tensor_is_contiguous(a)) {
        vt_simd_scale(a->data + a->offset, scalar, out->data + out->offset, (size_t)a->size);
    } else {
        for (int i = 0; i < a->size; i++)
            out->data[out->offset + i] = tensor_get_flat(a, i) * scalar;
    }
    return make_ok_nil(env);
}

/* =========================================================================
 * Retro / Fused Kernels
 * ========================================================================= */

/** nt_saturn_blend(Texture, Shade, Bias) -> {ok, RefC}
 * VDP1-inspired: result = texture + (shade - bias). Pure addition, no mul. */
ERL_NIF_TERM nt_saturn_blend(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *texture = get_tensor(env, argv[0]);
    NativeTensor *shade = get_tensor(env, argv[1]);
    if (!texture || !shade)
        return make_error(env, "invalid_tensor");
    if (texture->size != shade->size)
        return make_error(env, "size_mismatch");
    if (!tensor_is_contiguous(texture) || !tensor_is_contiguous(shade))
        return make_error(env, "non_contiguous");

    double bias;
    if (!enif_get_double(env, argv[2], &bias))
        return make_error(env, "invalid_bias");

    NativeTensor *c = alloc_tensor_uninit(texture->ndim, texture->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    vt_saturn_blend(texture->data + texture->offset, shade->data + shade->offset, bias, c->data,
                    (size_t)texture->size);
    return make_ok(env, make_tensor_term(env, c));
}

/** nt_fused_linear_relu(A, B, Bias, M, N, K) -> {ok, RefC}
 * Fused: C = max(0, A@B + bias). Uses BLAS for matmul + Zig SIMD for bias+relu.
 */
ERL_NIF_TERM nt_fused_linear_relu_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    NativeTensor *b = get_tensor(env, argv[1]);
    NativeTensor *bias = get_tensor(env, argv[2]);
    if (!a || !b || !bias)
        return make_error(env, "invalid_tensor");
    if (!tensor_is_contiguous(a) || !tensor_is_contiguous(b) || !tensor_is_contiguous(bias))
        return make_error(env, "non_contiguous");

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

    ERL_NIF_TERM result = fused_linear_relu_into_checked(env, c, a, b, bias, m, n, k);
    if (!enif_is_identical(result, make_ok_nil(env)))
        return result;

    return make_ok(env, make_tensor_term(env, c));
}

static ERL_NIF_TERM fused_linear_relu_into_checked(ErlNifEnv *env, NativeTensor *out,
                                                   NativeTensor *a, NativeTensor *b,
                                                   NativeTensor *bias, int m, int n, int k) {
    if (!out || !a || !b || !bias)
        return make_error(env, "invalid_tensor");
    if (!nt_can_write_into(out) || !tensor_is_contiguous(a) || !tensor_is_contiguous(b) ||
        !tensor_is_contiguous(bias))
        return make_error(env, "non_contiguous");
    if (a->size != m * k || b->size != k * n || bias->size != n || out->size != m * n)
        return make_error(env, "shape_mismatch");

    /* Step 1: out = A @ B via BLAS */
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a->data + a->offset, k,
                b->data + b->offset, n, 0.0, out->data + out->offset, n);
#else
    if (g_dgemm) {
        blas_dgemm(m, n, k, 1.0, a->data + a->offset, k, b->data + b->offset, n, 0.0,
                   out->data + out->offset, n);
    } else {
        return make_error(env, "no_blas_backend");
    }
#endif

    /* Step 2: out[i,j] += bias[j] for each row, then ReLU in-place */
    for (int i = 0; i < m; i++) {
        double *row = out->data + out->offset + i * n;
        vt_simd_add(row, bias->data + bias->offset, row, (size_t)n);
    }
    vt_simd_relu_mut(out->data + out->offset, (size_t)(m * n));

    return make_ok_nil(env);
}

/** nt_fused_linear_relu_into(Out, A, B, Bias, M, N, K) -> {ok, nil} */
ERL_NIF_TERM nt_fused_linear_relu_into(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *out = get_tensor(env, argv[0]);
    NativeTensor *a = get_tensor(env, argv[1]);
    NativeTensor *b = get_tensor(env, argv[2]);
    NativeTensor *bias = get_tensor(env, argv[3]);

    int m, n, k;
    if (!enif_get_int(env, argv[4], &m) || !enif_get_int(env, argv[5], &n) ||
        !enif_get_int(env, argv[6], &k))
        return make_error(env, "invalid_dims");

    return fused_linear_relu_into_checked(env, out, a, b, bias, m, n, k);
}
