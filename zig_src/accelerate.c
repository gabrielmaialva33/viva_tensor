/**
 * viva_tensor NIF - Apple Accelerate BLAS/vDSP bindings
 *
 * Provides hardware-accelerated tensor operations on macOS:
 * - matmul: Uses cblas_dgemm for matrix multiplication
 * - dot: Uses vDSP_dotprD for vector dot product
 *
 * Expected speedup: 10-50x for large matrices compared to pure Erlang
 */

#include <erl_nif.h>
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Helper Functions
 * ========================================================================== */

/**
 * Convert Erlang list of floats to C array
 * Returns 1 on success, 0 on failure
 */
static int list_to_doubles(ErlNifEnv *env, ERL_NIF_TERM list, double **out, size_t *len) {
    unsigned int length;
    if (!enif_get_list_length(env, list, &length)) {
        return 0;
    }

    *len = length;
    *out = (double *)malloc(length * sizeof(double));
    if (!*out)
        return 0;

    ERL_NIF_TERM head, tail = list;
    for (unsigned int i = 0; i < length; i++) {
        if (!enif_get_list_cell(env, tail, &head, &tail)) {
            free(*out);
            *out = NULL;
            return 0;
        }

        double val;
        if (!enif_get_double(env, head, &val)) {
            /* Try integer */
            int int_val;
            long long_val;
            if (enif_get_int(env, head, &int_val)) {
                val = (double)int_val;
            } else if (enif_get_int64(env, head, &long_val)) {
                val = (double)long_val;
            } else {
                free(*out);
                *out = NULL;
                return 0;
            }
        }
        (*out)[i] = val;
    }

    return 1;
}

/**
 * Convert C array of doubles to Erlang list
 */
static ERL_NIF_TERM doubles_to_list(ErlNifEnv *env, double *data, size_t len) {
    ERL_NIF_TERM *terms = (ERL_NIF_TERM *)malloc(len * sizeof(ERL_NIF_TERM));
    if (!terms) {
        return enif_make_list(env, 0);
    }

    for (size_t i = 0; i < len; i++) {
        terms[i] = enif_make_double(env, data[i]);
    }

    ERL_NIF_TERM result = enif_make_list_from_array(env, terms, len);
    free(terms);
    return result;
}

/**
 * Create error tuple: {error, Reason}
 */
static ERL_NIF_TERM make_error(ErlNifEnv *env, const char *reason) {
    return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, reason));
}

/**
 * Create ok tuple: {ok, Value}
 */
static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM value) {
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), value);
}

/* ==========================================================================
 * NIF: Matrix Multiplication using cblas_dgemm
 * A[m,k] @ B[k,n] -> C[m,n]
 * ========================================================================== */

static ERL_NIF_TERM nif_matmul(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 5) {
        return enif_make_badarg(env);
    }

    double *a_data = NULL, *b_data = NULL;
    size_t a_len, b_len;
    int m, n, k;

    /* Parse dimensions */
    if (!enif_get_int(env, argv[2], &m) || !enif_get_int(env, argv[3], &n) ||
        !enif_get_int(env, argv[4], &k)) {
        return enif_make_badarg(env);
    }

    /* Parse input lists */
    if (!list_to_doubles(env, argv[0], &a_data, &a_len) ||
        !list_to_doubles(env, argv[1], &b_data, &b_len)) {
        if (a_data)
            free(a_data);
        if (b_data)
            free(b_data);
        return make_error(env, "invalid_input");
    }

    /* Validate sizes */
    if ((int)a_len != m * k || (int)b_len != k * n) {
        free(a_data);
        free(b_data);
        return make_error(env, "size_mismatch");
    }

    /* Allocate output */
    double *c_data = (double *)calloc(m * n, sizeof(double));
    if (!c_data) {
        free(a_data);
        free(b_data);
        return make_error(env, "out_of_memory");
    }

    /*
     * Perform BLAS matrix multiplication
     * C = alpha * A * B + beta * C
     *
     * cblas_dgemm(order, transA, transB, m, n, k,
     *             alpha, A, lda, B, ldb, beta, C, ldc)
     *
     * For row-major A[m,k] @ B[k,n] -> C[m,n]:
     *   lda = k (columns of A)
     *   ldb = n (columns of B)
     *   ldc = n (columns of C)
     */
    cblas_dgemm(CblasRowMajor, /* Row-major order (C convention) */
                CblasNoTrans,  /* A is not transposed */
                CblasNoTrans,  /* B is not transposed */
                m,             /* Rows of A and C */
                n,             /* Cols of B and C */
                k,             /* Cols of A, rows of B */
                1.0,           /* alpha = 1.0 */
                a_data,        /* Matrix A */
                k,             /* Leading dimension of A */
                b_data,        /* Matrix B */
                n,             /* Leading dimension of B */
                0.0,           /* beta = 0.0 */
                c_data,        /* Output matrix C */
                n              /* Leading dimension of C */
    );

    /* Convert result to Erlang list */
    ERL_NIF_TERM result = doubles_to_list(env, c_data, m * n);

    /* Cleanup */
    free(a_data);
    free(b_data);
    free(c_data);

    return make_ok(env, result);
}

/* ==========================================================================
 * NIF: Dot Product using vDSP_dotprD
 * ========================================================================== */

static ERL_NIF_TERM nif_dot(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return enif_make_badarg(env);
    }

    double *a_data = NULL, *b_data = NULL;
    size_t a_len, b_len;

    if (!list_to_doubles(env, argv[0], &a_data, &a_len) ||
        !list_to_doubles(env, argv[1], &b_data, &b_len)) {
        if (a_data)
            free(a_data);
        if (b_data)
            free(b_data);
        return make_error(env, "invalid_input");
    }

    if (a_len != b_len) {
        free(a_data);
        free(b_data);
        return make_error(env, "length_mismatch");
    }

    double result;
    /* vDSP_dotprD(A, strideA, B, strideB, result, count) */
    vDSP_dotprD(a_data, 1, b_data, 1, &result, a_len);

    free(a_data);
    free(b_data);

    return make_ok(env, enif_make_double(env, result));
}

/* ==========================================================================
 * NIF: Vector Sum using vDSP_sveD
 * ========================================================================== */

static ERL_NIF_TERM nif_sum(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) {
        return enif_make_badarg(env);
    }

    double *data = NULL;
    size_t len;

    if (!list_to_doubles(env, argv[0], &data, &len)) {
        return make_error(env, "invalid_input");
    }

    double result;
    /* vDSP_sveD(A, strideA, result, count) */
    vDSP_sveD(data, 1, &result, len);

    free(data);

    return make_ok(env, enif_make_double(env, result));
}

/* ==========================================================================
 * NIF: Vector Scale using vDSP_vsmulD
 * ========================================================================== */

static ERL_NIF_TERM nif_scale(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) {
        return enif_make_badarg(env);
    }

    double *data = NULL;
    size_t len;
    double scalar;

    if (!list_to_doubles(env, argv[0], &data, &len)) {
        return make_error(env, "invalid_input");
    }

    if (!enif_get_double(env, argv[1], &scalar)) {
        int int_val;
        if (enif_get_int(env, argv[1], &int_val)) {
            scalar = (double)int_val;
        } else {
            free(data);
            return make_error(env, "invalid_scalar");
        }
    }

    double *result = (double *)malloc(len * sizeof(double));
    if (!result) {
        free(data);
        return make_error(env, "out_of_memory");
    }

    /* vDSP_vsmulD(A, strideA, scalar, C, strideC, count) */
    vDSP_vsmulD(data, 1, &scalar, result, 1, len);

    ERL_NIF_TERM result_list = doubles_to_list(env, result, len);

    free(data);
    free(result);

    return make_ok(env, result_list);
}

/* ==========================================================================
 * NIF Initialization
 * ========================================================================== */

static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    (void)env;
    (void)priv_data;
    (void)load_info;
    return 0;
}

static int upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
    (void)env;
    (void)priv_data;
    (void)old_priv_data;
    (void)load_info;
    return 0;
}

/* NIF function table */
static ErlNifFunc nif_funcs[] = {
    {"nif_matmul", 5, nif_matmul, 0},
    {"nif_dot", 2, nif_dot, 0},
    {"nif_sum", 1, nif_sum, 0},
    {"nif_scale", 2, nif_scale, 0},
};

ERL_NIF_INIT(viva_tensor_nif, nif_funcs, load, NULL, upgrade, NULL)
