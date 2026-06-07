/**
 * nif_fp8_cpu.c - CPU reference for FP8 (E4M3) quantization + emulated GEMM.
 *
 * No mainstream CPU has FP8 math units, so this is a *numerical* reference, not
 * an accelerator: it encodes the OFP8 E4M3 format exactly (Micikevicius et al.,
 * "FP8 Formats for Deep Learning", arXiv:2209.05433 — NVIDIA/Arm/Intel) and
 * emulates an FP8 GEMM with per-tensor current scaling (NVIDIA Transformer
 * Engine recipe) by up-converting to FP32 and calling SGEMM. This lets us
 * validate FP8 quantization error bands on any machine; the CUDA CUTLASS FP8
 * kernels remain the production accelerator.
 *
 * E4M3 (1 sign, 4 exp, 3 mantissa; bias 7; no infinities; single NaN pattern
 * S.1111.111):
 *   max normal  = S.1111.110 = 1.75 * 2^8  = 448
 *   min normal  = S.0001.000 = 2^-6
 *   subnormals  = (M/8) * 2^-6             (step 2^-9, min 2^-9)
 *
 * Per-tensor current scaling (Transformer Engine): s = FP8_MAX / amax, quantize
 * x -> round(x*s) in FP8, dequantize x ~= fp8(x*s) / s. GEMM result is divided
 * by s_a * s_b.
 */

#include "viva_nif.h"
#include <math.h>

#define E4M3_MAX 448.0f

/* float32 -> E4M3 byte. Round-to-nearest-even, saturate on overflow (the
 * inference convention: clamp to +-448 rather than emit NaN). */
static uint8_t f32_to_e4m3(float x) {
    if (isnan(x))
        return 0x7F; /* canonical NaN */
    uint8_t sign = 0;
    if (signbit(x)) {
        sign = 0x80;
        x = -x;
    }
    if (x >= E4M3_MAX)
        return sign | 0x7E; /* saturate to 448 (S.1111.110) */

    if (x >= 0x1p-6f) {          /* normal range [2^-6, 448) */
        int e;
        (void)frexpf(x, &e);     /* x in [2^(e-1), 2^e) => x = 1.m * 2^(e-1) */
        int E = e - 1;           /* unbiased exponent */
        float mant = x / ldexpf(1.0f, E);            /* significand in [1,2) */
        int mi = (int)rintf((mant - 1.0f) * 8.0f);   /* 3-bit mantissa, RTNE */
        if (mi == 8) {                                /* rounding carried out */
            mi = 0;
            E += 1;
        }
        if (E > 8)
            return sign | 0x7E; /* overflow after carry -> saturate */
        return sign | (uint8_t)(((E + 7) << 3) | mi);
    } else {                              /* subnormal / zero */
        int mi = (int)rintf(x / 0x1p-9f); /* steps of 2^-9, RTNE */
        if (mi <= 0)
            return sign;                  /* zero */
        if (mi >= 8)
            return sign | (1 << 3);       /* rounds up to min normal 2^-6 */
        return sign | (uint8_t)mi;        /* biased exp 0, mantissa mi */
    }
}

static float e4m3_to_f32(uint8_t b) {
    int sign = (b & 0x80) ? -1 : 1;
    int E = (b >> 3) & 0xF;
    int M = b & 0x7;
    if (E == 0xF && M == 0x7)
        return NAN;
    if (E == 0)
        return (float)sign * ldexpf((float)M / 8.0f, -6); /* subnormal */
    return (float)sign * ldexpf(1.0f + (float)M / 8.0f, E - 7); /* normal */
}

/* round-trip one value through E4M3 (encode then decode) */
static inline float e4m3_roundtrip(float x) {
    return e4m3_to_f32(f32_to_e4m3(x));
}

static double tensor_amax(const double *x, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > m)
            m = a;
    }
    return m;
}

/* ========================================================================= */
/* NIFs                                                                      */
/* ========================================================================= */

/** nt_quantize_e4m3(Ref) -> {ok, Ref}
 *  Fake-quantize an FP64 tensor through E4M3 with per-tensor current scaling
 *  and return the reconstructed FP64 tensor (same shape). Measures pure
 *  quantization error: ||x - dequant(quant(x))||.
 */
ERL_NIF_TERM nt_quantize_e4m3(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c)
        return make_error(env, "out_of_memory");

    const double *src = a->data + a->offset;
    double amax = tensor_amax(src, a->size);
    if (amax == 0.0) {
        memset(c->data, 0, a->size * sizeof(double));
        return make_ok(env, make_tensor_term(env, c));
    }
    float s = E4M3_MAX / (float)amax;
    double inv_s = 1.0 / (double)s;
    for (int i = 0; i < a->size; i++)
        c->data[i] = (double)e4m3_roundtrip((float)(src[i] * (double)s)) * inv_s;

    return make_ok(env, make_tensor_term(env, c));
}

/** nt_matmul_e4m3(RefA, RefB, M, N, K) -> {ok, RefC}
 *  Emulated FP8 GEMM: quantize A and B to E4M3 (per-tensor current scaling),
 *  SGEMM the dequantized FP32 values, rescale by 1/(s_a*s_b). Output is FP64.
 */
ERL_NIF_TERM nt_matmul_e4m3(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
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

    size_t sa_n = m * k, sb_n = k * n, sc_n = m * n;
    float *af = (float *)malloc(sa_n * sizeof(float));
    float *bf = (float *)malloc(sb_n * sizeof(float));
    float *cf = (float *)malloc(sc_n * sizeof(float));
    if (!af || !bf || !cf) {
        free(af);
        free(bf);
        free(cf);
        return make_error(env, "out_of_memory");
    }

    const double *ad = a->data + a->offset;
    const double *bd = b->data + b->offset;
    double amax_a = tensor_amax(ad, (int)sa_n);
    double amax_b = tensor_amax(bd, (int)sb_n);
    float s_a = amax_a == 0.0 ? 1.0f : E4M3_MAX / (float)amax_a;
    float s_b = amax_b == 0.0 ? 1.0f : E4M3_MAX / (float)amax_b;

    /* quantize to E4M3 (store dequantized-but-still-scaled FP32 value) */
    for (size_t i = 0; i < sa_n; i++)
        af[i] = e4m3_roundtrip((float)(ad[i] * (double)s_a));
    for (size_t i = 0; i < sb_n; i++)
        bf[i] = e4m3_roundtrip((float)(bd[i] * (double)s_b));

#if defined(_WIN32) || defined(USE_MKL_DIRECT) || defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0f, af, (int)k,
                bf, (int)n, 0.0f, cf, (int)n);
#else
    if (g_sgemm) {
        blas_sgemm((int)m, (int)n, (int)k, 1.0f, af, (int)k, bf, (int)n, 0.0f, cf, (int)n);
    } else {
        free(af);
        free(bf);
        free(cf);
        return make_error(env, "no_sgemm_backend");
    }
#endif

    NativeTensor *c = alloc_tensor_uninit(2, (int[]){m_int, n_int});
    if (!c) {
        free(af);
        free(bf);
        free(cf);
        return make_error(env, "out_of_memory");
    }

    /* undo the input scaling: real C = (scaled C) / (s_a * s_b) */
    double inv = 1.0 / ((double)s_a * (double)s_b);
    for (size_t i = 0; i < sc_n; i++)
        c->data[i] = (double)cf[i] * inv;

    free(af);
    free(bf);
    free(cf);
    return make_ok(env, make_tensor_term(env, c));
}
