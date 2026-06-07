/**
 * nif_turboquant.c - TurboQuant (MSE) data-oblivious quantization, CPU.
 *
 * Implements TurboQuant_mse from Zandieh, Silwal, Han, Mirrokni, Karbasi,
 * "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
 * (Google Research, ICLR 2026, arXiv:2504.19874).
 *
 * Idea: a random orthogonal rotation turns any vector's coordinates into
 * near-i.i.d. Gaussians (Beta-on-the-sphere, -> Normal in high dim). This
 * spreads outlier energy across all coordinates, so a *scalar* quantizer is
 * near-optimal per coordinate. We use:
 *   1. Randomized Hadamard Transform (RHT): y = (1/sqrt(n)) H (D x), with D a
 *      random +-1 diagonal (seeded). H is symmetric & orthogonal, so the
 *      normalized RHT is its own inverse.
 *   2. Lloyd-Max scalar quantizer for N(0,1), computed deterministically by
 *      fixed-point iteration over the Gaussian (optimal MSE codebook).
 *
 * Per row: normalize to unit norm, rotate, scale by sqrt(n) (so coords ~N(0,1)),
 * quantize each coord to b bits, dequantize, unscale, inverse-rotate, restore
 * the norm. This file provides a *fake-quant round-trip* to measure distortion;
 * the same machinery underlies a packed on-the-fly-dequant matmul.
 */

#include "viva_nif.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Gaussian pdf / cdf ------------------------------------------------- */

static double gauss_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

static double gauss_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

/* ---- Lloyd-Max codebook for N(0,1) (optimal MSE scalar quantizer) ------- */

/* Fill centroids[L] with the Lloyd-Max optimal reconstruction levels for a
 * standard normal source. Deterministic: thresholds and centroids are updated
 * via the closed-form conditional mean E[X | t_i < X < t_{i+1}]. */
static void lloyd_max_normal(int L, double *centroids) {
    double thr[256 + 1];
    /* init centroids at uniform quantiles of N(0,1) in [-3, 3] */
    for (int i = 0; i < L; i++)
        centroids[i] = -3.0 + 6.0 * ((double)i + 0.5) / (double)L;

    for (int iter = 0; iter < 64; iter++) {
        thr[0] = -40.0; /* practical -inf */
        thr[L] = 40.0;  /* practical +inf */
        for (int i = 1; i < L; i++)
            thr[i] = 0.5 * (centroids[i - 1] + centroids[i]);
        for (int i = 0; i < L; i++) {
            double num = gauss_pdf(thr[i]) - gauss_pdf(thr[i + 1]); /* -d/dx of pdf */
            double den = gauss_cdf(thr[i + 1]) - gauss_cdf(thr[i]);
            if (den > 1e-12)
                centroids[i] = num / den;
        }
    }
}

/* Quantize value y (assumed ~N(0,1)) to the nearest centroid index. */
static int quantize_to_codebook(double y, const double *centroids, int L) {
    int lo = 0, hi = L - 1;
    /* centroids are sorted ascending; binary search nearest */
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        double midpoint = 0.5 * (centroids[mid] + centroids[mid + 1]);
        if (y <= midpoint)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

/* ---- Randomized Hadamard Transform ------------------------------------- */

static int next_pow2(int n) {
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

/* In-place fast Walsh-Hadamard transform, length n must be a power of 2. */
static void fwht(double *a, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = i; j < i + len; j++) {
                double u = a[j];
                double v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
        }
    }
}

/* Deterministic +-1 sign for coordinate idx given a seed (splitmix-ish). */
static double rht_sign(uint64_t seed, int idx) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (z & 1ULL) ? 1.0 : -1.0;
}

/* y = (1/sqrt(n)) H (D x). buf length n (>= filled), tail already zero. */
static void rht_forward(double *buf, int n, uint64_t seed) {
    for (int i = 0; i < n; i++)
        buf[i] *= rht_sign(seed, i);
    fwht(buf, n);
    double inv = 1.0 / sqrt((double)n);
    for (int i = 0; i < n; i++)
        buf[i] *= inv;
}

/* inverse: x = D ((1/sqrt(n)) H y). Normalized RHT is its own inverse. */
static void rht_inverse(double *buf, int n, uint64_t seed) {
    fwht(buf, n);
    double inv = 1.0 / sqrt((double)n);
    for (int i = 0; i < n; i++)
        buf[i] *= inv;
    for (int i = 0; i < n; i++)
        buf[i] *= rht_sign(seed, i);
}

/* ---- NIF ---------------------------------------------------------------- */

/** nt_turboquant(Ref, Bits, Seed) -> {ok, Ref}
 *  TurboQuant_mse fake-quant round-trip on a 2D tensor [rows, cols], per row.
 *  Returns the reconstructed FP64 tensor (same shape). bits in 1..8.
 */
ERL_NIF_TERM nt_turboquant(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *a = get_tensor(env, argv[0]);
    if (!a)
        return make_error(env, "invalid_tensor");
    int bits;
    unsigned long seed_ul;
    if (!enif_get_int(env, argv[1], &bits))
        return make_error(env, "invalid_bits");
    if (!enif_get_ulong(env, argv[2], &seed_ul))
        return make_error(env, "invalid_seed");
    if (bits < 1 || bits > 8)
        return make_error(env, "bits_out_of_range");

    int rows, cols;
    if (a->ndim == 2) {
        rows = a->shape[0];
        cols = a->shape[1];
    } else if (a->ndim == 1) {
        rows = 1;
        cols = a->shape[0];
    } else {
        return make_error(env, "expected_1d_or_2d");
    }

    int L = 1 << bits;
    double centroids[256];
    lloyd_max_normal(L, centroids);

    int n = next_pow2(cols);
    double *buf = (double *)calloc((size_t)n, sizeof(double));
    if (!buf)
        return make_error(env, "out_of_memory");

    NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
    if (!c) {
        free(buf);
        return make_error(env, "out_of_memory");
    }

    const double *src = a->data + a->offset;
    uint64_t seed = (uint64_t)seed_ul;

    for (int r = 0; r < rows; r++) {
        const double *row = src + (size_t)r * cols;
        double *out = c->data + (size_t)r * cols;

        /* L2 norm of the row */
        double norm = 0.0;
        for (int j = 0; j < cols; j++)
            norm += row[j] * row[j];
        norm = sqrt(norm);
        if (norm < 1e-30) {
            for (int j = 0; j < cols; j++)
                out[j] = 0.0;
            continue;
        }

        /* load, unit-normalize, zero-pad tail */
        double inv_norm = 1.0 / norm;
        for (int j = 0; j < cols; j++)
            buf[j] = row[j] * inv_norm;
        for (int j = cols; j < n; j++)
            buf[j] = 0.0;

        /* rotate -> coords ~ N(0, 1/n) */
        rht_forward(buf, n, seed);

        /* Adaptive scale: map the rotated vector's max magnitude onto the
         * codebook's extreme level. This keeps the Lloyd-Max *shape* (optimal
         * for the Gaussian bulk) while never clamping residual outliers — the
         * weakness of a fixed N(0,1) codebook. */
        double amax = 0.0;
        for (int j = 0; j < n; j++) {
            double av = fabs(buf[j]);
            if (av > amax)
                amax = av;
        }
        double c_extreme = centroids[L - 1];
        if (c_extreme <= 0.0)
            c_extreme = 1.0;
        double scl = amax > 0.0 ? c_extreme / amax : 1.0;
        double inv_scl = 1.0 / scl;

        for (int j = 0; j < n; j++) {
            int idx = quantize_to_codebook(buf[j] * scl, centroids, L);
            buf[j] = centroids[idx] * inv_scl;
        }

        /* inverse-rotate, restore norm */
        rht_inverse(buf, n, seed);
        for (int j = 0; j < cols; j++)
            out[j] = buf[j] * norm;
    }

    free(buf);
    return make_ok(env, make_tensor_term(env, c));
}
