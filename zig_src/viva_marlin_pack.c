#include "viva_marlin_pack.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Generated from tmp/marlin/marlin/__init__.py:_get_perms(). */
static int kPerm[4096];
static const int kScalePerm[64] = {0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
                                   2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
                                   4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
                                   6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};
static const int kScalePermSingle[32] = {0,  1,  8,  9,  16, 17, 24, 25, 2,  3,  10,
                                         11, 18, 19, 26, 27, 4,  5,  12, 13, 20, 21,
                                         28, 29, 6,  7,  14, 15, 22, 23, 30, 31};
static int kPermInitialized = 0;

enum VivaMarlinPackError {
    VIVA_MARLIN_PACK_ERR_NULL = 1,
    VIVA_MARLIN_PACK_ERR_GROUPSIZE = 2,
    VIVA_MARLIN_PACK_ERR_SHAPE = 3,
    VIVA_MARLIN_PACK_ERR_ALLOC = 4,
    VIVA_MARLIN_PACK_ERR_SCALE = 5,
    VIVA_MARLIN_PACK_ERR_WEIGHT = 6
};

static int viva_marlin_pack_error(enum VivaMarlinPackError error) {
    switch (error) {
    case VIVA_MARLIN_PACK_ERR_NULL:
        return -1;
    case VIVA_MARLIN_PACK_ERR_GROUPSIZE:
        return -2;
    case VIVA_MARLIN_PACK_ERR_SHAPE:
        return -3;
    case VIVA_MARLIN_PACK_ERR_ALLOC:
        return -4;
    case VIVA_MARLIN_PACK_ERR_SCALE:
        return -5;
    case VIVA_MARLIN_PACK_ERR_WEIGHT:
        return -6;
    default:
        return -99;
    }
}

static void viva_marlin_init_perms(void) {
    int raw[1024];
    int perm1024[1024];
    const int interleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int idx = 0;
    int i;
    int j;

    if (kPermInitialized) {
        return;
    }

    for (i = 0; i < 32; i++) {
        int perm1[8];
        int col = i / 4;
        int m = i % 4;
        int p = 0;
        int block;

        for (block = 0; block < 2; block++) {
            perm1[p++] = 16 * (2 * m) + col + 8 * block;
            perm1[p++] = 16 * (2 * m + 1) + col + 8 * block;
            perm1[p++] = 16 * (2 * (m + 4)) + col + 8 * block;
            perm1[p++] = 16 * (2 * (m + 4) + 1) + col + 8 * block;
        }
        for (j = 0; j < 4; j++) {
            int base = 256 * j;
            int q;
            for (q = 0; q < 8; q++) {
                raw[idx++] = perm1[q] + base;
            }
        }
    }

    for (i = 0; i < 128; i++) {
        for (j = 0; j < 8; j++) {
            perm1024[i * 8 + j] = raw[i * 8 + interleave[j]];
        }
    }

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 1024; j++) {
            kPerm[i * 1024 + j] = i * 1024 + perm1024[j];
        }
    }

    kPermInitialized = 1;
}

static float viva_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x03ff);
    float val;

    if (exp == 0) {
        if (mant == 0) {
            val = 0.0f;
        } else {
            val = ldexpf((float)mant, -24);
        }
    } else if (exp == 31) {
        val = mant == 0 ? INFINITY : NAN;
    } else {
        val = ldexpf((float)(1024u + mant), (int)exp - 25);
    }

    return sign ? -val : val;
}

static uint16_t viva_fp32_to_fp16(float x) {
    uint32_t bits;
    uint32_t sign;
    int exp;
    uint32_t mant;
    uint32_t half_mant;
    uint32_t rem;

    memcpy(&bits, &x, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    mant = bits & 0x7fffffu;

    if (exp <= 0) {
        int shift;

        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000u;
        shift = 14 - exp;
        half_mant = mant >> shift;
        rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half_mant & 1u))) {
            half_mant++;
        }
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }

    half_mant = mant >> 13;
    rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            exp++;
            if (exp >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

static float viva_bf16_to_fp32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Read one weight element honoring layout + dtype, return fp16 bits. */
static uint16_t viva_pack_weight_at(const uint16_t *w, int k, int n, int K, int N,
                                    int weight_layout, int weight_dtype) {
    uint16_t raw = (weight_layout == 1) ? w[(size_t)n * (size_t)K + (size_t)k]
                                        : w[(size_t)k * (size_t)N + (size_t)n];
    if (weight_dtype == 1) {
        return viva_fp32_to_fp16(viva_bf16_to_fp32(raw));
    }
    return raw;
}

static int viva_round_nearest_even(float x) {
    float base = floorf(x);
    float frac = x - base;
    long v = (long)base;

    if (frac < 0.5f) {
        return (int)v;
    }
    if (frac > 0.5f) {
        return (int)(v + 1);
    }
    return (v & 1L) ? (int)(v + 1) : (int)v;
}

static int viva_quantize_w4(uint16_t w_h, uint16_t s_h, uint8_t *out) {
    float w = viva_fp16_to_fp32(w_h);
    float s = viva_fp16_to_fp32(s_h);
    int q;

    if (!isfinite(s) || s == 0.0f) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_SCALE);
    }
    if (!isfinite(w)) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_WEIGHT);
    }

    q = viva_round_nearest_even(viva_fp16_to_fp32(viva_fp32_to_fp16(w / s))) + 8;
    if (q < 0) {
        q = 0;
    } else if (q > 15) {
        q = 15;
    }
    *out = (uint8_t)q;
    return 0;
}

static uint16_t viva_scale_at_flat(const uint16_t *s_fp16, int groups, int N, int flat) {
    int row = flat / groups;
    int col = flat % groups;
    return s_fp16[col * N + row];
}

int viva_marlin_pack(const uint16_t *w_fp16, const uint16_t *s_fp16, int K, int N, int groupsize,
                     int weight_layout, int weight_dtype, uint32_t *out_B, uint16_t *out_s) {
    int effective_groupsize;
    int groups;
    uint8_t *w_q;
    int k;
    int n;

    if (!w_fp16 || !s_fp16 || !out_B || !out_s) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_NULL);
    }
    if (groupsize != -1 && groupsize != 128) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_GROUPSIZE);
    }

    effective_groupsize = groupsize == -1 ? K : groupsize;
    if (K <= 0 || N <= 0 || K % 128 != 0 || N % 256 != 0 || effective_groupsize <= 0 ||
        K % effective_groupsize != 0) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_SHAPE);
    }

    viva_marlin_init_perms();

    groups = K / effective_groupsize;
    w_q = (uint8_t *)malloc((size_t)K * (size_t)N);
    if (!w_q) {
        return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_ALLOC);
    }

    /* layout==1 (HF-native): the Erlang-side scales are unusable (sampled/global
     * and in a format that does not match the kernel). Recompute proper symmetric
     * per-group scales here, in fp16, from the (transposed-on-read) weight:
     *   gs[g, n] = max_k_in_group |W[k, n]| / 7,  group g along K (in-features).
     * A single global scale collapses outlier-dominated channels to zero, which
     * is what wrecked Marlin output; per-group (groupsize=128) preserves range. */
    uint16_t *gs = NULL;
    if (weight_layout == 1) {
        gs = (uint16_t *)malloc((size_t)groups * (size_t)N * sizeof(uint16_t));
        if (!gs) {
            free(w_q);
            return viva_marlin_pack_error(VIVA_MARLIN_PACK_ERR_ALLOC);
        }
        for (int g = 0; g < groups; g++) {
            int kbase = g * effective_groupsize;
            for (n = 0; n < N; n++) {
                float maxabs = 0.0f;
                for (int kk = 0; kk < effective_groupsize; kk++) {
                    uint16_t wh = viva_pack_weight_at(w_fp16, kbase + kk, n, K, N, 1, weight_dtype);
                    float v = viva_fp16_to_fp32(wh);
                    float a = v < 0.0f ? -v : v;
                    if (a > maxabs) {
                        maxabs = a;
                    }
                }
                float scale = maxabs / 7.0f;
                if (!(scale > 1.0e-4f)) {
                    scale = 1.0e-4f;
                }
                gs[(size_t)g * (size_t)N + n] = viva_fp32_to_fp16(scale);
            }
        }
    }

    for (k = 0; k < K; k++) {
        int group = k / effective_groupsize;
        int k_in_group = k - group * effective_groupsize;
        for (n = 0; n < N; n++) {
            uint16_t scale_h;
            int rc;

            if (weight_layout == 1) {
                scale_h = gs[(size_t)group * (size_t)N + n];
                (void)k_in_group;
            } else if (effective_groupsize != K) {
                int col = group * N + n;
                scale_h = viva_scale_at_flat(s_fp16, groups, N, col);
                (void)k_in_group;
            } else {
                scale_h = s_fp16[n];
            }

            rc = viva_quantize_w4(
                viva_pack_weight_at(w_fp16, k, n, K, N, weight_layout, weight_dtype), scale_h,
                &w_q[k * N + n]);
            if (rc != 0) {
                free(w_q);
                free(gs);
                return rc;
            }
        }
    }

    if (weight_layout == 1) {
        if (effective_groupsize != K) {
            int total = groups * N;
            int i;
            for (i = 0; i < total; i++) {
                int chunk = i / 64;
                int within = i - chunk * 64;
                out_s[i] = gs[chunk * 64 + kScalePerm[within]];
            }
        } else {
            int i;
            for (i = 0; i < N; i++) {
                int chunk = i / 32;
                int within = i - chunk * 32;
                out_s[i] = gs[chunk * 32 + kScalePermSingle[within]];
            }
        }
        free(gs);
    } else if (effective_groupsize != K) {
        int total = groups * N;
        int i;
        for (i = 0; i < total; i++) {
            int chunk = i / 64;
            int within = i - chunk * 64;
            out_s[i] = viva_scale_at_flat(s_fp16, groups, N, chunk * 64 + kScalePerm[within]);
        }
    } else {
        int i;
        for (i = 0; i < N; i++) {
            int chunk = i / 32;
            int within = i - chunk * 32;
            out_s[i] = s_fp16[chunk * 32 + kScalePermSingle[within]];
        }
    }

    for (k = 0; k < K / 16; k++) {
        int out_cols = N * 2;
        int j;
        for (j = 0; j < out_cols; j++) {
            uint32_t packed = 0;
            int lane;
            for (lane = 0; lane < 8; lane++) {
                int dst_res = j * 8 + lane;
                int block4096 = dst_res / 4096;
                int src_res = block4096 * 4096 + kPerm[dst_res - block4096 * 4096];
                int n_block = src_res / 256;
                int rem = src_res - n_block * 256;
                int k_inner = rem / 16;
                int n_inner = rem - k_inner * 16;
                int src_k = k * 16 + k_inner;
                int src_n = n_block * 16 + n_inner;
                packed |= ((uint32_t)w_q[src_k * N + src_n]) << (4 * lane);
            }
            out_B[k * out_cols + j] = packed;
        }
    }

    free(w_q);
    return 0;
}
