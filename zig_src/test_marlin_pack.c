#include "viva_marlin_pack.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum ScenarioMode { SCENARIO_SANITY_BITS = 0, SCENARIO_SIN_FP16 = 1 };

struct Scenario {
    const char *name;
    int K;
    int N;
    int groupsize;
    enum ScenarioMode mode;
    const char *path;
};

static uint16_t fp32_to_fp16(float x) {
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

static void fill_inputs(const struct Scenario *scenario, uint16_t *w_fp16, uint16_t *s_fp16,
                        size_t w_count, size_t s_count) {
    size_t i;

    if (scenario->mode == SCENARIO_SANITY_BITS) {
        for (i = 0; i < w_count; i++) {
            w_fp16[i] = (uint16_t)(i & 0x3fff);
        }
        for (i = 0; i < s_count; i++) {
            s_fp16[i] = 0x3c00u;
        }
        return;
    }

    for (i = 0; i < w_count; i++) {
        float w_f32 = sinf((float)i * 0.01f) * 0.5f;
        w_fp16[i] = fp32_to_fp16(w_f32);
    }
    for (i = 0; i < s_count; i++) {
        s_fp16[i] = fp32_to_fp16(0.05f);
    }
}

static int run_scenario(const struct Scenario *scenario) {
    const int groups = scenario->K / scenario->groupsize;
    const size_t w_count = (size_t)scenario->K * (size_t)scenario->N;
    const size_t s_count = (size_t)groups * (size_t)scenario->N;
    const size_t b_count = (size_t)(scenario->K / 16) * (size_t)(scenario->N * 2);
    uint16_t *w_fp16 = (uint16_t *)malloc(w_count * sizeof(uint16_t));
    uint16_t *s_fp16 = (uint16_t *)malloc(s_count * sizeof(uint16_t));
    uint32_t *out_B = (uint32_t *)malloc(b_count * sizeof(uint32_t));
    uint16_t *out_s = (uint16_t *)malloc(s_count * sizeof(uint16_t));
    FILE *f;
    int rc;
    size_t i;

    if (!w_fp16 || !s_fp16 || !out_B || !out_s) {
        fprintf(stderr, "%s: allocation failed\n", scenario->name);
        free(w_fp16);
        free(s_fp16);
        free(out_B);
        free(out_s);
        return 1;
    }

    fill_inputs(scenario, w_fp16, s_fp16, w_count, s_count);

    rc = viva_marlin_pack(w_fp16, s_fp16, scenario->K, scenario->N, scenario->groupsize, out_B,
                          out_s);
    if (rc != 0) {
        fprintf(stderr, "%s: viva_marlin_pack failed: %d\n", scenario->name, rc);
        free(w_fp16);
        free(s_fp16);
        free(out_B);
        free(out_s);
        return 1;
    }

    printf("%s out_B:", scenario->name);
    for (i = 0; i < 16; i++) {
        printf(" %08x", out_B[i]);
    }
    printf("\n%s out_s:", scenario->name);
    for (i = 0; i < 16; i++) {
        printf(" %04x", out_s[i]);
    }
    printf("\n");

    f = fopen(scenario->path, "wb");
    if (!f) {
        perror("fopen");
        free(w_fp16);
        free(s_fp16);
        free(out_B);
        free(out_s);
        return 1;
    }
    if (fwrite(out_B, sizeof(uint32_t), b_count, f) != b_count) {
        perror("fwrite");
        fclose(f);
        free(w_fp16);
        free(s_fp16);
        free(out_B);
        free(out_s);
        return 1;
    }
    fclose(f);

    free(w_fp16);
    free(s_fp16);
    free(out_B);
    free(out_s);
    return 0;
}

int main(void) {
    const struct Scenario scenarios[] = {
        {
            "scenario1",
            128,
            256,
            128,
            SCENARIO_SANITY_BITS,
            "/tmp/marlin_pack_c.bin",
        },
        {
            "scenario2",
            128,
            256,
            128,
            SCENARIO_SIN_FP16,
            "/tmp/marlin_pack_c2.bin",
        },
        {
            "scenario3",
            256,
            512,
            128,
            SCENARIO_SIN_FP16,
            "/tmp/marlin_pack_c3.bin",
        },
    };
    const size_t scenario_count = sizeof(scenarios) / sizeof(scenarios[0]);
    size_t i;

    for (i = 0; i < scenario_count; i++) {
        int rc = run_scenario(&scenarios[i]);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}
