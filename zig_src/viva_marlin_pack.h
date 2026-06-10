#ifndef VIVA_MARLIN_PACK_H
#define VIVA_MARLIN_PACK_H

#include <stdint.h>

/* weight_layout: 0 = w is [K, N] row-major; 1 = w is HF-native [N, K] (out, in),
 *                read transposed so B becomes the logical [K=in, N=out] matrix.
 * weight_dtype:  0 = w is fp16; 1 = w is bf16 (converted to fp16 on read). */
int viva_marlin_pack(const uint16_t *w_fp16, const uint16_t *s_fp16, int K, int N, int groupsize,
                     int weight_layout, int weight_dtype, uint32_t *out_B, uint16_t *out_s);

#endif
