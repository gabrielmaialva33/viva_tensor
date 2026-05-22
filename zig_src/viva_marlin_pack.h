#ifndef VIVA_MARLIN_PACK_H
#define VIVA_MARLIN_PACK_H

#include <stdint.h>

int viva_marlin_pack(
  const uint16_t *w_fp16,
  const uint16_t *s_fp16,
  int K,
  int N,
  int groupsize,
  uint32_t *out_B,
  uint16_t *out_s
);

#endif
