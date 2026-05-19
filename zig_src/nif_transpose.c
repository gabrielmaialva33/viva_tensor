/**
 * nif_transpose.c - Fast row-major FP32 matrix transpose NIF.
 *
 * Replaces the pure-Erlang `transpose_fp32/3` hot path used by the
 * safetensors loader. The Erlang version walks `binary_part/3` in a
 * comprehension and is O(Rows*Cols) with allocator pressure — 20s for
 * the 32000x2048 lm_head weight on TinyLlama-1.1B.
 *
 * This NIF does a tiled cache-friendly transpose in C:
 *   Out[c, r] = In[r, c],  c in [0..Cols), r in [0..Rows)
 *
 * Input is a row-major FP32 binary of shape (Rows, Cols), output is
 * a fresh row-major FP32 binary of shape (Cols, Rows).
 *
 * Single-threaded — sequential C with a 32x32 element tile is already
 * dramatically faster than the Erlang fallback (target: <500ms for the
 * 32000x2048 lm_head, was ~20s).
 *
 * Arity: nt_transpose_fp32(Binary, Rows, Cols) -> Binary
 *   - Returns the raw transposed binary (NOT wrapped in {ok, _}).
 *   - On size mismatch / invalid args, raises a badarg via
 *     enif_make_badarg(env). The Erlang wrapper catches this and
 *     falls back to the pure-Erlang implementation.
 */

#include "viva_nif.h"

/* Tile size: 32x32 f32 = 4 KiB per tile (per direction).
 * Fits comfortably in L1 even alongside the row/column working set. */
#define VT_TRANSPOSE_TILE 32

static inline void transpose_tile_fp32(const float *src, float *dst,
                                       int rows, int cols, int r0, int c0,
                                       int tile_rows, int tile_cols) {
  for (int r = 0; r < tile_rows; ++r) {
    const float *src_row = src + (size_t)(r0 + r) * (size_t)cols + (size_t)c0;
    for (int c = 0; c < tile_cols; ++c) {
      /* Out[c0+c, r0+r] = In[r0+r, c0+c] */
      dst[(size_t)(c0 + c) * (size_t)rows + (size_t)(r0 + r)] = src_row[c];
    }
  }
}

ERL_NIF_TERM nt_transpose_fp32(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;

  ErlNifBinary in_bin;
  if (!enif_inspect_binary(env, argv[0], &in_bin))
    return enif_make_badarg(env);

  int rows = 0, cols = 0;
  if (!enif_get_int(env, argv[1], &rows) || rows <= 0)
    return enif_make_badarg(env);
  if (!enif_get_int(env, argv[2], &cols) || cols <= 0)
    return enif_make_badarg(env);

  size_t n_elems = (size_t)rows * (size_t)cols;
  if (in_bin.size != n_elems * sizeof(float))
    return enif_make_badarg(env);

  ErlNifBinary out_bin;
  if (!enif_alloc_binary(n_elems * sizeof(float), &out_bin))
    return enif_make_badarg(env);

  const float *src = (const float *)in_bin.data;
  float *dst = (float *)out_bin.data;

  const int TS = VT_TRANSPOSE_TILE;

  for (int r0 = 0; r0 < rows; r0 += TS) {
    int tr = (r0 + TS <= rows) ? TS : (rows - r0);
    for (int c0 = 0; c0 < cols; c0 += TS) {
      int tc = (c0 + TS <= cols) ? TS : (cols - c0);
      transpose_tile_fp32(src, dst, rows, cols, r0, c0, tr, tc);
    }
  }

  return enif_make_binary(env, &out_bin);
}
