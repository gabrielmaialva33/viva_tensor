/**
 * nif_packed_weight.h — shared PackedWeight resource for inference NIFs.
 *
 * Used by:
 *   - nif_prepack_fp8.c        (this agent)
 *   - nif_linear_fp8.c         (this agent)
 *   - nif_prepack_int_sparse.c (agent B)
 *   - nif_linear_int_sparse.c  (agent B)
 *   - nif_linear_swiglu_fp8.c  (agent C)
 *
 * Include from your NIF source, not from viva_nif.h (keeps the legacy
 * header small).
 */

#ifndef VIVA_NIF_PACKED_WEIGHT_H
#define VIVA_NIF_PACKED_WEIGHT_H

#include "viva_nif.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PW_FP8 = 0,
  PW_INT8_SPARSE = 1,
  PW_INT4_SPARSE = 2,
} PackedWeightDType;

typedef struct {
  PackedWeightDType dtype;

  void *d_weight;       /* quantized weight on device (FP8/INT8/INT4) */
  size_t weight_bytes;

  void *d_metadata;     /* 2:4 sparsity metadata `E` — NULL for FP8 */
  size_t metadata_bytes;

  void *d_scales;       /* FP32 dequant scales. Layout depends on block_size:
                         *   block_size == 0: [out_features] (per-channel, legacy)
                         *   block_size  > 0: [out_features * (in_features/block_size)]
                         *                    indexed by [n * num_blocks + (k/block_size)]
                         */
  size_t scales_count;

  void *d_fp16_cache;   /* Optional persistent W8A16 dequant cache. */
  size_t fp16_cache_bytes;
  int fp16_cache_ready;

  int in_features;
  int out_features;
  int block_size;       /* 0 = per-channel (default); 32/128 = per-block along K. */

  /* cuSPARSELt-only fields. Populated by INT8/INT4 sparse prepack paths.
   * `plan_destroy` is a deleter installed by the prepack code (typically
   * a wrapper around cusparseLtMatmulPlanDestroy) so the destructor in
   * nif_packed_weight.c can clean up without dragging cusparseLt.h into
   * every TU. NULL when not in use. */
  void *cusparselt_plan;
  void (*plan_destroy)(void *plan);
  void *d_compressed;
  size_t compressed_bytes;
} PackedWeight;

typedef struct {
  void *d_weight;       /* FP16 embedding table on device, row-major [vocab, hidden]. */
  size_t bytes;
  int vocab;
  int hidden;
} EmbeddingTable;

extern ErlNifResourceType *PACKED_WEIGHT_RES;
extern ErlNifResourceType *EMBEDDING_TABLE_RES;

void packed_weight_destructor(ErlNifEnv *env, void *obj);
int register_packed_weight_resource(ErlNifEnv *env);
void embedding_table_destructor(ErlNifEnv *env, void *obj);
int register_embedding_table_resource(ErlNifEnv *env);
PackedWeight *alloc_packed_weight(void);
ERL_NIF_TERM make_packed_weight_term(ErlNifEnv *env, PackedWeight *w);
PackedWeight *get_packed_weight(ErlNifEnv *env, ERL_NIF_TERM term);
EmbeddingTable *get_embedding_table(ErlNifEnv *env, ERL_NIF_TERM term);

/* NIF entry points (defined in the per-dtype files). Each is registered
 * in nif_funcs[] inside nif_entry.c. Stubs in viva_tensor_zig.erl. */
ERL_NIF_TERM nt_prepack_fp8(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_prepack_fp8_blocked(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_embedding_table_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_embedding_row(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_linear_fp8(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_linear_fp8_w8a16(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_linear_gelu_fp8(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]);

#ifdef __cplusplus
}
#endif

#endif /* VIVA_NIF_PACKED_WEIGHT_H */
