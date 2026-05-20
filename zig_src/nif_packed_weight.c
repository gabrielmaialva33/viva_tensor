/**
 * nif_packed_weight.c — PackedWeight resource for stable inference API.
 *
 * Background: the championship benchmark NIFs (`cutlass_fp8_bench`, ...)
 * allocate device memory at the start of every call and free it at the
 * end. That is fine for measuring throughput but ruinous for inference:
 * a real forward pass calls `linear` thousands of times across many
 * layers, and each cudaMalloc/cudaFree is a ~50 µs penalty.
 *
 * The fix: a long-lived `PackedWeight` resource that owns the quantized
 * weight buffer, the per-tensor / per-channel scale buffer, the sparsity
 * metadata (for 2:4 INT8 / INT4 backends), and an optional cuSPARSELt
 * matmul plan. The user prepacks once, then calls `linear_fp8(input, w)`
 * repeatedly with the same `w` until the model is unloaded.
 *
 * Lifetime: when the BEAM garbage-collects the resource handle, the
 * `packed_weight_destructor` runs and frees every owned device buffer.
 *
 * Layout decisions:
 *   - `d_weight` always holds the quantized weight (FP8 bytes, INT8
 *     bytes, INT4 nibbles packed two-per-byte) in the layout the kernel
 *     expects. For CUTLASS that's column-major (`B[K,N]` colmajor =
 *     `B^T` in row-major); for cuSPARSELt the dense input order is
 *     plan-defined.
 *   - `d_scales` holds dequantization scales. FP8 = one FP32. INT8/INT4
 *     = `out_features` FP32 values (per-output-channel).
 *   - `d_metadata` holds the 2:4 sparsity metadata `E` for the sparse
 *     backends; NULL for FP8 (dense).
 *   - `cusparselt_plan` is allocated via `enif_alloc` (so it can be
 *     destroyed cleanly) and only populated by the cuSPARSELt prepack
 *     paths. `d_compressed` is the cuSPARSELt-side compressed buffer.
 *
 * Other agents (B, C) populate INT8 / INT4 / SwiGLU paths and reuse
 * this same resource type via the `dtype` discriminant.
 */

#include "viva_nif.h"
#include "nif_packed_weight.h"

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
#include <cuda_runtime.h>
#endif

ErlNifResourceType *PACKED_WEIGHT_RES = NULL;
ErlNifResourceType *EMBEDDING_TABLE_RES = NULL;

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
extern uint16_t float_to_half(float f);

static float bf16_to_float(uint16_t u) {
  uint32_t bits = ((uint32_t)u) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}
#endif

void packed_weight_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  PackedWeight *w = (PackedWeight *)obj;
  if (!w) return;

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
  if (w->d_weight) {
    cudaFree(w->d_weight);
    w->d_weight = NULL;
  }
  if (w->d_metadata) {
    cudaFree(w->d_metadata);
    w->d_metadata = NULL;
  }
  if (w->d_scales) {
    cudaFree(w->d_scales);
    w->d_scales = NULL;
  }
  if (w->d_fp16_cache) {
    cudaFree(w->d_fp16_cache);
    w->d_fp16_cache = NULL;
    w->fp16_cache_bytes = 0;
    w->fp16_cache_ready = 0;
  }
  if (w->d_compressed) {
    cudaFree(w->d_compressed);
    w->d_compressed = NULL;
  }
  /* cusparselt_plan is an opaque pointer allocated by the sparse prepack
   * NIFs (agents B/C). We can't call cusparseLtMatmulPlanDestroy here
   * without pulling in <cusparseLt.h> in every translation unit, so we
   * forward the destroy through a helper that the sparse code installs.
   * For now: free the host-side allocation. The sparse implementation
   * MUST drain device-side cuSPARSELt state inside its prepack path or
   * provide a destructor callback that gets registered here.
   *
   * If cusparselt_plan is set, the owning code is expected to have
   * stored a `void (*)(void*)` deleter into `plan_destroy` and the
   * actual plan in `cusparselt_plan`. We honor that contract. */
  if (w->plan_destroy && w->cusparselt_plan) {
    w->plan_destroy(w->cusparselt_plan);
  }
  w->cusparselt_plan = NULL;
  w->plan_destroy = NULL;
#endif
}

void embedding_table_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  EmbeddingTable *t = (EmbeddingTable *)obj;
  if (!t) return;
#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
  if (t->d_weight) {
    cudaFree(t->d_weight);
    t->d_weight = NULL;
  }
#endif
}

int register_packed_weight_resource(ErlNifEnv *env) {
  PACKED_WEIGHT_RES = enif_open_resource_type(
      env, NULL, "PackedWeight", packed_weight_destructor,
      ERL_NIF_RT_CREATE, NULL);
  return PACKED_WEIGHT_RES != NULL ? 0 : -1;
}

int register_embedding_table_resource(ErlNifEnv *env) {
  EMBEDDING_TABLE_RES = enif_open_resource_type(
      env, NULL, "EmbeddingTable", embedding_table_destructor,
      ERL_NIF_RT_CREATE, NULL);
  return EMBEDDING_TABLE_RES != NULL ? 0 : -1;
}

PackedWeight *alloc_packed_weight(void) {
  PackedWeight *w = (PackedWeight *)enif_alloc_resource(
      PACKED_WEIGHT_RES, sizeof(PackedWeight));
  if (!w) return NULL;
  /* Zero-initialise — destructor relies on NULL pointers for skipped slots. */
  memset(w, 0, sizeof(PackedWeight));
  return w;
}

ERL_NIF_TERM make_packed_weight_term(ErlNifEnv *env, PackedWeight *w) {
  ERL_NIF_TERM term = enif_make_resource(env, w);
  enif_release_resource(w);
  return term;
}

PackedWeight *get_packed_weight(ErlNifEnv *env, ERL_NIF_TERM term) {
  PackedWeight *w = NULL;
  if (!enif_get_resource(env, term, PACKED_WEIGHT_RES, (void **)&w))
    return NULL;
  return w;
}

EmbeddingTable *get_embedding_table(ErlNifEnv *env, ERL_NIF_TERM term) {
  EmbeddingTable *t = NULL;
  if (!enif_get_resource(env, term, EMBEDDING_TABLE_RES, (void **)&t))
    return NULL;
  return t;
}

ERL_NIF_TERM nt_embedding_table_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 3) return make_error(env, "bad_arity");

  ErlNifBinary bf16_bin;
  int vocab = 0, hidden = 0;
  if (!enif_inspect_binary(env, argv[0], &bf16_bin)) return make_error(env, "invalid_embedding");
  if (!enif_get_int(env, argv[1], &vocab) || vocab <= 0) return make_error(env, "invalid_vocab");
  if (!enif_get_int(env, argv[2], &hidden) || hidden <= 0) return make_error(env, "invalid_hidden");

  size_t elems = (size_t)vocab * (size_t)hidden;
  if (bf16_bin.size != elems * sizeof(uint16_t)) return make_error(env, "embedding_size_mismatch");

  uint16_t *host_fp16 = (uint16_t *)enif_alloc(elems * sizeof(uint16_t));
  if (!host_fp16) return make_error(env, "alloc_embedding_host_failed");
  const uint16_t *src = (const uint16_t *)bf16_bin.data;
  for (size_t i = 0; i < elems; ++i) {
    host_fp16[i] = float_to_half(bf16_to_float(src[i]));
  }

  EmbeddingTable *table = (EmbeddingTable *)enif_alloc_resource(
      EMBEDDING_TABLE_RES, sizeof(EmbeddingTable));
  if (!table) {
    enif_free(host_fp16);
    return make_error(env, "resource_alloc_failed");
  }
  memset(table, 0, sizeof(*table));
  table->vocab = vocab;
  table->hidden = hidden;
  table->bytes = elems * sizeof(uint16_t);

  if (cudaMalloc(&table->d_weight, table->bytes) != cudaSuccess ||
      cudaMemcpy(table->d_weight, host_fp16, table->bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    enif_free(host_fp16);
    enif_release_resource(table);
    return make_error(env, "cuda_upload_embedding_failed");
  }
  enif_free(host_fp16);

  ERL_NIF_TERM term = enif_make_resource(env, table);
  enif_release_resource(table);
  return make_ok(env, term);
#endif
}

ERL_NIF_TERM nt_embedding_row(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 2) return make_error(env, "bad_arity");
  EmbeddingTable *table = get_embedding_table(env, argv[0]);
  int token_id = 0;
  if (!table) return make_error(env, "invalid_embedding");
  if (!enif_get_int(env, argv[1], &token_id) || token_id < 0 || token_id >= table->vocab)
    return make_error(env, "invalid_token");
  size_t row_bytes = (size_t)table->hidden * sizeof(uint16_t);
  ErlNifBinary out;
  if (!enif_alloc_binary(row_bytes, &out)) return make_error(env, "alloc_row_failed");
  const uint8_t *row = (const uint8_t *)table->d_weight + (size_t)token_id * row_bytes;
  if (cudaMemcpy(out.data, row, row_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    enif_release_binary(&out);
    return make_error(env, "download_row_failed");
  }
  return make_ok(env, enif_make_binary(env, &out));
#endif
}
