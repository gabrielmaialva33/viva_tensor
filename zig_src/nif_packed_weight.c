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

int register_packed_weight_resource(ErlNifEnv *env) {
  PACKED_WEIGHT_RES = enif_open_resource_type(
      env, NULL, "PackedWeight", packed_weight_destructor,
      ERL_NIF_RT_CREATE, NULL);
  return PACKED_WEIGHT_RES != NULL ? 0 : -1;
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
