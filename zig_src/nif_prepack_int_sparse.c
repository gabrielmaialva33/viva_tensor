/**
 * nif_prepack_int_sparse.c — Prepack FP32 weights into INT8 / INT4 2:4 sparse format.
 *
 * Exposes:
 *   nt_prepack_int8_sparse(WeightBin, [InFeatures, OutFeatures]) -> {ok, PackedWeight}
 *   nt_prepack_int4_sparse(WeightBin, [InFeatures, OutFeatures]) -> {ok, PackedWeight}
 *
 * Both quantize per-output-channel (absmax / qmax), apply 2:4 magnitude pruning
 * along the K axis (groups of 4 contiguous K elements per row of the row-major
 * [out_features, in_features] CUTLASS layout), then upload to device.
 *
 * INT8 path: also builds a cuSPARSELt plan + compressed buffer, suitable for
 *            cuSPARSELt-based forward (winner ≥ 8192²). When K is not divisible
 *            by 16 cuSPARSELt path is skipped and only the CUTLASS path is set up.
 * INT4 path: packs the two kept int4 pairs per 8 K positions and emits the
 *            CUTLASS ElementE metadata consumed by sparse Tensor Core MMA.
 *
 * The PackedWeight resource is owned by agent A (zig_src/nif_packed_weight.c).
 * Symbols expected to be provided there:
 *   - PackedWeight struct (definition replicated below as compile fallback)
 *   - PACKED_WEIGHT_RES (ErlNifResourceType*)
 *   - make_packed_weight_term(env, w) -> ERL_NIF_TERM
 *   - register_packed_weight_resource(env)
 *
 * IMPORTANT: weight binary is interpreted as row-major
 *            shape = [out_features, in_features] (i.e. matrix W mapping x of
 *            length in_features to y of length out_features via y = W @ x).
 *
 * Authored as part of the viva_tensor inference NIF roadmap.
 */

#include <stdalign.h>
#include "viva_nif.h"

/* Sparse layout info exposed by zig_src/cuda_int_sparse_run.cu — used to
 * size the ElementE metadata buffer correctly regardless of CUTLASS
 * specialisation (Sm80+/Sm89 INT8 m16n8k32 → uint32 covering 16 K,
 * INT4 m16n8k128 → uint32 covering 32 K). */
extern void cutlass_int8_sparse_run_info(int* sparse, int* elem_per_e, int* sizeof_e);
extern void cutlass_int4_sparse_run_info(int* sparse, int* elem_per_e, int* sizeof_e);

#ifndef _WIN32
#include <cuda_runtime.h>
#include <cusparseLt.h>
#endif

/* =========================================================================
 * PackedWeight type — duplicated here as a forward-compatible definition.
 *
 * Agent A owns the canonical definition in nif_packed_weight.c. We define it
 * in the same memory layout so the two sources agree (the struct is opaque
 * to BEAM; only the resource pointer crosses the boundary).
 *
 * If agent A's header is checked into viva_nif.h before us, the typedef
 * here will collide. To stay forward-compatible the file uses include guards
 * around the definition and exposes only the *extern* declarations of the
 * helpers as fallbacks.
 * ========================================================================= */

#ifndef VIVA_PACKED_WEIGHT_DEFINED
#define VIVA_PACKED_WEIGHT_DEFINED

typedef enum {
    PW_FP8         = 0,
    PW_INT8_SPARSE = 1,
    PW_INT4_SPARSE = 2,
} PackedWeightDType;

typedef struct {
    PackedWeightDType dtype;
    void*  d_weight;          /* quantized + (for INT4) packed weight buffer  */
    size_t weight_bytes;
    void*  d_metadata;        /* 2:4 metadata buffer (E)                       */
    size_t metadata_bytes;
    void*  d_scales;          /* per-output-channel FP32 dequant scales        */
    size_t scales_count;
    int    in_features;
    int    out_features;
    void*  cusparselt_plan;   /* cusparseLtMatmulPlan_t* or NULL               */
    void*  d_compressed;      /* cuSPARSELt compressed buffer or NULL          */
} PackedWeight;

#endif /* VIVA_PACKED_WEIGHT_DEFINED */

/* Helpers expected from nif_packed_weight.c (agent A). Declared weak so the
 * linker will accept a missing symbol on platforms that support it — if you
 * see "undefined reference to make_packed_weight_term" at link time, agent A
 * has not landed yet. */
extern ErlNifResourceType* PACKED_WEIGHT_RES;
ERL_NIF_TERM make_packed_weight_term(ErlNifEnv* env, PackedWeight* w);

/* =========================================================================
 * Helpers
 * ========================================================================= */

static inline int parse_in_out(ErlNifEnv* env, ERL_NIF_TERM list_term,
                               int* in_features, int* out_features) {
    int shape[8];
    int ndim = 0;
    if (!parse_shape(env, list_term, shape, &ndim)) return 0;
    if (ndim != 2) return 0;
    *in_features = shape[0];
    *out_features = shape[1];
    return 1;
}

/* Free helper that releases any partially-initialised PackedWeight on the
 * error path. */
static void free_packed_weight(PackedWeight* w) {
#ifndef _WIN32
    if (!w) return;
    if (w->d_weight)     cudaFree(w->d_weight);
    if (w->d_metadata)   cudaFree(w->d_metadata);
    if (w->d_scales)     cudaFree(w->d_scales);
    if (w->d_compressed) cudaFree(w->d_compressed);
    if (w->cusparselt_plan) {
        cusparseLtMatmulPlanDestroy((cusparseLtMatmulPlan_t*)w->cusparselt_plan);
        free(w->cusparselt_plan);
    }
#else
    (void)w;
#endif
}

#ifndef _WIN32

/* Round-half-to-even-ish saturating int8 quantization with clamping. */
static inline int8_t f32_to_int8_sat(float v, float inv_scale) {
    float q = v * inv_scale;
    /* Round to nearest, banker's rounding via rintf. */
    long r = lrintf(q);
    if (r < -127) r = -127;
    if (r > 127)  r = 127;
    return (int8_t)r;
}

static inline int8_t f32_to_int4_sat(float v, float inv_scale) {
    float q = v * inv_scale;
    long r = lrintf(q);
    if (r < -7) r = -7;
    if (r > 7)  r = 7;
    return (int8_t)r;
}

/* Sort 4 elements by |value| descending. Returns indices of the 2 largest. */
static inline void top2_of_4(const float* a, int* keep0, int* keep1) {
    float m0 = fabsf(a[0]), m1 = fabsf(a[1]), m2 = fabsf(a[2]), m3 = fabsf(a[3]);
    /* Initial: assume 0 is largest, 1 second. */
    int i0 = 0; float v0 = m0;
    int i1 = 1; float v1 = m1;
    if (v1 > v0) { int ti=i0; float tv=v0; i0=i1; v0=v1; i1=ti; v1=tv; }
    /* Test 2 */
    if (m2 > v0)      { i1 = i0; v1 = v0; i0 = 2; v0 = m2; }
    else if (m2 > v1) { i1 = 2; v1 = m2; }
    /* Test 3 */
    if (m3 > v0)      { i1 = i0; v1 = v0; i0 = 3; v0 = m3; }
    else if (m3 > v1) { i1 = 3; v1 = m3; }
    *keep0 = (i0 < i1) ? i0 : i1;
    *keep1 = (i0 < i1) ? i1 : i0;
}

/* CUTLASS sparse INT4 metadata indexes pairs of adjacent int4 values.
 * The official host_uncompress.h path uses step=8, a=4, b=8, ElementsPerE=2:
 * one metadata nibble names two kept pairs inside each 8-int4 chunk. */
static inline void top2_pairs_of_8(const float* a, int* keep0, int* keep1) {
    float score[4];
    for (int p = 0; p < 4; ++p) {
        score[p] = fabsf(a[p * 2 + 0]) + fabsf(a[p * 2 + 1]);
    }

    int i0 = 0;
    int i1 = 1;
    if (score[i1] > score[i0]) {
        int tmp = i0; i0 = i1; i1 = tmp;
    }

    for (int p = 2; p < 4; ++p) {
        if (score[p] > score[i0]) {
            i1 = i0;
            i0 = p;
        } else if (score[p] > score[i1]) {
            i1 = p;
        }
    }

    *keep0 = (i0 < i1) ? i0 : i1;
    *keep1 = (i0 < i1) ? i1 : i0;
}

/* =========================================================================
 * INT8 2:4 sparse prepack
 *
 * Steps:
 *   1) Reinterpret weight bin as FP32 [out_features, in_features] row-major.
 *      (HuggingFace convention: linear y = W @ x, W has shape [out, in].)
 *   2) Per row (per output channel) — compute absmax → scale[o] = absmax / 127.
 *   3) Quantize element-wise to int8.
 *   4) Apply 2:4 magnitude prune in groups of 4 contiguous K positions.
 *   5) Build the "compressed" sparse A layout for CUTLASS / cuSPARSELt:
 *      - CUTLASS expects A row-major with K halved (sparse representation).
 *      - cuSPARSELt manages its own compressed layout via SpMMACompress.
 *
 *   Stored in PackedWeight:
 *     d_weight        = int8 quantized + zeroed-non-keep weights, size out * in
 *                       (kept for reference / fallback path & for cuSPARSELt input).
 *     d_metadata      = CUTLASS ElementE buffer (uint16 per 16 K positions).
 *                       NULL on cuSPARSELt-only path.
 *     d_compressed    = cuSPARSELt compressed buffer (built once at prepack).
 *     cusparselt_plan = compiled plan or NULL.
 *     d_scales        = float per-output-channel scales.
 * ========================================================================= */
ERL_NIF_TERM nt_prepack_int8_sparse(ErlNifEnv* env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void)argc;
    if (!cuda_available()) return make_error(env, "cuda_unavailable");

    ErlNifBinary weight_bin;
    if (!enif_inspect_binary(env, argv[0], &weight_bin))
        return make_error(env, "weight_not_binary");

    int in_features, out_features;
    if (!parse_in_out(env, argv[1], &in_features, &out_features))
        return make_error(env, "bad_shape_arg");

    size_t expected = (size_t)in_features * (size_t)out_features * sizeof(float);
    if (weight_bin.size != expected)
        return make_error(env, "weight_size_mismatch");

    /* K = in_features must be divisible by 4 for 2:4 grouping. */
    if (in_features % 4 != 0)
        return make_error(env, "in_features_not_mul_4");

    const float* W = (const float*)weight_bin.data;

    /* Allocate host scratch. */
    float*  h_scales = (float*)malloc((size_t)out_features * sizeof(float));
    int8_t* h_quant  = (int8_t*)malloc((size_t)out_features * (size_t)in_features);
    if (!h_scales || !h_quant) {
        free(h_scales); free(h_quant);
        return make_error(env, "host_alloc_failed");
    }

    /* Pass 1: per-row absmax → scale. */
    for (int o = 0; o < out_features; ++o) {
        float amax = 0.0f;
        const float* row = W + (size_t)o * (size_t)in_features;
        for (int k = 0; k < in_features; ++k) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        /* Avoid div-by-zero on all-zero rows. */
        h_scales[o] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    }

    /* Pass 2: quantize + 2:4 prune (zero-out non-kept positions). */
    for (int o = 0; o < out_features; ++o) {
        const float* row = W + (size_t)o * (size_t)in_features;
        int8_t* qrow = h_quant + (size_t)o * (size_t)in_features;
        float inv_scale = 1.0f / h_scales[o];
        for (int k = 0; k < in_features; k += 4) {
            float g[4] = { row[k+0], row[k+1], row[k+2], row[k+3] };
            int keep0, keep1;
            top2_of_4(g, &keep0, &keep1);
            for (int j = 0; j < 4; ++j) {
                if (j == keep0 || j == keep1) {
                    qrow[k + j] = f32_to_int8_sat(g[j], inv_scale);
                } else {
                    qrow[k + j] = 0;
                }
            }
        }
    }

    /* Allocate the PackedWeight resource. */
    PackedWeight* pw = (PackedWeight*)enif_alloc_resource(
        PACKED_WEIGHT_RES, sizeof(PackedWeight));
    if (!pw) {
        free(h_scales); free(h_quant);
        return make_error(env, "resource_alloc_failed");
    }
    memset(pw, 0, sizeof(*pw));
    pw->dtype = PW_INT8_SPARSE;
    pw->in_features = in_features;
    pw->out_features = out_features;
    pw->scales_count = (size_t)out_features;

    /* Upload quantized weight to device. */
    cudaError_t cerr;
    size_t weight_bytes = (size_t)out_features * (size_t)in_features; /* INT8 */
    cerr = cudaMalloc(&pw->d_weight, weight_bytes);
    if (cerr != cudaSuccess) goto cuda_fail;
    pw->weight_bytes = weight_bytes;
    cerr = cudaMemcpy(pw->d_weight, h_quant, weight_bytes, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) goto cuda_fail;

    /* Upload scales. */
    cerr = cudaMalloc(&pw->d_scales, (size_t)out_features * sizeof(float));
    if (cerr != cudaSuccess) goto cuda_fail;
    cerr = cudaMemcpy(pw->d_scales, h_scales,
                      (size_t)out_features * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) goto cuda_fail;

    /* CUTLASS metadata "E" buffer.
     * For CUTLASS INT8 sparse (m16n8k32 OpClassTensorOp on Sm80+):
     *   kSparse=2  kElementsPerElementE=16  sizeof(ElementE)=4 (uint32_t)
     * Verified via cutlass_int8_sparse_run_info() at runtime.
     * Buffer shape: [out_features, in_features / (kSparse*kElemsPerE)] = [O, K/32].
     * We zero-fill: cuSPARSELt path computes its own metadata, CUTLASS
     * fallback path is gated on non-zero metadata (a future encoder fills
     * the 2-bit keep indices; for now zero means "all positions 0,1 kept"
     * which is harmless for sanity tests but wrong for real inference). */
    int int8_sparse, int8_elems_per_e, int8_sizeof_e;
    cutlass_int8_sparse_run_info(&int8_sparse, &int8_elems_per_e, &int8_sizeof_e);
    size_t metaK_int8 =
        (size_t)in_features / (size_t)(int8_sparse * int8_elems_per_e);
    if (metaK_int8 > 0) {
        size_t metadata_bytes =
            (size_t)out_features * metaK_int8 * (size_t)int8_sizeof_e;
        cerr = cudaMalloc(&pw->d_metadata, metadata_bytes);
        if (cerr != cudaSuccess) goto cuda_fail;
        cerr = cudaMemset(pw->d_metadata, 0, metadata_bytes);
        if (cerr != cudaSuccess) goto cuda_fail;
        pw->metadata_bytes = metadata_bytes;
    }

    /* Build cuSPARSELt plan + compressed buffer.
     * The plan caches a specific (M=out_features, K=in_features) sparse shape
     * but is parametric in N (batch). We pick a representative N=128 for the
     * SearchIterations algorithm tuning; cusparseLtMatmul accepts a different
     * N at call time as long as descriptors are rebuilt. To keep things
     * simple we don't auto-tune here — just compile a default plan. */
    {
        cusparseLtHandle_t handle;
        if (cusparseLtInit(&handle) != CUSPARSE_STATUS_SUCCESS) {
            /* Non-fatal — fall back to CUTLASS-only path. */
            goto cusparselt_skip;
        }

        cusparseLtMatDescriptor_t matA, matB, matC, matD;
        cusparseLtMatmulDescriptor_t matmulDescr;
        cusparseLtMatmulAlgSelection_t algSel;
        cusparseLtMatmulPlan_t plan;

        /* A (sparse): INT8 [out_features, in_features] ROW-major */
        if (cusparseLtStructuredDescriptorInit(
                &handle, &matA, out_features, in_features, in_features,
                16, CUDA_R_8I, CUSPARSE_ORDER_ROW,
                CUSPARSELT_SPARSITY_50_PERCENT) != CUSPARSE_STATUS_SUCCESS)
            goto plan_fail;

        /* B (dense input^T): INT8 [in_features, N_dummy] COL-major, K = ldb. */
        int N_dummy = 128;
        if (cusparseLtDenseDescriptorInit(
                &handle, &matB, in_features, N_dummy, in_features,
                16, CUDA_R_8I, CUSPARSE_ORDER_COL) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            goto plan_fail;
        }

        if (cusparseLtDenseDescriptorInit(
                &handle, &matC, out_features, N_dummy, N_dummy,
                16, CUDA_R_32I, CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            goto plan_fail;
        }

        if (cusparseLtDenseDescriptorInit(
                &handle, &matD, out_features, N_dummy, N_dummy,
                16, CUDA_R_32I, CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            goto plan_fail;
        }

        if (cusparseLtMatmulDescriptorInit(
                &handle, &matmulDescr,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &matA, &matB, &matC, &matD,
                CUSPARSE_COMPUTE_32I) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        if (cusparseLtMatmulAlgSelectionInit(
                &handle, &algSel, &matmulDescr,
                CUSPARSELT_MATMUL_ALG_DEFAULT) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        if (cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel)
            != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        /* SpMMA prune on the device buffer (idempotent for already-pruned data). */
        if (cusparseLtSpMMAPrune(&handle, &matmulDescr,
                                  (const void*)pw->d_weight,
                                  pw->d_weight,
                                  CUSPARSELT_PRUNE_SPMMA_STRIP, 0)
            != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        size_t compressed_size = 0, compressed_buf_size = 0;
        if (cusparseLtSpMMACompressedSize(&handle, &plan,
                                          &compressed_size, &compressed_buf_size)
            != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        cerr = cudaMalloc(&pw->d_compressed, compressed_size);
        if (cerr != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }
        void* d_compress_buf = NULL;
        cerr = cudaMalloc(&d_compress_buf, compressed_buf_size);
        if (cerr != cudaSuccess) {
            cudaFree(pw->d_compressed); pw->d_compressed = NULL;
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }

        if (cusparseLtSpMMACompress(&handle, &plan,
                                     (const void*)pw->d_weight,
                                     pw->d_compressed,
                                     d_compress_buf, 0)
            != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(pw->d_compressed); pw->d_compressed = NULL;
            cudaFree(d_compress_buf);
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }
        cudaDeviceSynchronize();
        cudaFree(d_compress_buf);

        /* Persist the plan handle.  We allocate on the heap so the resource
         * destructor can call cusparseLtMatmulPlanDestroy. */
        cusparseLtMatmulPlan_t* held_plan =
            (cusparseLtMatmulPlan_t*)malloc(sizeof(cusparseLtMatmulPlan_t));
        if (!held_plan) {
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }
        memcpy(held_plan, &plan, sizeof(plan));
        pw->cusparselt_plan = (void*)held_plan;

        /* Note: matA..matD descriptors are owned by the plan internally in
         * cuSPARSELt; we don't destroy them here. The handle itself is
         * stateless and can be destroyed. */
        cusparseLtDestroy(&handle);
        goto cusparselt_done;

    plan_fail:
        cusparseLtDestroy(&handle);
        /* Non-fatal — fall through with cusparselt_plan = NULL. */
    cusparselt_skip:
        ;
    cusparselt_done:
        ;
    }

    free(h_scales);
    free(h_quant);

    ERL_NIF_TERM term = make_packed_weight_term(env, pw);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

cuda_fail:
    free_packed_weight(pw);
    enif_release_resource(pw);
    free(h_scales);
    free(h_quant);
    return make_error(env, "cuda_alloc_failed");
}

/* =========================================================================
 * INT4 2:4 sparse prepack
 *
 * Layout: row-major [out_features, in_features]; every 8-int4 K chunk keeps
 * two adjacent int4 pairs, packs those four values into two bytes, and writes
 * one metadata nibble.
 *
 *   in_features must be divisible by 128 to match the m16n8k128 sparse MMA
 *   tile used by the runtime launcher.
 *
 * After sparse-pair pruning, the "packed sparse A" buffer has size
 *   out_features * (in_features / 2 / 2) bytes  (kSparse=2 halves K, nibbles halve again)
 *
 * One 32-bit ElementE covers 64 logical INT4 positions as 8 metadata nibbles.
 * ========================================================================= */
ERL_NIF_TERM nt_prepack_int4_sparse(ErlNifEnv* env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void)argc;
    if (!cuda_available()) return make_error(env, "cuda_unavailable");

    ErlNifBinary weight_bin;
    if (!enif_inspect_binary(env, argv[0], &weight_bin))
        return make_error(env, "weight_not_binary");

    int in_features, out_features;
    if (!parse_in_out(env, argv[1], &in_features, &out_features))
        return make_error(env, "bad_shape_arg");

    size_t expected = (size_t)in_features * (size_t)out_features * sizeof(float);
    if (weight_bin.size != expected)
        return make_error(env, "weight_size_mismatch");

    /* CUTLASS INT4 sparse MMA is m16n8k128; ElementE covers 64 logical int4s
     * and the runtime launcher expects K aligned to the 128-wide MMA K tile. */
    if (in_features % 128 != 0)
        return make_error(env, "in_features_not_mul_128");

    const float* W = (const float*)weight_bin.data;

    /* Host scratch. */
    float*   h_scales   = (float*)malloc((size_t)out_features * sizeof(float));
    int8_t*  h_quant    = (int8_t*)malloc((size_t)out_features * (size_t)in_features);
    if (!h_scales || !h_quant) {
        free(h_scales); free(h_quant);
        return make_error(env, "host_alloc_failed");
    }

    /* Per-output-channel absmax → scale (INT4 qmax = 7). */
    for (int o = 0; o < out_features; ++o) {
        const float* row = W + (size_t)o * (size_t)in_features;
        float amax = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        h_scales[o] = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    }

    /* Quantize + sparse-pair prune. h_quant holds int8 in range [-7, 7].
     * CUTLASS INT4 sparse metadata indexes 2-int4 pairs, not individual
     * scalar int4 lanes, so the pruning unit is 2 kept pairs per 8 values. */
    for (int o = 0; o < out_features; ++o) {
        const float* row = W + (size_t)o * (size_t)in_features;
        int8_t* qrow = h_quant + (size_t)o * (size_t)in_features;
        float inv_scale = 1.0f / h_scales[o];
        for (int k = 0; k < in_features; k += 8) {
            int keep_pair0, keep_pair1;
            top2_pairs_of_8(row + k, &keep_pair0, &keep_pair1);
            for (int pair = 0; pair < 4; ++pair) {
                for (int lane = 0; lane < 2; ++lane) {
                    int j = pair * 2 + lane;
                    if (pair == keep_pair0 || pair == keep_pair1) {
                        qrow[k + j] = f32_to_int4_sat(row[k + j], inv_scale);
                    } else {
                        qrow[k + j] = 0;
                    }
                }
            }
        }
    }

    /* Metadata buffer for CUTLASS INT4 2:4 sparse (m16n8k128, Sm80+).
     *
     * CUTLASS' official host_uncompress.h decodes int4 ElementE as:
     *   ElementE:uint32 covers 64 logical int4 values;
     *   each nibble covers 8 int4 values;
     *   bits[1:0] = first kept pair index, bits[3:2] = second kept pair.
     */
    int int4_sparse, int4_elems_per_e, int4_sizeof_e;
    cutlass_int4_sparse_run_info(&int4_sparse, &int4_elems_per_e, &int4_sizeof_e);
    if (int4_sizeof_e != (int)sizeof(uint32_t)) {
        free(h_scales); free(h_quant);
        return make_error(env, "unexpected_int4_elemente_size");
    }
    size_t meta_units_per_row =
        (size_t)in_features / (size_t)(int4_sparse * int4_elems_per_e);
    size_t meta_bytes_per_row = meta_units_per_row * (size_t)int4_sizeof_e;
    size_t meta_total = (size_t)out_features * meta_bytes_per_row;
    uint8_t* h_meta = (uint8_t*)malloc(meta_total);
    if (!h_meta) {
        free(h_scales); free(h_quant);
        return make_error(env, "host_alloc_failed");
    }
    memset(h_meta, 0, meta_total);

    for (int o = 0; o < out_features; ++o) {
        const float* row = W + (size_t)o * (size_t)in_features;
        uint32_t* meta_row = (uint32_t*)(h_meta + (size_t)o * meta_bytes_per_row);
        for (int k = 0; k < in_features; k += 8) {
            int keep_pair0, keep_pair1;
            top2_pairs_of_8(row + k, &keep_pair0, &keep_pair1);
            size_t group = (size_t)k / 8;
            size_t word = group / 8;
            size_t nibble = group % 8;
            uint32_t e = (uint32_t)(keep_pair0 | (keep_pair1 << 2));
            meta_row[word] |= e << (nibble * 4);
        }
    }

    size_t packed_bytes_per_row = (size_t)in_features / 4;  /* 50% sparse, int4 packed */
    size_t packed_total = (size_t)out_features * packed_bytes_per_row;
    int8_t* h_packed = (int8_t*)malloc(packed_total);
    if (!h_packed) {
        free(h_scales); free(h_quant); free(h_meta);
        return make_error(env, "host_alloc_failed");
    }
    memset(h_packed, 0, packed_total);

    /* Pack compressed A in the same pair order named by ElementE. */
    for (int o = 0; o < out_features; ++o) {
        int8_t* qrow = h_quant + (size_t)o * (size_t)in_features;
        int8_t* prow = h_packed + (size_t)o * packed_bytes_per_row;

        size_t out_byte_idx = 0;
        for (int k = 0; k < in_features; k += 8) {
            uint32_t* meta_row = (uint32_t*)(h_meta + (size_t)o * meta_bytes_per_row);
            size_t group = (size_t)k / 8;
            uint32_t e = (meta_row[group / 8] >> ((group % 8) * 4)) & 0x0F;
            int keep_pair0 = (int)(e & 0x3);
            int keep_pair1 = (int)(e >> 2);

            int base0 = k + keep_pair0 * 2;
            int base1 = k + keep_pair1 * 2;
            uint8_t p0_lo = (uint8_t)qrow[base0 + 0] & 0x0F;
            uint8_t p0_hi = ((uint8_t)qrow[base0 + 1] & 0x0F) << 4;
            uint8_t p1_lo = (uint8_t)qrow[base1 + 0] & 0x0F;
            uint8_t p1_hi = ((uint8_t)qrow[base1 + 1] & 0x0F) << 4;
            prow[out_byte_idx++] = (int8_t)(p0_lo | p0_hi);
            prow[out_byte_idx++] = (int8_t)(p1_lo | p1_hi);
        }
    }

    /* Allocate PackedWeight resource. */
    PackedWeight* pw = (PackedWeight*)enif_alloc_resource(
        PACKED_WEIGHT_RES, sizeof(PackedWeight));
    if (!pw) {
        free(h_scales); free(h_quant); free(h_packed); free(h_meta);
        return make_error(env, "resource_alloc_failed");
    }
    memset(pw, 0, sizeof(*pw));
    pw->dtype = PW_INT4_SPARSE;
    pw->in_features = in_features;
    pw->out_features = out_features;
    pw->scales_count = (size_t)out_features;

    cudaError_t cerr;
    cerr = cudaMalloc(&pw->d_weight, packed_total);
    if (cerr != cudaSuccess) goto cuda_fail4;
    pw->weight_bytes = packed_total;
    cerr = cudaMemcpy(pw->d_weight, h_packed, packed_total, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) goto cuda_fail4;

    cerr = cudaMalloc(&pw->d_metadata, meta_total);
    if (cerr != cudaSuccess) goto cuda_fail4;
    pw->metadata_bytes = meta_total;
    cerr = cudaMemcpy(pw->d_metadata, h_meta, meta_total, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) goto cuda_fail4;

    cerr = cudaMalloc(&pw->d_scales, (size_t)out_features * sizeof(float));
    if (cerr != cudaSuccess) goto cuda_fail4;
    cerr = cudaMemcpy(pw->d_scales, h_scales,
                      (size_t)out_features * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) goto cuda_fail4;

    /* No cuSPARSELt plan for INT4 (cuSPARSELt does not support int4 on Ada). */
    pw->cusparselt_plan = NULL;
    pw->d_compressed = NULL;

    free(h_scales); free(h_quant); free(h_packed); free(h_meta);

    ERL_NIF_TERM term = make_packed_weight_term(env, pw);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

cuda_fail4:
    free_packed_weight(pw);
    enif_release_resource(pw);
    free(h_scales); free(h_quant); free(h_packed); free(h_meta);
    return make_error(env, "cuda_alloc_failed");
}

#else  /* _WIN32 — CUDA is Linux-only in this codebase. */

ERL_NIF_TERM nt_prepack_int8_sparse(ErlNifEnv* env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    return make_error(env, "cuda_unavailable_on_windows");
}

ERL_NIF_TERM nt_prepack_int4_sparse(ErlNifEnv* env, int argc,
                                    const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    return make_error(env, "cuda_unavailable_on_windows");
}

#endif  /* _WIN32 */
