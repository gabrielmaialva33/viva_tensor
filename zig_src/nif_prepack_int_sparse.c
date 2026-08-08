/**
 * nif_prepack_int_sparse.c — Prepack FP32 weights into INT8 2:4 / INT4 pair-4:8 format.
 *
 * Exposes:
 *   nt_prepack_int8_sparse(WeightBin, [InFeatures, OutFeatures]) -> {ok, PackedWeight}
 *   nt_prepack_int4_sparse(WeightBin, [InFeatures, OutFeatures]) -> {ok, PackedWeight}
 *   nt_prepack_int4_sparse_pair_4_8(WeightBin, Shape, MaskBin) -> {ok, PackedWeight}
 *
 * INT8 applies scalar 2:4 magnitude pruning. INT4 keeps two adjacent pairs per
 * 8 K lanes, either selected by magnitude or supplied by an authoritative mask,
 * then uploads the matching compressed values and metadata to the device.
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
#include "nif_packed_weight.h"

/* Sparse layout info exposed by zig_src/cuda_int_sparse_run.cu — used to
 * size the ElementE metadata buffer correctly regardless of CUTLASS
 * specialisation (Sm80+/Sm89 INT8 m16n8k32 → uint32 covering 16 K,
 * INT4 m16n8k128 → uint32 covering 32 K). */
extern void cutlass_int8_sparse_run_info(int *sparse, int *elem_per_e, int *sizeof_e);
extern void cutlass_int4_sparse_run_info(int *sparse, int *elem_per_e, int *sizeof_e);

#ifndef _WIN32
#include <cuda_runtime.h>
#include <cusparseLt.h>
#endif

/* =========================================================================
 * Helpers
 * ========================================================================= */

static inline int parse_in_out(ErlNifEnv *env, ERL_NIF_TERM list_term, int *in_features,
                               int *out_features) {
    int shape[8];
    int ndim = 0;
    if (!parse_shape(env, list_term, shape, &ndim))
        return 0;
    if (ndim != 2)
        return 0;
    *in_features = shape[0];
    *out_features = shape[1];
    return 1;
}

#ifndef _WIN32

static void destroy_cusparselt_plan(void *opaque_plan) {
    if (!opaque_plan)
        return;
    cusparseLtMatmulPlanDestroy((cusparseLtMatmulPlan_t *)opaque_plan);
    free(opaque_plan);
}

/* Round-half-to-even-ish saturating int8 quantization with clamping. */
static inline int8_t f32_to_int8_sat(float v, float inv_scale) {
    float q = v * inv_scale;
    /* Round to nearest, banker's rounding via rintf. */
    long r = lrintf(q);
    if (r < -127)
        r = -127;
    if (r > 127)
        r = 127;
    return (int8_t)r;
}

static inline int8_t f32_to_int4_sat(float v, float inv_scale) {
    float q = v * inv_scale;
    long r = lrintf(q);
    if (r < -7)
        r = -7;
    if (r > 7)
        r = 7;
    return (int8_t)r;
}

/* Sort 4 elements by |value| descending. Returns indices of the 2 largest. */
static inline void top2_of_4(const float *a, int *keep0, int *keep1) {
    float m0 = fabsf(a[0]), m1 = fabsf(a[1]), m2 = fabsf(a[2]), m3 = fabsf(a[3]);
    /* Initial: assume 0 is largest, 1 second. */
    int i0 = 0;
    float v0 = m0;
    int i1 = 1;
    float v1 = m1;
    if (v1 > v0) {
        int ti = i0;
        float tv = v0;
        i0 = i1;
        v0 = v1;
        i1 = ti;
        v1 = tv;
    }
    /* Test 2 */
    if (m2 > v0) {
        i1 = i0;
        v1 = v0;
        i0 = 2;
        v0 = m2;
    } else if (m2 > v1) {
        i1 = 2;
        v1 = m2;
    }
    /* Test 3 */
    if (m3 > v0) {
        i1 = i0;
        v1 = v0;
        i0 = 3;
        v0 = m3;
    } else if (m3 > v1) {
        i1 = 3;
        v1 = m3;
    }
    *keep0 = (i0 < i1) ? i0 : i1;
    *keep1 = (i0 < i1) ? i1 : i0;
}

/* CUTLASS sparse INT4 metadata indexes pairs of adjacent int4 values.
 * The official host_uncompress.h path uses step=8, a=4, b=8, ElementsPerE=2:
 * one metadata nibble names two kept pairs inside each 8-int4 chunk. */
static inline void top2_pairs_of_8(const float *a, int *keep0, int *keep1) {
    float score[4];
    for (int p = 0; p < 4; ++p) {
        score[p] = fabsf(a[p * 2 + 0]) + fabsf(a[p * 2 + 1]);
    }

    int i0 = 0;
    int i1 = 1;
    if (score[i1] > score[i0]) {
        int tmp = i0;
        i0 = i1;
        i1 = tmp;
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

/* Provided by cuda_int_sparse_run.cu — calls cutlass::reorder_meta()
 * directly against a RowMajor src and a ColumnMajorInterleaved<2> dst,
 * eliminating any layout/permutation transcription error. */
extern void cutlass_int4_sparse_reorder_meta_e(uint32_t *dst, const uint32_t *src, int M,
                                               int K_words);

/* Round-trip uncompress (debug-only). */
extern void cutlass_int4_sparse_uncompress_to_dense(int8_t *dst_int8_per_cell,
                                                    const int8_t *compressed_a,
                                                    const uint32_t *meta, int M, int K);

/* Pure CUTLASS-driven self-test (debug only). Returns 0 on success. */
extern int cutlass_int4_sparse_self_test(int M, int N, int K, int *out_max_diff);

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
ERL_NIF_TERM nt_prepack_int8_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    if (!cuda_available())
        return make_error(env, "cuda_unavailable");

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

    const float *W = (const float *)weight_bin.data;

    /* Allocate host scratch. */
    float *h_scales = (float *)malloc((size_t)out_features * sizeof(float));
    int8_t *h_quant = (int8_t *)malloc((size_t)out_features * (size_t)in_features);
    if (!h_scales || !h_quant) {
        free(h_scales);
        free(h_quant);
        return make_error(env, "host_alloc_failed");
    }

    /* Pass 1: per-row absmax → scale. */
    for (int o = 0; o < out_features; ++o) {
        float amax = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_features;
        for (int k = 0; k < in_features; ++k) {
            float a = fabsf(row[k]);
            if (a > amax)
                amax = a;
        }
        /* Avoid div-by-zero on all-zero rows. */
        h_scales[o] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    }

    /* Pass 2: quantize + 2:4 prune (zero-out non-kept positions). */
    for (int o = 0; o < out_features; ++o) {
        const float *row = W + (size_t)o * (size_t)in_features;
        int8_t *qrow = h_quant + (size_t)o * (size_t)in_features;
        float inv_scale = 1.0f / h_scales[o];
        for (int k = 0; k < in_features; k += 4) {
            float g[4] = {row[k + 0], row[k + 1], row[k + 2], row[k + 3]};
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
    PackedWeight *pw = alloc_packed_weight();
    if (!pw) {
        free(h_scales);
        free(h_quant);
        return make_error(env, "resource_alloc_failed");
    }
    pw->dtype = PW_INT8_SPARSE;
    pw->in_features = in_features;
    pw->out_features = out_features;
    pw->scales_count = (size_t)out_features;

    /* Upload quantized weight to device. */
    cudaError_t cerr;
    size_t weight_bytes = (size_t)out_features * (size_t)in_features; /* INT8 */
    cerr = cudaMalloc(&pw->d_weight, weight_bytes);
    if (cerr != cudaSuccess)
        goto cuda_fail;
    pw->weight_bytes = weight_bytes;
    cerr = cudaMemcpy(pw->d_weight, h_quant, weight_bytes, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
        goto cuda_fail;

    /* Upload scales. */
    cerr = cudaMalloc(&pw->d_scales, (size_t)out_features * sizeof(float));
    if (cerr != cudaSuccess)
        goto cuda_fail;
    cerr = cudaMemcpy(pw->d_scales, h_scales, (size_t)out_features * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
        goto cuda_fail;

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
    size_t metaK_int8 = (size_t)in_features / (size_t)(int8_sparse * int8_elems_per_e);
    if (metaK_int8 > 0) {
        size_t metadata_bytes = (size_t)out_features * metaK_int8 * (size_t)int8_sizeof_e;
        cerr = cudaMalloc(&pw->d_metadata, metadata_bytes);
        if (cerr != cudaSuccess)
            goto cuda_fail;
        cerr = cudaMemset(pw->d_metadata, 0, metadata_bytes);
        if (cerr != cudaSuccess)
            goto cuda_fail;
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
        if (cusparseLtStructuredDescriptorInit(&handle, &matA, out_features, in_features,
                                               in_features, 16, CUDA_R_8I, CUSPARSE_ORDER_ROW,
                                               CUSPARSELT_SPARSITY_50_PERCENT) !=
            CUSPARSE_STATUS_SUCCESS)
            goto plan_fail;

        /* B (dense input^T): INT8 [in_features, N_dummy] COL-major, K = ldb. */
        int N_dummy = 128;
        if (cusparseLtDenseDescriptorInit(&handle, &matB, in_features, N_dummy, in_features, 16,
                                          CUDA_R_8I,
                                          CUSPARSE_ORDER_COL) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            goto plan_fail;
        }

        if (cusparseLtDenseDescriptorInit(&handle, &matC, out_features, N_dummy, N_dummy, 16,
                                          CUDA_R_32I,
                                          CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            goto plan_fail;
        }

        if (cusparseLtDenseDescriptorInit(&handle, &matD, out_features, N_dummy, N_dummy, 16,
                                          CUDA_R_32I,
                                          CUSPARSE_ORDER_ROW) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            goto plan_fail;
        }

        if (cusparseLtMatmulDescriptorInit(&handle, &matmulDescr, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC,
                                           &matD,
                                           CUSPARSE_COMPUTE_32I) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        if (cusparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmulDescr,
                                             CUSPARSELT_MATMUL_ALG_DEFAULT) !=
            CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        if (cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel) !=
            CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        /* SpMMA prune on the device buffer (idempotent for already-pruned data). */
        if (cusparseLtSpMMAPrune(&handle, &matmulDescr, (const void *)pw->d_weight, pw->d_weight,
                                 CUSPARSELT_PRUNE_SPMMA_STRIP, 0) != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matB);
            cusparseLtMatDescriptorDestroy(&matC);
            cusparseLtMatDescriptorDestroy(&matD);
            goto plan_fail;
        }

        size_t compressed_size = 0, compressed_buf_size = 0;
        if (cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buf_size) !=
            CUSPARSE_STATUS_SUCCESS) {
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
        pw->compressed_bytes = compressed_size;
        void *d_compress_buf = NULL;
        cerr = cudaMalloc(&d_compress_buf, compressed_buf_size);
        if (cerr != cudaSuccess) {
            cudaFree(pw->d_compressed);
            pw->d_compressed = NULL;
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }

        if (cusparseLtSpMMACompress(&handle, &plan, (const void *)pw->d_weight, pw->d_compressed,
                                    d_compress_buf, 0) != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(pw->d_compressed);
            pw->d_compressed = NULL;
            cudaFree(d_compress_buf);
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }
        cudaDeviceSynchronize();
        cudaFree(d_compress_buf);

        /* Persist the plan handle.  We allocate on the heap so the resource
         * destructor can call cusparseLtMatmulPlanDestroy. */
        cusparseLtMatmulPlan_t *held_plan =
            (cusparseLtMatmulPlan_t *)malloc(sizeof(cusparseLtMatmulPlan_t));
        if (!held_plan) {
            cusparseLtMatmulPlanDestroy(&plan);
            goto plan_fail;
        }
        memcpy(held_plan, &plan, sizeof(plan));
        pw->cusparselt_plan = (void *)held_plan;
        pw->plan_destroy = destroy_cusparselt_plan;

        /* Note: matA..matD descriptors are owned by the plan internally in
         * cuSPARSELt; we don't destroy them here. The handle itself is
         * stateless and can be destroyed. */
        cusparseLtDestroy(&handle);
        goto cusparselt_done;

    plan_fail:
        cusparseLtDestroy(&handle);
        /* Non-fatal — fall through with cusparselt_plan = NULL. */
    cusparselt_skip:;
    cusparselt_done:;
    }

    free(h_scales);
    free(h_quant);

    ERL_NIF_TERM term = make_packed_weight_term(env, pw);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

cuda_fail:
    enif_release_resource(pw);
    free(h_scales);
    free(h_quant);
    return make_error(env, "cuda_alloc_failed");
}

/* Decode one authoritative pair-4:8 mask byte. Each bit names one scalar K
 * lane; valid masks keep exactly two complete adjacent pairs. */
static int decode_pair48_lane_mask(uint8_t lane_mask, int *keep_pair0, int *keep_pair1) {
    int kept[2] = {-1, -1};
    int count = 0;
    for (int pair = 0; pair < 4; ++pair) {
        uint8_t pair_bits = (lane_mask >> (pair * 2)) & 0x3u;
        if (pair_bits == 0x3u) {
            if (count >= 2)
                return 0;
            kept[count++] = pair;
        } else if (pair_bits != 0) {
            return 0;
        }
    }
    if (count != 2)
        return 0;
    *keep_pair0 = kept[0];
    *keep_pair1 = kept[1];
    return 1;
}

/* =========================================================================
 * INT4 pair-4:8 sparse prepack
 *
 * The public weight is row-major [in_features, out_features]. Internally A is
 * materialised as [out_features, in_features]. Every 8-int4 K chunk keeps two
 * adjacent pairs, packs those four values into two bytes, and writes one
 * metadata nibble.
 *
 * strict_mask, when present, is [out_features, in_features / 8] with one byte
 * per group. Bit j marks logical lane j as kept. It is authoritative: this
 * path validates and preserves it instead of selecting pairs by magnitude.
 * ========================================================================= */
static ERL_NIF_TERM prepack_int4_sparse_impl(ErlNifEnv *env, ERL_NIF_TERM weight_term,
                                             ERL_NIF_TERM shape_term,
                                             const ErlNifBinary *strict_mask) {
    ErlNifBinary weight_bin;
    if (!enif_inspect_binary(env, weight_term, &weight_bin))
        return make_error(env, "weight_not_binary");

    int in_features, out_features;
    if (!parse_in_out(env, shape_term, &in_features, &out_features))
        return make_error(env, "bad_shape_arg");

    size_t expected = (size_t)in_features * (size_t)out_features * sizeof(float);
    if (weight_bin.size != expected)
        return make_error(env, "weight_size_mismatch");

    /* CUTLASS INT4 sparse MMA is m16n8k128; ElementE covers 64 logical int4s
     * and the runtime launcher expects K aligned to the 128-wide MMA K tile. */
    if (in_features % 128 != 0)
        return make_error(env, "in_features_not_mul_128");

    size_t mask_groups_per_row = (size_t)in_features / 8;
    size_t mask_total = (size_t)out_features * mask_groups_per_row;
    if (strict_mask && strict_mask->size != mask_total)
        return make_error(env, "pair48_mask_size_mismatch");

    if (strict_mask) {
        for (size_t idx = 0; idx < mask_total; ++idx) {
            int keep_pair0, keep_pair1;
            if (!decode_pair48_lane_mask(strict_mask->data[idx], &keep_pair0, &keep_pair1)) {
                char reason[128];
                size_t row = idx / mask_groups_per_row;
                size_t group = idx % mask_groups_per_row;
                snprintf(reason, sizeof(reason), "invalid_pair48_mask_at_row_%zu_group_%zu", row,
                         group);
                return make_error(env, reason);
            }
        }
    }

    if (!cuda_available())
        return make_error(env, "cuda_unavailable");

    const float *W = (const float *)weight_bin.data;

    /* Host scratch. h_lane_masks is used for both legacy magnitude selection
     * and the strict explicit-mask path so metadata never has to infer kept
     * pairs from quantized zeros. */
    float *h_scales = (float *)malloc((size_t)out_features * sizeof(float));
    int8_t *h_quant = (int8_t *)malloc((size_t)out_features * (size_t)in_features);
    uint8_t *h_lane_masks = (uint8_t *)malloc(mask_total);
    if (!h_scales || !h_quant || !h_lane_masks) {
        free(h_scales);
        free(h_quant);
        free(h_lane_masks);
        return make_error(env, "host_alloc_failed");
    }

    if (strict_mask) {
        memcpy(h_lane_masks, strict_mask->data, mask_total);
    } else {
        for (int o = 0; o < out_features; ++o) {
            uint8_t *mask_row = h_lane_masks + (size_t)o * mask_groups_per_row;
            for (int k = 0; k < in_features; k += 8) {
                float group[8];
                for (int j = 0; j < 8; ++j)
                    group[j] = W[(size_t)(k + j) * (size_t)out_features + (size_t)o];
                int keep_pair0, keep_pair1;
                top2_pairs_of_8(group, &keep_pair0, &keep_pair1);
                mask_row[k / 8] =
                    (uint8_t)((0x3u << (keep_pair0 * 2)) | (0x3u << (keep_pair1 * 2)));
            }
        }
    }

    /* Per-output-channel absmax → scale (INT4 qmax = 7). The strict path
     * only considers lanes selected by the authoritative mask; a discarded
     * outlier must not collapse the usable INT4 range. */
    for (int o = 0; o < out_features; ++o) {
        float amax = 0.0f;
        const uint8_t *mask_row = h_lane_masks + (size_t)o * mask_groups_per_row;
        for (int k = 0; k < in_features; ++k) {
            if (strict_mask && !(mask_row[k / 8] & (1u << (k % 8))))
                continue;
            float a = fabsf(W[(size_t)k * (size_t)out_features + (size_t)o]);
            if (a > amax)
                amax = a;
        }
        h_scales[o] = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    }

    /* Quantize exactly the selected lanes. h_quant holds one signed int4
     * value per int8 cell until the compressed packing step. */
    for (int o = 0; o < out_features; ++o) {
        int8_t *qrow = h_quant + (size_t)o * (size_t)in_features;
        const uint8_t *mask_row = h_lane_masks + (size_t)o * mask_groups_per_row;
        float inv_scale = 1.0f / h_scales[o];
        for (int k = 0; k < in_features; ++k) {
            if (mask_row[k / 8] & (1u << (k % 8))) {
                float value = W[(size_t)k * (size_t)out_features + (size_t)o];
                qrow[k] = f32_to_int4_sat(value, inv_scale);
            } else {
                qrow[k] = 0;
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
        free(h_scales);
        free(h_quant);
        free(h_lane_masks);
        return make_error(env, "unexpected_int4_elemente_size");
    }
    size_t meta_units_per_row = (size_t)in_features / (size_t)(int4_sparse * int4_elems_per_e);
    size_t meta_bytes_per_row = meta_units_per_row * (size_t)int4_sizeof_e;
    size_t meta_total = (size_t)out_features * meta_bytes_per_row;
    uint8_t *h_meta = (uint8_t *)malloc(meta_total);
    if (!h_meta) {
        free(h_scales);
        free(h_quant);
        free(h_lane_masks);
        return make_error(env, "host_alloc_failed");
    }
    memset(h_meta, 0, meta_total);

    /* Derive metadata from the authoritative lane masks. Inferring it from
     * quantized zeros loses the distinction between a discarded lane and a
     * selected weight that legitimately quantizes to zero. */
    for (int o = 0; o < out_features; ++o) {
        const uint8_t *mask_row = h_lane_masks + (size_t)o * mask_groups_per_row;
        uint32_t *meta_row = (uint32_t *)(h_meta + (size_t)o * meta_bytes_per_row);
        for (int k = 0; k < in_features; k += 8) {
            int keep_pair0, keep_pair1;
            (void)decode_pair48_lane_mask(mask_row[k / 8], &keep_pair0, &keep_pair1);

            size_t group = (size_t)k / 8;
            size_t word = group / 8;
            size_t nibble = group % 8;
            uint32_t e = (uint32_t)(keep_pair0 | (keep_pair1 << 2));
            meta_row[word] |= e << (nibble * 4);
        }
    }

    size_t packed_bytes_per_row = (size_t)in_features / 4; /* 50% sparse, int4 packed */
    size_t packed_total = (size_t)out_features * packed_bytes_per_row;
    int8_t *h_packed = (int8_t *)malloc(packed_total);
    if (!h_packed) {
        free(h_scales);
        free(h_quant);
        free(h_lane_masks);
        free(h_meta);
        return make_error(env, "host_alloc_failed");
    }
    memset(h_packed, 0, packed_total);

    /* Pack compressed A in the same pair order named by ElementE. */
    for (int o = 0; o < out_features; ++o) {
        int8_t *qrow = h_quant + (size_t)o * (size_t)in_features;
        int8_t *prow = h_packed + (size_t)o * packed_bytes_per_row;

        size_t out_byte_idx = 0;
        for (int k = 0; k < in_features; k += 8) {
            uint32_t *meta_row = (uint32_t *)(h_meta + (size_t)o * meta_bytes_per_row);
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
    PackedWeight *pw = alloc_packed_weight();
    if (!pw) {
        free(h_scales);
        free(h_quant);
        free(h_lane_masks);
        free(h_packed);
        free(h_meta);
        return make_error(env, "resource_alloc_failed");
    }
    pw->dtype = PW_INT4_SPARSE;
    pw->in_features = in_features;
    pw->out_features = out_features;
    pw->scales_count = (size_t)out_features;

    cudaError_t cerr;
    cerr = cudaMalloc(&pw->d_weight, packed_total);
    if (cerr != cudaSuccess)
        goto cuda_fail4;
    pw->weight_bytes = packed_total;
    cerr = cudaMemcpy(pw->d_weight, h_packed, packed_total, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
        goto cuda_fail4;

    cerr = cudaMalloc(&pw->d_metadata, meta_total);
    if (cerr != cudaSuccess)
        goto cuda_fail4;
    pw->metadata_bytes = meta_total;

    /* Sm80 GemmSparseUniversal expects ElementE in an interleaved (warp-lane)
     * layout. Reorder our logical row-major metadata before upload. */
    {
        uint32_t *h_meta_reordered = (uint32_t *)malloc(meta_total);
        if (!h_meta_reordered)
            goto cuda_fail4;
        memset(h_meta_reordered, 0, meta_total);
        int meta_words_per_row = (int)(meta_bytes_per_row / sizeof(uint32_t));
        cutlass_int4_sparse_reorder_meta_e(h_meta_reordered, (const uint32_t *)h_meta, out_features,
                                           meta_words_per_row);
        cerr = cudaMemcpy(pw->d_metadata, h_meta_reordered, meta_total, cudaMemcpyHostToDevice);
        free(h_meta_reordered);
        if (cerr != cudaSuccess)
            goto cuda_fail4;
    }

    cerr = cudaMalloc(&pw->d_scales, (size_t)out_features * sizeof(float));
    if (cerr != cudaSuccess)
        goto cuda_fail4;
    cerr = cudaMemcpy(pw->d_scales, h_scales, (size_t)out_features * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
        goto cuda_fail4;

    /* No cuSPARSELt plan for INT4 (cuSPARSELt does not support int4 on Ada). */
    pw->cusparselt_plan = NULL;
    pw->d_compressed = NULL;

    free(h_scales);
    free(h_quant);
    free(h_lane_masks);
    free(h_packed);
    free(h_meta);

    ERL_NIF_TERM term = make_packed_weight_term(env, pw);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

cuda_fail4:
    enif_release_resource(pw);
    free(h_scales);
    free(h_quant);
    free(h_lane_masks);
    free(h_packed);
    free(h_meta);
    return make_error(env, "cuda_alloc_failed");
}

ERL_NIF_TERM nt_prepack_int4_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    return prepack_int4_sparse_impl(env, argv[0], argv[1], NULL);
}

ERL_NIF_TERM nt_prepack_int4_sparse_pair_4_8(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary mask_bin;
    if (!enif_inspect_binary(env, argv[2], &mask_bin))
        return make_error(env, "pair48_mask_not_binary");
    return prepack_int4_sparse_impl(env, argv[0], argv[1], &mask_bin);
}

#else /* _WIN32 — CUDA is Linux-only in this codebase. */

ERL_NIF_TERM nt_prepack_int8_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_error(env, "cuda_unavailable_on_windows");
}

ERL_NIF_TERM nt_prepack_int4_sparse(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_error(env, "cuda_unavailable_on_windows");
}

ERL_NIF_TERM nt_prepack_int4_sparse_pair_4_8(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;
    return make_error(env, "cuda_unavailable_on_windows");
}

#endif /* _WIN32 */
