/**
 * nif_linear_fp8.c — FP8 linear forward against a prepacked weight.
 *
 * Two entry points:
 *   nt_linear_fp8(InputBinary, PackedWeight, BiasOrNil, Epilogue)
 *       Plain GEMM (CUTLASS f16acc, 660 TOPS) when bias is nil and
 *       Epilogue == 1 (DEFAULT). Otherwise falls through to cublasLt
 *       with the requested epilogue (`4`=BIAS, `6`=BIAS+RELU,
 *       `36`=BIAS+GELU).
 *   nt_linear_gelu_fp8(InputBinary, PackedWeight, BiasOrNil, _Epilogue)
 *       Always epilogue=36 (BIAS+GELU) when bias present, else 32 (GELU).
 *       Forces the cublasLt path.
 *
 * Input contract:
 *   InputBinary = FP16 row-major bytes, shape [B, in_features].
 *   PackedWeight.d_weight = FP8 col-major, shape [in_features, out_features].
 *   Output = FP16 row-major bytes, shape [B, out_features].
 *
 * Why FP16 input vs FP8 input? The cuBLASLt FP8 matmul on Ada accepts
 * E4M3 for both operands, but the Gleam-side API only commits FP16
 * input data today (cheap activations stay FP16; only the static
 * weights get quantized). We quantize the input to FP8 on-the-fly when
 * we route through the FP8 GEMM (CUTLASS or cublasLt). The activation
 * quantization is a per-tensor absmax over each forward call — this
 * matches what TransformerEngine / vllm do.
 *
 * NOTE: For an MVP we keep the activation-side scale derivation cheap
 * (one device pass via thrust would be cleaner but pulls in a heavy
 * dependency). The current implementation uses a CPU absmax on the
 * decoded host bytes before upload. For typical batch sizes
 * (B*in_features <= 1M) this is well below 1 ms on a modern desktop
 * CPU. For larger batches a CUDA absmax kernel should replace this.
 */

#include "viva_nif.h"
#include "nif_packed_weight.h"

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
#include <cuda_runtime.h>
/* NOTE: we deliberately do NOT include <cuda_fp16.h>. It requires nvcc-only
 * intrinsics that the Zig clang front-end refuses. Other NIF files in this
 * tree treat FP16 device pointers as raw `uint16_t*` and delegate the
 * actual half math to the `.cu` files where nvcc handles `uint16_t`. We
 * follow the same pattern below. */
#include <cublasLt.h>
#endif

/* Mirror of nif_prepack_fp8.c — must match. */
/* FP8 E4M3 max finite value: 1.75 × 2^8 = 448. Previously this was set to
 * 128, which silently dropped 8× of the dynamic range and made FP8-quantized
 * outputs ~50% smaller than HF reference. Confirmed via bisect against
 * transformers in dev/hf_bisect.py. */
#define FP8_E4M3_MAX 448.0f

/* FP8 E4M3 quantization (host-side). Local duplicate of the prepack
 * routine — kept here to avoid an extra header file. If the math is
 * ever revised it MUST stay in sync with nif_prepack_fp8.c. */
static inline uint8_t lin_float_to_fp8_e4m3(float val) {
  if (val == 0.0f) return 0x00;
  if (val != val) return 0x7F;

  uint32_t bits;
  memcpy(&bits, &val, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t f32_exp = (int32_t)((bits >> 23) & 0xFF) - 127;
  uint32_t f32_mant = bits & 0x7FFFFF;

  if (f32_exp >= 8) return (uint8_t)((sign << 7) | 0x7E);
  if (f32_exp < -9) return (uint8_t)(sign << 7);

  int32_t e_exp;
  uint32_t e_mant;
  if (f32_exp >= -6) {
    e_exp = f32_exp + 7;
    uint32_t round_bit = (f32_mant >> 19) & 0x1;
    uint32_t sticky = (f32_mant & 0x7FFFF) != 0;
    e_mant = (f32_mant >> 20) & 0x7;
    if (round_bit && (sticky || (e_mant & 0x1))) {
      e_mant += 1;
      if (e_mant == 8) {
        e_mant = 0;
        e_exp += 1;
      }
    }
    if (e_exp >= 15) return (uint8_t)((sign << 7) | 0x7E);
  } else {
    int32_t shift = -6 - f32_exp;
    uint32_t mant_with_implicit = f32_mant | 0x800000;
    e_exp = 0;
    uint32_t total_shift = 20 + shift;
    uint32_t round_bit = (mant_with_implicit >> (total_shift - 1)) & 0x1;
    uint32_t sticky =
        (mant_with_implicit & ((1u << (total_shift - 1)) - 1)) != 0;
    e_mant = (mant_with_implicit >> total_shift) & 0x7;
    if (round_bit && (sticky || (e_mant & 0x1))) {
      e_mant += 1;
      if (e_mant == 8) {
        e_mant = 0;
        e_exp = 1;
      }
    }
  }
  return (uint8_t)((sign << 7) | ((uint32_t)e_exp << 3) | e_mant);
}

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)

/* Cached cublasLt handle + workspace shared across calls. The first
 * NIF call lazily creates the context. We never destroy it explicitly
 * — the BEAM exits and CUDA tears the process down anyway. */
static cublasLtHandle_t g_lt_ctx = NULL;
static void *g_lt_workspace = NULL;
static const size_t g_lt_workspace_size = 32 * 1024 * 1024;

static int ensure_lt_ctx(void) {
  if (g_lt_ctx) return 0;
  if (cublasLtCreate(&g_lt_ctx) != CUBLAS_STATUS_SUCCESS) return -1;
  if (cudaMalloc(&g_lt_workspace, g_lt_workspace_size) != cudaSuccess) {
    cublasLtDestroy(g_lt_ctx);
    g_lt_ctx = NULL;
    return -2;
  }
  return 0;
}

/* =========================================================================
 * Input upload helper.
 *
 * Decodes the input binary as FP16 row-major [batch, in_features], runs
 * an absmax pass on host to derive the activation scale, quantizes to
 * FP8 E4M3, and uploads to a freshly allocated device buffer.
 *
 * Out parameters:
 *   d_input_fp8     [batch * in_features] bytes, row-major
 *   d_input_fp16    [batch * in_features] half, row-major (kept around in
 *                   case the caller prefers an FP16 input path)
 *   act_scale       per-tensor dequant scale (output FP32 = q * act_scale)
 *
 * Caller is responsible for cudaFree on both device buffers (or NULL on
 * failure).
 * ========================================================================= */
static int upload_input_fp8(const ErlNifBinary *bin, int batch,
                             int in_features, uint8_t **out_d_input,
                             float **out_row_scales) {
  *out_d_input = NULL;
  *out_row_scales = NULL;

  size_t n = (size_t)batch * (size_t)in_features;
  if (bin->size != n * sizeof(uint16_t)) return -1;

  const uint16_t *src_half = (const uint16_t *)bin->data;

  /* Per-row activation quantization (ggml-style block_q8_0 with block ==
   * row): each input batch row gets its own absmax + scale. Outliers in
   * one row don't compress the dynamic range of the others — important
   * for attention scores where one sequence position can dominate. */
  float *row_scales = (float *)malloc((size_t)batch * sizeof(float));
  if (!row_scales) return -2;

  for (int b = 0; b < batch; ++b) {
    float absmax = 0.0f;
    for (int k = 0; k < in_features; ++k) {
      float a = fabsf(f16_to_f32(src_half[(size_t)b * in_features + k]));
      if (a > absmax) absmax = a;
    }
    row_scales[b] = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
  }

  uint8_t *h_packed = (uint8_t *)malloc(n);
  if (!h_packed) { free(row_scales); return -2; }
  for (int b = 0; b < batch; ++b) {
    float inv = 1.0f / row_scales[b];
    for (int k = 0; k < in_features; ++k) {
      float v = f16_to_f32(src_half[(size_t)b * in_features + k]) * inv;
      h_packed[(size_t)b * in_features + k] = lin_float_to_fp8_e4m3(v);
    }
  }

  uint8_t *d_input = NULL;
  if (cudaMalloc((void **)&d_input, n) != cudaSuccess) {
    free(h_packed); free(row_scales);
    return -3;
  }
  if (cudaMemcpy(d_input, h_packed, n, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    free(h_packed); free(row_scales);
    cudaFree(d_input);
    return -4;
  }
  free(h_packed);
  *out_d_input = d_input;
  *out_row_scales = row_scales;
  return 0;
}

/* Pull the per-tensor weight scale (1 FP32) back from device memory.
 * Cheap (4 bytes, sync copy) — only runs once per linear call. */
/* Read the (avg of) weight scales into a single float — used by the
 * cublasLt fused path which takes a single alpha. The CUTLASS path
 * uses `read_weight_scales_per_channel` below to do per-channel dequant
 * on the host. */
static int read_weight_scale(const PackedWeight *w, float *out_scale) {
  if (!w->d_scales) return -1;
  size_t count = w->scales_count > 0 ? w->scales_count : 1;
  float *tmp = (float *)malloc(count * sizeof(float));
  if (!tmp) return -2;
  if (cudaMemcpy(tmp, w->d_scales, count * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    free(tmp);
    return -3;
  }
  float avg = 0.0f;
  for (size_t i = 0; i < count; ++i) avg += tmp[i];
  *out_scale = (count > 0) ? (avg / (float)count) : 1.0f;
  free(tmp);
  return 0;
}

/* Allocate + populate a host array with the per-output-channel scales. */
static int read_weight_scales_per_channel(const PackedWeight *w,
                                            float **out_arr,
                                            size_t *out_count) {
  if (!w->d_scales || w->scales_count == 0) return -1;
  float *arr = (float *)malloc(w->scales_count * sizeof(float));
  if (!arr) return -2;
  if (cudaMemcpy(arr, w->d_scales, w->scales_count * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    free(arr);
    return -3;
  }
  *out_arr = arr;
  *out_count = w->scales_count;
  return 0;
}


/* =========================================================================
 * Path A: CUTLASS FP8 GEMM (no bias, no activation, 660 TOPS on Ada).
 *
 * We hand off to the same `cutlass_fp8_gemm_f16acc` symbol that the
 * benchmark NIFs already use, against device-resident `d_A` (input,
 * FP8 row-major), `d_B` (prepacked weight, FP8 col-major), and a fresh
 * `d_C` (FP16 row-major output we allocate here).
 *
 * Scaling: CUTLASS multiplies q_a * q_b in FP16 accumulators. We
 * compensate with `output_fp32 = c_fp16 * (act_scale * weight_scale)`
 * applied during the FP16 -> output binary conversion step.
 * ========================================================================= */
static int run_cutlass_path(const PackedWeight *w,
                             const uint8_t *d_input,
                             int batch,
                             float **out_d_C) {
  /* FP32 output buffer eliminates the FP16 cast saturation that
   * previously capped end-to-end precision at L2 ~13% on K=4096.
   * Caller applies per-row × per-channel dequant on the FP32 values
   * before the final FP16 binary encode on the host. */
  float *d_C = NULL;
  size_t bytes_C = (size_t)batch * (size_t)w->out_features * sizeof(float);
  if (cudaMalloc((void **)&d_C, bytes_C) != cudaSuccess) return -1;
  if (cudaMemset(d_C, 0, bytes_C) != cudaSuccess) {
    cudaFree(d_C);
    return -2;
  }
  int rc = cutlass_fp8_gemm_f32acc_out_f32(
      batch, w->out_features, w->in_features,
      (const void *)d_input, (const void *)w->d_weight, d_C);
  if (rc != 0) {
    cudaFree(d_C);
    return -100 + rc;
  }
  *out_d_C = d_C;
  return 0;
}

/* =========================================================================
 * Path B: cublasLt FP8 with epilogue fusion.
 *
 * Used when the caller asks for bias and/or activation. cublasLt on
 * GeForce is capped at ~330 TOPS for FP8 (FP32 accumulator half-rate),
 * but the saved HBM bandwidth from fused bias/GELU often wins back
 * the difference, especially for skinny matmuls.
 *
 * Epilogue codes (cublasLtEpilogue_t):
 *    1 = DEFAULT
 *    4 = BIAS
 *    6 = BIAS + RELU
 *   32 = GELU
 *   36 = BIAS + GELU
 *
 * Layout: CUTLASS prepacks B as col-major, which corresponds exactly
 * to cublasLt's TN configuration (op_A=N, op_B=T over col-major
 * descriptors). To match the descriptors used by `cublaslt_fp8_algo_sweep`
 * we keep the same TN swap (m_lt=N, n_lt=M, k_lt=K).
 * ========================================================================= */
static int run_cublaslt_path(const PackedWeight *w,
                              const uint8_t *d_input,
                              int batch,
                              const uint16_t *d_bias /* may be NULL */,
                              int epilogue,
                              float **out_d_C) {
  if (ensure_lt_ctx() != 0) return -1;

  /* FP32 output buffer eliminates the FP16 cast saturation that the
   * previous FP16-output path suffered at large K — same fix already
   * applied to the CUTLASS f32acc_out_f32 path. Per-row × per-channel
   * dequant runs on FP32 host afterwards. */
  float *d_C = NULL;
  size_t bytes_C = (size_t)batch * (size_t)w->out_features * sizeof(float);
  if (cudaMalloc((void **)&d_C, bytes_C) != cudaSuccess) return -2;
  if (cudaMemset(d_C, 0, bytes_C) != cudaSuccess) {
    cudaFree(d_C);
    return -3;
  }

  cublasLtMatmulDesc_t desc;
  if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_C);
    return -4;
  }
  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t,
                                  sizeof(op_t));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n,
                                  sizeof(op_n));
  cublasLtEpilogue_t ep = (cublasLtEpilogue_t)epilogue;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep,
                                  sizeof(ep));
  if (d_bias && (epilogue == CUBLASLT_EPILOGUE_BIAS ||
                  epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
                  epilogue == CUBLASLT_EPILOGUE_GELU_BIAS)) {
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                    &d_bias, sizeof(d_bias));
  }

  cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
  cublasLtMatrixLayoutCreate(&layout_bt, CUDA_R_8F_E4M3,
                              (uint64_t)w->in_features,
                              (uint64_t)w->out_features,
                              (int64_t)w->in_features);
  cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_8F_E4M3,
                              (uint64_t)w->in_features,
                              (uint64_t)batch,
                              (int64_t)w->in_features);
  cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32F,
                              (uint64_t)w->out_features,
                              (uint64_t)batch,
                              (int64_t)w->out_features);

  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref,
                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &g_lt_workspace_size,
                                        sizeof(g_lt_workspace_size));

  cublasLtMatmulHeuristicResult_t heur;
  int returned = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
      g_lt_ctx, desc, layout_bt, layout_a, layout_c, layout_c, pref, 1, &heur,
      &returned);
  if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_bt);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(d_C);
    return -5;
  }

  float alpha = 1.0f, beta = 0.0f;
  st = cublasLtMatmul(g_lt_ctx, desc, &alpha,
                       w->d_weight, layout_bt,
                       d_input,     layout_a,
                       &beta,
                       d_C, layout_c, d_C, layout_c,
                       &heur.algo, g_lt_workspace, g_lt_workspace_size,
                       (cudaStream_t)0);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(layout_bt);
  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatrixLayoutDestroy(layout_c);
  cublasLtMatmulDescDestroy(desc);

  if (st != CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_C);
    return -1000 - (int)st;
  }
  *out_d_C = d_C;
  return 0;
}

/* Download an FP32 device buffer (output of the CUTLASS f32acc_out_f32
 * GEMM), apply per-row × per-channel dequant on FP32, cast to FP16, and
 * emit a FP16 binary. The FP32 accumulator skips the cast saturation
 * that the older FP16-output path suffered at large K. */
static int download_fp32_and_make_binary(ErlNifEnv *env, float *d_C, int batch,
                                          int out_features,
                                          const float *act_scales,
                                          const float *weight_scales,
                                          ERL_NIF_TERM *out_term) {
  size_t n = (size_t)batch * (size_t)out_features;
  size_t in_bytes = n * sizeof(float);
  size_t out_bytes = n * sizeof(uint16_t);
  float *h_C = (float *)malloc(in_bytes);
  if (!h_C) return -1;
  if (cudaMemcpy(h_C, d_C, in_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    free(h_C);
    return -2;
  }

  ErlNifBinary outbin;
  if (!enif_alloc_binary(out_bytes, &outbin)) {
    free(h_C);
    return -3;
  }
  uint16_t *dst = (uint16_t *)outbin.data;

  for (int b = 0; b < batch; ++b) {
    float a = act_scales ? act_scales[b] : 1.0f;
    for (int c = 0; c < out_features; ++c) {
      size_t i = (size_t)b * (size_t)out_features + (size_t)c;
      float wsc = weight_scales ? weight_scales[c] : 1.0f;
      dst[i] = float_to_half(h_C[i] * a * wsc);
    }
  }
  free(h_C);

  *out_term = enif_make_binary(env, &outbin);
  return 0;
}

/* Download the FP16 device buffer (legacy / cublasLt epilogue path),
 * apply per-row × per-channel dequant, and emit a new FP16 binary.
 * `act_scales` may be NULL (single scalar fallback in
 * `legacy_combined_scale`). `weight_scales` may be NULL when only the
 * legacy path is needed. */
static int download_and_make_binary(ErlNifEnv *env, uint16_t *d_C, int batch,
                                     int out_features,
                                     const float *act_scales,
                                     const float *weight_scales,
                                     float legacy_combined_scale,
                                     ERL_NIF_TERM *out_term) {
  size_t n = (size_t)batch * (size_t)out_features;
  size_t bytes = n * sizeof(uint16_t);
  uint16_t *h_C = (uint16_t *)malloc(bytes);
  if (!h_C) return -1;
  if (cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    free(h_C);
    return -2;
  }

  ErlNifBinary outbin;
  if (!enif_alloc_binary(bytes, &outbin)) {
    free(h_C);
    return -3;
  }
  uint16_t *dst = (uint16_t *)outbin.data;

  if (weight_scales != NULL && act_scales != NULL) {
    /* Full per-row × per-channel dequant. */
    for (int b = 0; b < batch; ++b) {
      float a = act_scales[b];
      for (int c = 0; c < out_features; ++c) {
        size_t i = (size_t)b * (size_t)out_features + (size_t)c;
        float v = f16_to_f32(h_C[i]) * a * weight_scales[c];
        dst[i] = float_to_half(v);
      }
    }
  } else {
    /* Legacy single combined scale path. */
    for (size_t i = 0; i < n; ++i) {
      float v = f16_to_f32(h_C[i]) * legacy_combined_scale;
      dst[i] = float_to_half(v);
    }
  }
  free(h_C);

  *out_term = enif_make_binary(env, &outbin);
  return 0;
}

#endif /* !_WIN32 && !VIVA_NO_CUDA */

/* =========================================================================
 * NIF entry: nt_linear_fp8(InputBin, PackedWeight, BiasOrNil, Epilogue)
 * ========================================================================= */
ERL_NIF_TERM nt_linear_fp8(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  return make_error(env, "cuda_not_available");
#else
  ErlNifBinary input_bin;
  if (!enif_inspect_binary(env, argv[0], &input_bin))
    return make_error(env, "invalid_input_binary");

  PackedWeight *w = get_packed_weight(env, argv[1]);
  if (!w) return make_error(env, "invalid_packed_weight");
  if (w->dtype != PW_FP8) return make_error(env, "weight_not_fp8");

  /* Derive batch from binary size: batch = bytes / (in_features * 2). */
  size_t expected_per_row = (size_t)w->in_features * sizeof(uint16_t);
  if (expected_per_row == 0 || input_bin.size % expected_per_row != 0)
    return make_error(env, "input_size_mismatch");
  int batch = (int)(input_bin.size / expected_per_row);
  if (batch <= 0) return make_error(env, "input_batch_zero");

  /* Bias: nil | binary (FP16). For now only binary is consumed. */
  ErlNifBinary bias_bin;
  int has_bias = enif_inspect_binary(env, argv[2], &bias_bin);
  if (has_bias && bias_bin.size != (size_t)w->out_features * sizeof(uint16_t))
    return make_error(env, "bias_size_mismatch");

  int epilogue = 1;
  if (!enif_get_int(env, argv[3], &epilogue))
    return make_error(env, "invalid_epilogue");

  /* Quantize + upload input. */
  uint8_t *d_input = NULL;
  float *act_scales = NULL;
  int rc = upload_input_fp8(&input_bin, batch, w->in_features, &d_input,
                            &act_scales);
  if (rc != 0) return make_error(env, "input_upload_failed");

  /* Upload bias if present. */
  uint16_t *d_bias = NULL;
  if (has_bias) {
    if (cudaMalloc((void **)&d_bias, bias_bin.size) != cudaSuccess) {
      cudaFree(d_input);
      return make_error(env, "cuda_malloc_bias_failed");
    }
    if (cudaMemcpy(d_bias, bias_bin.data, bias_bin.size,
                    cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(d_bias);
      cudaFree(d_input);
      return make_error(env, "cuda_upload_bias_failed");
    }
  }

  /* Both CUTLASS and cublasLt now produce FP32 output buffers — the
   * dequant + FP16 cast happens uniformly on host afterwards. */
  float *d_C_fp32 = NULL;
  int use_cublaslt = has_bias || (epilogue != 1);
  if (use_cublaslt) {
    rc = run_cublaslt_path(w, d_input, batch, d_bias, epilogue, &d_C_fp32);
  } else {
    rc = run_cutlass_path(w, d_input, batch, &d_C_fp32);
  }
  cudaFree(d_input);
  if (rc != 0) {
    if (d_bias) cudaFree(d_bias);
    if (d_C_fp32) cudaFree(d_C_fp32);
    char err[64];
    snprintf(err, sizeof(err), "gemm_failed_%d", rc);
    return make_error(env, err);
  }

  /* Per-output-channel dequant. */
  float *w_scales = NULL;
  size_t w_scales_count = 0;
  if (read_weight_scales_per_channel(w, &w_scales, &w_scales_count) != 0
      || w_scales_count != (size_t)w->out_features) {
    if (d_bias) cudaFree(d_bias);
    if (d_C_fp32) cudaFree(d_C_fp32);
    if (w_scales) free(w_scales);
    return make_error(env, "weight_scale_read_failed");
  }


  ERL_NIF_TERM out_term;
  rc = download_fp32_and_make_binary(env, d_C_fp32, batch, w->out_features,
                                      act_scales, w_scales, &out_term);
  free(act_scales);
  free(w_scales);
  if (d_bias) cudaFree(d_bias);
  if (d_C_fp32) cudaFree(d_C_fp32);
  if (rc != 0) return make_error(env, "output_download_failed");

  return make_ok(env, out_term);
#endif
}

/* =========================================================================
 * NIF entry: nt_linear_gelu_fp8(InputBin, PackedWeight, BiasOrNil, _Ep)
 *
 * Always routes through cublasLt with BIAS+GELU (36) or GELU (32). The
 * 4th argument is ignored, kept for arity parity with `nt_linear_fp8`.
 * ========================================================================= */
ERL_NIF_TERM nt_linear_gelu_fp8(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  return make_error(env, "cuda_not_available");
#else
  ErlNifBinary input_bin;
  if (!enif_inspect_binary(env, argv[0], &input_bin))
    return make_error(env, "invalid_input_binary");

  PackedWeight *w = get_packed_weight(env, argv[1]);
  if (!w) return make_error(env, "invalid_packed_weight");
  if (w->dtype != PW_FP8) return make_error(env, "weight_not_fp8");

  size_t expected_per_row = (size_t)w->in_features * sizeof(uint16_t);
  if (expected_per_row == 0 || input_bin.size % expected_per_row != 0)
    return make_error(env, "input_size_mismatch");
  int batch = (int)(input_bin.size / expected_per_row);
  if (batch <= 0) return make_error(env, "input_batch_zero");

  ErlNifBinary bias_bin;
  int has_bias = enif_inspect_binary(env, argv[2], &bias_bin);
  if (has_bias && bias_bin.size != (size_t)w->out_features * sizeof(uint16_t))
    return make_error(env, "bias_size_mismatch");

  uint8_t *d_input = NULL;
  float *act_scales = NULL;
  int rc = upload_input_fp8(&input_bin, batch, w->in_features, &d_input,
                            &act_scales);
  if (rc != 0) return make_error(env, "input_upload_failed");

  uint16_t *d_bias = NULL;
  if (has_bias) {
    if (cudaMalloc((void **)&d_bias, bias_bin.size) != cudaSuccess) {
      cudaFree(d_input);
      return make_error(env, "cuda_malloc_bias_failed");
    }
    if (cudaMemcpy(d_bias, bias_bin.data, bias_bin.size,
                    cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(d_bias);
      cudaFree(d_input);
      return make_error(env, "cuda_upload_bias_failed");
    }
  }

  int epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS
                          : CUBLASLT_EPILOGUE_GELU;
  float *d_C = NULL;
  rc = run_cublaslt_path(w, d_input, batch, d_bias, epilogue, &d_C);
  cudaFree(d_input);
  if (rc != 0) {
    if (d_bias) cudaFree(d_bias);
    if (d_C) cudaFree(d_C);
    char err[64];
    snprintf(err, sizeof(err), "gelu_gemm_failed_%d", rc);
    return make_error(env, err);
  }

  /* FP32 output buffer; per-row × per-channel dequant happens on host. */
  float *w_scales2 = NULL;
  size_t w_scales2_count = 0;
  if (read_weight_scales_per_channel(w, &w_scales2, &w_scales2_count) != 0
      || w_scales2_count != (size_t)w->out_features) {
    if (d_bias) cudaFree(d_bias);
    cudaFree(d_C);
    if (w_scales2) free(w_scales2);
    return make_error(env, "weight_scale_read_failed");
  }

  ERL_NIF_TERM out_term;
  rc = download_fp32_and_make_binary(env, d_C, batch, w->out_features,
                                      act_scales, w_scales2, &out_term);
  free(act_scales);
  free(w_scales2);
  if (d_bias) cudaFree(d_bias);
  cudaFree(d_C);
  if (rc != 0) return make_error(env, "output_download_failed");

  return make_ok(env, out_term);
#endif
}
