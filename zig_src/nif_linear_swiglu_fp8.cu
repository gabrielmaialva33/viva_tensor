/**
 * nif_linear_swiglu_fp8.cu - Fused FP8 SwiGLU NIF
 *
 * Implements `nt_linear_swiglu_fp8/4`:
 *
 *   output = silu(input @ gate_weight) * (input @ up_weight)   [+ bias]
 *           where silu(x) = x * sigmoid(x)
 *
 * Pipeline (single forward pass):
 *   1. Parse input list-of-doubles into a host FP32 [B, in_features] buffer.
 *   2. Quantize input to FP8 E4M3 using the same per-tensor absmax/448 scheme
 *      agent A used for prepack_fp8_weight (kept in sync via `prepack_fp8_quantize_input`).
 *   3. Allocate device buffers: gate_out, up_out, final out (FP32 [B, out_features]).
 *   4. Call `cutlass_fp8_gemm_f32acc_out_f32` twice — once with the prepacked
 *      gate weight, once with the prepacked up weight. Both produce FP32 outputs.
 *   5. Launch the `silu_mul_kernel` fused element-wise kernel: it reads
 *      gate_out + up_out, applies dequant scales before SiLU, computes in FP32,
 *      optionally adds bias, and writes FP32 to `out`.
 *   6. Download FP32 output to host, convert to list-of-doubles, return.
 *
 * Why fused: avoids two extra HBM round-trips (one for silu(gate), one for
 * silu(gate)*up). The fused kernel is ~3% of total runtime, so a saved trip
 * is a real speedup.
 *
 * Contract with agent A (nif_packed_weight.c):
 *   The C struct that backs the `PackedWeight*` resource MUST expose:
 *     void* packed_weight_device_ptr(PackedWeight*);
 *     int   packed_weight_in_features(PackedWeight*);
 *     int   packed_weight_out_features(PackedWeight*);
 *     float packed_weight_scale(PackedWeight*);            // FP8 dequant scale
 *     ErlNifResourceType* packed_weight_resource_type();
 *   And the FP8 input quantization helper:
 *     void quantize_fp8_e4m3(uint8_t* d_out, const float* h_in,
 *                            size_t n, float* out_scale);
 *
 * BSD-3-Clause (parts derived from CUTLASS examples)
 */

#include "viva_nif.h"
#include "nif_packed_weight.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

/* Proper FP8 E4M3 host quantizer with round-to-nearest-even mantissa and
 * full subnormal support (down to 2^-9). Mirrors `lin_float_to_fp8_e4m3`
 * in nif_linear_fp8.c verbatim, so the dense linear and the fused SwiGLU
 * paths produce bit-identical FP8 input bytes for the same activation.
 *
 * DO NOT call `float_to_fp8_e4m3_batch` from cuda_sage.c here. That
 * helper (a) truncates the mantissa instead of rounding, biasing
 * |q| downward, and (b) flushes everything with |x| < 2^-6 to zero,
 * skipping the entire E4M3 subnormal range 2^-9..2^-7. Combined with
 * the FP8_E4M3_MAX=448 fix, the truncation+subnormal-flush produced a
 * lopsided distribution on the gate/up FFN GEMMs whose nonlinearity
 * via silu(g)*u landed at 1.35x of the HF reference (bisected
 * against transformers — see dev/hf_bisect.py). */
static inline uint8_t swiglu_float_to_fp8_e4m3(float val) {
  if (val == 0.0f) return 0x00;
  if (val != val) return 0x7F;

  uint32_t bits;
  memcpy(&bits, &val, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t f32_exp = (int32_t)((bits >> 23) & 0xFF) - 127;
  uint32_t f32_mant = bits & 0x7FFFFF;

  /* Hard overflow: f32_exp >= 9 means |x| >= 512, definitely beyond
   * E4M3 max finite 448 → saturate.
   * Underflow: f32_exp < -9 means |x| < 2^-9, below the smallest E4M3
   * subnormal → flush to zero. */
  if (f32_exp >= 9) return (uint8_t)((sign << 7) | 0x7E);
  if (f32_exp < -9) return (uint8_t)(sign << 7);

  int32_t e_exp;
  uint32_t e_mant;
  if (f32_exp >= -6) {
    /* Normal range covers fp8_exp 1..15 (i.e. f32_exp -6..8). At
     * f32_exp=8 the mantissa selects 256, 288, ..., 448. Round to
     * nearest even, then saturate to 0x7E (mantissa=6 → 448) if the
     * rounded mantissa exceeds 6 — i.e. true |x| in [464, 512). */
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
    if (e_exp >= 15 && e_mant > 6) {
      return (uint8_t)((sign << 7) | 0x7E);
    }
    if (e_exp > 15) return (uint8_t)((sign << 7) | 0x7E);
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

/* Inline per-tensor E4M3 host quantization (same as in nif_linear_fp8.c).
 * input_scale = absmax / 448, then qx = swiglu_float_to_fp8_e4m3(x / input_scale).
 * Returns 0 on success. */
static int swiglu_quantize_fp8_e4m3_host(uint8_t* d_out,
                                          const float* h_in,
                                          size_t n,
                                          float* out_scale) {
  float amax = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float a = fabsf(h_in[i]);
    if (a > amax) amax = a;
  }
  float s = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
  *out_scale = s;

  uint8_t* h_q = (uint8_t*)malloc(n);
  if (!h_q) return -1;
  float inv = 1.0f / s;
  for (size_t i = 0; i < n; ++i) {
    h_q[i] = swiglu_float_to_fp8_e4m3(h_in[i] * inv);
  }

  cudaError_t cerr = cudaMemcpy(d_out, h_q, n, cudaMemcpyHostToDevice);
  free(h_q);
  return (cerr == cudaSuccess) ? 0 : -2;
}

/* ============================================================================
 * Fused silu+mul kernel
 *
 * out[i] = silu(gate[i]) * up[i] + (bias_present ? bias[i % out_features] : 0)
 *
 * Computes silu in FP32 for numerical stability, writes FP32.
 * One thread per output element. The block size is a safe 256 — well within
 * the 1024 max for SM89 and gives us ~12 SM tiles for a 4096-wide row.
 * ============================================================================ */

__device__ __forceinline__ float silu_f32(float x) {
  /* silu(x) = x * sigmoid(x). Compute via exp for stability with large |x|. */
  /* sigmoid(x) = 1 / (1 + exp(-x))  — branchless variant works for all x. */
  return x / (1.0f + __expf(-x));
}

__global__ void silu_mul_kernel(const float*  __restrict__ gate,
                                const float*  __restrict__ up,
                                const __half* __restrict__ bias,  /* may be NULL */
                                const float*  __restrict__ gate_scales,
                                const float*  __restrict__ up_scales,
                                float input_scale,
                                float*        __restrict__ out,
                                int total,
                                int out_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;

  int col = i % out_features;
  float g = gate[i] * input_scale * gate_scales[col];
  float u = up[i] * input_scale * up_scales[col];
  float y = silu_f32(g) * u;
  if (bias) {
    y += __half2float(bias[col]);
  }
  out[i] = y;
}

static int launch_silu_mul(const float* d_gate, const float* d_up,
                            const __half* d_bias,
                            const float* d_gate_scales,
                            const float* d_up_scales,
                            float input_scale,
                            float* d_out,
                            int batch, int out_features) {
  int total = batch * out_features;
  int threads = 256;
  int blocks  = (total + threads - 1) / threads;
  silu_mul_kernel<<<blocks, threads>>>(d_gate, d_up, d_bias,
                                       d_gate_scales, d_up_scales,
                                       input_scale, d_out,
                                       total, out_features);
  cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? 0 : -1;
}

/* ============================================================================
 * NIF entry point: nt_linear_swiglu_fp8/4
 *
 * argv:
 *   [0] input_data     :: list(float)      -- FP32 host data, length B*in_features
 *   [1] input_shape    :: list(int)        -- typically [B, in_features]
 *   [2] gate_weight    :: PackedWeight ref
 *   [3] up_weight      :: PackedWeight ref
 *   [4] bias_or_nil    :: list(float) | nil -- length out_features
 *
 * NOTE: argc is 5. The Erlang stub is registered with arity 5 in nif_entry.c.
 * ============================================================================ */

extern "C" ERL_NIF_TERM
nt_linear_swiglu_fp8_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) return enif_make_badarg(env);

  /* --- Parse input shape --- */
  int shape[8];
  int ndim = 0;
  int shape_ok = parse_shape(env, argv[1], shape, &ndim);
  if (!shape_ok || ndim < 2) {
    return make_error(env, "bad_input_shape");
  }
  int in_features = shape[ndim - 1];
  int batch = 1;
  for (int i = 0; i < ndim - 1; ++i) batch *= shape[i];
  if (batch <= 0 || in_features <= 0) {
    return make_error(env, "bad_input_dims");
  }

  /* --- Parse input data (FP32 host) --- */
  unsigned data_len;
  double* input_doubles = list_to_doubles(env, argv[0], &data_len);
  if (!input_doubles || (int)data_len != batch * in_features) {
    if (input_doubles) free(input_doubles);
    return make_error(env, "input_data_shape_mismatch");
  }

  float* h_input_fp32 = (float*)malloc(sizeof(float) * data_len);
  if (!h_input_fp32) { free(input_doubles); return make_error(env, "oom_host"); }
  for (unsigned i = 0; i < data_len; ++i) h_input_fp32[i] = (float)input_doubles[i];
  free(input_doubles);

  /* --- Fetch packed weights --- */
  PackedWeight* gate_w = get_packed_weight(env, argv[2]);
  PackedWeight* up_w   = get_packed_weight(env, argv[3]);
  if (!gate_w || !up_w) {
    free(h_input_fp32);
    return make_error(env, "bad_packed_weight");
  }

  if (gate_w->dtype != PW_FP8 || up_w->dtype != PW_FP8) {
    free(h_input_fp32);
    return make_error(env, "weight_not_fp8");
  }

  if (gate_w->in_features != in_features || up_w->in_features != in_features) {
    free(h_input_fp32);
    return make_error(env, "in_features_mismatch");
  }
  if (gate_w->out_features != up_w->out_features) {
    free(h_input_fp32);
    return make_error(env, "out_features_mismatch");
  }
  int out_features = gate_w->out_features;
  if (!gate_w->d_scales || !up_w->d_scales ||
      gate_w->scales_count != (size_t)out_features ||
      up_w->scales_count != (size_t)out_features) {
    free(h_input_fp32);
    return make_error(env, "weight_scale_read_failed");
  }

  void* d_gate_weight = gate_w->d_weight;
  void* d_up_weight   = up_w->d_weight;
  if (!d_gate_weight || !d_up_weight) {
    free(h_input_fp32);
    return make_error(env, "packed_weight_device_ptr_null");
  }

  /* --- Quantize input to FP8 E4M3 + upload --- */
  size_t input_elems = (size_t)batch * in_features;
  uint8_t* d_input_fp8 = NULL;
  if (cudaMalloc(&d_input_fp8, input_elems) != cudaSuccess) {
    free(h_input_fp32);
    return make_error(env, "cuda_malloc_input");
  }
  float input_scale = 1.0f;
  if (swiglu_quantize_fp8_e4m3_host(d_input_fp8, h_input_fp32, input_elems, &input_scale) != 0) {
    cudaFree(d_input_fp8);
    free(h_input_fp32);
    return make_error(env, "input_quantize_failed");
  }
  free(h_input_fp32);

  /* --- Allocate device output buffers --- */
  size_t fp32_out_bytes = (size_t)batch * out_features * sizeof(float);
  float* d_gate_out  = NULL;
  float* d_up_out    = NULL;
  float* d_out       = NULL;
  if (cudaMalloc(&d_gate_out, fp32_out_bytes) != cudaSuccess) {
    cudaFree(d_input_fp8);
    return make_error(env, "cuda_malloc_gate_out");
  }
  if (cudaMalloc(&d_up_out, fp32_out_bytes) != cudaSuccess) {
    cudaFree(d_gate_out); cudaFree(d_input_fp8);
    return make_error(env, "cuda_malloc_up_out");
  }
  if (cudaMalloc(&d_out, fp32_out_bytes) != cudaSuccess) {
    cudaFree(d_up_out); cudaFree(d_gate_out); cudaFree(d_input_fp8);
    return make_error(env, "cuda_malloc_out");
  }

  /* --- Two FP8 GEMMs: gate and up --- */
  int rc1 = cutlass_fp8_gemm_f32acc_out_f32(batch, out_features, in_features,
                                             d_input_fp8, d_gate_weight, d_gate_out);
  if (rc1 != 0) {
    cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out); cudaFree(d_input_fp8);
    return make_error(env, "cutlass_gate_gemm_failed");
  }
  int rc2 = cutlass_fp8_gemm_f32acc_out_f32(batch, out_features, in_features,
                                             d_input_fp8, d_up_weight, d_up_out);
  if (rc2 != 0) {
    cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out); cudaFree(d_input_fp8);
    return make_error(env, "cutlass_up_gemm_failed");
  }
  cudaFree(d_input_fp8);

  /* --- Optional bias upload (FP16 [out_features]) --- */
  __half* d_bias = NULL;
  ERL_NIF_TERM bias_arg = argv[4];
  int has_bias = !enif_is_identical(bias_arg, enif_make_atom(env, "bias_nil")) &&
                 !enif_is_identical(bias_arg, enif_make_atom(env, "nil"));
  if (has_bias) {
    int bias_tuple_arity = 0;
    const ERL_NIF_TERM* bias_tuple = NULL;
    if (enif_get_tuple(env, bias_arg, &bias_tuple_arity, &bias_tuple)) {
      if (bias_tuple_arity != 2 ||
          !enif_is_identical(bias_tuple[0], enif_make_atom(env, "bias_list"))) {
        cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out);
        return make_error(env, "invalid_bias_arg");
      }
      bias_arg = bias_tuple[1];
    }
    unsigned bias_len;
    double* bias_doubles = list_to_doubles(env, bias_arg, &bias_len);
    if (!bias_doubles || (int)bias_len != out_features) {
      if (bias_doubles) free(bias_doubles);
      cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out);
      return make_error(env, "bias_shape_mismatch");
    }
    __half* h_bias = (__half*)malloc(sizeof(__half) * out_features);
    if (!h_bias) {
      free(bias_doubles);
      cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out);
      return make_error(env, "oom_bias_host");
    }
    for (int i = 0; i < out_features; ++i) {
      h_bias[i] = __float2half_rn((float)bias_doubles[i]);
    }
    free(bias_doubles);
    if (cudaMalloc(&d_bias, sizeof(__half) * out_features) != cudaSuccess) {
      free(h_bias);
      cudaFree(d_out); cudaFree(d_up_out); cudaFree(d_gate_out);
      return make_error(env, "cuda_malloc_bias");
    }
    cudaMemcpy(d_bias, h_bias, sizeof(__half) * out_features, cudaMemcpyHostToDevice);
    free(h_bias);
  }

  /* --- DEBUG: dump gate_out / up_out mean_abs post-dequant --- */
  if (getenv("VIVA_SWIGLU_DEBUG")) {
    size_t n_out = (size_t)batch * out_features;
    float* h_g = (float*)malloc(sizeof(float) * n_out);
    float* h_u = (float*)malloc(sizeof(float) * n_out);
    float* h_gs = (float*)malloc(sizeof(float) * out_features);
    float* h_us = (float*)malloc(sizeof(float) * out_features);
    cudaMemcpy(h_g, d_gate_out, sizeof(float) * n_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u, d_up_out, sizeof(float) * n_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gs, gate_w->d_scales, sizeof(float) * out_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_us, up_w->d_scales, sizeof(float) * out_features, cudaMemcpyDeviceToHost);
    double sum_g_raw = 0.0, sum_u_raw = 0.0;
    double sum_g_deq = 0.0, sum_u_deq = 0.0;
    double sum_sg = 0.0, sum_su = 0.0;
    for (size_t i = 0; i < n_out; ++i) {
      int col = i % out_features;
      sum_g_raw += fabs(h_g[i]);
      sum_u_raw += fabs(h_u[i]);
      double dg = h_g[i] * input_scale * h_gs[col];
      double du = h_u[i] * input_scale * h_us[col];
      sum_g_deq += fabs(dg);
      sum_u_deq += fabs(du);
      sum_sg += fabs(h_gs[col]);
      sum_su += fabs(h_us[col]);
    }
    double sum_g_w = 0.0, sum_u_w = 0.0;
    for (int c = 0; c < out_features; ++c) {
      sum_g_w += fabs(h_gs[c]);
      sum_u_w += fabs(h_us[c]);
    }
    fprintf(stderr, "[swiglu] in_scale=%.6e gate_raw_mean=%.6e up_raw_mean=%.6e gate_deq_mean=%.6e up_deq_mean=%.6e mean_gscale=%.6e mean_uscale=%.6e\n",
            (double)input_scale,
            sum_g_raw / n_out, sum_u_raw / n_out,
            sum_g_deq / n_out, sum_u_deq / n_out,
            sum_g_w / out_features, sum_u_w / out_features);
    fprintf(stderr, "[swiglu] first5 gate_deq: ");
    for (int k = 0; k < 5; ++k) {
      double dg = h_g[k] * input_scale * h_gs[k];
      fprintf(stderr, "%.6f ", dg);
    }
    fprintf(stderr, "\n[swiglu] first5 up_deq:   ");
    for (int k = 0; k < 5; ++k) {
      double du = h_u[k] * input_scale * h_us[k];
      fprintf(stderr, "%.6f ", du);
    }
    fprintf(stderr, "\n[swiglu] first5 gscale: ");
    for (int k = 0; k < 5; ++k) fprintf(stderr, "%.6e ", h_gs[k]);
    fprintf(stderr, "\n[swiglu] first5 gate_raw_fp32: ");
    for (int k = 0; k < 5; ++k) fprintf(stderr, "%.4f ", h_g[k]);
    fprintf(stderr, "\n");
    free(h_g); free(h_u); free(h_gs); free(h_us);
  }

  /* --- Fused silu+mul (+optional bias) --- */
  int kc = launch_silu_mul(d_gate_out, d_up_out, d_bias,
                           (const float*)gate_w->d_scales,
                           (const float*)up_w->d_scales,
                           input_scale,
                           d_out, batch, out_features);
  cudaFree(d_gate_out);
  cudaFree(d_up_out);
  if (d_bias) cudaFree(d_bias);
  if (kc != 0) {
    cudaFree(d_out);
    return make_error(env, "silu_mul_kernel_failed");
  }

  /* --- Download output --- */
  size_t total = (size_t)batch * out_features;
  float* h_out = (float*)malloc(sizeof(float) * total);
  if (!h_out) {
    cudaFree(d_out);
    return make_error(env, "oom_output_host");
  }
  if (cudaMemcpy(h_out, d_out, sizeof(float) * total, cudaMemcpyDeviceToHost) != cudaSuccess) {
    free(h_out);
    cudaFree(d_out);
    return make_error(env, "cuda_memcpy_d2h_fp32");
  }
  cudaFree(d_out);

  double* h_out_doubles = (double*)malloc(sizeof(double) * total);
  if (!h_out_doubles) {
    free(h_out);
    return make_error(env, "oom_output_doubles");
  }
  for (size_t i = 0; i < total; ++i) {
    double v = (double)h_out[i];
    if (!isfinite(v)) v = (v < 0.0) ? -DBL_MAX : DBL_MAX;
    h_out_doubles[i] = v;
  }
  free(h_out);

  ERL_NIF_TERM out_list = doubles_to_list(env, h_out_doubles, (unsigned)total);
  free(h_out_doubles);

  return make_ok(env, out_list);
}
