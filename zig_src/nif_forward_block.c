#include "viva_nif.h"
#include "nif_packed_weight.h"

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
#include <cuda_runtime.h>
#include <cublasLt.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)

#define LLAMA_HEAD_DIM 64
#define LLAMA_EPS 1.0e-5f

extern int vt_fp16_to_fp32_cast(const void *in, float *out, int n);
extern int vt_fp32_to_fp16_cast(const float *in, void *out, int n);
extern int vt_rmsnorm_fp32(const float *x, const float *gamma, float *out, int n, float eps);
extern int vt_residual_add_fp32(const float *a, const float *b, float *out, int n);
extern int vt_rope_apply_fp32(float *x, const float *freqs, int pos, int num_heads, int head_dim);
extern int vt_silu_mul_fp32(const float *gate, const float *up, float *out, int n);
extern int vt_gqa_attn_single_token(const float *q, const float *new_k, const float *new_v,
                                    const void *k_cache, const void *v_cache, float *out,
                                    int past_len, int num_heads, int num_kv_heads,
                                    int head_dim);

typedef struct {
  void *ptr;
  size_t cap;
} BlockBuf;

static cublasLtHandle_t g_block_lt = NULL;
static void *g_block_workspace = NULL;
static const size_t g_block_workspace_size = 32 * 1024 * 1024;

static BlockBuf b_hidden16 = {0}, b_hidden32 = {0}, b_norm16 = {0};
static BlockBuf b_norm1 = {0}, b_norm2 = {0}, b_rope = {0};
static BlockBuf b_weight16 = {0}, b_q = {0}, b_k = {0}, b_v = {0};
static BlockBuf b_attn = {0}, b_attn16 = {0}, b_o = {0}, b_h1 = {0};
static BlockBuf b_x2 = {0}, b_x2_16 = {0}, b_gate = {0}, b_up = {0};
static BlockBuf b_sw = {0}, b_sw16 = {0}, b_down = {0}, b_hout16 = {0};
static BlockBuf b_k_cache = {0}, b_v_cache = {0}, b_k_append = {0}, b_v_append = {0};

static int ensure_block_buf(BlockBuf *buf, size_t needed) {
  if (buf->cap >= needed) return 0;
  size_t new_cap = needed + (needed >> 1) + 4096;
  void *fresh = NULL;
  if (cudaMalloc(&fresh, new_cap) != cudaSuccess) return -1;
  if (buf->ptr) cudaFree(buf->ptr);
  buf->ptr = fresh;
  buf->cap = new_cap;
  return 0;
}

static int ensure_block_lt(void) {
  if (g_block_lt) return 0;
  if (cublasLtCreate(&g_block_lt) != CUBLAS_STATUS_SUCCESS) return -1;
  if (cudaMalloc(&g_block_workspace, g_block_workspace_size) != cudaSuccess) {
    cublasLtDestroy(g_block_lt);
    g_block_lt = NULL;
    return -2;
  }
  return 0;
}

static int upload_binary(BlockBuf *buf, const ErlNifBinary *bin) {
  if (ensure_block_buf(buf, bin->size) != 0) return -1;
  if (bin->size == 0) return 0;
  return cudaMemcpy(buf->ptr, bin->data, bin->size, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -2;
}

static int dequant_weight_fp16(const PackedWeight *w, uint16_t **out_weight) {
  size_t bytes = (size_t)w->in_features * (size_t)w->out_features * sizeof(uint16_t);
  if (ensure_block_buf(&b_weight16, bytes) != 0) return -1;
  uint16_t *d_weight = (uint16_t *)b_weight16.ptr;
  int rc;
  if (w->block_size > 0) {
    rc = cuda_fp8_colmajor_dequant_to_fp16_blocked(w->d_weight, (const float *)w->d_scales,
                                                   d_weight, w->in_features, w->out_features,
                                                   w->block_size);
  } else {
    rc = cuda_fp8_colmajor_dequant_to_fp16(w->d_weight, (const float *)w->d_scales,
                                           d_weight, w->in_features, w->out_features);
  }
  if (rc != 0) return -20 + rc;
  *out_weight = d_weight;
  return 0;
}

static int gemm_w8a16_dequant(const PackedWeight *w, const uint16_t *d_input,
                              int batch, float *d_out) {
  if (ensure_block_lt() != 0) return -1;

  uint16_t *d_weight = NULL;
  int rc = dequant_weight_fp16(w, &d_weight);
  if (rc != 0) return rc;

  size_t bytes_C = (size_t)batch * (size_t)w->out_features * sizeof(float);
  if (cudaMemset(d_out, 0, bytes_C) != cudaSuccess) return -2;

  cublasLtMatmulDesc_t desc;
  if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
    return -3;

  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

  cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
  cublasStatus_t st = cublasLtMatrixLayoutCreate(&layout_bt, CUDA_R_16F,
      (uint64_t)w->in_features, (uint64_t)w->out_features, (int64_t)w->in_features);
  if (st != CUBLAS_STATUS_SUCCESS) { cublasLtMatmulDescDestroy(desc); return -4; }
  st = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F,
      (uint64_t)w->in_features, (uint64_t)batch, (int64_t)w->in_features);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layout_bt); cublasLtMatmulDescDestroy(desc); return -5;
  }
  st = cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32F,
      (uint64_t)w->out_features, (uint64_t)batch, (int64_t)w->out_features);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatrixLayoutDestroy(layout_bt); cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(desc); return -6;
  }

  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &g_block_workspace_size, sizeof(g_block_workspace_size));

  cublasLtMatmulHeuristicResult_t heur;
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(g_block_lt, desc, layout_bt, layout_a, layout_c,
                                      layout_c, pref, 1, &heur, &returned);
  if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_bt); cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_c); cublasLtMatmulDescDestroy(desc);
    return -7;
  }

  float alpha = 1.0f, beta = 0.0f;
  st = cublasLtMatmul(g_block_lt, desc, &alpha,
                      d_weight, layout_bt,
                      d_input, layout_a,
                      &beta,
                      d_out, layout_c, d_out, layout_c,
                      &heur.algo, g_block_workspace, g_block_workspace_size,
                      (cudaStream_t)0);

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(layout_bt);
  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatrixLayoutDestroy(layout_c);
  cublasLtMatmulDescDestroy(desc);

  return st == CUBLAS_STATUS_SUCCESS ? 0 : (-1000 - (int)st);
}

static int validate_fp8_weight(const PackedWeight *w) {
  return w && w->dtype == PW_FP8 && w->d_weight && w->d_scales &&
         w->in_features > 0 && w->out_features > 0;
}

static ERL_NIF_TERM make_block_error(ErlNifEnv *env, const char *prefix, int rc) {
  char msg[96];
  snprintf(msg, sizeof(msg), "%s_%d", prefix, rc);
  return make_error(env, msg);
}

#endif

ERL_NIF_TERM nt_forward_block_w8a16(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  (void)argc;
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  return make_error(env, "cuda_not_available");
#else
  if (argc != 14) return make_error(env, "bad_arity");

  ErlNifBinary hidden_bin, norm1_bin, norm2_bin, rope_bin, k_cache_bin, v_cache_bin;
  if (!enif_inspect_binary(env, argv[0], &hidden_bin)) return make_error(env, "invalid_hidden");
  if (!enif_inspect_binary(env, argv[8], &norm1_bin)) return make_error(env, "invalid_norm1");
  if (!enif_inspect_binary(env, argv[9], &norm2_bin)) return make_error(env, "invalid_norm2");
  if (!enif_inspect_binary(env, argv[11], &rope_bin)) return make_error(env, "invalid_rope");
  if (!enif_inspect_binary(env, argv[12], &k_cache_bin)) return make_error(env, "invalid_k_cache");
  if (!enif_inspect_binary(env, argv[13], &v_cache_bin)) return make_error(env, "invalid_v_cache");

  int pos = 0;
  if (!enif_get_int(env, argv[10], &pos) || pos < 0) return make_error(env, "invalid_pos");

  PackedWeight *q = get_packed_weight(env, argv[1]);
  PackedWeight *k = get_packed_weight(env, argv[2]);
  PackedWeight *v = get_packed_weight(env, argv[3]);
  PackedWeight *o = get_packed_weight(env, argv[4]);
  PackedWeight *gate = get_packed_weight(env, argv[5]);
  PackedWeight *up = get_packed_weight(env, argv[6]);
  PackedWeight *down = get_packed_weight(env, argv[7]);
  if (!validate_fp8_weight(q) || !validate_fp8_weight(k) || !validate_fp8_weight(v) ||
      !validate_fp8_weight(o) || !validate_fp8_weight(gate) || !validate_fp8_weight(up) ||
      !validate_fp8_weight(down)) {
    return make_error(env, "invalid_packed_weight");
  }

  int hidden = q->in_features;
  int kv_dim = k->out_features;
  int ffn = gate->out_features;
  int num_heads = hidden / LLAMA_HEAD_DIM;
  int num_kv_heads = kv_dim / LLAMA_HEAD_DIM;
  if (hidden <= 0 || kv_dim <= 0 || ffn <= 0 ||
      hidden % LLAMA_HEAD_DIM != 0 || kv_dim % LLAMA_HEAD_DIM != 0 ||
      q->out_features != hidden || k->in_features != hidden || v->in_features != hidden ||
      v->out_features != kv_dim || o->in_features != hidden || o->out_features != hidden ||
      gate->in_features != hidden || up->in_features != hidden || up->out_features != ffn ||
      down->in_features != ffn || down->out_features != hidden) {
    return make_error(env, "shape_mismatch");
  }

  size_t hidden16_bytes = (size_t)hidden * sizeof(uint16_t);
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  size_t kv16_bytes = (size_t)kv_dim * sizeof(uint16_t);
  size_t kv32_bytes = (size_t)kv_dim * sizeof(float);
  size_t ffn16_bytes = (size_t)ffn * sizeof(uint16_t);
  size_t ffn32_bytes = (size_t)ffn * sizeof(float);
  if (hidden_bin.size != hidden16_bytes || norm1_bin.size != hidden32_bytes ||
      norm2_bin.size != hidden32_bytes || rope_bin.size != (size_t)(LLAMA_HEAD_DIM / 2) * sizeof(float)) {
    return make_error(env, "input_size_mismatch");
  }
  if (k_cache_bin.size != v_cache_bin.size || (kv16_bytes > 0 && k_cache_bin.size % kv16_bytes != 0)) {
    return make_error(env, "cache_size_mismatch");
  }
  int past_len = (int)(k_cache_bin.size / kv16_bytes);

  int rc = 0;
  if ((rc = upload_binary(&b_hidden16, &hidden_bin)) != 0) return make_block_error(env, "upload_hidden", rc);
  if ((rc = upload_binary(&b_norm1, &norm1_bin)) != 0) return make_block_error(env, "upload_norm1", rc);
  if ((rc = upload_binary(&b_norm2, &norm2_bin)) != 0) return make_block_error(env, "upload_norm2", rc);
  if ((rc = upload_binary(&b_rope, &rope_bin)) != 0) return make_block_error(env, "upload_rope", rc);
  if ((rc = upload_binary(&b_k_cache, &k_cache_bin)) != 0) return make_block_error(env, "upload_k_cache", rc);
  if ((rc = upload_binary(&b_v_cache, &v_cache_bin)) != 0) return make_block_error(env, "upload_v_cache", rc);

  BlockBuf *bufs[] = {&b_hidden32, &b_norm16, &b_q, &b_o, &b_h1, &b_x2, &b_x2_16,
                      &b_attn, &b_attn16, &b_down, &b_hout16};
  for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
    if (ensure_block_buf(bufs[i], hidden32_bytes) != 0) return make_error(env, "cuda_malloc_hidden_failed");
  }
  if (ensure_block_buf(&b_k, kv32_bytes) != 0 ||
      ensure_block_buf(&b_v, kv32_bytes) != 0 ||
      ensure_block_buf(&b_k_append, kv16_bytes) != 0 ||
      ensure_block_buf(&b_v_append, kv16_bytes) != 0) {
    return make_error(env, "cuda_malloc_kv_failed");
  }
  if (ensure_block_buf(&b_gate, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_up, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw16, ffn16_bytes) != 0) {
    return make_error(env, "cuda_malloc_ffn_failed");
  }

  if ((rc = vt_fp16_to_fp32_cast(b_hidden16.ptr, (float *)b_hidden32.ptr, hidden)) != 0)
    return make_block_error(env, "cast_hidden", rc);
  if ((rc = vt_rmsnorm_fp32((float *)b_hidden32.ptr, (float *)b_norm1.ptr,
                            (float *)b_x2.ptr, hidden, LLAMA_EPS)) != 0)
    return make_block_error(env, "rmsnorm1", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_norm16.ptr, hidden)) != 0)
    return make_block_error(env, "cast_norm1", rc);

  if ((rc = gemm_w8a16_dequant(q, (uint16_t *)b_norm16.ptr, 1, (float *)b_q.ptr)) != 0)
    return make_block_error(env, "gemm_q", rc);
  if ((rc = gemm_w8a16_dequant(k, (uint16_t *)b_norm16.ptr, 1, (float *)b_k.ptr)) != 0)
    return make_block_error(env, "gemm_k", rc);
  if ((rc = gemm_w8a16_dequant(v, (uint16_t *)b_norm16.ptr, 1, (float *)b_v.ptr)) != 0)
    return make_block_error(env, "gemm_v", rc);

  if ((rc = vt_rope_apply_fp32((float *)b_q.ptr, (float *)b_rope.ptr, pos,
                               num_heads, LLAMA_HEAD_DIM)) != 0)
    return make_block_error(env, "rope_q", rc);
  if ((rc = vt_rope_apply_fp32((float *)b_k.ptr, (float *)b_rope.ptr, pos,
                               num_kv_heads, LLAMA_HEAD_DIM)) != 0)
    return make_block_error(env, "rope_k", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_k.ptr, b_k_append.ptr, kv_dim)) != 0)
    return make_block_error(env, "cast_k_append", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_v.ptr, b_v_append.ptr, kv_dim)) != 0)
    return make_block_error(env, "cast_v_append", rc);

  if ((rc = vt_gqa_attn_single_token((float *)b_q.ptr, (float *)b_k.ptr, (float *)b_v.ptr,
                                     b_k_cache.ptr, b_v_cache.ptr, (float *)b_attn.ptr,
                                     past_len, num_heads, num_kv_heads, LLAMA_HEAD_DIM)) != 0)
    return make_block_error(env, "attention", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_attn.ptr, b_attn16.ptr, hidden)) != 0)
    return make_block_error(env, "cast_attn", rc);
  if ((rc = gemm_w8a16_dequant(o, (uint16_t *)b_attn16.ptr, 1, (float *)b_o.ptr)) != 0)
    return make_block_error(env, "gemm_o", rc);
  if ((rc = vt_residual_add_fp32((float *)b_hidden32.ptr, (float *)b_o.ptr,
                                 (float *)b_h1.ptr, hidden)) != 0)
    return make_block_error(env, "residual1", rc);

  if ((rc = vt_rmsnorm_fp32((float *)b_h1.ptr, (float *)b_norm2.ptr,
                            (float *)b_x2.ptr, hidden, LLAMA_EPS)) != 0)
    return make_block_error(env, "rmsnorm2", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_x2_16.ptr, hidden)) != 0)
    return make_block_error(env, "cast_norm2", rc);
  if ((rc = gemm_w8a16_dequant(gate, (uint16_t *)b_x2_16.ptr, 1, (float *)b_gate.ptr)) != 0)
    return make_block_error(env, "gemm_gate", rc);
  if ((rc = gemm_w8a16_dequant(up, (uint16_t *)b_x2_16.ptr, 1, (float *)b_up.ptr)) != 0)
    return make_block_error(env, "gemm_up", rc);
  if ((rc = vt_silu_mul_fp32((float *)b_gate.ptr, (float *)b_up.ptr, (float *)b_sw.ptr, ffn)) != 0)
    return make_block_error(env, "silu_mul", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_sw.ptr, b_sw16.ptr, ffn)) != 0)
    return make_block_error(env, "cast_sw", rc);
  if ((rc = gemm_w8a16_dequant(down, (uint16_t *)b_sw16.ptr, 1, (float *)b_down.ptr)) != 0)
    return make_block_error(env, "gemm_down", rc);
  if ((rc = vt_residual_add_fp32((float *)b_h1.ptr, (float *)b_down.ptr,
                                 (float *)b_x2.ptr, hidden)) != 0)
    return make_block_error(env, "residual2", rc);
  if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_hout16.ptr, hidden)) != 0)
    return make_block_error(env, "cast_hout", rc);

  ErlNifBinary out_hidden, out_k, out_v;
  if (!enif_alloc_binary(hidden16_bytes, &out_hidden) ||
      !enif_alloc_binary(kv16_bytes, &out_k) ||
      !enif_alloc_binary(kv16_bytes, &out_v)) {
    return make_error(env, "alloc_output_failed");
  }
  if (cudaMemcpy(out_hidden.data, b_hout16.ptr, hidden16_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(out_k.data, b_k_append.ptr, kv16_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(out_v.data, b_v_append.ptr, kv16_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    enif_release_binary(&out_hidden);
    enif_release_binary(&out_k);
    enif_release_binary(&out_v);
    return make_error(env, "download_output_failed");
  }

  ERL_NIF_TERM tuple = enif_make_tuple3(env,
      enif_make_binary(env, &out_hidden),
      enif_make_binary(env, &out_k),
      enif_make_binary(env, &out_v));
  return make_ok(env, tuple);
#endif
}
