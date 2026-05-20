#include "viva_nif.h"
#include "nif_packed_weight.h"

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
#include <cuda_runtime.h>
#include <cublasLt.h>
#endif

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)

#define LLAMA_HEAD_DIM 64
#define LLAMA_EPS 1.0e-5f

extern int vt_fp16_to_fp32_cast(const void *in, float *out, int n);
extern int vt_fp32_to_fp16_cast(const float *in, void *out, int n);
extern int vt_rmsnorm_fp32(const float *x, const float *gamma, float *out, int n, float eps);
extern int vt_residual_add_fp32(const float *a, const float *b, float *out, int n);
extern int vt_rope_apply_fp32(float *x, const float *freqs, int pos, int num_heads, int head_dim);
extern int vt_rope_apply_fp32_dyn(float *x, const float *freqs, const int *pos_ptr,
                                  int num_heads, int head_dim);
extern int vt_silu_mul_fp32(const float *gate, const float *up, float *out, int n);
extern int vt_argmax_fp32_as_fp16(const float *x, int n, int *out_idx);
extern int vt_embedding_lookup_fp16(const void *table, const int *token_id, void *out,
                                    int hidden, int vocab);
extern int vt_w8a16_mmv_blocked_k16(const void *d_weight, const float *d_scales,
                                    const void *d_input, float *d_out,
                                    int in_features, int out_features);
extern int vt_gqa_attn_single_token(const float *q, const float *new_k, const float *new_v,
                                     const void *k_cache, const void *v_cache, float *out,
                                     int past_len, int num_heads, int num_kv_heads,
                                     int head_dim);
extern int vt_gqa_attn_single_token_dyn(const float *q, const float *new_k, const float *new_v,
                                         const void *k_cache, const void *v_cache, float *out,
                                         const int *past_len_ptr, int num_heads,
                                         int num_kv_heads, int head_dim);
extern int vt_kv_cache_append_fp16_dyn(void *k_cache, void *v_cache, const void *k_append,
                                       const void *v_append, const int *past_len_ptr,
                                       int kv_dim);
extern void vt_block_set_stream(void *stream);
extern void cuda_fp8_dequant_set_stream(void *stream);
extern uint16_t float_to_half(float f);

typedef struct {
  void *ptr;
  size_t cap;
} BlockBuf;

static cublasLtHandle_t g_block_lt = NULL;
static void *g_block_workspace = NULL;
static const size_t g_block_workspace_size = 32 * 1024 * 1024;
static cudaStream_t g_block_stream = NULL;

static BlockBuf b_hidden16 = {0}, b_hidden32 = {0}, b_norm16 = {0};
static BlockBuf b_norm1 = {0}, b_norm2 = {0}, b_rope = {0};
static BlockBuf b_weight16 = {0}, b_q = {0}, b_k = {0}, b_v = {0};
static BlockBuf b_attn = {0}, b_attn16 = {0}, b_o = {0}, b_h1 = {0};
static BlockBuf b_x2 = {0}, b_x2_16 = {0}, b_gate = {0}, b_up = {0};
static BlockBuf b_sw = {0}, b_sw16 = {0}, b_down = {0}, b_hout16 = {0};
static BlockBuf b_k_cache = {0}, b_v_cache = {0}, b_k_append = {0}, b_v_append = {0};
static BlockBuf b_logits = {0}, b_qkv = {0}, b_gate_up = {0}, b_argmax = {0};
static BlockBuf b_token_id = {0}, b_pos_id = {0}, b_past_len_id = {0};
static void *g_attn_k_cache_ptr = NULL;
static void *g_attn_v_cache_ptr = NULL;
static float *g_q_ptr = NULL, *g_k_ptr = NULL, *g_v_ptr = NULL;
static float *g_gate_ptr = NULL, *g_up_ptr = NULL;
static float *g_norm1_ptr = NULL, *g_norm2_ptr = NULL, *g_rope_ptr = NULL;
static int *g_dyn_pos_ptr = NULL, *g_dyn_past_len_ptr = NULL;
static int g_full_decode_capture = 0;

typedef struct {
  void *d_k;
  void *d_v;
  int max_seq;
  int kv_dim;
  int len;
} BlockKvCache;

static ErlNifResourceType *BLOCK_KV_CACHE_RES = NULL;

typedef enum {
  BLOCK_GRAPH_NORM1 = 1,
  BLOCK_GRAPH_ROPE_ATTN = 2,
  BLOCK_GRAPH_POST_ATTN = 3,
  BLOCK_GRAPH_FFN = 4,
  BLOCK_GRAPH_OUT = 5
} BlockGraphKind;

typedef struct {
  int kind;
  int hidden;
  int kv_dim;
  int ffn;
  int pos;
  int past_len;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  unsigned launches;
} BlockGraphEntry;

#define BLOCK_GRAPH_MAX 256
static BlockGraphEntry g_block_graphs[BLOCK_GRAPH_MAX];
static int g_block_graph_count = 0;
static int g_block_graph_disabled = 0;

typedef struct {
  int in_features;
  int out_features;
  int batch;
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t layout_bt;
  cublasLtMatrixLayout_t layout_a;
  cublasLtMatrixLayout_t layout_c;
  cublasLtMatmulHeuristicResult_t heur;
} BlockGemmPlan;

#define BLOCK_GEMM_PLAN_MAX 16
static BlockGemmPlan g_block_gemm_plans[BLOCK_GEMM_PLAN_MAX];
static int g_block_gemm_plan_count = 0;

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
  if (!g_block_lt) {
    if (cublasLtCreate(&g_block_lt) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cudaMalloc(&g_block_workspace, g_block_workspace_size) != cudaSuccess) {
      cublasLtDestroy(g_block_lt);
      g_block_lt = NULL;
      return -2;
    }
  }
  if (!g_block_stream) {
    if (cudaStreamCreateWithFlags(&g_block_stream, cudaStreamNonBlocking) != cudaSuccess) {
      return -3;
    }
    vt_block_set_stream((void *)g_block_stream);
    cuda_fp8_dequant_set_stream((void *)g_block_stream);
  }
  return 0;
}

static int upload_binary(BlockBuf *buf, const ErlNifBinary *bin) {
  if (ensure_block_buf(buf, bin->size) != 0) return -1;
  if (bin->size == 0) return 0;
  return cudaMemcpy(buf->ptr, bin->data, bin->size, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -2;
}

static int upload_binary_async(BlockBuf *buf, const ErlNifBinary *bin) {
  if (ensure_block_buf(buf, bin->size) != 0) return -1;
  if (bin->size == 0) return 0;
  return cudaMemcpyAsync(buf->ptr, bin->data, bin->size, cudaMemcpyHostToDevice,
                         g_block_stream) == cudaSuccess ? 0 : -2;
}

static int dequant_weight_fp16(const PackedWeight *w, uint16_t **out_weight) {
  size_t bytes = (size_t)w->in_features * (size_t)w->out_features * sizeof(uint16_t);
  PackedWeight *mw = (PackedWeight *)w;
  if (mw->d_fp16_cache && mw->fp16_cache_ready && mw->fp16_cache_bytes == bytes) {
    *out_weight = (uint16_t *)mw->d_fp16_cache;
    return 0;
  }
  if (mw->d_fp16_cache && mw->fp16_cache_bytes != bytes) {
    cudaFree(mw->d_fp16_cache);
    mw->d_fp16_cache = NULL;
    mw->fp16_cache_bytes = 0;
    mw->fp16_cache_ready = 0;
  }
  if (!mw->d_fp16_cache) {
    if (cudaMalloc(&mw->d_fp16_cache, bytes) != cudaSuccess) return -1;
    mw->fp16_cache_bytes = bytes;
    mw->fp16_cache_ready = 0;
  }
  uint16_t *d_weight = (uint16_t *)mw->d_fp16_cache;
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
  mw->fp16_cache_ready = 1;
  *out_weight = d_weight;
  return 0;
}

static int get_block_gemm_plan(const PackedWeight *w, int batch, BlockGemmPlan **out_plan) {
  for (int i = 0; i < g_block_gemm_plan_count; ++i) {
    BlockGemmPlan *p = &g_block_gemm_plans[i];
    if (p->in_features == w->in_features && p->out_features == w->out_features &&
        p->batch == batch) {
      *out_plan = p;
      return 0;
    }
  }
  if (g_block_gemm_plan_count >= BLOCK_GEMM_PLAN_MAX) return -1;

  BlockGemmPlan *p = &g_block_gemm_plans[g_block_gemm_plan_count];
  memset(p, 0, sizeof(*p));
  p->in_features = w->in_features;
  p->out_features = w->out_features;
  p->batch = batch;

  if (cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
    return -2;
  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
  cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

  cublasStatus_t st = cublasLtMatrixLayoutCreate(&p->layout_bt, CUDA_R_16F,
      (uint64_t)w->in_features, (uint64_t)w->out_features, (int64_t)w->in_features);
  if (st != CUBLAS_STATUS_SUCCESS) return -3;
  st = cublasLtMatrixLayoutCreate(&p->layout_a, CUDA_R_16F,
      (uint64_t)w->in_features, (uint64_t)batch, (int64_t)w->in_features);
  if (st != CUBLAS_STATUS_SUCCESS) return -4;
  st = cublasLtMatrixLayoutCreate(&p->layout_c, CUDA_R_32F,
      (uint64_t)w->out_features, (uint64_t)batch, (int64_t)w->out_features);
  if (st != CUBLAS_STATUS_SUCCESS) return -5;

  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &g_block_workspace_size, sizeof(g_block_workspace_size));
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(g_block_lt, p->desc, p->layout_bt, p->layout_a,
                                      p->layout_c, p->layout_c, pref, 1, &p->heur, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS || returned == 0) return -6;

  g_block_gemm_plan_count++;
  *out_plan = p;
  return 0;
}

static int gemm_w8a16_dequant(const PackedWeight *w, const uint16_t *d_input,
                              int batch, float *d_out) {
  if (ensure_block_lt() != 0) return -1;

  if (batch == 1 && w->block_size == 16) {
    if (w->scales_count != (size_t)w->out_features * (size_t)(w->in_features / 16)) {
      return -40;
    }
    return vt_w8a16_mmv_blocked_k16(w->d_weight, (const float *)w->d_scales,
                                    d_input, d_out, w->in_features,
                                    w->out_features);
  }

  uint16_t *d_weight = NULL;
  int rc = dequant_weight_fp16(w, &d_weight);
  if (rc != 0) return rc;

  size_t bytes_C = (size_t)batch * (size_t)w->out_features * sizeof(float);
  if (cudaMemsetAsync(d_out, 0, bytes_C, g_block_stream) != cudaSuccess) return -2;

  BlockGemmPlan *plan = NULL;
  int plan_rc = get_block_gemm_plan(w, batch, &plan);
  if (plan_rc != 0) return -7 + plan_rc;

  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st = cublasLtMatmul(g_block_lt, plan->desc, &alpha,
                      d_weight, plan->layout_bt,
                      d_input, plan->layout_a,
                      &beta,
                      d_out, plan->layout_c, d_out, plan->layout_c,
                      &plan->heur.algo, g_block_workspace, g_block_workspace_size,
                      g_block_stream);

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

static BlockGraphEntry *find_block_graph(int kind, int hidden, int kv_dim, int ffn,
                                         int pos, int past_len) {
  for (int i = 0; i < g_block_graph_count; ++i) {
    BlockGraphEntry *e = &g_block_graphs[i];
    if (e->kind == kind && e->hidden == hidden && e->kv_dim == kv_dim &&
        e->ffn == ffn && e->pos == pos && e->past_len == past_len) {
      return e;
    }
  }
  return NULL;
}

static BlockGraphEntry *alloc_block_graph(int kind, int hidden, int kv_dim, int ffn,
                                          int pos, int past_len) {
  if (g_block_graph_count >= BLOCK_GRAPH_MAX) return NULL;
  BlockGraphEntry *e = &g_block_graphs[g_block_graph_count++];
  memset(e, 0, sizeof(*e));
  e->kind = kind;
  e->hidden = hidden;
  e->kv_dim = kv_dim;
  e->ffn = ffn;
  e->pos = pos;
  e->past_len = past_len;
  return e;
}

static int run_helper_sequence(int kind, int hidden, int kv_dim, int ffn,
                               int pos, int past_len, int num_heads,
                               int num_kv_heads) {
  int rc;
  switch (kind) {
    case BLOCK_GRAPH_NORM1:
      if ((rc = vt_fp16_to_fp32_cast(b_hidden16.ptr, (float *)b_hidden32.ptr, hidden)) != 0) return -100 + rc;
      if ((rc = vt_rmsnorm_fp32((float *)b_hidden32.ptr,
                                g_norm1_ptr ? g_norm1_ptr : (float *)b_norm1.ptr,
                                 (float *)b_x2.ptr, hidden, LLAMA_EPS)) != 0) return -200 + rc;
      if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_norm16.ptr, hidden)) != 0) return -300 + rc;
      return 0;
    case BLOCK_GRAPH_ROPE_ATTN:
      if (g_dyn_pos_ptr) {
        if ((rc = vt_rope_apply_fp32_dyn(g_q_ptr ? g_q_ptr : (float *)b_q.ptr,
                                         g_rope_ptr ? g_rope_ptr : (float *)b_rope.ptr,
                                         g_dyn_pos_ptr, num_heads, LLAMA_HEAD_DIM)) != 0)
          return -400 + rc;
        if ((rc = vt_rope_apply_fp32_dyn(g_k_ptr ? g_k_ptr : (float *)b_k.ptr,
                                         g_rope_ptr ? g_rope_ptr : (float *)b_rope.ptr,
                                         g_dyn_pos_ptr, num_kv_heads, LLAMA_HEAD_DIM)) != 0)
          return -500 + rc;
      } else {
        if ((rc = vt_rope_apply_fp32(g_q_ptr ? g_q_ptr : (float *)b_q.ptr,
                                     g_rope_ptr ? g_rope_ptr : (float *)b_rope.ptr, pos,
                                     num_heads, LLAMA_HEAD_DIM)) != 0) return -400 + rc;
        if ((rc = vt_rope_apply_fp32(g_k_ptr ? g_k_ptr : (float *)b_k.ptr,
                                     g_rope_ptr ? g_rope_ptr : (float *)b_rope.ptr, pos,
                                     num_kv_heads, LLAMA_HEAD_DIM)) != 0) return -500 + rc;
      }
      if ((rc = vt_fp32_to_fp16_cast(g_k_ptr ? g_k_ptr : (float *)b_k.ptr, b_k_append.ptr, kv_dim)) != 0) return -600 + rc;
      if ((rc = vt_fp32_to_fp16_cast(g_v_ptr ? g_v_ptr : (float *)b_v.ptr, b_v_append.ptr, kv_dim)) != 0) return -700 + rc;
      if (g_dyn_past_len_ptr) {
        if ((rc = vt_gqa_attn_single_token_dyn(g_q_ptr ? g_q_ptr : (float *)b_q.ptr,
                                               g_k_ptr ? g_k_ptr : (float *)b_k.ptr,
                                               g_v_ptr ? g_v_ptr : (float *)b_v.ptr,
                                               g_attn_k_cache_ptr, g_attn_v_cache_ptr,
                                               (float *)b_attn.ptr, g_dyn_past_len_ptr,
                                               num_heads, num_kv_heads, LLAMA_HEAD_DIM)) != 0)
          return -800 + rc;
      } else if ((rc = vt_gqa_attn_single_token(g_q_ptr ? g_q_ptr : (float *)b_q.ptr,
                                                g_k_ptr ? g_k_ptr : (float *)b_k.ptr,
                                                g_v_ptr ? g_v_ptr : (float *)b_v.ptr,
                                                g_attn_k_cache_ptr, g_attn_v_cache_ptr,
                                                (float *)b_attn.ptr, past_len, num_heads,
                                                num_kv_heads, LLAMA_HEAD_DIM)) != 0) return -800 + rc;
      if ((rc = vt_fp32_to_fp16_cast((float *)b_attn.ptr, b_attn16.ptr, hidden)) != 0) return -900 + rc;
      return 0;
    case BLOCK_GRAPH_POST_ATTN:
      if ((rc = vt_residual_add_fp32((float *)b_hidden32.ptr, (float *)b_o.ptr,
                                      (float *)b_h1.ptr, hidden)) != 0) return -1000 + rc;
      if ((rc = vt_rmsnorm_fp32((float *)b_h1.ptr,
                                g_norm2_ptr ? g_norm2_ptr : (float *)b_norm2.ptr,
                                 (float *)b_x2.ptr, hidden, LLAMA_EPS)) != 0) return -1100 + rc;
      if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_x2_16.ptr, hidden)) != 0) return -1200 + rc;
      return 0;
    case BLOCK_GRAPH_FFN:
      if ((rc = vt_silu_mul_fp32(g_gate_ptr ? g_gate_ptr : (float *)b_gate.ptr,
                                 g_up_ptr ? g_up_ptr : (float *)b_up.ptr,
                                 (float *)b_sw.ptr, ffn)) != 0) return -1300 + rc;
      if ((rc = vt_fp32_to_fp16_cast((float *)b_sw.ptr, b_sw16.ptr, ffn)) != 0) return -1400 + rc;
      return 0;
    case BLOCK_GRAPH_OUT:
      if ((rc = vt_residual_add_fp32((float *)b_h1.ptr, (float *)b_down.ptr,
                                     (float *)b_x2.ptr, hidden)) != 0) return -1500 + rc;
      if ((rc = vt_fp32_to_fp16_cast((float *)b_x2.ptr, b_hout16.ptr, hidden)) != 0) return -1600 + rc;
      return 0;
    default:
      return -9999;
  }
}

static int run_helper_graph(int kind, int hidden, int kv_dim, int ffn,
                            int pos, int past_len, int num_heads,
                            int num_kv_heads) {
  if (g_block_graph_disabled || g_full_decode_capture) {
    return run_helper_sequence(kind, hidden, kv_dim, ffn, pos, past_len, num_heads, num_kv_heads);
  }

  BlockGraphEntry *e = find_block_graph(kind, hidden, kv_dim, ffn, pos, past_len);
  if (e && e->exec) {
    cudaError_t err = cudaGraphLaunch(e->exec, g_block_stream);
    if (err == cudaSuccess) {
      e->launches++;
      return 0;
    }
    return -2000 - (int)err;
  }

  e = alloc_block_graph(kind, hidden, kv_dim, ffn, pos, past_len);
  if (!e) {
    g_block_graph_disabled = 1;
    return run_helper_sequence(kind, hidden, kv_dim, ffn, pos, past_len, num_heads, num_kv_heads);
  }

  cudaError_t err = cudaStreamBeginCapture(g_block_stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    g_block_graph_disabled = 1;
    return run_helper_sequence(kind, hidden, kv_dim, ffn, pos, past_len, num_heads, num_kv_heads);
  }

  int rc = run_helper_sequence(kind, hidden, kv_dim, ffn, pos, past_len, num_heads, num_kv_heads);
  cudaGraph_t graph = NULL;
  err = cudaStreamEndCapture(g_block_stream, &graph);
  if (rc != 0 || err != cudaSuccess || !graph) {
    if (graph) cudaGraphDestroy(graph);
    g_block_graph_disabled = 1;
    return rc != 0 ? rc : (-2100 - (int)err);
  }

  cudaGraphExec_t exec = NULL;
  err = cudaGraphInstantiate(&exec, graph, 0);
  if (err != cudaSuccess || !exec) {
    cudaGraphDestroy(graph);
    g_block_graph_disabled = 1;
    return run_helper_sequence(kind, hidden, kv_dim, ffn, pos, past_len, num_heads, num_kv_heads);
  }

  e->graph = graph;
  e->exec = exec;
  e->launches = 1;
  err = cudaGraphLaunch(exec, g_block_stream);
  return err == cudaSuccess ? 0 : (-2200 - (int)err);
}

static float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1u;
  uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
  uint32_t frac = (uint32_t)h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign << 31;
    } else {
      exp = 1;
      while ((frac & 0x400u) == 0) {
        frac <<= 1;
        exp--;
      }
      frac &= 0x3ffu;
      bits = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
    }
  } else if (exp == 0x1fu) {
    bits = (sign << 31) | 0x7f800000u | (frac << 13);
  } else {
    bits = (sign << 31) | ((exp + 112u) << 23) | (frac << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

static int argmax_logits_like_erlang(const float *logits, int n) {
  int best_i = 0;
  float best = half_to_float(float_to_half(logits[0]));
  for (int i = 1; i < n; ++i) {
    float v = half_to_float(float_to_half(logits[i]));
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  return best_i;
}

static int map_get_binary(ErlNifEnv *env, ERL_NIF_TERM map, const char *key,
                          ErlNifBinary *out) {
  ERL_NIF_TERM val;
  if (!enif_get_map_value(env, map, enif_make_atom(env, key), &val)) return 0;
  return enif_inspect_binary(env, val, out);
}

static PackedWeight *map_get_weight(ErlNifEnv *env, ERL_NIF_TERM map, const char *key) {
  ERL_NIF_TERM val;
  if (!enif_get_map_value(env, map, enif_make_atom(env, key), &val)) return NULL;
  return get_packed_weight(env, val);
}

static int run_decode_block_device(PackedWeight *q, PackedWeight *k, PackedWeight *v,
                                    PackedWeight *o, PackedWeight *gate,
                                    PackedWeight *up, PackedWeight *down,
                                    PackedWeight *qkv, PackedWeight *gate_up,
                                    const ErlNifBinary *norm1_bin,
                                    const ErlNifBinary *norm2_bin,
                                    const ErlNifBinary *rope_bin,
                                   BlockKvCache *kv_cache_res, int pos) {
  if (!validate_fp8_weight(q) || !validate_fp8_weight(k) || !validate_fp8_weight(v) ||
      !validate_fp8_weight(o) || !validate_fp8_weight(gate) || !validate_fp8_weight(up) ||
      !validate_fp8_weight(down)) {
    return -10;
  }

  int hidden = q->in_features;
  int kv_dim = k->out_features;
  int ffn = gate->out_features;
  int use_qkv = qkv && validate_fp8_weight(qkv);
  int use_gate_up = gate_up && validate_fp8_weight(gate_up);
  int num_heads = hidden / LLAMA_HEAD_DIM;
  int num_kv_heads = kv_dim / LLAMA_HEAD_DIM;
  if (hidden <= 0 || kv_dim <= 0 || ffn <= 0 ||
      hidden % LLAMA_HEAD_DIM != 0 || kv_dim % LLAMA_HEAD_DIM != 0 ||
      q->out_features != hidden || k->in_features != hidden || v->in_features != hidden ||
      v->out_features != kv_dim || o->in_features != hidden || o->out_features != hidden ||
      gate->in_features != hidden || up->in_features != hidden || up->out_features != ffn ||
      down->in_features != ffn || down->out_features != hidden) {
    return -11;
  }
  if (use_qkv && (qkv->in_features != hidden || qkv->out_features != hidden + kv_dim + kv_dim)) {
    use_qkv = 0;
  }
  if (use_gate_up && (gate_up->in_features != hidden || gate_up->out_features != ffn + ffn)) {
    use_gate_up = 0;
  }

  size_t hidden16_bytes = (size_t)hidden * sizeof(uint16_t);
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  size_t kv16_bytes = (size_t)kv_dim * sizeof(uint16_t);
  size_t kv32_bytes = (size_t)kv_dim * sizeof(float);
  size_t ffn16_bytes = (size_t)ffn * sizeof(uint16_t);
  size_t ffn32_bytes = (size_t)ffn * sizeof(float);
  if (norm1_bin->size != hidden32_bytes || norm2_bin->size != hidden32_bytes ||
      rope_bin->size != (size_t)(LLAMA_HEAD_DIM / 2) * sizeof(float)) {
    return -12;
  }
  if (!kv_cache_res || kv_cache_res->kv_dim != kv_dim || kv_cache_res->len < 0 ||
      kv_cache_res->len >= kv_cache_res->max_seq) {
    return -13;
  }
  int past_len = kv_cache_res->len;

  int rc = 0;
  if ((rc = upload_binary_async(&b_norm1, norm1_bin)) != 0) return -100 + rc;
  if ((rc = upload_binary_async(&b_norm2, norm2_bin)) != 0) return -120 + rc;
  if ((rc = upload_binary_async(&b_rope, rope_bin)) != 0) return -140 + rc;
  g_norm1_ptr = (float *)b_norm1.ptr;
  g_norm2_ptr = (float *)b_norm2.ptr;
  g_rope_ptr = (float *)b_rope.ptr;
  g_dyn_pos_ptr = NULL;
  g_dyn_past_len_ptr = NULL;

  BlockBuf *bufs[] = {&b_hidden32, &b_norm16, &b_q, &b_o, &b_h1, &b_x2, &b_x2_16,
                      &b_attn, &b_attn16, &b_down, &b_hout16};
  for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
    if (ensure_block_buf(bufs[i], hidden32_bytes) != 0) return -20;
  }
  if (ensure_block_buf(&b_k, kv32_bytes) != 0 ||
      ensure_block_buf(&b_v, kv32_bytes) != 0 ||
      ensure_block_buf(&b_k_append, kv16_bytes) != 0 ||
      ensure_block_buf(&b_v_append, kv16_bytes) != 0) {
    return -21;
  }
  if (ensure_block_buf(&b_gate, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_up, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw16, ffn16_bytes) != 0) {
    return -22;
  }
  if (use_qkv && ensure_block_buf(&b_qkv, (size_t)(hidden + kv_dim + kv_dim) * sizeof(float)) != 0) {
    return -23;
  }
  if (use_gate_up && ensure_block_buf(&b_gate_up, (size_t)(ffn + ffn) * sizeof(float)) != 0) {
    return -24;
  }

  g_attn_k_cache_ptr = kv_cache_res->d_k;
  g_attn_v_cache_ptr = kv_cache_res->d_v;
  g_q_ptr = NULL;
  g_k_ptr = NULL;
  g_v_ptr = NULL;
  g_gate_ptr = NULL;
  g_up_ptr = NULL;

  if ((rc = run_helper_graph(BLOCK_GRAPH_NORM1, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -200 + rc;
  if (use_qkv) {
    if ((rc = gemm_w8a16_dequant(qkv, (uint16_t *)b_norm16.ptr, 1, (float *)b_qkv.ptr)) != 0)
      return -300 + rc;
    g_q_ptr = (float *)b_qkv.ptr;
    g_k_ptr = g_q_ptr + hidden;
    g_v_ptr = g_k_ptr + kv_dim;
  } else {
    if ((rc = gemm_w8a16_dequant(q, (uint16_t *)b_norm16.ptr, 1, (float *)b_q.ptr)) != 0)
      return -300 + rc;
    if ((rc = gemm_w8a16_dequant(k, (uint16_t *)b_norm16.ptr, 1, (float *)b_k.ptr)) != 0)
      return -320 + rc;
    if ((rc = gemm_w8a16_dequant(v, (uint16_t *)b_norm16.ptr, 1, (float *)b_v.ptr)) != 0)
      return -340 + rc;
  }
  if ((rc = run_helper_sequence(BLOCK_GRAPH_ROPE_ATTN, hidden, kv_dim, ffn, pos,
                                 past_len, num_heads, num_kv_heads)) != 0)
    return -400 + rc;
  if ((rc = gemm_w8a16_dequant(o, (uint16_t *)b_attn16.ptr, 1, (float *)b_o.ptr)) != 0)
    return -500 + rc;
  if ((rc = run_helper_graph(BLOCK_GRAPH_POST_ATTN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -600 + rc;
  if (use_gate_up) {
    if ((rc = gemm_w8a16_dequant(gate_up, (uint16_t *)b_x2_16.ptr, 1, (float *)b_gate_up.ptr)) != 0)
      return -700 + rc;
    g_gate_ptr = (float *)b_gate_up.ptr;
    g_up_ptr = g_gate_ptr + ffn;
  } else {
    if ((rc = gemm_w8a16_dequant(gate, (uint16_t *)b_x2_16.ptr, 1, (float *)b_gate.ptr)) != 0)
      return -700 + rc;
    if ((rc = gemm_w8a16_dequant(up, (uint16_t *)b_x2_16.ptr, 1, (float *)b_up.ptr)) != 0)
      return -720 + rc;
  }
  if ((rc = run_helper_graph(BLOCK_GRAPH_FFN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -800 + rc;
  if ((rc = gemm_w8a16_dequant(down, (uint16_t *)b_sw16.ptr, 1, (float *)b_down.ptr)) != 0)
    return -900 + rc;
  if ((rc = run_helper_graph(BLOCK_GRAPH_OUT, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -1000 + rc;

  size_t offset = (size_t)past_len * kv16_bytes;
  if (cudaMemcpyAsync((uint8_t *)kv_cache_res->d_k + offset, b_k_append.ptr, kv16_bytes,
                      cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess ||
      cudaMemcpyAsync((uint8_t *)kv_cache_res->d_v + offset, b_v_append.ptr, kv16_bytes,
                      cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess) {
    return -30;
  }
  kv_cache_res->len = past_len + 1;

  if (cudaMemcpyAsync(b_hidden16.ptr, b_hout16.ptr, hidden16_bytes,
                      cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess) {
    return -31;
  }
  return 0;
}

#define DECODE_MAX_LAYERS 64
#define DECODE_GRAPH_MAX 4096

typedef struct {
  PackedWeight *q, *k, *v, *o, *gate, *up, *down, *qkv, *gate_up;
  BlockKvCache *cache;
  float *norm1;
  float *norm2;
} DecodeLayerParams;

typedef struct {
  int layer_count;
  uintptr_t signature;
  DecodeLayerParams layers[DECODE_MAX_LAYERS];
} ModelLayers;

typedef struct {
  const uint8_t *host_ptr;
  size_t size;
  BlockBuf buf;
} DecodeConstBuf;

typedef struct {
  int pos;
  int past_len;
  int hidden;
  int vocab;
  int layer_count;
  uintptr_t signature;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  unsigned launches;
} DecodeGraphEntry;

static DecodeConstBuf g_decode_norm1[DECODE_MAX_LAYERS];
static DecodeConstBuf g_decode_norm2[DECODE_MAX_LAYERS];
static DecodeConstBuf g_decode_rope = {0};
static DecodeConstBuf g_decode_final_norm = {0};
static DecodeGraphEntry g_decode_graphs[DECODE_GRAPH_MAX];
static ErlNifResourceType *MODEL_LAYERS_RES = NULL;
static int g_decode_graph_count = 0;
static int g_decode_graph_disabled = 0;
static unsigned g_decode_graph_captures = 0;
static unsigned g_decode_graph_hits = 0;
static unsigned g_decode_graph_updates = 0;
static unsigned g_decode_graph_update_failures = 0;
static unsigned g_decode_graph_failures = 0;
static cudaError_t g_decode_last_capture_error = cudaSuccess;
static cudaGraphExec_t g_decode_rolling_exec = NULL;
static int g_decode_rolling_hidden = 0;
static int g_decode_rolling_vocab = 0;
static int g_decode_rolling_layer_count = 0;
static uintptr_t g_decode_rolling_signature = 0;

static int decode_graph_debug_enabled(void) {
  const char *v = getenv("VIVA_DECODE_GRAPH_DEBUG");
  return v && v[0] && v[0] != '0';
}

static void decode_graph_debug(const char *msg, int a, int b, int c) {
  if (decode_graph_debug_enabled()) {
    fprintf(stderr, "[decode_graph] %s %d %d %d\n", msg, a, b, c);
  }
}

static uintptr_t mix_ptr(uintptr_t h, const void *p) {
  uintptr_t x = (uintptr_t)p;
  h ^= x + (uintptr_t)0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h;
}

static void keep_weight(PackedWeight *w) {
  if (w) enif_keep_resource(w);
}

static void release_weight(PackedWeight *w) {
  if (w) enif_release_resource(w);
}

static void release_model_layers(ModelLayers *ml) {
  if (!ml) return;
  for (int i = 0; i < ml->layer_count; ++i) {
    DecodeLayerParams *l = &ml->layers[i];
    release_weight(l->q);
    release_weight(l->k);
    release_weight(l->v);
    release_weight(l->o);
    release_weight(l->gate);
    release_weight(l->up);
    release_weight(l->down);
    release_weight(l->qkv);
    release_weight(l->gate_up);
    if (l->cache) enif_release_resource(l->cache);
    if (l->norm1) cudaFree(l->norm1);
    if (l->norm2) cudaFree(l->norm2);
  }
  memset(ml, 0, sizeof(*ml));
}

static int upload_model_norm(const ErlNifBinary *bin, float **out) {
  float *ptr = NULL;
  if (cudaMalloc(&ptr, bin->size) != cudaSuccess) return -1;
  if (cudaMemcpy(ptr, bin->data, bin->size, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(ptr);
    return -2;
  }
  *out = ptr;
  return 0;
}

static int validate_decode_layer_shapes(DecodeLayerParams *l) {
  if (!validate_fp8_weight(l->q) || !validate_fp8_weight(l->k) ||
      !validate_fp8_weight(l->v) || !validate_fp8_weight(l->o) ||
      !validate_fp8_weight(l->gate) || !validate_fp8_weight(l->up) ||
      !validate_fp8_weight(l->down)) {
    return -10;
  }
  int hidden = l->q->in_features;
  int kv_dim = l->k->out_features;
  int ffn = l->gate->out_features;
  if (hidden <= 0 || kv_dim <= 0 || ffn <= 0 ||
      hidden % LLAMA_HEAD_DIM != 0 || kv_dim % LLAMA_HEAD_DIM != 0 ||
      l->q->out_features != hidden || l->k->in_features != hidden ||
      l->v->in_features != hidden || l->v->out_features != kv_dim ||
      l->o->in_features != hidden || l->o->out_features != hidden ||
      l->gate->in_features != hidden || l->up->in_features != hidden ||
      l->up->out_features != ffn || l->down->in_features != ffn ||
      l->down->out_features != hidden) {
    return -11;
  }
  if (l->qkv && validate_fp8_weight(l->qkv) &&
      (l->qkv->in_features != hidden ||
       l->qkv->out_features != hidden + kv_dim + kv_dim)) {
    l->qkv = NULL;
  }
  if (l->gate_up && validate_fp8_weight(l->gate_up) &&
      (l->gate_up->in_features != hidden ||
       l->gate_up->out_features != ffn + ffn)) {
    l->gate_up = NULL;
  }
  if (!l->cache || l->cache->kv_dim != kv_dim || l->cache->len < 0 ||
      l->cache->len >= l->cache->max_seq) {
    return -13;
  }
  return 0;
}

static int ensure_decode_const(DecodeConstBuf *dst, const ErlNifBinary *src) {
  if (ensure_block_buf(&dst->buf, src->size) != 0) return -1;
  if (src->size == 0) return 0;
  if (dst->host_ptr == src->data && dst->size == src->size) return 0;
  if (cudaMemcpyAsync(dst->buf.ptr, src->data, src->size, cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess) {
    return -2;
  }
  dst->host_ptr = src->data;
  dst->size = src->size;
  return 0;
}

static DecodeGraphEntry *find_decode_graph(int pos, int past_len, int hidden, int vocab,
                                           int layer_count, uintptr_t signature) {
  for (int i = 0; i < g_decode_graph_count; ++i) {
    DecodeGraphEntry *e = &g_decode_graphs[i];
    if (e->pos == pos && e->past_len == past_len && e->hidden == hidden &&
        e->vocab == vocab && e->layer_count == layer_count && e->signature == signature) {
      return e;
    }
  }
  return NULL;
}

static DecodeGraphEntry *alloc_decode_graph(int pos, int past_len, int hidden, int vocab,
                                            int layer_count, uintptr_t signature) {
  if (g_decode_graph_count >= DECODE_GRAPH_MAX) return NULL;
  DecodeGraphEntry *e = &g_decode_graphs[g_decode_graph_count++];
  memset(e, 0, sizeof(*e));
  e->pos = pos;
  e->past_len = past_len;
  e->hidden = hidden;
  e->vocab = vocab;
  e->layer_count = layer_count;
  e->signature = signature;
  return e;
}

static int rolling_decode_graph_compatible(int hidden, int vocab, int layer_count,
                                           uintptr_t signature) {
  return g_decode_rolling_exec && g_decode_rolling_hidden == hidden &&
         g_decode_rolling_vocab == vocab &&
         g_decode_rolling_layer_count == layer_count &&
         g_decode_rolling_signature == signature;
}

static cudaError_t update_decode_rolling_exec(cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
  cudaGraphExecUpdateResultInfo info;
  memset(&info, 0, sizeof(info));
  return cudaGraphExecUpdate(g_decode_rolling_exec, graph, &info);
#else
  cudaGraphNode_t error_node = NULL;
  cudaGraphExecUpdateResult result = cudaGraphExecUpdateSuccess;
  return cudaGraphExecUpdate(g_decode_rolling_exec, graph, &error_node, &result);
#endif
}

static void set_decode_rolling_exec(cudaGraphExec_t exec, int hidden, int vocab,
                                    int layer_count, uintptr_t signature) {
  if (g_decode_rolling_exec && g_decode_rolling_exec != exec) {
    cudaGraphExecDestroy(g_decode_rolling_exec);
  }
  g_decode_rolling_exec = exec;
  g_decode_rolling_hidden = hidden;
  g_decode_rolling_vocab = vocab;
  g_decode_rolling_layer_count = layer_count;
  g_decode_rolling_signature = signature;
}

static int all_decode_weights_captureable(DecodeLayerParams *layers, int layer_count,
                                          PackedWeight *lm_head) {
  if (!lm_head || lm_head->block_size != 16) return 0;
  for (int i = 0; i < layer_count; ++i) {
    DecodeLayerParams *l = &layers[i];
    PackedWeight *ws[] = {l->q, l->k, l->v, l->o, l->gate, l->up, l->down};
    for (unsigned j = 0; j < sizeof(ws) / sizeof(ws[0]); ++j) {
      if (!ws[j] || ws[j]->block_size != 16) return 0;
    }
    if (l->qkv && validate_fp8_weight(l->qkv) && l->qkv->block_size != 16) return 0;
    if (l->gate_up && validate_fp8_weight(l->gate_up) && l->gate_up->block_size != 16) return 0;
  }
  return 1;
}

static int ensure_decode_layer_scratch(PackedWeight *q, PackedWeight *k, PackedWeight *gate,
                                       PackedWeight *qkv, PackedWeight *gate_up) {
  if (!q || !k || !gate) return -10;
  int hidden = q->in_features;
  int kv_dim = k->out_features;
  int ffn = gate->out_features;
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  size_t kv16_bytes = (size_t)kv_dim * sizeof(uint16_t);
  size_t kv32_bytes = (size_t)kv_dim * sizeof(float);
  size_t ffn16_bytes = (size_t)ffn * sizeof(uint16_t);
  size_t ffn32_bytes = (size_t)ffn * sizeof(float);
  BlockBuf *bufs[] = {&b_hidden32, &b_norm16, &b_q, &b_o, &b_h1, &b_x2, &b_x2_16,
                      &b_attn, &b_attn16, &b_down, &b_hout16};
  for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
    if (ensure_block_buf(bufs[i], hidden32_bytes) != 0) return -20;
  }
  if (ensure_block_buf(&b_k, kv32_bytes) != 0 ||
      ensure_block_buf(&b_v, kv32_bytes) != 0 ||
      ensure_block_buf(&b_k_append, kv16_bytes) != 0 ||
      ensure_block_buf(&b_v_append, kv16_bytes) != 0) return -21;
  if (ensure_block_buf(&b_gate, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_up, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw16, ffn16_bytes) != 0) return -22;
  if (qkv && validate_fp8_weight(qkv) &&
      ensure_block_buf(&b_qkv, (size_t)(hidden + kv_dim + kv_dim) * sizeof(float)) != 0)
    return -23;
  if (gate_up && validate_fp8_weight(gate_up) &&
      ensure_block_buf(&b_gate_up, (size_t)(ffn + ffn) * sizeof(float)) != 0)
    return -24;
  return 0;
}

static int run_decode_block_device_preloaded(DecodeLayerParams *l, float *rope, int pos) {
  if (!validate_fp8_weight(l->q) || !validate_fp8_weight(l->k) || !validate_fp8_weight(l->v) ||
      !validate_fp8_weight(l->o) || !validate_fp8_weight(l->gate) ||
      !validate_fp8_weight(l->up) || !validate_fp8_weight(l->down)) {
    return -10;
  }

  int hidden = l->q->in_features;
  int kv_dim = l->k->out_features;
  int ffn = l->gate->out_features;
  int use_qkv = l->qkv && validate_fp8_weight(l->qkv);
  int use_gate_up = l->gate_up && validate_fp8_weight(l->gate_up);
  int num_heads = hidden / LLAMA_HEAD_DIM;
  int num_kv_heads = kv_dim / LLAMA_HEAD_DIM;
  if (hidden <= 0 || kv_dim <= 0 || ffn <= 0 ||
      hidden % LLAMA_HEAD_DIM != 0 || kv_dim % LLAMA_HEAD_DIM != 0 ||
      l->q->out_features != hidden || l->k->in_features != hidden ||
      l->v->in_features != hidden || l->v->out_features != kv_dim ||
      l->o->in_features != hidden || l->o->out_features != hidden ||
      l->gate->in_features != hidden || l->up->in_features != hidden ||
      l->up->out_features != ffn || l->down->in_features != ffn ||
      l->down->out_features != hidden) {
    return -11;
  }
  if (use_qkv && (l->qkv->in_features != hidden ||
                  l->qkv->out_features != hidden + kv_dim + kv_dim)) {
    use_qkv = 0;
  }
  if (use_gate_up && (l->gate_up->in_features != hidden ||
                      l->gate_up->out_features != ffn + ffn)) {
    use_gate_up = 0;
  }
  if (!l->cache || l->cache->kv_dim != kv_dim || l->cache->len < 0 ||
      l->cache->len >= l->cache->max_seq) {
    return -13;
  }
  int past_len = l->cache->len;
  size_t hidden16_bytes = (size_t)hidden * sizeof(uint16_t);
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  size_t kv16_bytes = (size_t)kv_dim * sizeof(uint16_t);
  size_t kv32_bytes = (size_t)kv_dim * sizeof(float);
  size_t ffn16_bytes = (size_t)ffn * sizeof(uint16_t);
  size_t ffn32_bytes = (size_t)ffn * sizeof(float);

  BlockBuf *bufs[] = {&b_hidden32, &b_norm16, &b_q, &b_o, &b_h1, &b_x2, &b_x2_16,
                      &b_attn, &b_attn16, &b_down, &b_hout16};
  for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i) {
    if (ensure_block_buf(bufs[i], hidden32_bytes) != 0) return -20;
  }
  if (ensure_block_buf(&b_k, kv32_bytes) != 0 ||
      ensure_block_buf(&b_v, kv32_bytes) != 0 ||
      ensure_block_buf(&b_k_append, kv16_bytes) != 0 ||
      ensure_block_buf(&b_v_append, kv16_bytes) != 0) return -21;
  if (ensure_block_buf(&b_gate, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_up, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw, ffn32_bytes) != 0 ||
      ensure_block_buf(&b_sw16, ffn16_bytes) != 0) return -22;
  if (use_qkv && ensure_block_buf(&b_qkv, (size_t)(hidden + kv_dim + kv_dim) * sizeof(float)) != 0)
    return -23;
  if (use_gate_up && ensure_block_buf(&b_gate_up, (size_t)(ffn + ffn) * sizeof(float)) != 0)
    return -24;

  g_attn_k_cache_ptr = l->cache->d_k;
  g_attn_v_cache_ptr = l->cache->d_v;
  g_norm1_ptr = l->norm1;
  g_norm2_ptr = l->norm2;
  g_rope_ptr = rope;
  g_q_ptr = NULL;
  g_k_ptr = NULL;
  g_v_ptr = NULL;
  g_gate_ptr = NULL;
  g_up_ptr = NULL;

  int rc = 0;
  if ((rc = run_helper_graph(BLOCK_GRAPH_NORM1, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -200 + rc;
  if (use_qkv) {
    if ((rc = gemm_w8a16_dequant(l->qkv, (uint16_t *)b_norm16.ptr, 1,
                                 (float *)b_qkv.ptr)) != 0) return -300 + rc;
    g_q_ptr = (float *)b_qkv.ptr;
    g_k_ptr = g_q_ptr + hidden;
    g_v_ptr = g_k_ptr + kv_dim;
  } else {
    if ((rc = gemm_w8a16_dequant(l->q, (uint16_t *)b_norm16.ptr, 1,
                                 (float *)b_q.ptr)) != 0) return -300 + rc;
    if ((rc = gemm_w8a16_dequant(l->k, (uint16_t *)b_norm16.ptr, 1,
                                 (float *)b_k.ptr)) != 0) return -320 + rc;
    if ((rc = gemm_w8a16_dequant(l->v, (uint16_t *)b_norm16.ptr, 1,
                                 (float *)b_v.ptr)) != 0) return -340 + rc;
  }
  if ((rc = run_helper_sequence(BLOCK_GRAPH_ROPE_ATTN, hidden, kv_dim, ffn, pos,
                                past_len, num_heads, num_kv_heads)) != 0) return -400 + rc;
  if ((rc = gemm_w8a16_dequant(l->o, (uint16_t *)b_attn16.ptr, 1,
                               (float *)b_o.ptr)) != 0) return -500 + rc;
  if ((rc = run_helper_graph(BLOCK_GRAPH_POST_ATTN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -600 + rc;
  if (use_gate_up) {
    if ((rc = gemm_w8a16_dequant(l->gate_up, (uint16_t *)b_x2_16.ptr, 1,
                                 (float *)b_gate_up.ptr)) != 0) return -700 + rc;
    g_gate_ptr = (float *)b_gate_up.ptr;
    g_up_ptr = g_gate_ptr + ffn;
  } else {
    if ((rc = gemm_w8a16_dequant(l->gate, (uint16_t *)b_x2_16.ptr, 1,
                                 (float *)b_gate.ptr)) != 0) return -700 + rc;
    if ((rc = gemm_w8a16_dequant(l->up, (uint16_t *)b_x2_16.ptr, 1,
                                 (float *)b_up.ptr)) != 0) return -720 + rc;
  }
  if ((rc = run_helper_graph(BLOCK_GRAPH_FFN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -800 + rc;
  if ((rc = gemm_w8a16_dequant(l->down, (uint16_t *)b_sw16.ptr, 1,
                               (float *)b_down.ptr)) != 0) return -900 + rc;
  if ((rc = run_helper_graph(BLOCK_GRAPH_OUT, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0) return -1000 + rc;

  if (g_dyn_past_len_ptr) {
    rc = vt_kv_cache_append_fp16_dyn(l->cache->d_k, l->cache->d_v, b_k_append.ptr,
                                     b_v_append.ptr, g_dyn_past_len_ptr, kv_dim);
    if (rc != 0) return -30 + rc;
  } else {
    size_t offset = (size_t)past_len * kv16_bytes;
    if (cudaMemcpyAsync((uint8_t *)l->cache->d_k + offset, b_k_append.ptr, kv16_bytes,
                        cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess ||
        cudaMemcpyAsync((uint8_t *)l->cache->d_v + offset, b_v_append.ptr, kv16_bytes,
                        cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess) return -30;
  }
  l->cache->len = past_len + 1;
  if (cudaMemcpyAsync(b_hidden16.ptr, b_hout16.ptr, hidden16_bytes,
                      cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess) return -31;
  return 0;
}

static int run_decode_token_sequence(EmbeddingTable *embed, int token_id,
                                     DecodeLayerParams *layers, int layer_count,
                                     float *final_norm, PackedWeight *lm_head,
                                     float *rope, int pos) {
  int rc = vt_embedding_lookup_fp16(embed->d_weight, (const int *)b_token_id.ptr,
                                    b_hidden16.ptr, embed->hidden, embed->vocab);
  if (rc != 0) return -10000 + rc;
  for (int i = 0; i < layer_count; ++i) {
    rc = run_decode_block_device_preloaded(&layers[i], rope, pos);
    if (rc != 0) return -11000 + rc;
  }
  g_norm1_ptr = final_norm;
  g_norm2_ptr = NULL;
  g_rope_ptr = rope;
  rc = run_helper_graph(BLOCK_GRAPH_NORM1, embed->hidden, 0, 0, -1, -1, 0, 0);
  if (rc != 0) return -12000 + rc;
  rc = gemm_w8a16_dequant(lm_head, (uint16_t *)b_norm16.ptr, 1,
                          (float *)b_logits.ptr);
  if (rc != 0) return -13000 + rc;
  rc = vt_argmax_fp32_as_fp16((float *)b_logits.ptr, lm_head->out_features,
                              (int *)b_argmax.ptr);
  if (rc != 0) return -14000 + rc;
  (void)token_id;
  return 0;
}

#endif

void block_kv_cache_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
  BlockKvCache *cache = (BlockKvCache *)obj;
  if (!cache) return;
  if (cache->d_k) {
    cudaFree(cache->d_k);
    cache->d_k = NULL;
  }
  if (cache->d_v) {
    cudaFree(cache->d_v);
    cache->d_v = NULL;
  }
#else
  (void)obj;
#endif
}

int register_block_kv_cache_resource(ErlNifEnv *env) {
  BLOCK_KV_CACHE_RES = enif_open_resource_type(
      env, NULL, "BlockKvCache", block_kv_cache_destructor,
      ERL_NIF_RT_CREATE, NULL);
  return BLOCK_KV_CACHE_RES != NULL ? 0 : -1;
}

void model_layers_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
#if !defined(_WIN32) && !defined(VIVA_NO_CUDA)
  release_model_layers((ModelLayers *)obj);
#else
  (void)obj;
#endif
}

int register_model_layers_resource(ErlNifEnv *env) {
  MODEL_LAYERS_RES = enif_open_resource_type(
      env, NULL, "ModelLayers", model_layers_destructor,
      ERL_NIF_RT_CREATE, NULL);
  return MODEL_LAYERS_RES != NULL ? 0 : -1;
}

ERL_NIF_TERM nt_model_layers_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 2) return make_error(env, "bad_arity");
  if (!MODEL_LAYERS_RES) return make_error(env, "model_layers_resource_unavailable");
  ModelLayers *ml = (ModelLayers *)enif_alloc_resource(MODEL_LAYERS_RES, sizeof(ModelLayers));
  if (!ml) return make_error(env, "resource_alloc_failed");
  memset(ml, 0, sizeof(*ml));
  ml->signature = (uintptr_t)0xcbf29ce484222325ull;

  ERL_NIF_TERM layers_tail = argv[0];
  ERL_NIF_TERM caches_tail = argv[1];
  ERL_NIF_TERM layer_term, cache_term;
  while (enif_get_list_cell(env, layers_tail, &layer_term, &layers_tail)) {
    int i = ml->layer_count;
    if (i >= DECODE_MAX_LAYERS) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_error(env, "too_many_layers");
    }
    if (!enif_get_list_cell(env, caches_tail, &cache_term, &caches_tail)) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_error(env, "cache_count_mismatch");
    }

    ErlNifBinary norm1_bin, norm2_bin;
    if (!map_get_binary(env, layer_term, "norm1_bin", &norm1_bin) ||
        !map_get_binary(env, layer_term, "norm2_bin", &norm2_bin)) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_error(env, "invalid_layer_norm");
    }

    DecodeLayerParams *l = &ml->layers[i];
    l->q = map_get_weight(env, layer_term, "q");
    l->k = map_get_weight(env, layer_term, "k");
    l->v = map_get_weight(env, layer_term, "v");
    l->o = map_get_weight(env, layer_term, "o");
    l->gate = map_get_weight(env, layer_term, "gate");
    l->up = map_get_weight(env, layer_term, "up");
    l->down = map_get_weight(env, layer_term, "down");
    l->qkv = map_get_weight(env, layer_term, "qkv");
    l->gate_up = map_get_weight(env, layer_term, "gate_up");
    if (!enif_get_resource(env, cache_term, BLOCK_KV_CACHE_RES, (void **)&l->cache)) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_error(env, "invalid_kv_cache");
    }

    int rc = validate_decode_layer_shapes(l);
    if (rc != 0) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_block_error(env, "model_layer_shape", rc);
    }
    size_t hidden32_bytes = (size_t)l->q->in_features * sizeof(float);
    if (norm1_bin.size != hidden32_bytes || norm2_bin.size != hidden32_bytes) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_error(env, "layer_norm_size_mismatch");
    }
    rc = upload_model_norm(&norm1_bin, &l->norm1);
    if (rc != 0) {
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_block_error(env, "upload_model_norm1", rc);
    }
    rc = upload_model_norm(&norm2_bin, &l->norm2);
    if (rc != 0) {
      if (l->norm1) {
        cudaFree(l->norm1);
        l->norm1 = NULL;
      }
      release_model_layers(ml);
      enif_release_resource(ml);
      return make_block_error(env, "upload_model_norm2", rc);
    }

    keep_weight(l->q);
    keep_weight(l->k);
    keep_weight(l->v);
    keep_weight(l->o);
    keep_weight(l->gate);
    keep_weight(l->up);
    keep_weight(l->down);
    keep_weight(l->qkv);
    keep_weight(l->gate_up);
    enif_keep_resource(l->cache);

    ml->signature = mix_ptr(ml->signature, l->cache->d_k);
    ml->signature = mix_ptr(ml->signature, l->cache->d_v);
    ml->signature = mix_ptr(ml->signature, l->q ? l->q->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->k ? l->k->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->v ? l->v->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->o ? l->o->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->gate ? l->gate->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->up ? l->up->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->down ? l->down->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->qkv ? l->qkv->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->gate_up ? l->gate_up->d_weight : NULL);
    ml->signature = mix_ptr(ml->signature, l->norm1);
    ml->signature = mix_ptr(ml->signature, l->norm2);
    ml->layer_count++;
  }
  if (!enif_is_empty_list(env, caches_tail)) {
    release_model_layers(ml);
    enif_release_resource(ml);
    return make_error(env, "cache_count_mismatch");
  }
  if (ml->layer_count <= 0) {
    release_model_layers(ml);
    enif_release_resource(ml);
    return make_error(env, "empty_layers");
  }

  ERL_NIF_TERM term = enif_make_resource(env, ml);
  enif_release_resource(ml);
  return make_ok(env, term);
#endif
}

ERL_NIF_TERM nt_kv_cache_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 2) return make_error(env, "bad_arity");
  int max_seq = 0, kv_dim = 0;
  if (!enif_get_int(env, argv[0], &max_seq) || max_seq <= 0) return make_error(env, "invalid_max_seq");
  if (!enif_get_int(env, argv[1], &kv_dim) || kv_dim <= 0) return make_error(env, "invalid_kv_dim");
  BlockKvCache *cache = (BlockKvCache *)enif_alloc_resource(BLOCK_KV_CACHE_RES, sizeof(BlockKvCache));
  if (!cache) return make_error(env, "resource_alloc_failed");
  memset(cache, 0, sizeof(*cache));
  cache->max_seq = max_seq;
  cache->kv_dim = kv_dim;
  size_t bytes = (size_t)max_seq * (size_t)kv_dim * sizeof(uint16_t);
  if (cudaMalloc(&cache->d_k, bytes) != cudaSuccess ||
      cudaMalloc(&cache->d_v, bytes) != cudaSuccess) {
    enif_release_resource(cache);
    return make_error(env, "cuda_malloc_kv_cache_failed");
  }
  ERL_NIF_TERM term = enif_make_resource(env, cache);
  enif_release_resource(cache);
  return make_ok(env, term);
#endif
}

ERL_NIF_TERM nt_forward_block_w8a16(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  (void)argc;
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  return make_error(env, "cuda_not_available");
#else
  if (argc != 14) return make_error(env, "bad_arity");

  ErlNifBinary hidden_bin, norm1_bin, norm2_bin, rope_bin, k_cache_bin, v_cache_bin;
  BlockKvCache *kv_cache_res = NULL;
  int use_kv_resource = 0;
  if (!enif_inspect_binary(env, argv[0], &hidden_bin)) return make_error(env, "invalid_hidden");
  if (!enif_inspect_binary(env, argv[8], &norm1_bin)) return make_error(env, "invalid_norm1");
  if (!enif_inspect_binary(env, argv[9], &norm2_bin)) return make_error(env, "invalid_norm2");
  if (!enif_inspect_binary(env, argv[11], &rope_bin)) return make_error(env, "invalid_rope");
  if (enif_get_resource(env, argv[12], BLOCK_KV_CACHE_RES, (void **)&kv_cache_res)) {
    use_kv_resource = 1;
    memset(&k_cache_bin, 0, sizeof(k_cache_bin));
    memset(&v_cache_bin, 0, sizeof(v_cache_bin));
  } else {
    if (!enif_inspect_binary(env, argv[12], &k_cache_bin)) return make_error(env, "invalid_k_cache");
    if (!enif_inspect_binary(env, argv[13], &v_cache_bin)) return make_error(env, "invalid_v_cache");
  }

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
  int past_len = 0;
  if (use_kv_resource) {
    if (kv_cache_res->kv_dim != kv_dim || kv_cache_res->len < 0 || kv_cache_res->len >= kv_cache_res->max_seq) {
      return make_error(env, "kv_resource_mismatch");
    }
    past_len = kv_cache_res->len;
  } else {
    if (k_cache_bin.size != v_cache_bin.size || (kv16_bytes > 0 && k_cache_bin.size % kv16_bytes != 0)) {
      return make_error(env, "cache_size_mismatch");
    }
    past_len = (int)(k_cache_bin.size / kv16_bytes);
  }

  int rc = 0;
  if ((rc = upload_binary(&b_hidden16, &hidden_bin)) != 0) return make_block_error(env, "upload_hidden", rc);
  if ((rc = upload_binary(&b_norm1, &norm1_bin)) != 0) return make_block_error(env, "upload_norm1", rc);
  if ((rc = upload_binary(&b_norm2, &norm2_bin)) != 0) return make_block_error(env, "upload_norm2", rc);
  if ((rc = upload_binary(&b_rope, &rope_bin)) != 0) return make_block_error(env, "upload_rope", rc);
  if (use_kv_resource) {
    g_attn_k_cache_ptr = kv_cache_res->d_k;
    g_attn_v_cache_ptr = kv_cache_res->d_v;
  } else {
    if ((rc = upload_binary(&b_k_cache, &k_cache_bin)) != 0) return make_block_error(env, "upload_k_cache", rc);
    if ((rc = upload_binary(&b_v_cache, &v_cache_bin)) != 0) return make_block_error(env, "upload_v_cache", rc);
    g_attn_k_cache_ptr = b_k_cache.ptr;
    g_attn_v_cache_ptr = b_v_cache.ptr;
  }

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

  if (ensure_block_lt() != 0) return make_error(env, "cuda_stream_init_failed");
  vt_block_set_stream((void *)g_block_stream);
  cuda_fp8_dequant_set_stream((void *)g_block_stream);

  if ((rc = run_helper_graph(BLOCK_GRAPH_NORM1, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0)
    return make_block_error(env, "graph_norm1", rc);

  if ((rc = gemm_w8a16_dequant(q, (uint16_t *)b_norm16.ptr, 1, (float *)b_q.ptr)) != 0)
    return make_block_error(env, "gemm_q", rc);
  if ((rc = gemm_w8a16_dequant(k, (uint16_t *)b_norm16.ptr, 1, (float *)b_k.ptr)) != 0)
    return make_block_error(env, "gemm_k", rc);
  if ((rc = gemm_w8a16_dequant(v, (uint16_t *)b_norm16.ptr, 1, (float *)b_v.ptr)) != 0)
    return make_block_error(env, "gemm_v", rc);

  if (use_kv_resource) {
    rc = run_helper_sequence(BLOCK_GRAPH_ROPE_ATTN, hidden, kv_dim, ffn, pos, past_len,
                             num_heads, num_kv_heads);
  } else {
    rc = run_helper_graph(BLOCK_GRAPH_ROPE_ATTN, hidden, kv_dim, ffn, pos, past_len,
                          num_heads, num_kv_heads);
  }
  if (rc != 0)
    return make_block_error(env, "graph_rope_attn", rc);
  if ((rc = gemm_w8a16_dequant(o, (uint16_t *)b_attn16.ptr, 1, (float *)b_o.ptr)) != 0)
    return make_block_error(env, "gemm_o", rc);
  if ((rc = run_helper_graph(BLOCK_GRAPH_POST_ATTN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0)
    return make_block_error(env, "graph_post_attn", rc);
  if ((rc = gemm_w8a16_dequant(gate, (uint16_t *)b_x2_16.ptr, 1, (float *)b_gate.ptr)) != 0)
    return make_block_error(env, "gemm_gate", rc);
  if ((rc = gemm_w8a16_dequant(up, (uint16_t *)b_x2_16.ptr, 1, (float *)b_up.ptr)) != 0)
    return make_block_error(env, "gemm_up", rc);
  if ((rc = run_helper_graph(BLOCK_GRAPH_FFN, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0)
    return make_block_error(env, "graph_ffn", rc);
  if ((rc = gemm_w8a16_dequant(down, (uint16_t *)b_sw16.ptr, 1, (float *)b_down.ptr)) != 0)
    return make_block_error(env, "gemm_down", rc);
  if ((rc = run_helper_graph(BLOCK_GRAPH_OUT, hidden, kv_dim, ffn, -1, -1,
                             num_heads, num_kv_heads)) != 0)
    return make_block_error(env, "graph_out", rc);

  if (use_kv_resource) {
    size_t offset = (size_t)past_len * kv16_bytes;
    if (cudaMemcpyAsync((uint8_t *)kv_cache_res->d_k + offset, b_k_append.ptr, kv16_bytes,
                        cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess ||
        cudaMemcpyAsync((uint8_t *)kv_cache_res->d_v + offset, b_v_append.ptr, kv16_bytes,
                        cudaMemcpyDeviceToDevice, g_block_stream) != cudaSuccess) {
      return make_error(env, "kv_cache_append_failed");
    }
    kv_cache_res->len = past_len + 1;
  }

  if (cudaStreamSynchronize(g_block_stream) != cudaSuccess)
    return make_error(env, "block_stream_sync_failed");
  vt_block_set_stream(NULL);
  cuda_fp8_dequant_set_stream(NULL);

  ErlNifBinary out_hidden, out_k, out_v;
  if (!enif_alloc_binary(hidden16_bytes, &out_hidden) ||
      !enif_alloc_binary(use_kv_resource ? 0 : kv16_bytes, &out_k) ||
      !enif_alloc_binary(use_kv_resource ? 0 : kv16_bytes, &out_v)) {
    return make_error(env, "alloc_output_failed");
  }
  int download_ok = cudaMemcpy(out_hidden.data, b_hout16.ptr, hidden16_bytes,
                               cudaMemcpyDeviceToHost) == cudaSuccess;
  if (download_ok && !use_kv_resource) {
    download_ok =
        cudaMemcpy(out_k.data, b_k_append.ptr, kv16_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(out_v.data, b_v_append.ptr, kv16_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  if (!download_ok) {
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

ERL_NIF_TERM nt_forward_decode_step(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 8) return make_error(env, "bad_arity");

  int token_id = 0, pos = 0;
  if (!enif_get_int(env, argv[0], &token_id) || token_id < 0) return make_error(env, "invalid_token");
  EmbeddingTable *embed = get_embedding_table(env, argv[1]);
  if (!embed || token_id >= embed->vocab || embed->hidden <= 0) return make_error(env, "invalid_embedding");
  if (!enif_get_int(env, argv[6], &pos) || pos < 0) return make_error(env, "invalid_pos");

  ErlNifBinary final_norm_bin, rope_bin;
  if (!enif_inspect_binary(env, argv[3], &final_norm_bin)) return make_error(env, "invalid_final_norm");
  PackedWeight *lm_head = get_packed_weight(env, argv[4]);
  if (!validate_fp8_weight(lm_head)) return make_error(env, "invalid_lm_head");
  if (!enif_inspect_binary(env, argv[7], &rope_bin)) return make_error(env, "invalid_rope");

  int hidden = embed->hidden;
  if (lm_head->in_features != hidden || final_norm_bin.size != (size_t)hidden * sizeof(float)) {
    return make_error(env, "shape_mismatch");
  }
  if (ensure_block_lt() != 0) return make_error(env, "cuda_stream_init_failed");
  vt_block_set_stream((void *)g_block_stream);
  cuda_fp8_dequant_set_stream((void *)g_block_stream);

  size_t hidden16_bytes = (size_t)hidden * sizeof(uint16_t);
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  if (ensure_block_buf(&b_hidden16, hidden16_bytes) != 0 ||
      ensure_block_buf(&b_hidden32, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_norm1, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_x2, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_norm16, hidden16_bytes) != 0 ||
      ensure_block_buf(&b_logits, (size_t)lm_head->out_features * sizeof(float)) != 0 ||
      ensure_block_buf(&b_argmax, sizeof(int)) != 0 ||
      ensure_block_buf(&b_token_id, sizeof(int)) != 0 ||
      ensure_block_buf(&b_pos_id, sizeof(int)) != 0 ||
      ensure_block_buf(&b_past_len_id, sizeof(int)) != 0) {
    return make_error(env, "cuda_malloc_decode_failed");
  }

  ERL_NIF_TERM layers_tail = argv[2];
  ERL_NIF_TERM caches_tail = argv[5];
  ERL_NIF_TERM layer_term, cache_term;
  int layer_count = 0;
  int past_len = -1;
  uintptr_t signature = (uintptr_t)0xcbf29ce484222325ull;
  DecodeLayerParams decode_layers[DECODE_MAX_LAYERS];
  ErlNifBinary norm1_bins[DECODE_MAX_LAYERS];
  ErlNifBinary norm2_bins[DECODE_MAX_LAYERS];
  memset(decode_layers, 0, sizeof(decode_layers));
  while (enif_get_list_cell(env, layers_tail, &layer_term, &layers_tail)) {
    if (layer_count >= DECODE_MAX_LAYERS) return make_error(env, "too_many_layers");
    if (!enif_get_list_cell(env, caches_tail, &cache_term, &caches_tail)) {
      return make_error(env, "cache_count_mismatch");
    }

    ErlNifBinary norm1_bin, norm2_bin;
    if (!map_get_binary(env, layer_term, "norm1_bin", &norm1_bin) ||
        !map_get_binary(env, layer_term, "norm2_bin", &norm2_bin)) {
      return make_error(env, "invalid_layer_norm");
    }
    PackedWeight *q = map_get_weight(env, layer_term, "q");
    PackedWeight *k = map_get_weight(env, layer_term, "k");
    PackedWeight *v = map_get_weight(env, layer_term, "v");
    PackedWeight *o = map_get_weight(env, layer_term, "o");
    PackedWeight *gate = map_get_weight(env, layer_term, "gate");
    PackedWeight *up = map_get_weight(env, layer_term, "up");
    PackedWeight *down = map_get_weight(env, layer_term, "down");
    PackedWeight *qkv = map_get_weight(env, layer_term, "qkv");
    PackedWeight *gate_up = map_get_weight(env, layer_term, "gate_up");
    BlockKvCache *cache = NULL;
    if (!enif_get_resource(env, cache_term, BLOCK_KV_CACHE_RES, (void **)&cache)) {
      return make_error(env, "invalid_kv_cache");
    }

    if (!cache || cache->len < 0) return make_error(env, "invalid_kv_cache");
    if (past_len < 0) {
      past_len = cache->len;
    } else if (cache->len != past_len) {
      return make_error(env, "kv_cache_len_mismatch");
    }

    int rc = ensure_decode_const(&g_decode_norm1[layer_count], &norm1_bin);
    if (rc != 0) return make_block_error(env, "upload_decode_norm1", rc);
    rc = ensure_decode_const(&g_decode_norm2[layer_count], &norm2_bin);
    if (rc != 0) return make_block_error(env, "upload_decode_norm2", rc);
    rc = ensure_decode_layer_scratch(q, k, gate, qkv, gate_up);
    if (rc != 0) return make_block_error(env, "decode_scratch", rc);

    decode_layers[layer_count].q = q;
    decode_layers[layer_count].k = k;
    decode_layers[layer_count].v = v;
    decode_layers[layer_count].o = o;
    decode_layers[layer_count].gate = gate;
    decode_layers[layer_count].up = up;
    decode_layers[layer_count].down = down;
    decode_layers[layer_count].qkv = qkv;
    decode_layers[layer_count].gate_up = gate_up;
    decode_layers[layer_count].cache = cache;
    decode_layers[layer_count].norm1 = (float *)g_decode_norm1[layer_count].buf.ptr;
    decode_layers[layer_count].norm2 = (float *)g_decode_norm2[layer_count].buf.ptr;
    norm1_bins[layer_count] = norm1_bin;
    norm2_bins[layer_count] = norm2_bin;

    signature = mix_ptr(signature, cache->d_k);
    signature = mix_ptr(signature, cache->d_v);
    signature = mix_ptr(signature, q ? q->d_weight : NULL);
    signature = mix_ptr(signature, k ? k->d_weight : NULL);
    signature = mix_ptr(signature, v ? v->d_weight : NULL);
    signature = mix_ptr(signature, o ? o->d_weight : NULL);
    signature = mix_ptr(signature, gate ? gate->d_weight : NULL);
    signature = mix_ptr(signature, up ? up->d_weight : NULL);
    signature = mix_ptr(signature, down ? down->d_weight : NULL);
    signature = mix_ptr(signature, qkv ? qkv->d_weight : NULL);
    signature = mix_ptr(signature, gate_up ? gate_up->d_weight : NULL);
    layer_count++;
  }
  if (!enif_is_empty_list(env, caches_tail)) return make_error(env, "cache_count_mismatch");
  if (layer_count <= 0) return make_error(env, "empty_layers");
  if (past_len < 0) return make_error(env, "invalid_past_len");

  int const_rc = ensure_decode_const(&g_decode_final_norm, &final_norm_bin);
  if (const_rc != 0) return make_block_error(env, "upload_final_norm", const_rc);
  const_rc = ensure_decode_const(&g_decode_rope, &rope_bin);
  if (const_rc != 0) return make_block_error(env, "upload_decode_rope", const_rc);

  if (cudaMemcpyAsync(b_token_id.ptr, &token_id, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess) {
    return make_error(env, "upload_token_id_failed");
  }
  if (cudaMemcpyAsync(b_pos_id.ptr, &pos, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess ||
      cudaMemcpyAsync(b_past_len_id.ptr, &past_len, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess) {
    return make_error(env, "upload_decode_scalar_failed");
  }

  signature = mix_ptr(signature, embed->d_weight);
  signature = mix_ptr(signature, lm_head->d_weight);
  signature = mix_ptr(signature, lm_head->d_scales);
  signature = mix_ptr(signature, g_decode_final_norm.buf.ptr);
  signature = mix_ptr(signature, g_decode_rope.buf.ptr);

  const char *disable_graph = getenv("VIVA_DECODE_GRAPH_DISABLE");
  int graph_ok = !(disable_graph && disable_graph[0] && disable_graph[0] != '0') &&
                 !g_decode_graph_disabled &&
                 all_decode_weights_captureable(decode_layers, layer_count, lm_head);
  int rc = 0;
  if (graph_ok) {
    int graph_pos_key = -1;
    int graph_past_key = -1;
    DecodeGraphEntry *entry = find_decode_graph(graph_pos_key, graph_past_key, hidden, embed->vocab,
                                                layer_count, signature);
    if (entry && entry->exec) {
      cudaError_t launch_err = cudaGraphLaunch(entry->exec, g_block_stream);
      if (launch_err != cudaSuccess) {
        return make_block_error(env, "decode_graph_launch", -2000 - (int)launch_err);
      }
      entry->launches++;
      g_decode_graph_hits++;
      for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len + 1;
    } else {
      cudaError_t err = cudaStreamBeginCapture(g_block_stream, cudaStreamCaptureModeThreadLocal);
      if (err != cudaSuccess) {
        g_decode_graph_disabled = 1;
        g_decode_graph_failures++;
        g_decode_last_capture_error = err;
        graph_ok = 0;
      } else {
        g_full_decode_capture = 1;
        g_dyn_pos_ptr = (int *)b_pos_id.ptr;
        g_dyn_past_len_ptr = (int *)b_past_len_id.ptr;
        rc = run_decode_token_sequence(embed, token_id, decode_layers, layer_count,
                                       (float *)g_decode_final_norm.buf.ptr, lm_head,
                                       (float *)g_decode_rope.buf.ptr, pos);
        g_dyn_pos_ptr = NULL;
        g_dyn_past_len_ptr = NULL;
        g_full_decode_capture = 0;
        cudaGraph_t graph = NULL;
        err = cudaStreamEndCapture(g_block_stream, &graph);
        if (rc != 0 || err != cudaSuccess || !graph) {
          if (graph) cudaGraphDestroy(graph);
          for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len;
          (void)cudaGetLastError();
          g_decode_graph_disabled = 1;
          g_decode_graph_failures++;
          g_decode_last_capture_error = err;
          decode_graph_debug("capture_failed", rc, (int)err, past_len);
          graph_ok = 0;
        } else {
          g_decode_graph_captures++;
          if (rolling_decode_graph_compatible(hidden, embed->vocab, layer_count, signature)) {
            err = update_decode_rolling_exec(graph);
            if (err == cudaSuccess) {
              g_decode_graph_updates++;
              err = cudaGraphLaunch(g_decode_rolling_exec, g_block_stream);
              cudaGraphDestroy(graph);
              if (err != cudaSuccess) {
                return make_block_error(env, "decode_graph_launch", -2200 - (int)err);
              }
              decode_graph_debug("update_ok", g_decode_graph_count,
                                 g_decode_graph_updates, past_len);
            } else {
              g_decode_graph_update_failures++;
              if (err == cudaErrorGraphExecUpdateFailure) (void)cudaGetLastError();
              cudaGraphExec_t exec = NULL;
              err = cudaGraphInstantiate(&exec, graph, 0);
              if (err != cudaSuccess || !exec) {
                cudaGraphDestroy(graph);
                for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len;
                g_decode_graph_disabled = 1;
                g_decode_graph_failures++;
                g_decode_last_capture_error = err;
                decode_graph_debug("instantiate_failed", 0, (int)err, past_len);
                graph_ok = 0;
              } else {
                set_decode_rolling_exec(exec, hidden, embed->vocab, layer_count, signature);
                err = cudaGraphLaunch(exec, g_block_stream);
                cudaGraphDestroy(graph);
                if (err != cudaSuccess) {
                  return make_block_error(env, "decode_graph_launch", -2300 - (int)err);
                }
              }
            }
          } else {
            cudaGraphExec_t exec = NULL;
            err = cudaGraphInstantiate(&exec, graph, 0);
            if (err != cudaSuccess || !exec) {
              cudaGraphDestroy(graph);
              for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len;
              (void)cudaGetLastError();
              g_decode_graph_disabled = 1;
              g_decode_graph_failures++;
              g_decode_last_capture_error = err;
              decode_graph_debug("instantiate_failed", 0, (int)err, past_len);
              graph_ok = 0;
            } else {
              cudaGraphExec_t rolling_exec = NULL;
              cudaError_t roll_err = cudaGraphInstantiate(&rolling_exec, graph, 0);
              if (roll_err == cudaSuccess && rolling_exec) {
                set_decode_rolling_exec(rolling_exec, hidden, embed->vocab,
                                        layer_count, signature);
              }
              entry = alloc_decode_graph(graph_pos_key, graph_past_key, hidden, embed->vocab,
                                         layer_count, signature);
              if (entry) {
                entry->graph = graph;
                entry->exec = exec;
                entry->launches = 1;
              } else {
                cudaGraphDestroy(graph);
              }
              err = cudaGraphLaunch(exec, g_block_stream);
              if (err != cudaSuccess) {
                return make_block_error(env, "decode_graph_launch", -2100 - (int)err);
              }
              decode_graph_debug("capture_ok", g_decode_graph_count, g_decode_graph_captures,
                                 past_len);
            }
          }
        }
      }
    }
  }

  if (!graph_ok) {
    g_dyn_pos_ptr = NULL;
    g_dyn_past_len_ptr = NULL;
    const uint8_t *row = (const uint8_t *)embed->d_weight + (size_t)token_id * hidden16_bytes;
    if (cudaMemcpyAsync(b_hidden16.ptr, row, hidden16_bytes, cudaMemcpyDeviceToDevice,
                        g_block_stream) != cudaSuccess) {
      return make_error(env, "embedding_lookup_failed");
    }
    for (int i = 0; i < layer_count; ++i) {
      rc = run_decode_block_device(decode_layers[i].q, decode_layers[i].k,
                                   decode_layers[i].v, decode_layers[i].o,
                                   decode_layers[i].gate, decode_layers[i].up,
                                   decode_layers[i].down, decode_layers[i].qkv,
                                   decode_layers[i].gate_up, &norm1_bins[i],
                                   &norm2_bins[i], &rope_bin, decode_layers[i].cache, pos);
      if (rc != 0) return make_block_error(env, "decode_block", rc);
    }
    if (upload_binary_async(&b_norm1, &final_norm_bin) != 0)
      return make_error(env, "upload_final_norm_failed");
    int final_rc = run_helper_graph(BLOCK_GRAPH_NORM1, hidden, 0, 0, -1, -1, 0, 0);
    if (final_rc != 0) return make_block_error(env, "final_norm_gpu", final_rc);
    if ((rc = gemm_w8a16_dequant(lm_head, (uint16_t *)b_norm16.ptr, 1,
                                 (float *)b_logits.ptr)) != 0)
      return make_block_error(env, "lm_head", rc);
    int argmax_rc = vt_argmax_fp32_as_fp16((float *)b_logits.ptr, lm_head->out_features,
                                           (int *)b_argmax.ptr);
    if (argmax_rc != 0) return make_block_error(env, "argmax_gpu", argmax_rc);
  }

  int next_token = 0;
  if (cudaStreamSynchronize(g_block_stream) != cudaSuccess ||
      cudaMemcpy(&next_token, b_argmax.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return make_error(env, "download_argmax_failed");
  }
  vt_block_set_stream(NULL);
  cuda_fp8_dequant_set_stream(NULL);
  if (decode_graph_debug_enabled()) {
    fprintf(stderr,
            "[decode_graph] stats cached=%d captures=%u hits=%u updates=%u update_failures=%u failures=%u last_err=%d\n",
            g_decode_graph_count, g_decode_graph_captures, g_decode_graph_hits,
            g_decode_graph_updates, g_decode_graph_update_failures, g_decode_graph_failures,
            (int)g_decode_last_capture_error);
  }

  return make_ok(env, enif_make_int(env, next_token));
#endif
}

ERL_NIF_TERM nt_forward_decode_step_v2(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
#if defined(_WIN32) || defined(VIVA_NO_CUDA)
  (void)argc;
  (void)argv;
  return make_error(env, "cuda_not_available");
#else
  if (argc != 7) return make_error(env, "bad_arity");

  int token_id = 0, pos = 0;
  if (!enif_get_int(env, argv[0], &token_id) || token_id < 0) return make_error(env, "invalid_token");
  EmbeddingTable *embed = get_embedding_table(env, argv[1]);
  if (!embed || token_id >= embed->vocab || embed->hidden <= 0) return make_error(env, "invalid_embedding");
  ModelLayers *ml = NULL;
  if (!enif_get_resource(env, argv[2], MODEL_LAYERS_RES, (void **)&ml) || !ml) {
    return make_error(env, "invalid_model_layers");
  }
  if (!enif_get_int(env, argv[5], &pos) || pos < 0) return make_error(env, "invalid_pos");

  ErlNifBinary final_norm_bin, rope_bin;
  if (!enif_inspect_binary(env, argv[3], &final_norm_bin)) return make_error(env, "invalid_final_norm");
  PackedWeight *lm_head = get_packed_weight(env, argv[4]);
  if (!validate_fp8_weight(lm_head)) return make_error(env, "invalid_lm_head");
  if (!enif_inspect_binary(env, argv[6], &rope_bin)) return make_error(env, "invalid_rope");

  int hidden = embed->hidden;
  if (lm_head->in_features != hidden || final_norm_bin.size != (size_t)hidden * sizeof(float) ||
      rope_bin.size != (size_t)(LLAMA_HEAD_DIM / 2) * sizeof(float)) {
    return make_error(env, "shape_mismatch");
  }
  if (ml->layer_count <= 0 || ml->layer_count > DECODE_MAX_LAYERS) {
    return make_error(env, "invalid_model_layers");
  }

  int past_len = -1;
  for (int i = 0; i < ml->layer_count; ++i) {
    DecodeLayerParams *l = &ml->layers[i];
    if (!l->cache || l->cache->len < 0 || l->cache->len >= l->cache->max_seq) {
      return make_error(env, "invalid_kv_cache");
    }
    if (past_len < 0) {
      past_len = l->cache->len;
    } else if (l->cache->len != past_len) {
      return make_error(env, "kv_cache_len_mismatch");
    }
  }
  if (past_len < 0) return make_error(env, "invalid_past_len");

  if (ensure_block_lt() != 0) return make_error(env, "cuda_stream_init_failed");
  vt_block_set_stream((void *)g_block_stream);
  cuda_fp8_dequant_set_stream((void *)g_block_stream);

  size_t hidden16_bytes = (size_t)hidden * sizeof(uint16_t);
  size_t hidden32_bytes = (size_t)hidden * sizeof(float);
  if (ensure_block_buf(&b_hidden16, hidden16_bytes) != 0 ||
      ensure_block_buf(&b_hidden32, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_norm1, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_x2, hidden32_bytes) != 0 ||
      ensure_block_buf(&b_norm16, hidden16_bytes) != 0 ||
      ensure_block_buf(&b_logits, (size_t)lm_head->out_features * sizeof(float)) != 0 ||
      ensure_block_buf(&b_argmax, sizeof(int)) != 0 ||
      ensure_block_buf(&b_token_id, sizeof(int)) != 0 ||
      ensure_block_buf(&b_pos_id, sizeof(int)) != 0 ||
      ensure_block_buf(&b_past_len_id, sizeof(int)) != 0) {
    return make_error(env, "cuda_malloc_decode_failed");
  }

  int const_rc = ensure_decode_const(&g_decode_final_norm, &final_norm_bin);
  if (const_rc != 0) return make_block_error(env, "upload_final_norm", const_rc);
  const_rc = ensure_decode_const(&g_decode_rope, &rope_bin);
  if (const_rc != 0) return make_block_error(env, "upload_decode_rope", const_rc);

  if (cudaMemcpyAsync(b_token_id.ptr, &token_id, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess) {
    return make_error(env, "upload_token_id_failed");
  }
  if (cudaMemcpyAsync(b_pos_id.ptr, &pos, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess ||
      cudaMemcpyAsync(b_past_len_id.ptr, &past_len, sizeof(int), cudaMemcpyHostToDevice,
                      g_block_stream) != cudaSuccess) {
    return make_error(env, "upload_decode_scalar_failed");
  }

  uintptr_t signature = ml->signature;
  signature = mix_ptr(signature, embed->d_weight);
  signature = mix_ptr(signature, lm_head->d_weight);
  signature = mix_ptr(signature, lm_head->d_scales);
  signature = mix_ptr(signature, g_decode_final_norm.buf.ptr);
  signature = mix_ptr(signature, g_decode_rope.buf.ptr);

  DecodeLayerParams *decode_layers = ml->layers;
  int layer_count = ml->layer_count;
  const char *disable_graph = getenv("VIVA_DECODE_GRAPH_DISABLE");
  int graph_ok = !(disable_graph && disable_graph[0] && disable_graph[0] != '0') &&
                 !g_decode_graph_disabled &&
                 all_decode_weights_captureable(decode_layers, layer_count, lm_head);
  int rc = 0;
  if (graph_ok) {
    DecodeGraphEntry *entry = find_decode_graph(-1, -1, hidden, embed->vocab,
                                                layer_count, signature);
    if (entry && entry->exec) {
      cudaError_t launch_err = cudaGraphLaunch(entry->exec, g_block_stream);
      if (launch_err != cudaSuccess) {
        return make_block_error(env, "decode_graph_launch", -2000 - (int)launch_err);
      }
      entry->launches++;
      g_decode_graph_hits++;
      for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len + 1;
    } else {
      cudaError_t err = cudaStreamBeginCapture(g_block_stream, cudaStreamCaptureModeThreadLocal);
      if (err != cudaSuccess) {
        g_decode_graph_disabled = 1;
        g_decode_graph_failures++;
        g_decode_last_capture_error = err;
        graph_ok = 0;
      } else {
        g_full_decode_capture = 1;
        g_dyn_pos_ptr = (int *)b_pos_id.ptr;
        g_dyn_past_len_ptr = (int *)b_past_len_id.ptr;
        rc = run_decode_token_sequence(embed, token_id, decode_layers, layer_count,
                                       (float *)g_decode_final_norm.buf.ptr, lm_head,
                                       (float *)g_decode_rope.buf.ptr, pos);
        g_dyn_pos_ptr = NULL;
        g_dyn_past_len_ptr = NULL;
        g_full_decode_capture = 0;
        cudaGraph_t graph = NULL;
        err = cudaStreamEndCapture(g_block_stream, &graph);
        if (rc != 0 || err != cudaSuccess || !graph) {
          if (graph) cudaGraphDestroy(graph);
          for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len;
          (void)cudaGetLastError();
          g_decode_graph_disabled = 1;
          g_decode_graph_failures++;
          g_decode_last_capture_error = err;
          decode_graph_debug("capture_failed", rc, (int)err, past_len);
          graph_ok = 0;
        } else {
          g_decode_graph_captures++;
          cudaGraphExec_t exec = NULL;
          err = cudaGraphInstantiate(&exec, graph, 0);
          if (err != cudaSuccess || !exec) {
            cudaGraphDestroy(graph);
            for (int i = 0; i < layer_count; ++i) decode_layers[i].cache->len = past_len;
            (void)cudaGetLastError();
            g_decode_graph_disabled = 1;
            g_decode_graph_failures++;
            g_decode_last_capture_error = err;
            decode_graph_debug("instantiate_failed", 0, (int)err, past_len);
            graph_ok = 0;
          } else {
            cudaGraphExec_t rolling_exec = NULL;
            cudaError_t roll_err = cudaGraphInstantiate(&rolling_exec, graph, 0);
            if (roll_err == cudaSuccess && rolling_exec) {
              set_decode_rolling_exec(rolling_exec, hidden, embed->vocab,
                                      layer_count, signature);
            }
            entry = alloc_decode_graph(-1, -1, hidden, embed->vocab, layer_count, signature);
            if (entry) {
              entry->graph = graph;
              entry->exec = exec;
              entry->launches = 1;
            } else {
              cudaGraphDestroy(graph);
            }
            err = cudaGraphLaunch(exec, g_block_stream);
            if (err != cudaSuccess) {
              return make_block_error(env, "decode_graph_launch", -2100 - (int)err);
            }
            decode_graph_debug("capture_ok", g_decode_graph_count, g_decode_graph_captures,
                               past_len);
          }
        }
      }
    }
  }

  if (!graph_ok) {
    g_dyn_pos_ptr = NULL;
    g_dyn_past_len_ptr = NULL;
    rc = run_decode_token_sequence(embed, token_id, decode_layers, layer_count,
                                   (float *)g_decode_final_norm.buf.ptr, lm_head,
                                   (float *)g_decode_rope.buf.ptr, pos);
    if (rc != 0) return make_block_error(env, "decode_token", rc);
  }

  int next_token = 0;
  if (cudaStreamSynchronize(g_block_stream) != cudaSuccess ||
      cudaMemcpy(&next_token, b_argmax.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return make_error(env, "download_argmax_failed");
  }
  vt_block_set_stream(NULL);
  cuda_fp8_dequant_set_stream(NULL);
  if (decode_graph_debug_enabled()) {
    fprintf(stderr,
            "[decode_graph] stats cached=%d captures=%u hits=%u updates=%u update_failures=%u failures=%u last_err=%d\n",
            g_decode_graph_count, g_decode_graph_captures, g_decode_graph_hits,
            g_decode_graph_updates, g_decode_graph_update_failures, g_decode_graph_failures,
            (int)g_decode_last_capture_error);
  }

  return make_ok(env, enif_make_int(env, next_token));
#endif
}
