#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

static cudaStream_t g_vt_block_stream = 0;

void vt_block_set_stream(void *stream) {
  g_vt_block_stream = (cudaStream_t)stream;
}

__global__ void fp16_to_fp32_cast_kernel(const uint16_t *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __half2float(reinterpret_cast<const __half *>(in)[i]);
}

__global__ void fp32_to_fp16_cast_kernel(const float *in, uint16_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) reinterpret_cast<__half *>(out)[i] = __float2half_rn(in[i]);
}

__global__ void rmsnorm_fp32_kernel(const float *x, const float *gamma, float *out,
                                    int n, float eps) {
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = x[i];
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }

  float inv = rsqrtf(partial[0] / (float)n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = x[i] * inv * gamma[i];
  }
}

__global__ void residual_add_fp32_kernel(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void rope_apply_fp32_kernel(float *x, const float *freqs, int pos,
                                       int num_heads, int head_dim) {
  int half = head_dim >> 1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * half;
  if (i >= total) return;

  int head = i / half;
  int j = i - head * half;
  int base = head * head_dim;
  float a = x[base + j];
  float b = x[base + j + half];
  float angle = (float)pos * freqs[j];
  float c = cosf(angle);
  float s = sinf(angle);
  x[base + j] = a * c - b * s;
  x[base + j + half] = a * s + b * c;
}

__global__ void silu_mul_fp32_kernel(const float *gate, const float *up, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
  }
}

__global__ void argmax_fp32_as_fp16_kernel(const float *x, int n, int *out_idx) {
  __shared__ float vals[256];
  __shared__ int idxs[256];
  int tid = threadIdx.x;
  float best = -INFINITY;
  int best_idx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    float v = __half2float(__float2half_rn(x[i]));
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }
  vals[tid] = best;
  idxs[tid] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other = vals[tid + stride];
      int other_idx = idxs[tid + stride];
      if (other > vals[tid] || (other == vals[tid] && other_idx < idxs[tid])) {
        vals[tid] = other;
        idxs[tid] = other_idx;
      }
    }
    __syncthreads();
  }
  if (tid == 0) *out_idx = idxs[0];
}

__global__ void embedding_lookup_fp16_kernel(const uint16_t *table, const int *token_id,
                                             uint16_t *out, int hidden, int vocab) {
  int tok = *token_id;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok < 0 || tok >= vocab || i >= hidden) return;
  out[i] = table[(size_t)tok * (size_t)hidden + (size_t)i];
}

__global__ void gqa_attn_naive_single_token_kernel(const float *q, const float *new_k,
                                                   const float *new_v,
                                                   const uint16_t *k_cache,
                                                   const uint16_t *v_cache,
                                                   float *out, int past_len,
                                                   int num_heads, int num_kv_heads,
                                                   int head_dim) {
  int qh = blockIdx.x;
  if (qh >= num_heads || threadIdx.x != 0) return;

  int q_per_kv = num_heads / num_kv_heads;
  int kvh = qh / q_per_kv;
  int seq_len = past_len + 1;
  float scale = rsqrtf((float)head_dim);
  float max_score = -INFINITY;

  extern __shared__ float scratch[];
  float *scores = scratch;

  for (int t = 0; t < seq_len; ++t) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      float qv = q[qh * head_dim + d];
      float kv;
      if (t < past_len) {
        size_t idx = ((size_t)t * num_kv_heads + kvh) * head_dim + d;
        kv = __half2float(reinterpret_cast<const __half *>(k_cache)[idx]);
      } else {
        kv = new_k[kvh * head_dim + d];
      }
      dot += qv * kv;
    }
    float score = dot * scale;
    scores[t] = score;
    if (score > max_score) max_score = score;
  }

  float denom = 0.0f;
  for (int t = 0; t < seq_len; ++t) {
    float e = expf(scores[t] - max_score);
    scores[t] = e;
    denom += e;
  }
  float inv_denom = 1.0f / denom;

  for (int d = 0; d < head_dim; ++d) {
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      float vv;
      if (t < past_len) {
        size_t idx = ((size_t)t * num_kv_heads + kvh) * head_dim + d;
        vv = __half2float(reinterpret_cast<const __half *>(v_cache)[idx]);
      } else {
        vv = new_v[kvh * head_dim + d];
      }
      acc += scores[t] * inv_denom * vv;
    }
    out[qh * head_dim + d] = acc;
  }
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
  uint32_t sign = ((uint32_t)x & 0x80u) << 24;
  uint32_t exp = ((uint32_t)x >> 3) & 0x0fu;
  uint32_t mant = (uint32_t)x & 0x07u;

  if (exp == 0) {
    float v = (float)mant * 0.001953125f;
    return sign ? -v : v;
  }

  if (exp == 0x0fu && mant == 0x07u) {
    return sign ? -448.0f : 448.0f;
  }

  union {
    uint32_t u;
    float f;
  } bits;
  bits.u = sign | ((exp + 120u) << 23) | (mant << 20);
  return bits.f;
}

__global__ void w8a16_mmv_blocked_k16_kernel(const uint8_t *weight,
                                             const float *scales,
                                             const uint16_t *input,
                                             float *out,
                                             int in_features,
                                             int num_blocks) {
  int out_col = blockIdx.x;
  int lane = threadIdx.x & 31;
  const uint8_t *w_row = weight + (size_t)out_col * (size_t)in_features;
  const float *s_row = scales + (size_t)out_col * (size_t)num_blocks;
  const __half *x = reinterpret_cast<const __half *>(input);

  float acc = 0.0f;
  for (int b = lane; b < num_blocks; b += 32) {
    int base = b * 16;
    float scale = s_row[b];
    float block_acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      float wv = fp8_e4m3_to_float(w_row[base + j]);
      float xv = __half2float(x[base + j]);
      block_acc = fmaf(wv, xv, block_acc);
    }
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) out[out_col] = acc;
}

__global__ void gqa_attn_flash_single_token_kernel(const float *q, const float *new_k,
                                                   const float *new_v,
                                                   const uint16_t *k_cache,
                                                   const uint16_t *v_cache,
                                                   float *out, int past_len,
                                                   int num_heads, int num_kv_heads,
                                                   int head_dim) {
  int qh = blockIdx.x;
  int lane = threadIdx.x & 31;
  if (qh >= num_heads || threadIdx.x >= 32 || head_dim != 64) return;

  int q_per_kv = num_heads / num_kv_heads;
  int kvh = qh / q_per_kv;
  int seq_len = past_len + 1;
  int d0 = lane << 1;
  int d1 = d0 + 1;
  int q_base = qh * head_dim;
  int kv_base = kvh * head_dim;
  float q0 = q[q_base + d0];
  float q1 = q[q_base + d1];
  float scale = rsqrtf((float)head_dim);

  float m = -INFINITY;
  float l = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    float k0, k1, v0, v1;
    if (t < past_len) {
      size_t pair_idx = (((size_t)t * num_kv_heads + kvh) * head_dim + d0) >> 1;
      __half2 k2 = reinterpret_cast<const __half2 *>(k_cache)[pair_idx];
      __half2 v2 = reinterpret_cast<const __half2 *>(v_cache)[pair_idx];
      float2 kf = __half22float2(k2);
      float2 vf = __half22float2(v2);
      k0 = kf.x; k1 = kf.y;
      v0 = vf.x; v1 = vf.y;
    } else {
      k0 = new_k[kv_base + d0];
      k1 = new_k[kv_base + d1];
      v0 = new_v[kv_base + d0];
      v1 = new_v[kv_base + d1];
    }

    float dot = warp_sum(q0 * k0 + q1 * k1);
    dot = __shfl_sync(0xffffffffu, dot, 0);
    float score = dot * scale;
    float new_m = fmaxf(m, score);
    float alpha = expf(m - new_m);
    float beta = expf(score - new_m);
    acc0 = acc0 * alpha + beta * v0;
    acc1 = acc1 * alpha + beta * v1;
    l = l * alpha + beta;
    m = new_m;
  }

  float inv_l = 1.0f / l;
  out[q_base + d0] = acc0 * inv_l;
  out[q_base + d1] = acc1 * inv_l;
}

int vt_fp16_to_fp32_cast(const void *in, float *out, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  fp16_to_fp32_cast_kernel<<<grid, block, 0, g_vt_block_stream>>>((const uint16_t *)in, out, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_fp32_to_fp16_cast(const float *in, void *out, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  fp32_to_fp16_cast_kernel<<<grid, block, 0, g_vt_block_stream>>>(in, (uint16_t *)out, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_rmsnorm_fp32(const float *x, const float *gamma, float *out, int n, float eps) {
  rmsnorm_fp32_kernel<<<1, 256, 0, g_vt_block_stream>>>(x, gamma, out, n, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_residual_add_fp32(const float *a, const float *b, float *out, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  residual_add_fp32_kernel<<<grid, block, 0, g_vt_block_stream>>>(a, b, out, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_rope_apply_fp32(float *x, const float *freqs, int pos, int num_heads, int head_dim) {
  int total = num_heads * (head_dim / 2);
  int block = 128;
  int grid = (total + block - 1) / block;
  rope_apply_fp32_kernel<<<grid, block, 0, g_vt_block_stream>>>(x, freqs, pos, num_heads, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_silu_mul_fp32(const float *gate, const float *up, float *out, int n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  silu_mul_fp32_kernel<<<grid, block, 0, g_vt_block_stream>>>(gate, up, out, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_gqa_attn_single_token(const float *q, const float *new_k, const float *new_v,
                              const void *k_cache, const void *v_cache, float *out,
                              int past_len, int num_heads, int num_kv_heads,
                              int head_dim) {
  int seq_len = past_len + 1;
  size_t shared = (size_t)seq_len * sizeof(float);
  if (head_dim == 64) {
    (void)seq_len;
    (void)shared;
    gqa_attn_flash_single_token_kernel<<<num_heads, 32, 0, g_vt_block_stream>>>(
        q, new_k, new_v, (const uint16_t *)k_cache, (const uint16_t *)v_cache, out,
        past_len, num_heads, num_kv_heads, head_dim);
  } else {
    gqa_attn_naive_single_token_kernel<<<num_heads, 1, shared, g_vt_block_stream>>>(
        q, new_k, new_v, (const uint16_t *)k_cache, (const uint16_t *)v_cache, out,
        past_len, num_heads, num_kv_heads, head_dim);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_w8a16_mmv_blocked_k16(const void *d_weight, const float *d_scales,
                             const void *d_input, float *d_out,
                             int in_features, int out_features) {
  if (!d_weight || !d_scales || !d_input || !d_out || in_features <= 0 ||
      out_features <= 0 || (in_features % 16) != 0) {
    return -2;
  }
  int num_blocks = in_features / 16;
  w8a16_mmv_blocked_k16_kernel<<<out_features, 32, 0, g_vt_block_stream>>>(
      (const uint8_t *)d_weight, d_scales, (const uint16_t *)d_input, d_out,
      in_features, num_blocks);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_argmax_fp32_as_fp16(const float *x, int n, int *out_idx) {
  if (!x || !out_idx || n <= 0) return -2;
  argmax_fp32_as_fp16_kernel<<<1, 256, 0, g_vt_block_stream>>>(x, n, out_idx);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int vt_embedding_lookup_fp16(const void *table, const int *token_id, void *out,
                             int hidden, int vocab) {
  if (!table || !token_id || !out || hidden <= 0 || vocab <= 0) return -2;
  int block = 256;
  int grid = (hidden + block - 1) / block;
  embedding_lookup_fp16_kernel<<<grid, block, 0, g_vt_block_stream>>>(
      (const uint16_t *)table, token_id, (uint16_t *)out, hidden, vocab);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}
