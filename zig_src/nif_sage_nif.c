/**
 * nif_sage_nif.c - SageAttention NIFs (INT8 QK^T + FP8 PV)
 * CPU path via per-block quantization + MKL, GPU path via cuBLAS SGEMM.
 */

#include "viva_nif.h"

#ifndef _WIN32

/** sage_available() -> true | false */
ERL_NIF_TERM nif_sage_available(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc; (void)argv;
  return sage_available() ? enif_make_atom(env, "true")
                          : enif_make_atom(env, "false");
}

/** sage_fp8_available() -> true | false */
ERL_NIF_TERM nif_sage_fp8_available(ErlNifEnv *env, int argc,
                                           const ERL_NIF_TERM argv[]) {
  (void)argc; (void)argv;
  return sage_fp8_available() ? enif_make_atom(env, "true")
                              : enif_make_atom(env, "false");
}

/** sage_quant_int8(Tensor, BlockSize) -> {ok, {QuantTensor, Scales}}
 *  Per-block INT8 quantization for SageAttention.
 */
ERL_NIF_TERM nif_sage_quant_int8(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  if (argc != 2) return enif_make_badarg(env);

  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_tensor");

  int block_size;
  if (!enif_get_int(env, argv[1], &block_size) || block_size <= 0)
    return make_error(env, "invalid_block_size");

  size_t n = (size_t)t->size;
  size_t num_blocks = (n + block_size - 1) / block_size;

  int8_t *int8_data = malloc(n);
  float *scales = malloc(num_blocks * sizeof(float));
  float *fp32_data = malloc(n * sizeof(float));

  if (!int8_data || !scales || !fp32_data) {
    free(int8_data); free(scales); free(fp32_data);
    return make_error(env, "alloc_failed");
  }

  for (size_t i = 0; i < n; i++) {
    fp32_data[i] = (float)t->data[i];
  }

  int result = quant_int8_per_block_cpu(int8_data, scales, fp32_data, n, block_size);
  free(fp32_data);

  if (result != 0) {
    free(int8_data); free(scales);
    return make_error(env, "quant_failed");
  }

  int quant_shape[1] = {(int)n};
  NativeTensor *quant_tensor = alloc_tensor_uninit(1, quant_shape);

  int scale_shape[1] = {(int)num_blocks};
  NativeTensor *scale_tensor = alloc_tensor_uninit(1, scale_shape);

  if (!quant_tensor || !scale_tensor) {
    free(int8_data); free(scales);
    if (quant_tensor) free(quant_tensor);
    if (scale_tensor) free(scale_tensor);
    return make_error(env, "alloc_tensor_failed");
  }

  for (size_t i = 0; i < n; i++) {
    quant_tensor->data[i] = (double)int8_data[i];
  }
  for (size_t i = 0; i < num_blocks; i++) {
    scale_tensor->data[i] = (double)scales[i];
  }

  free(int8_data);
  free(scales);

  ERL_NIF_TERM quant_ref = make_tensor_term(env, quant_tensor);
  ERL_NIF_TERM scale_ref = make_tensor_term(env, scale_tensor);

  return make_ok(env, enif_make_tuple2(env, quant_ref, scale_ref));
}

/** sage_softmax(Tensor, Dim) -> {ok, Tensor}
 *  Numerically stable softmax over last dimension.
 */
ERL_NIF_TERM nif_sage_softmax(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  if (argc != 2) return enif_make_badarg(env);

  NativeTensor *t = get_tensor(env, argv[0]);
  if (!t) return make_error(env, "invalid_tensor");

  int dim;
  if (!enif_get_int(env, argv[1], &dim) || dim <= 0)
    return make_error(env, "invalid_dim");

  size_t n = (size_t)t->size;
  if (n % dim != 0) return make_error(env, "size_not_divisible_by_dim");

  size_t batch = n / dim;

  float *in_fp32 = malloc(n * sizeof(float));
  float *out_fp32 = malloc(n * sizeof(float));

  if (!in_fp32 || !out_fp32) {
    free(in_fp32); free(out_fp32);
    return make_error(env, "alloc_failed");
  }

  for (size_t i = 0; i < n; i++) {
    in_fp32[i] = (float)t->data[i];
  }

  int result = softmax_cpu(out_fp32, in_fp32, batch, dim);
  free(in_fp32);

  if (result != 0) {
    free(out_fp32);
    return make_error(env, "softmax_failed");
  }

  NativeTensor *out = alloc_tensor_uninit(t->ndim, t->shape);
  if (!out) {
    free(out_fp32);
    return make_error(env, "alloc_tensor_failed");
  }

  for (size_t i = 0; i < n; i++) {
    out->data[i] = (double)out_fp32[i];
  }
  free(out_fp32);

  return make_ok(env, make_tensor_term(env, out));
}

/** sage_attention(Q, K, V, Batch, Heads, SeqQ, SeqK, HeadDim) -> {ok, Output}
 *  SageAttention CPU path: INT8 QK^T + FP32 softmax + FP32 PV.
 */
ERL_NIF_TERM nif_sage_attention(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  if (argc != 8) return enif_make_badarg(env);

  NativeTensor *q = get_tensor(env, argv[0]);
  NativeTensor *k = get_tensor(env, argv[1]);
  NativeTensor *v = get_tensor(env, argv[2]);

  if (!q || !k || !v) return make_error(env, "invalid_tensor");

  int batch, heads, seq_q, seq_k, head_dim;
  if (!enif_get_int(env, argv[3], &batch) ||
      !enif_get_int(env, argv[4], &heads) ||
      !enif_get_int(env, argv[5], &seq_q) ||
      !enif_get_int(env, argv[6], &seq_k) ||
      !enif_get_int(env, argv[7], &head_dim)) {
    return make_error(env, "invalid_dimensions");
  }

  size_t q_expected = (size_t)batch * heads * seq_q * head_dim;
  size_t k_expected = (size_t)batch * heads * seq_k * head_dim;
  size_t v_expected = k_expected;

  if ((size_t)q->size != q_expected) return make_error(env, "q_size_mismatch");
  if ((size_t)k->size != k_expected) return make_error(env, "k_size_mismatch");
  if ((size_t)v->size != v_expected) return make_error(env, "v_size_mismatch");

  float *q_fp32 = malloc(q_expected * sizeof(float));
  float *k_fp32 = malloc(k_expected * sizeof(float));
  float *v_fp32 = malloc(v_expected * sizeof(float));
  float *o_fp32 = malloc(q_expected * sizeof(float));

  if (!q_fp32 || !k_fp32 || !v_fp32 || !o_fp32) {
    free(q_fp32); free(k_fp32); free(v_fp32); free(o_fp32);
    return make_error(env, "alloc_failed");
  }

  for (size_t i = 0; i < q_expected; i++) q_fp32[i] = (float)q->data[i];
  for (size_t i = 0; i < k_expected; i++) k_fp32[i] = (float)k->data[i];
  for (size_t i = 0; i < v_expected; i++) v_fp32[i] = (float)v->data[i];

  float sm_scale = 1.0f / sqrtf((float)head_dim);

  int result = sage_attention_cpu(o_fp32, q_fp32, k_fp32, v_fp32,
                                   batch, heads, seq_q, seq_k, head_dim, sm_scale);

  free(q_fp32); free(k_fp32); free(v_fp32);

  if (result != 0) {
    free(o_fp32);
    return make_error(env, "sage_attention_failed");
  }

  int out_shape[4] = {batch, heads, seq_q, head_dim};
  NativeTensor *out = alloc_tensor_uninit(4, out_shape);
  if (!out) {
    free(o_fp32);
    return make_error(env, "alloc_tensor_failed");
  }

  for (size_t i = 0; i < q_expected; i++) {
    out->data[i] = (double)o_fp32[i];
  }
  free(o_fp32);

  return make_ok(env, make_tensor_term(env, out));
}

/** sage_attention_ct(Q, K, V, Batch, Heads, SeqQ, SeqK, HeadDim) -> {ok, Output}
 *  SageAttention GPU path via cuBLAS SGEMM on CudaTensors.
 */
ERL_NIF_TERM sage_attention_ct(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  if (argc != 8) return enif_make_badarg(env);

  if (!cuda_available())
    return make_error(env, "cuda_not_available");

  CudaTensor *q, *k, *v;
  if (!enif_get_resource(env, argv[0], CUDA_TENSOR_RESOURCE, (void**)&q) ||
      !enif_get_resource(env, argv[1], CUDA_TENSOR_RESOURCE, (void**)&k) ||
      !enif_get_resource(env, argv[2], CUDA_TENSOR_RESOURCE, (void**)&v)) {
    return make_error(env, "invalid_cuda_tensor");
  }

  int batch, heads, seq_q, seq_k, head_dim;
  if (!enif_get_int(env, argv[3], &batch) ||
      !enif_get_int(env, argv[4], &heads) ||
      !enif_get_int(env, argv[5], &seq_q) ||
      !enif_get_int(env, argv[6], &seq_k) ||
      !enif_get_int(env, argv[7], &head_dim)) {
    return make_error(env, "invalid_dimensions");
  }

  size_t q_expected = (size_t)batch * heads * seq_q * head_dim;
  size_t k_expected = (size_t)batch * heads * seq_k * head_dim;
  size_t v_expected = k_expected;

  if ((size_t)q->size != q_expected) return make_error(env, "q_size_mismatch");
  if ((size_t)k->size != k_expected) return make_error(env, "k_size_mismatch");
  if ((size_t)v->size != v_expected) return make_error(env, "v_size_mismatch");

  CudaTensor *out = (CudaTensor *)enif_alloc_resource(CUDA_TENSOR_RESOURCE, sizeof(CudaTensor));
  if (!out) return make_error(env, "alloc_resource_failed");

  out->d_data = cuda_tensor_alloc(q_expected);
  if (!out->d_data) {
    enif_release_resource(out);
    return make_error(env, "gpu_alloc_failed");
  }

  out->ndim = 4;
  out->shape = (int *)malloc(4 * sizeof(int));
  if (!out->shape) {
    cuda_tensor_free(out->d_data);
    enif_release_resource(out);
    return make_error(env, "alloc_shape_failed");
  }
  out->shape[0] = batch;
  out->shape[1] = heads;
  out->shape[2] = seq_q;
  out->shape[3] = head_dim;
  out->size = (int)q_expected;

  float sm_scale = 1.0f / sqrtf((float)head_dim);

  int result = sage_attention_gpu(out->d_data, q->d_data, k->d_data, v->d_data,
                                   batch, heads, seq_q, seq_k, head_dim, sm_scale);

  if (result != 0) {
    cuda_tensor_free(out->d_data);
    free(out->shape);
    enif_release_resource(out);
    char err_msg[64];
    snprintf(err_msg, sizeof(err_msg), "sage_attention_gpu_failed_%d", result);
    return make_error(env, err_msg);
  }

  return make_ok(env, make_cuda_tensor_term(env, out));
}

#endif /* !_WIN32 */
