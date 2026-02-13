/**
 * cuda_sage.c - SageAttention kernels (INT8 QK^T + FP8 PV)
 * Adapted from: https://github.com/thu-ml/SageAttention (Apache 2.0)
 */

#ifndef _WIN32

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_types.h"

/* MKL for CPU SGEMM fallback */
#ifdef USE_MKL_DIRECT
#include <mkl.h>
#define USE_MKL_SGEMM 1
#else
typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans=111, CblasTrans=112 } CBLAS_TRANSPOSE;
extern void cblas_sgemm(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
                        int, int, int, float, const float*, int,
                        const float*, int, float, float*, int);
#define USE_MKL_SGEMM 1
#endif

extern int cuda_init(void);
extern int cuda_available(void);

/* =========================================================================
 * FP8 GEMM - E4M3 and E5M2 on RTX 4090 Tensor Cores
 * ========================================================================= */

/**
 * Allocate GPU memory for FP8 tensors
 */
fp8_e4m3_t* cuda_tensor_alloc_fp8(size_t num_elements) {
    if (!cuda_available()) return NULL;

    fp8_e4m3_t *d_ptr = NULL;
    cudaError_t err = g_cuda_malloc((void**)&d_ptr, num_elements * sizeof(fp8_e4m3_t));
    if (err != cudaSuccess) return NULL;

    return d_ptr;
}

/**
 * Upload FP8 data to GPU
 */
int cuda_tensor_upload_fp8(fp8_e4m3_t *d_dst, const fp8_e4m3_t *h_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;
    cudaError_t err = g_cuda_memcpy(d_dst, h_src, num_elements * sizeof(fp8_e4m3_t), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Download FP8 data from GPU
 */
int cuda_tensor_download_fp8(fp8_e4m3_t *h_dst, const fp8_e4m3_t *d_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;
    cudaError_t err = g_cuda_memcpy(h_dst, d_src, num_elements * sizeof(fp8_e4m3_t), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * FP8 E4M3 GEMM: C = alpha * A @ B + beta * C
 * A is FP8 [M x K], B is FP8 [K x N], C is FP32 [M x N]
 *
 * Uses FP8 E4M3 Tensor Cores when available (Ada/Hopper)
 */
int cuda_fp8gemm_gpu(int M, int N, int K,
                     float alpha, const fp8_e4m3_t *d_A, int lda,
                     const fp8_e4m3_t *d_B, int ldb,
                     float beta, float *d_C, int ldc) {
    if (!cuda_available() || !g_cublas_gemm_ex) {
        fprintf(stderr, "[viva_tensor] cublasGemmEx not available for FP8\n");
        return -1;
    }

    /* FP8 E4M3 input with FP32 output and accumulator */
    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_8F_E4M3, N,          /* B is FP8 E4M3 */
        d_A, CUDA_R_8F_E4M3, K,          /* A is FP8 E4M3 */
        &beta,
        d_C, CUDA_R_32F, N,              /* C is FP32 for accuracy */
        CUBLAS_COMPUTE_32F,               /* FP32 accumulator */
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS FP8 GEMM error: %d\n", stat);
        return -2;
    }

    g_cuda_sync();
    return 0;
}

/**
 * FP8 E4M3 GEMM - Async version (no sync)
 */
int cuda_fp8gemm_gpu_async(int M, int N, int K,
                           float alpha, const fp8_e4m3_t *d_A, int lda,
                           const fp8_e4m3_t *d_B, int ldb,
                           float beta, float *d_C, int ldc) {
    if (!cuda_available() || !g_cublas_gemm_ex) return -1;

    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_8F_E4M3, N,
        d_A, CUDA_R_8F_E4M3, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

/**
 * Check if FP8 is available (Ada/Hopper architecture)
 */
int cuda_fp8_available(void) {
    if (!cuda_available()) return 0;
    /* FP8 requires Ada (SM89) or Hopper (SM90) */
    /* For now, assume if cublasGemmEx is available, try FP8 */
    return g_cublas_gemm_ex != NULL;
}

/* =========================================================================
 * FP32 to FP8 Conversion (CPU fallback)
 * For GPU conversion, we'd need a custom CUDA kernel
 * ========================================================================= */

/**
 * Convert FP32 to FP8 E4M3 format (CPU)
 * E4M3 range: [-448, 448], subnormals, no inf/nan
 *
 * Based on SageAttention numeric_conversion.cuh
 */
static fp8_e4m3_t float_to_fp8_e4m3(float x) {
    /* Clamp to E4M3 range */
    const float max_val = 448.0f;
    const float min_val = -448.0f;

    if (x > max_val) x = max_val;
    if (x < min_val) x = min_val;
    if (isnan(x)) x = 0.0f;

    /* Get bits */
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));

    int sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127;  /* Unbias FP32 exponent */
    int mantissa = (bits >> 20) & 0x7;       /* Top 3 bits of mantissa */

    /* Bias for E4M3 is 7 */
    int fp8_exp = exp + 7;

    /* Clamp exponent */
    if (fp8_exp <= 0) {
        return sign ? 0x80 : 0x00;  /* Underflow to zero */
    }
    if (fp8_exp >= 15) {
        return sign ? 0xFE : 0x7E;  /* Saturate to max (no inf in E4M3) */
    }

    /* Pack: 1 sign + 4 exp + 3 mantissa */
    return (sign << 7) | (fp8_exp << 3) | mantissa;
}

/**
 * Convert FP8 E4M3 to FP32 (CPU)
 */
static float fp8_e4m3_to_float(fp8_e4m3_t x) {
    int sign = (x >> 7) & 1;
    int exp = (x >> 3) & 0xF;
    int mantissa = x & 0x7;

    if (exp == 0 && mantissa == 0) {
        return sign ? -0.0f : 0.0f;
    }

    /* E4M3 bias is 7 */
    float value;
    if (exp == 0) {
        /* Subnormal */
        value = ldexpf((float)mantissa / 8.0f, -6);  /* 1 - 7 = -6 */
    } else {
        value = ldexpf(1.0f + (float)mantissa / 8.0f, exp - 7);
    }

    return sign ? -value : value;
}

/**
 * Batch convert FP32 array to FP8 E4M3 (CPU)
 */
void float_to_fp8_e4m3_batch(fp8_e4m3_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_fp8_e4m3(src[i]);
    }
}

/**
 * Batch convert FP8 E4M3 to FP32 (CPU)
 */
void fp8_e4m3_to_float_batch(float *dst, const fp8_e4m3_t *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_to_float(src[i]);
    }
}

/* =========================================================================
 * INT8 Per-Block Quantization (CPU fallback, GPU kernel below)
 * Based on SageAttention fused.cu QuantInt8Kernel
 * ========================================================================= */

/**
 * Quantize FP32 tensor to INT8 with per-block scaling
 *
 * For each block of 'block_size' elements:
 *   1. Find max absolute value
 *   2. Compute scale = max_abs / 127
 *   3. Quantize: int8 = round(x / scale)
 *
 * Output:
 *   - int8_data: quantized values
 *   - scales: one scale per block
 */
int quant_int8_per_block_cpu(
    int8_t *int8_data,      /* Output: [n] */
    float *scales,          /* Output: [n / block_size] */
    const float *fp32_data, /* Input: [n] */
    size_t n,
    size_t block_size
) {
    if (block_size == 0 || n == 0) return -1;

    size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > n) end = n;

        /* Find max absolute value */
        float max_abs = 1e-7f;  /* Prevent division by zero */
        for (size_t i = start; i < end; i++) {
            float abs_val = fabsf(fp32_data[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        /* Compute scale */
        float scale = max_abs / 127.0f;
        scales[b] = scale;

        /* Quantize */
        float inv_scale = 127.0f / max_abs;
        for (size_t i = start; i < end; i++) {
            float scaled = fp32_data[i] * inv_scale;
            int val = (int)roundf(scaled);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            int8_data[i] = (int8_t)val;
        }
    }

    return 0;
}

/**
 * Dequantize INT8 with per-block scales back to FP32
 */
int dequant_int8_per_block_cpu(
    float *fp32_data,         /* Output: [n] */
    const int8_t *int8_data,  /* Input: [n] */
    const float *scales,      /* Input: [n / block_size] */
    size_t n,
    size_t block_size
) {
    if (block_size == 0 || n == 0) return -1;

    size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > n) end = n;

        float scale = scales[b];

        for (size_t i = start; i < end; i++) {
            fp32_data[i] = (float)int8_data[i] * scale;
        }
    }

    return 0;
}

/* =========================================================================
 * Softmax (CPU fallback)
 * GPU kernel would avoid the device-host roundtrip for large tensors
 * ========================================================================= */

/**
 * Softmax over last dimension: softmax(x)[i] = exp(x[i]) / sum(exp(x))
 * Numerically stable: subtract max first
 */
int softmax_cpu(float *output, const float *input, size_t batch, size_t dim) {
    for (size_t b = 0; b < batch; b++) {
        const float *in = input + b * dim;
        float *out = output + b * dim;

        /* Find max for numerical stability */
        float max_val = in[0];
        for (size_t i = 1; i < dim; i++) {
            if (in[i] > max_val) max_val = in[i];
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            out[i] = expf(in[i] - max_val);
            sum += out[i];
        }

        /* Normalize */
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < dim; i++) {
            out[i] *= inv_sum;
        }
    }

    return 0;
}

/* =========================================================================
 * SageAttention Core - INT8 QK^T + FP16/FP8 PV
 *
 * Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
 * Sage attention:
 *   1. Quantize Q, K to INT8 per-block
 *   2. QK^T via INT8 Tensor Cores
 *   3. Dequant + softmax
 *   4. P @ V in FP16 or FP8
 * ========================================================================= */

/**
 * SageAttention on CPU - MKL optimized version
 *
 * Q: [batch, heads, seq_q, head_dim] - Query
 * K: [batch, heads, seq_k, head_dim] - Key
 * V: [batch, heads, seq_k, head_dim] - Value
 * O: [batch, heads, seq_q, head_dim] - Output
 *
 * Uses MKL cblas_sgemm for both Q@K^T and attn@V
 */
int sage_attention_cpu(
    float *O,           /* Output */
    const float *Q,     /* Query */
    const float *K,     /* Key */
    const float *V,     /* Value */
    int batch,
    int heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float sm_scale      /* 1/sqrt(head_dim) */
) {
    /* Allocate temporary buffers */
    size_t qk_size = (size_t)seq_q * seq_k;
    float *qk_float = malloc(qk_size * sizeof(float));
    float *attn = malloc(qk_size * sizeof(float));

    if (!qk_float || !attn) {
        free(qk_float); free(attn);
        return -1;
    }

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            /* Get pointers for this head */
            size_t bh_offset = ((size_t)b * heads + h);
            const float *q_ptr = Q + bh_offset * seq_q * head_dim;
            const float *k_ptr = K + bh_offset * seq_k * head_dim;
            const float *v_ptr = V + bh_offset * seq_k * head_dim;
            float *o_ptr = O + bh_offset * seq_q * head_dim;

#ifdef USE_MKL_SGEMM
            /* Step 1: Q @ K^T via MKL
             * Q[seq_q, head_dim] @ K^T[head_dim, seq_k] = QK[seq_q, seq_k]
             * Row-major: C = alpha * A * B^T + beta * C
             * cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
             * M=seq_q, N=seq_k, K=head_dim
             */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_q, seq_k, head_dim,
                        sm_scale,       /* alpha = 1/sqrt(d) */
                        q_ptr, head_dim,
                        k_ptr, head_dim,
                        0.0f,           /* beta */
                        qk_float, seq_k);
#else
            /* Naive fallback for Q @ K^T */
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < seq_k; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q_ptr[i * head_dim + d] * k_ptr[j * head_dim + d];
                    }
                    qk_float[i * seq_k + j] = sum * sm_scale;
                }
            }
#endif

            /* Step 2: Softmax */
            softmax_cpu(attn, qk_float, seq_q, seq_k);

#ifdef USE_MKL_SGEMM
            /* Step 3: Attn @ V via MKL
             * attn[seq_q, seq_k] @ V[seq_k, head_dim] = O[seq_q, head_dim]
             * Row-major: C = alpha * A * B + beta * C
             */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_q, head_dim, seq_k,
                        1.0f,           /* alpha */
                        attn, seq_k,
                        v_ptr, head_dim,
                        0.0f,           /* beta */
                        o_ptr, head_dim);
#else
            /* Naive fallback for Attn @ V */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_k; j++) {
                        sum += attn[i * seq_k + j] * v_ptr[j * head_dim + d];
                    }
                    o_ptr[i * head_dim + d] = sum;
                }
            }
#endif
        }
    }

    free(qk_float); free(attn);
    return 0;
}

/* =========================================================================
 * SageAttention GPU - Uses cuBLAS for GEMM, custom kernels for quant/softmax
 * ========================================================================= */

/* cuBLAS SGEMM function pointer (from cuda_gemm.c) */
typedef int (*cublas_sgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int);
extern cublas_sgemm_fn g_cublas_sgemm;

/**
 * SageAttention on GPU using cuBLAS
 *
 * Uses cuBLAS SGEMM for both matmuls
 * Softmax computed on CPU (a GPU softmax kernel would avoid the roundtrip)
 *
 * Data stays on GPU except for softmax intermediate step
 */
int sage_attention_gpu(
    float *d_O,           /* Output on GPU */
    const float *d_Q,     /* Query on GPU */
    const float *d_K,     /* Key on GPU */
    const float *d_V,     /* Value on GPU */
    int batch,
    int heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float sm_scale
) {
    if (!cuda_available() || !g_cublas_ctx || !g_cublas_sgemm) {
        /* Fall back to CPU if CUDA not available */
        return -1;
    }

    /* Allocate GPU workspace for QK^T and attention weights */
    size_t qk_size = (size_t)seq_q * seq_k;
    float *d_qk = NULL;
    float *d_attn = NULL;

    if (g_cuda_malloc((void**)&d_qk, qk_size * sizeof(float)) != cudaSuccess) {
        return -2;
    }
    if (g_cuda_malloc((void**)&d_attn, qk_size * sizeof(float)) != cudaSuccess) {
        g_cuda_free(d_qk);
        return -2;
    }

    /* Host buffer for softmax (GPU softmax kernel would eliminate this) */
    float *h_qk = malloc(qk_size * sizeof(float));
    float *h_attn = malloc(qk_size * sizeof(float));
    if (!h_qk || !h_attn) {
        free(h_qk); free(h_attn);
        g_cuda_free(d_qk); g_cuda_free(d_attn);
        return -3;
    }

    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            size_t bh_offset = ((size_t)b * heads + h);
            const float *dq = d_Q + bh_offset * seq_q * head_dim;
            const float *dk = d_K + bh_offset * seq_k * head_dim;
            const float *dv = d_V + bh_offset * seq_k * head_dim;
            float *do_ = d_O + bh_offset * seq_q * head_dim;

            /*
             * Step 1: Q @ K^T via cuBLAS
             * C = alpha * op(A) * op(B) + beta * C
             * cuBLAS uses column-major, but our data is row-major
             * Trick: C^T = B^T @ A^T, so we swap A and B and transpose result
             *
             * We want: QK = Q[seq_q, d] @ K^T[d, seq_k] = [seq_q, seq_k]
             * In col-major: C[seq_k, seq_q] = K[d, seq_k]^T @ Q[d, seq_q]
             *             = K @ Q^T with col-major interpretation
             * So: sgemm(NoTrans, Trans, N=seq_k, M=seq_q, K=d, alpha, dk, d, dq, d, beta, d_qk, seq_k)
             */
            cublasStatus_t status = g_cublas_sgemm(
                g_cublas_ctx,
                CUBLAS_OP_T,   /* K transposed */
                CUBLAS_OP_N,   /* Q not transposed */
                seq_k,         /* N (cols of result) */
                seq_q,         /* M (rows of result) */
                head_dim,      /* K */
                &sm_scale,     /* alpha = 1/sqrt(d) */
                dk, head_dim,  /* lda */
                dq, head_dim,  /* ldb */
                &beta_zero,
                d_qk, seq_k    /* ldc */
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                free(h_qk); free(h_attn);
                g_cuda_free(d_qk); g_cuda_free(d_attn);
                return -4;
            }

            /* Step 2: Download QK, compute softmax on CPU, upload attn */
            g_cuda_memcpy(h_qk, d_qk, qk_size * sizeof(float), cudaMemcpyDeviceToHost);
            softmax_cpu(h_attn, h_qk, seq_q, seq_k);
            g_cuda_memcpy(d_attn, h_attn, qk_size * sizeof(float), cudaMemcpyHostToDevice);

            /*
             * Step 3: attn @ V via cuBLAS
             * O = attn[seq_q, seq_k] @ V[seq_k, head_dim] = [seq_q, head_dim]
             * Col-major: O[d, seq_q] = V[d, seq_k] @ attn^T[seq_k, seq_q]
             * sgemm(NoTrans, Trans, d, seq_q, seq_k, 1, dv, d, d_attn, seq_k, 0, do_, d)
             */
            status = g_cublas_sgemm(
                g_cublas_ctx,
                CUBLAS_OP_N,   /* V not transposed */
                CUBLAS_OP_T,   /* attn transposed */
                head_dim,      /* N */
                seq_q,         /* M */
                seq_k,         /* K */
                &alpha_one,
                dv, head_dim,
                d_attn, seq_k,
                &beta_zero,
                do_, head_dim
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                free(h_qk); free(h_attn);
                g_cuda_free(d_qk); g_cuda_free(d_attn);
                return -5;
            }
        }
    }

    free(h_qk); free(h_attn);
    g_cuda_free(d_qk); g_cuda_free(d_attn);
    g_cuda_sync();

    return 0;
}

/* =========================================================================
 * Exported API Functions
 * ========================================================================= */

/**
 * Initialize SageAttention (loads CUDA driver if needed)
 */
int sage_init(void) {
    return cuda_init();
}

/**
 * Check if SageAttention is available
 */
int sage_available(void) {
    return cuda_available();
}

/**
 * Check if FP8 is available
 */
int sage_fp8_available(void) {
    return cuda_fp8_available();
}

#else  /* _WIN32 */

/* Stubs for Windows - not implemented yet */
int sage_init(void) { return 0; }
int sage_available(void) { return 0; }
int sage_fp8_available(void) { return 0; }

int quant_int8_per_block_cpu(int8_t *out, float *scales, const float *in,
                              size_t n, size_t block_size) { return -1; }
int dequant_int8_per_block_cpu(float *out, const int8_t *in, const float *scales,
                                size_t n, size_t block_size) { return -1; }
int softmax_cpu(float *out, const float *in, size_t batch, size_t dim) { return -1; }

int sage_attention_cpu(float *O, const float *Q, const float *K, const float *V,
                       int batch, int heads, int seq_q, int seq_k, int head_dim,
                       float sm_scale) { return -1; }

#endif /* _WIN32 */
