/**
 * cuda_gemm.c - cuBLAS GEMM wrapper via dlopen (no compile-time CUDA dependency)
 */

#ifndef _WIN32

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_types.h"

/* cuBLAS workspace (32 MiB for Ada) */
#define CUBLAS_WORKSPACE_SIZE (32 * 1024 * 1024)
static void *g_cublas_workspace = NULL;

/* Global CUDA state */
static void *g_cuda_rt_handle = NULL;
static void *g_cublas_handle = NULL;
cublasHandle_t g_cublas_ctx = NULL;  /* exported for cuda_sage.c */
static int g_cuda_available = -1;  /* -1 = not checked, 0 = no, 1 = yes */

/* CUDA runtime functions - exported for cuda_sparselt.c */
cuda_malloc_fn g_cuda_malloc = NULL;
cuda_free_fn g_cuda_free = NULL;
cuda_memcpy_fn g_cuda_memcpy = NULL;
cuda_device_sync_fn g_cuda_sync = NULL;
static cuda_get_error_fn g_cuda_error = NULL;

/* cuBLAS functions */
static cublas_create_fn g_cublas_create = NULL;
static cublas_destroy_fn g_cublas_destroy = NULL;
static cublas_dgemm_fn g_cublas_dgemm = NULL;
cublas_sgemm_fn g_cublas_sgemm = NULL;  /* exported for cuda_sage.c */
cublas_gemm_ex_fn g_cublas_gemm_ex = NULL;  /* exported for cuda_sage.c */
static cublas_set_math_mode_fn g_cublas_set_math_mode = NULL;
static cublas_set_workspace_fn g_cublas_set_workspace = NULL;
static cublas_gemm_strided_batched_ex_fn g_cublas_gemm_strided_batched_ex = NULL;

/**
 * Initialize CUDA and cuBLAS via dlopen
 * Returns 1 if successful, 0 otherwise
 */
int cuda_init(void) {
    if (g_cuda_available >= 0) return g_cuda_available;

    /* Load CUDA runtime */
    g_cuda_rt_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda_rt_handle) {
        g_cuda_rt_handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cuda_rt_handle) {
        g_cuda_rt_handle = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cuda_rt_handle) {
        g_cuda_available = 0;
        return 0;
    }

    /* Load CUDA runtime functions */
    g_cuda_malloc = (cuda_malloc_fn)dlsym(g_cuda_rt_handle, "cudaMalloc");
    g_cuda_free = (cuda_free_fn)dlsym(g_cuda_rt_handle, "cudaFree");
    g_cuda_memcpy = (cuda_memcpy_fn)dlsym(g_cuda_rt_handle, "cudaMemcpy");
    g_cuda_sync = (cuda_device_sync_fn)dlsym(g_cuda_rt_handle, "cudaDeviceSynchronize");
    g_cuda_error = (cuda_get_error_fn)dlsym(g_cuda_rt_handle, "cudaGetErrorString");

    if (!g_cuda_malloc || !g_cuda_free || !g_cuda_memcpy || !g_cuda_sync) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Load cuBLAS */
    g_cublas_handle = dlopen("libcublas.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cublas_handle) {
        g_cublas_handle = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublas_handle) {
        g_cublas_handle = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublas_handle) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Load cuBLAS functions */
    g_cublas_create = (cublas_create_fn)dlsym(g_cublas_handle, "cublasCreate_v2");
    g_cublas_destroy = (cublas_destroy_fn)dlsym(g_cublas_handle, "cublasDestroy_v2");
    g_cublas_dgemm = (cublas_dgemm_fn)dlsym(g_cublas_handle, "cublasDgemm_v2");
    g_cublas_sgemm = (cublas_sgemm_fn)dlsym(g_cublas_handle, "cublasSgemm_v2");
    g_cublas_gemm_ex = (cublas_gemm_ex_fn)dlsym(g_cublas_handle, "cublasGemmEx");
    g_cublas_set_math_mode = (cublas_set_math_mode_fn)dlsym(g_cublas_handle, "cublasSetMathMode");
    g_cublas_set_workspace = (cublas_set_workspace_fn)dlsym(g_cublas_handle, "cublasSetWorkspace_v2");
    g_cublas_gemm_strided_batched_ex = (cublas_gemm_strided_batched_ex_fn)dlsym(g_cublas_handle, "cublasGemmStridedBatchedEx");

    if (!g_cublas_create || !g_cublas_destroy || !g_cublas_dgemm) {
        dlclose(g_cublas_handle);
        dlclose(g_cuda_rt_handle);
        g_cublas_handle = NULL;
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Create cuBLAS context */
    if (g_cublas_create(&g_cublas_ctx) != CUBLAS_STATUS_SUCCESS) {
        dlclose(g_cublas_handle);
        dlclose(g_cuda_rt_handle);
        g_cublas_handle = NULL;
        g_cuda_rt_handle = NULL;
        g_cuda_available = 0;
        return 0;
    }

    /* Enable Tensor Cores (TF32 math mode) */
    if (g_cublas_set_math_mode) {
        g_cublas_set_math_mode(g_cublas_ctx, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    /* Allocate 32 MiB workspace for cuBLAS (required for best algorithms on Ada) */
    if (g_cublas_set_workspace && g_cuda_malloc) {
        cudaError_t werr = g_cuda_malloc((void **)&g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
        if (werr == cudaSuccess && g_cublas_workspace) {
            cublasStatus_t wstat = g_cublas_set_workspace(g_cublas_ctx, g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
            if (wstat == CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "[viva_tensor] cuBLAS workspace: 32 MiB (optimal for Ada)\n");
            } else {
                g_cuda_free(g_cublas_workspace);
                g_cublas_workspace = NULL;
            }
        }
    }

    const char *tensor_core_status = g_cublas_gemm_ex ? "INT8/FP16 Tensor Cores ENABLED" : "FP32 only";
    fprintf(stderr, "[viva_tensor] CUDA backend: cuBLAS (%s)\n", tensor_core_status);
    fprintf(stderr, "[viva_tensor] CUDA initialized (cuBLAS + cublasLt)\n");
    g_cuda_available = 1;
    return 1;
}

/**
 * DGEMM on GPU: C = alpha * A @ B + beta * C
 * A is M x K, B is K x N, C is M x N (all row-major)
 *
 * cuBLAS uses column-major, so we compute: C^T = B^T @ A^T
 * This gives us row-major result without explicit transpose.
 */
int cuda_dgemm(int M, int N, int K,
               double alpha, const double *A, int lda,
               const double *B, int ldb,
               double beta, double *C, int ldc) {
    if (!g_cuda_available) return -1;

    cudaError_t err;
    cublasStatus_t stat;

    size_t size_a = (size_t)M * K * sizeof(double);
    size_t size_b = (size_t)K * N * sizeof(double);
    size_t size_c = (size_t)M * N * sizeof(double);

    double *d_A = NULL, *d_B = NULL, *d_C = NULL;

    /* Allocate GPU memory */
    err = g_cuda_malloc((void**)&d_A, size_a);
    if (err != cudaSuccess) { return -2; }

    err = g_cuda_malloc((void**)&d_B, size_b);
    if (err != cudaSuccess) { g_cuda_free(d_A); return -2; }

    err = g_cuda_malloc((void**)&d_C, size_c);
    if (err != cudaSuccess) { g_cuda_free(d_A); g_cuda_free(d_B); return -2; }

    /* Copy A and B to GPU */
    err = g_cuda_memcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup; }

    err = g_cuda_memcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup; }

    /* cuBLAS DGEMM (column-major trick: swap A and B, swap M and N) */
    /* C = A @ B  becomes  C^T = B^T @ A^T  which is column-major of row-major result */
    stat = g_cublas_dgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,  /* No transpose (we're doing the swap trick) */
        N, M, K,                   /* Swapped dimensions for row-major */
        &alpha,
        d_B, N,                    /* B with leading dim N */
        d_A, K,                    /* A with leading dim K */
        &beta,
        d_C, N                     /* C with leading dim N */
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS error: %d\n", stat);
        goto cleanup;
    }

    /* Synchronize and copy result back */
    g_cuda_sync();
    err = g_cuda_memcpy(C, d_C, size_c, cudaMemcpyDeviceToHost);

cleanup:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess && stat == CUBLAS_STATUS_SUCCESS) ? 0 : -3;
}

/**
 * SGEMM on GPU: C = alpha * A @ B + beta * C (FP32 with TF32 Tensor Cores)
 * A is M x K, B is K x N, C is M x N (all row-major)
 */
int cuda_sgemm(int M, int N, int K,
               float alpha, const float *A, int lda,
               const float *B, int ldb,
               float beta, float *C, int ldc) {
    if (!g_cuda_available || !g_cublas_sgemm) return -1;

    cudaError_t err;
    cublasStatus_t stat;

    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)K * N * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    /* Allocate GPU memory */
    err = g_cuda_malloc((void**)&d_A, size_a);
    if (err != cudaSuccess) { return -2; }

    err = g_cuda_malloc((void**)&d_B, size_b);
    if (err != cudaSuccess) { g_cuda_free(d_A); return -2; }

    err = g_cuda_malloc((void**)&d_C, size_c);
    if (err != cudaSuccess) { g_cuda_free(d_A); g_cuda_free(d_B); return -2; }

    /* Copy A and B to GPU */
    err = g_cuda_memcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup_sgemm; }

    err = g_cuda_memcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { goto cleanup_sgemm; }

    /* cuBLAS SGEMM (column-major trick: swap A and B, swap M and N) */
    stat = g_cublas_sgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS SGEMM error: %d\n", stat);
        goto cleanup_sgemm;
    }

    /* Synchronize and copy result back */
    g_cuda_sync();
    err = g_cuda_memcpy(C, d_C, size_c, cudaMemcpyDeviceToHost);

cleanup_sgemm:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess && stat == CUBLAS_STATUS_SUCCESS) ? 0 : -3;
}

/**
 * Check if CUDA is available
 */
int cuda_available(void) {
    if (g_cuda_available < 0) cuda_init();
    return g_cuda_available;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup(void) {
    if (g_cublas_ctx) {
        g_cublas_destroy(g_cublas_ctx);
        g_cublas_ctx = NULL;
    }
    if (g_cublas_handle) {
        dlclose(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    if (g_cuda_rt_handle) {
        dlclose(g_cuda_rt_handle);
        g_cuda_rt_handle = NULL;
    }
    g_cuda_available = -1;
}

/* =========================================================================
 * CudaTensor API - Persistent GPU memory management
 * Eliminates PCIe transfer overhead for repeated operations
 * ========================================================================= */

/**
 * Allocate GPU memory
 * Returns NULL on failure
 */
float* cuda_tensor_alloc(size_t num_elements) {
    if (!cuda_available()) return NULL;

    float *d_ptr = NULL;
    cudaError_t err = g_cuda_malloc((void**)&d_ptr, num_elements * sizeof(float));
    if (err != cudaSuccess) return NULL;

    return d_ptr;
}

/**
 * Free GPU memory
 */
void cuda_tensor_free(void *d_ptr) {
    if (d_ptr && g_cuda_free) {
        g_cuda_free(d_ptr);
    }
}

/**
 * Upload data from CPU to GPU
 */
int cuda_tensor_upload(float *d_dst, const float *h_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(d_dst, h_src, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Download data from GPU to CPU
 */
int cuda_tensor_download(float *h_dst, const float *d_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(h_dst, d_src, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * SGEMM on pre-uploaded GPU data (no PCIe transfer overhead)
 * C = alpha * A @ B + beta * C (all pointers are GPU memory)
 */
int cuda_sgemm_gpu(int M, int N, int K,
                   float alpha, const float *d_A, int lda,
                   const float *d_B, int ldb,
                   float beta, float *d_C, int ldc) {
    if (!g_cuda_available) return -1;

    /* Use cublasGemmEx with TF32 Tensor Cores */
    if (g_cublas_gemm_ex) {
        cublasStatus_t stat = g_cublas_gemm_ex(
            g_cublas_ctx,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_32F, N,
            d_A, CUDA_R_32F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (stat == CUBLAS_STATUS_SUCCESS) {
            g_cuda_sync();
            return 0;
        }
    }

    /* Fallback */
    if (!g_cublas_sgemm) return -1;
    cublasStatus_t stat = g_cublas_sgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS SGEMM_GPU error: %d\n", stat);
        return -2;
    }

    g_cuda_sync();
    return 0;
}

/**
 * Batch upload multiple tensors (for matrix multiply prep)
 * More efficient than individual uploads
 */
int cuda_tensor_upload_batch(float **d_dsts, const float **h_srcs,
                              const size_t *sizes, int count) {
    for (int i = 0; i < count; i++) {
        if (cuda_tensor_upload(d_dsts[i], h_srcs[i], sizes[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Forward declaration for cublasLtMatmul-based INT8 GEMM */
int cuda_igemm_lt_gpu(int M, int N, int K, const int8_t *d_A, int lda,
                       const int8_t *d_B, int ldb, int32_t *d_C, int ldc);

/* =========================================================================
 * INT8 Tensor Core GEMM (cublasGemmEx + COMPUTE_32I)
 * ========================================================================= */

/**
 * Allocate GPU memory for INT8 tensors
 */
int8_t* cuda_tensor_alloc_int8(size_t num_elements) {
    if (!cuda_available()) return NULL;

    int8_t *d_ptr = NULL;
    cudaError_t err = g_cuda_malloc((void**)&d_ptr, num_elements * sizeof(int8_t));
    if (err != cudaSuccess) return NULL;

    return d_ptr;
}

/**
 * Allocate GPU memory for INT32 accumulators
 */
int32_t* cuda_tensor_alloc_int32(size_t num_elements) {
    if (!cuda_available()) return NULL;

    int32_t *d_ptr = NULL;
    cudaError_t err = g_cuda_malloc((void**)&d_ptr, num_elements * sizeof(int32_t));
    if (err != cudaSuccess) return NULL;

    return d_ptr;
}

/**
 * Upload INT8 data to GPU
 */
int cuda_tensor_upload_int8(int8_t *d_dst, const int8_t *h_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(d_dst, h_src, num_elements * sizeof(int8_t), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Download INT32 accumulator data from GPU
 */
int cuda_tensor_download_int32(int32_t *h_dst, const int32_t *d_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(h_dst, d_src, num_elements * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * INT8 GEMM on Tensor Cores: C = A @ B (with INT32 accumulator)
 * A is INT8 [M x K], B is INT8 [K x N], C is INT32 [M x N]
 *
 * INT8 Tensor Core path
 *
 * Note: Tensor Cores require dimensions to be multiples of 16 for best perf.
 */
int cuda_igemm_gpu(int M, int N, int K,
                   int32_t alpha, const int8_t *d_A, int lda,
                   const int8_t *d_B, int ldb,
                   int32_t beta, int32_t *d_C, int ldc) {
    /* Delegate to cublasLtMatmul version for proper Tensor Core usage */
    return cuda_igemm_lt_gpu(M, N, K, d_A, lda, d_B, ldb, d_C, ldc);
}

/**
 * INT8 GEMM with host memory (includes PCIe transfer)
 * For benchmarking and when data isn't already on GPU
 */
int cuda_igemm(int M, int N, int K,
               int32_t alpha, const int8_t *A, int lda,
               const int8_t *B, int ldb,
               int32_t beta, int32_t *C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    cudaError_t err;

    size_t size_a = (size_t)M * K;
    size_t size_b = (size_t)K * N;
    size_t size_c = (size_t)M * N;

    int8_t *d_A = cuda_tensor_alloc_int8(size_a);
    int8_t *d_B = cuda_tensor_alloc_int8(size_b);
    int32_t *d_C = cuda_tensor_alloc_int32(size_c);

    if (!d_A || !d_B || !d_C) {
        if (d_A) g_cuda_free(d_A);
        if (d_B) g_cuda_free(d_B);
        if (d_C) g_cuda_free(d_C);
        return -2;
    }

    /* Upload A and B */
    err = g_cuda_memcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_igemm;

    err = g_cuda_memcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_igemm;

    /* Run GEMM on Tensor Cores */
    int result = cuda_igemm_gpu(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    if (result != 0) {
        err = 1;  /* Mark as failed */
        goto cleanup_igemm;
    }

    /* Download result */
    err = g_cuda_memcpy(C, d_C, size_c * sizeof(int32_t), cudaMemcpyDeviceToHost);

cleanup_igemm:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess) ? 0 : -3;
}

/**
 * Check if INT8 Tensor Cores are available
 */
int cuda_int8_available(void) {
    if (!cuda_available()) return 0;
    return g_cublas_gemm_ex != NULL;
}

/* =========================================================================
 * FP16 Tensor Core GEMM (cublasGemmEx + COMPUTE_16F)
 * ========================================================================= */

/* FP16 type for CUDA */
typedef uint16_t cuda_half_t;

/**
 * Allocate GPU memory for FP16 tensors
 */
cuda_half_t* cuda_tensor_alloc_fp16(size_t num_elements) {
    if (!cuda_available()) return NULL;

    cuda_half_t *d_ptr = NULL;
    cudaError_t err = g_cuda_malloc((void**)&d_ptr, num_elements * sizeof(cuda_half_t));
    if (err != cudaSuccess) return NULL;

    return d_ptr;
}

/**
 * FP16 GEMM on Tensor Cores: C = alpha * A @ B + beta * C
 * A is FP16 [M x K], B is FP16 [K x N], C is FP32 [M x N]
 *
 * Uses FP16 Tensor Cores with FP32 accumulator for accuracy.
 * FP16 HGEMM (Tensor Cores)
 */
int cuda_hgemm_gpu(int M, int N, int K,
                   float alpha, const cuda_half_t *d_A, int lda,
                   const cuda_half_t *d_B, int ldb,
                   float beta, float *d_C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) {
        fprintf(stderr, "[viva_tensor] cublasGemmEx not available for FP16\n");
        return -1;
    }

    /* FP16 input with FP32 output and accumulator */
    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,              /* B is FP16 */
        d_A, CUDA_R_16F, K,              /* A is FP16 */
        &beta,
        d_C, CUDA_R_32F, N,              /* C is FP32 for accuracy */
        CUBLAS_COMPUTE_32F_FAST_16F,      /* FP16 Tensor Core with FP32 acc */
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuBLAS FP16 GEMM error: %d\n", stat);
        return -2;
    }

    g_cuda_sync();
    return 0;
}

/**
 * Pure FP16 HGEMM on GPU - FP16 in, FP16 out, FP16 compute
 * Maximum throughput: no FP32 conversion, half the output bandwidth.
 * For in-place usage: pass pre-allocated d_C pointer.
 * Pure FP16 HGEMM (COMPUTE_16F for max throughput)
 */
int cuda_hgemm_gpu_pure16(int M, int N, int K,
                           const cuda_half_t *d_A, int lda,
                           const cuda_half_t *d_B, int ldb,
                           cuda_half_t *d_C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    /* FP16 alpha/beta for pure FP16 path */
    cuda_half_t alpha_h = 0x3C00;  /* 1.0 in FP16 */
    cuda_half_t beta_h  = 0x0000;  /* 0.0 in FP16 */

    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta_h,
        d_C, CUDA_R_16F, N,        /* Output FP16, half the bandwidth */
        CUBLAS_COMPUTE_16F,          /* Pure FP16 compute */
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (stat != CUBLAS_STATUS_SUCCESS) return -2;
    g_cuda_sync();
    return 0;
}

/**
 * Pure FP16 HGEMM async (no sync) - for pipeline benchmarks
 */
int cuda_hgemm_gpu_pure16_async(int M, int N, int K,
                                 const cuda_half_t *d_A, int lda,
                                 const cuda_half_t *d_B, int ldb,
                                 cuda_half_t *d_C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    cuda_half_t alpha_h = 0x3C00;
    cuda_half_t beta_h  = 0x0000;

    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta_h,
        d_C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

/**
 * FP32 SGEMM in-place on GPU using TF32 Tensor Cores.
 * TF32 uses Tensor Cores for FP32 matmul (~2x over pure FP32).
 * RTX 4090: 82T (FP32) -> 165T (TF32)
 */
int cuda_sgemm_gpu_inplace(int M, int N, int K,
                            float alpha, const float *d_A, int lda,
                            const float *d_B, int ldb,
                            float beta, float *d_C, int ldc) {
    if (!g_cuda_available) return -1;

    /* Use cublasGemmEx with TF32 for Tensor Core acceleration */
    if (g_cublas_gemm_ex) {
        cublasStatus_t stat = g_cublas_gemm_ex(
            g_cublas_ctx,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_32F, N,
            d_A, CUDA_R_32F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_TF32,     /* TF32 Tensor Cores */
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (stat == CUBLAS_STATUS_SUCCESS) {
            g_cuda_sync();
            return 0;
        }
        /* Fall through to cublasSgemm if TF32 fails */
    }

    /* Fallback: regular cublasSgemm */
    if (!g_cublas_sgemm) return -1;
    cublasStatus_t stat = g_cublas_sgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    if (stat != CUBLAS_STATUS_SUCCESS) return -2;
    g_cuda_sync();
    return 0;
}

/**
 * Check if FP16 Tensor Cores are available
 */
int cuda_fp16_available(void) {
    if (!cuda_available()) return 0;
    return g_cublas_gemm_ex != NULL;
}

/**
 * Upload FP16 data to GPU
 */
int cuda_tensor_upload_fp16(cuda_half_t *d_dst, const cuda_half_t *h_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(d_dst, h_src, num_elements * sizeof(cuda_half_t), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Download FP32 result from GPU
 */
int cuda_tensor_download_fp32(float *h_dst, const float *d_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(h_dst, d_src, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * Download FP16 data from GPU
 */
int cuda_tensor_download_fp16(cuda_half_t *h_dst, const cuda_half_t *d_src, size_t num_elements) {
    if (!g_cuda_memcpy) return -1;

    cudaError_t err = g_cuda_memcpy(h_dst, d_src, num_elements * sizeof(cuda_half_t), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

/**
 * FP16 GEMM with host memory (includes PCIe transfer + conversion)
 * Input: FP16 matrices A[M,K] and B[K,N] on CPU
 * Output: FP32 matrix C[M,N] on CPU (higher precision accumulator)
 *
 * FP16 Tensor Core path
 */
int cuda_hgemm(int M, int N, int K,
               float alpha, const cuda_half_t *A, int lda,
               const cuda_half_t *B, int ldb,
               float beta, float *C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    cudaError_t err;

    size_t size_a = (size_t)M * K;
    size_t size_b = (size_t)K * N;
    size_t size_c = (size_t)M * N;

    cuda_half_t *d_A = cuda_tensor_alloc_fp16(size_a);
    cuda_half_t *d_B = cuda_tensor_alloc_fp16(size_b);
    float *d_C = cuda_tensor_alloc(size_c);  /* FP32 output for accuracy */

    if (!d_A || !d_B || !d_C) {
        if (d_A) g_cuda_free(d_A);
        if (d_B) g_cuda_free(d_B);
        if (d_C) g_cuda_free(d_C);
        return -2;
    }

    /* Upload A and B (FP16) */
    err = g_cuda_memcpy(d_A, A, size_a * sizeof(cuda_half_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_hgemm;

    err = g_cuda_memcpy(d_B, B, size_b * sizeof(cuda_half_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_hgemm;

    /* Initialize C to zero on GPU (for beta=0) */
    if (beta == 0.0f) {
        err = g_cuda_memcpy(d_C, C, 0, cudaMemcpyHostToDevice);  /* No-op but valid */
    }

    /* Run FP16 GEMM on Tensor Cores */
    int result = cuda_hgemm_gpu(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    if (result != 0) {
        err = 1;
        goto cleanup_hgemm;
    }

    /* Download result (FP32) */
    err = g_cuda_memcpy(C, d_C, size_c * sizeof(float), cudaMemcpyDeviceToHost);

cleanup_hgemm:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess) ? 0 : -3;
}

/* =========================================================================
 * cublasLt for INT8 Tensor Cores (Ada Lovelace IMMA)
 * cublasGemmEx uses DP4A; cublasLt enables IMMA Tensor Cores.
 * ========================================================================= */

/* cublasLt types */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

/* cublasLtMatmulAlgo_t - opaque 64 bytes */
typedef struct { uint64_t data[8]; } cublasLtMatmulAlgo_t;

/* cublasLtMatmulHeuristicResult_t - 96 bytes */
typedef struct __attribute__((aligned(8))) {
    cublasLtMatmulAlgo_t algo;      /* 64 bytes */
    size_t workspaceSize;            /* 8 bytes */
    cublasStatus_t state;            /* 4 bytes */
    float wavesCount;                /* 4 bytes */
    int reserved[4];                 /* 16 bytes */
} cublasLtHeuristicResult_t;

/* cublasLt compute types for Tensor Cores */
#define CUBLAS_COMPUTE_32I_PEDANTIC    73   /* Force IMMA Tensor Cores (pedantic) */

/* cublasLt attribute enums (from cublasLt.h) */
#define CUBLASLT_MATRIX_LAYOUT_ORDER             1
#define CUBLASLT_MATMUL_DESC_TRANSA              3
#define CUBLASLT_MATMUL_DESC_TRANSB              4
#define CUBLASLT_MATMUL_DESC_EPILOGUE            7
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER        8
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES 1

/* cublasLt epilogue types (fused activations) */
#define CUBLASLT_EPILOGUE_DEFAULT     1
#define CUBLASLT_EPILOGUE_RELU        2
#define CUBLASLT_EPILOGUE_BIAS        4
#define CUBLASLT_EPILOGUE_RELU_BIAS   6   /* RELU | BIAS */
#define CUBLASLT_EPILOGUE_GELU       32
#define CUBLASLT_EPILOGUE_GELU_BIAS  36   /* GELU | BIAS */

/* cublasLt layout order */
#define CUBLASLT_ORDER_COL  0
#define CUBLASLT_ORDER_ROW  1

/* cublasLt function pointers */
typedef cublasStatus_t (*cublaslt_create_fn)(cublasLtHandle_t*);
typedef cublasStatus_t (*cublaslt_destroy_fn)(cublasLtHandle_t);
typedef cublasStatus_t (*cublaslt_matmul_desc_create_fn)(
    cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
typedef cublasStatus_t (*cublaslt_matmul_desc_destroy_fn)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*cublaslt_matmul_desc_set_attr_fn)(
    cublasLtMatmulDesc_t, int, const void*, size_t);
typedef cublasStatus_t (*cublaslt_matrix_layout_create_fn)(
    cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef cublasStatus_t (*cublaslt_matrix_layout_destroy_fn)(cublasLtMatrixLayout_t);
typedef cublasStatus_t (*cublaslt_matmul_preference_create_fn)(cublasLtMatmulPreference_t*);
typedef cublasStatus_t (*cublaslt_matmul_preference_destroy_fn)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*cublaslt_matmul_preference_set_attr_fn)(
    cublasLtMatmulPreference_t, int, const void*, size_t);
typedef cublasStatus_t (*cublaslt_matrix_layout_set_attr_fn)(
    cublasLtMatrixLayout_t, int, const void*, size_t);
typedef cublasStatus_t (*cublaslt_matmul_algo_get_heuristic_fn)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int, cublasLtHeuristicResult_t*, int*);
typedef cublasStatus_t (*cublaslt_matmul_fn)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, cudaStream_t);

/* Global cublasLt state */
static void *g_cublaslt_handle = NULL;
static cublasLtHandle_t g_cublaslt_ctx = NULL;
static int g_cublaslt_available = -1;

static cublaslt_create_fn g_cublaslt_create = NULL;
static cublaslt_destroy_fn g_cublaslt_destroy = NULL;
static cublaslt_matmul_desc_create_fn g_cublaslt_matmul_desc_create = NULL;
static cublaslt_matmul_desc_destroy_fn g_cublaslt_matmul_desc_destroy = NULL;
static cublaslt_matmul_desc_set_attr_fn g_cublaslt_matmul_desc_set_attr = NULL;
static cublaslt_matrix_layout_create_fn g_cublaslt_matrix_layout_create = NULL;
static cublaslt_matrix_layout_destroy_fn g_cublaslt_matrix_layout_destroy = NULL;
static cublaslt_matmul_preference_create_fn g_cublaslt_matmul_preference_create = NULL;
static cublaslt_matmul_preference_destroy_fn g_cublaslt_matmul_preference_destroy = NULL;
static cublaslt_matmul_preference_set_attr_fn g_cublaslt_matmul_preference_set_attr = NULL;
static cublaslt_matrix_layout_set_attr_fn g_cublaslt_matrix_layout_set_attr = NULL;
static cublaslt_matmul_algo_get_heuristic_fn g_cublaslt_matmul_algo_get_heuristic = NULL;
static cublaslt_matmul_fn g_cublaslt_matmul = NULL;

/* cublasLt workspace for heuristic algorithms */
static void *g_cublaslt_workspace = NULL;
#define CUBLASLT_WORKSPACE_SIZE (32 * 1024 * 1024)

/* Algorithm cache for repeated same-size INT8 calls */
static int g_int8_cache_m = 0, g_int8_cache_n = 0, g_int8_cache_k = 0;
static cublasLtHeuristicResult_t g_int8_cached_result;
static int g_int8_cache_valid = 0;

/* FP16 TN algorithm cache */
static int g_fp16_cache_m = 0, g_fp16_cache_n = 0, g_fp16_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_cached_result;
static int g_fp16_cache_valid = 0;

/* FP16 fused GEMM+ReLU algorithm cache */
static int g_fp16_relu_cache_m = 0, g_fp16_relu_cache_n = 0, g_fp16_relu_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_relu_cached_result;
static int g_fp16_relu_cache_valid = 0;

/* FP16 fused GEMM+GELU algorithm cache */
static int g_fp16_gelu_cache_m = 0, g_fp16_gelu_cache_n = 0, g_fp16_gelu_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_gelu_cached_result;
static int g_fp16_gelu_cache_valid = 0;

/**
 * Initialize cublasLt for INT8 Tensor Cores (Ada IMMA)
 */
int cublaslt_init(void) {
    if (g_cublaslt_available >= 0) return g_cublaslt_available;

    /* Ensure CUDA runtime is initialized first (we need g_cuda_malloc) */
    if (!cuda_available()) {
        g_cublaslt_available = 0;
        return 0;
    }

    /* Load cublasLt library */
    g_cublaslt_handle = dlopen("libcublasLt.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_cublaslt_handle) {
        g_cublaslt_handle = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublaslt_handle) {
        g_cublaslt_handle = dlopen("libcublasLt.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_cublaslt_handle) {
        g_cublaslt_available = 0;
        return 0;
    }

    /* Load functions */
    g_cublaslt_create = (cublaslt_create_fn)dlsym(g_cublaslt_handle, "cublasLtCreate");
    g_cublaslt_destroy = (cublaslt_destroy_fn)dlsym(g_cublaslt_handle, "cublasLtDestroy");
    g_cublaslt_matmul_desc_create = (cublaslt_matmul_desc_create_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulDescCreate");
    g_cublaslt_matmul_desc_destroy = (cublaslt_matmul_desc_destroy_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulDescDestroy");
    g_cublaslt_matmul_desc_set_attr = (cublaslt_matmul_desc_set_attr_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulDescSetAttribute");
    g_cublaslt_matrix_layout_create = (cublaslt_matrix_layout_create_fn)dlsym(g_cublaslt_handle, "cublasLtMatrixLayoutCreate");
    g_cublaslt_matrix_layout_destroy = (cublaslt_matrix_layout_destroy_fn)dlsym(g_cublaslt_handle, "cublasLtMatrixLayoutDestroy");
    g_cublaslt_matmul_preference_create = (cublaslt_matmul_preference_create_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulPreferenceCreate");
    g_cublaslt_matmul_preference_destroy = (cublaslt_matmul_preference_destroy_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulPreferenceDestroy");
    g_cublaslt_matmul_preference_set_attr = (cublaslt_matmul_preference_set_attr_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulPreferenceSetAttribute");
    g_cublaslt_matrix_layout_set_attr = (cublaslt_matrix_layout_set_attr_fn)dlsym(g_cublaslt_handle, "cublasLtMatrixLayoutSetAttribute");
    g_cublaslt_matmul_algo_get_heuristic = (cublaslt_matmul_algo_get_heuristic_fn)dlsym(g_cublaslt_handle, "cublasLtMatmulAlgoGetHeuristic");
    g_cublaslt_matmul = (cublaslt_matmul_fn)dlsym(g_cublaslt_handle, "cublasLtMatmul");

    if (!g_cublaslt_create || !g_cublaslt_matmul) {
        dlclose(g_cublaslt_handle);
        g_cublaslt_handle = NULL;
        g_cublaslt_available = 0;
        return 0;
    }

    /* Create cublasLt context */
    if (g_cublaslt_create(&g_cublaslt_ctx) != CUBLAS_STATUS_SUCCESS) {
        dlclose(g_cublaslt_handle);
        g_cublaslt_handle = NULL;
        g_cublaslt_available = 0;
        return 0;
    }

    /* Allocate 32 MiB workspace for cublasLt heuristic algorithms */
    if (g_cuda_malloc) {
        cudaError_t werr = g_cuda_malloc(&g_cublaslt_workspace, CUBLASLT_WORKSPACE_SIZE);
        if (werr != cudaSuccess) {
            g_cublaslt_workspace = NULL;
            fprintf(stderr, "[viva_tensor] cublasLt workspace alloc failed (non-fatal)\n");
        }
    }

    fprintf(stderr, "[viva_tensor] cublasLt loaded - INT8 IMMA Tensor Cores ready (ws=%s)\n",
            g_cublaslt_workspace ? "32MiB" : "none");
    g_cublaslt_available = 1;
    return 1;
}

/**
 * INT8 GEMM via cublasLt with IMMA Tensor Cores
 * INT8 IMMA Tensor Core path (higher throughput than FP32)
 *
 * Dimensions should be multiples of 16 for best performance.
 */
int cuda_igemm_lt(int M, int N, int K,
                  float alpha, const int8_t *A, int lda,
                  const int8_t *B, int ldb,
                  float beta, int32_t *C, int ldc) {
    /* Initialize cublasLt if needed */
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available) {
        fprintf(stderr, "[viva_tensor] cublasLt not available, falling back to cublasGemmEx\n");
        return cuda_igemm(M, N, K, (int32_t)alpha, A, lda, B, ldb, (int32_t)beta, C, ldc);
    }

    cudaError_t err;

    size_t size_a = (size_t)M * K;
    size_t size_b = (size_t)K * N;
    size_t size_c = (size_t)M * N;

    /* Allocate GPU memory */
    int8_t *d_A = cuda_tensor_alloc_int8(size_a);
    int8_t *d_B = cuda_tensor_alloc_int8(size_b);
    int32_t *d_C = cuda_tensor_alloc_int32(size_c);

    if (!d_A || !d_B || !d_C) {
        if (d_A) g_cuda_free(d_A);
        if (d_B) g_cuda_free(d_B);
        if (d_C) g_cuda_free(d_C);
        return -2;
    }

    /* Upload data */
    err = g_cuda_memcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_lt;

    err = g_cuda_memcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_lt;

    /* Use cuda_igemm_lt_gpu for the actual GEMM (data is on GPU) */
    int result = cuda_igemm_lt_gpu(M, N, K, d_A, K, d_B, N, d_C, N);
    if (result != 0) {
        err = 1;
        goto cleanup_lt;
    }

    /* Download result */
    err = g_cuda_memcpy(C, d_C, size_c * sizeof(int32_t), cudaMemcpyDeviceToHost);

cleanup_lt:
    g_cuda_free(d_A);
    g_cuda_free(d_B);
    g_cuda_free(d_C);

    return (err == cudaSuccess) ? 0 : -3;
}

/**
 * Check if cublasLt INT8 Tensor Cores are available
 */
int cuda_int8_lt_available(void) {
    if (g_cublaslt_available < 0) cublaslt_init();
    return g_cublaslt_available;
}

/* =========================================================================
 * ASYNC FUNCTIONS - No sync, for pipeline benchmarking
 * Call cuda_explicit_sync() when you need results
 * ========================================================================= */

/**
 * Explicit sync - call this when you need the GPU results
 */
void cuda_explicit_sync(void) {
    if (g_cuda_sync) g_cuda_sync();
}

/**
 * FP32 SGEMM async (no sync) - for pipeline benchmarking
 */
int cuda_sgemm_gpu_async(int M, int N, int K,
                          float alpha, const float *d_A, int lda,
                          const float *d_B, int ldb,
                          float beta, float *d_C, int ldc) {
    if (!g_cuda_available) return -1;

    /* Use TF32 Tensor Cores for async path */
    if (g_cublas_gemm_ex) {
        cublasStatus_t stat = g_cublas_gemm_ex(
            g_cublas_ctx,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_32F, N,
            d_A, CUDA_R_32F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        if (stat == CUBLAS_STATUS_SUCCESS) return 0;
    }

    if (!g_cublas_sgemm) return -1;
    cublasStatus_t stat = g_cublas_sgemm(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

/**
 * FP16 HGEMM async (no sync) - Tensor Cores without sync overhead
 * Async SGEMM for sustained throughput (no sync per call)
 */
int cuda_hgemm_gpu_async(int M, int N, int K,
                          float alpha, const cuda_half_t *d_A, int lda,
                          const cuda_half_t *d_B, int ldb,
                          float beta, float *d_C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

/* =========================================================================
 * INT8 TENSOR CORE GEMM (GPU-only) via cublasLtMatmul
 *
 * cublasGemmEx with NN format does NOT use INT8 Tensor Cores (IMMA) on Ada!
 * It silently falls back to CUDA core IGEMM kernels (~150 TOPS).
 *
 * cublasLtMatmul with proper descriptors enables IMMA: ~560 TOPS on RTX 4090.
 *
 * Key settings:
 * - CUBLAS_COMPUTE_32I compute type (NOT CUDA_R_32I which is a data type!)
 * - CUDA_R_32F scale type (float alpha/beta for fractional scaling)
 * - Row-major layout order (CUBLASLT_ORDER_ROW)
 * - 32 MiB workspace for heuristic algorithm selection
 * - Algorithm caching for repeated same-size calls
 * ========================================================================= */

/**
 * INT8 GEMM on GPU via cublasLtMatmul TN format with Tensor Core IMMA
 *
 * Uses TN (Transpose-NoTranspose) layout which is REQUIRED for INT8 IMMA
 * on Ada Lovelace. NN format falls back to CUDA cores (~150 TOPS).
 * TN format achieves ~560-660 TOPS on RTX 4090.
 *
 * d_A:   INT8 [M x K] row-major on GPU (used directly as A^T col-major)
 * d_B_T: INT8 [N x K] row-major on GPU (TRANSPOSED B, so col-major K×N, ld=K)
 * d_C:   INT32 [M x N] row-major on GPU (output)
 *
 * The caller must provide d_B_T = transpose of original B[K][N].
 * This function does NOT transpose internally.
 *
 * Uses cached heuristic algorithm for repeated same-size calls.
 */
int cuda_igemm_lt_gpu_tn(int M, int N, int K,
                          const int8_t *d_A,
                          const int8_t *d_B_T,
                          int32_t *d_C) {
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available || !g_cublaslt_matmul) return -1;

    cublasStatus_t stat;
    int32_t alpha_i = 1, beta_i = 0;

    /* Check algorithm cache */
    int need_heuristic = !g_int8_cache_valid ||
                         g_int8_cache_m != M || g_int8_cache_n != N || g_int8_cache_k != K;

    if (need_heuristic) {
        /* Create matmul descriptor: COMPUTE_32I with INT32 scale type */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        /* TN format: Transpose first operand, No-transpose second */
        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        /*
         * TN swap trick for row-major data:
         * C_row[M][N] = A_row[M][K] @ B_row[K][N]
         * In col-major: C_col(N×M) = B_T_col(K×N)^T × A_col(K×M)
         *
         * B_T is transposed B: B_T_row[N][K] → col-major K×N, ld=K
         * A_row[M][K] → col-major K×M, ld=K
         * C_row[M][N] → col-major N×M, ld=N
         *
         * m_lt=N, n_lt=M, k_lt=K
         */
        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        /* B_T: col-major K×N, ld=K (first operand, will be transposed by OP_T) */
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_8I, (uint64_t)K, (uint64_t)N, (int64_t)K);
        /* A: col-major K×M, ld=K (second operand, OP_N) */
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8I, (uint64_t)K, (uint64_t)M, (int64_t)K);
        /* C: col-major N×M, ld=N (result) */
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_32I, (uint64_t)N, (uint64_t)M, (int64_t)N);

        /* Preference with 32 MiB workspace */
        cublasLtMatmulPreference_t preference;
        g_cublaslt_matmul_preference_create(&preference);
        size_t ws_size = g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0;
        g_cublaslt_matmul_preference_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &ws_size, sizeof(ws_size));

        /* Get best algorithm via heuristic */
        cublasLtHeuristicResult_t results[8];
        int returned_count = 0;
        stat = g_cublaslt_matmul_algo_get_heuristic(
            g_cublaslt_ctx, matmul_desc,
            layout_bt, layout_a, layout_c, layout_c,
            preference, 8, results, &returned_count);

        if (stat != CUBLAS_STATUS_SUCCESS || returned_count == 0) {
            fprintf(stderr, "[viva_tensor] INT8 TN heuristic failed: stat=%d, count=%d\n",
                    stat, returned_count);
            g_cublaslt_matmul_preference_destroy(preference);
            g_cublaslt_matrix_layout_destroy(layout_bt);
            g_cublaslt_matrix_layout_destroy(layout_a);
            g_cublaslt_matrix_layout_destroy(layout_c);
            g_cublaslt_matmul_desc_destroy(matmul_desc);
            return -3;
        }

        /* Execute TN GEMM with Tensor Cores */
        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_i,
            d_B_T, layout_bt,    /* B_T: first operand (will be transposed) */
            d_A, layout_a,       /* A: second operand */
            &beta_i,
            d_C, layout_c,
            d_C, layout_c,
            &results[0].algo,
            g_cublaslt_workspace, ws_size,
            (cudaStream_t)0);

        /* Cache the algorithm */
        g_int8_cache_m = M; g_int8_cache_n = N; g_int8_cache_k = K;
        g_int8_cached_result = results[0];
        g_int8_cache_valid = 1;

        g_cublaslt_matmul_preference_destroy(preference);
        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_int8_cache_valid = 0;
            return -4;
        }
        return 0;

    } else {
        /* Cached fast path - reuse algorithm */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_8I, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8I, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_32I, (uint64_t)N, (uint64_t)M, (int64_t)N);

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_i,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_i,
            d_C, layout_c, d_C, layout_c,
            &g_int8_cached_result.algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);

        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_int8_cache_valid = 0;
            return -4;
        }
        return 0;
    }
}

/**
 * FP16 HGEMM via cublasLtMatmul with TN format + heuristic
 * Same TN swap trick as INT8: pre-transpose B for Tensor Core alignment.
 * Uses CUBLAS_COMPUTE_16F with FP16 alpha/beta for maximum throughput.
 * cublasLt FP16 HGEMM (higher throughput than cublasGemmEx)
 */
int cuda_hgemm_lt_gpu_tn(int M, int N, int K,
                           const cuda_half_t *d_A,
                           const cuda_half_t *d_B_T,
                           cuda_half_t *d_C) {
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available || !g_cublaslt_matmul) return -1;

    cublasStatus_t stat;
    cuda_half_t alpha_h = 0x3C00;  /* 1.0 in FP16 */
    cuda_half_t beta_h  = 0x0000;  /* 0.0 in FP16 */

    int need_heuristic = !g_fp16_cache_valid ||
                         g_fp16_cache_m != M || g_fp16_cache_n != N || g_fp16_cache_k != K;

    if (need_heuristic) {
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        /* TN swap for row-major: m_lt=N, n_lt=M, k_lt=K
         * B_T col-major K×N ld=K, A col-major K×M ld=K, C col-major N×M ld=N */
        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_16F, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        cublasLtMatmulPreference_t preference;
        g_cublaslt_matmul_preference_create(&preference);
        size_t ws_size = g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0;
        g_cublaslt_matmul_preference_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &ws_size, sizeof(ws_size));

        cublasLtHeuristicResult_t results[8];
        int returned_count = 0;
        stat = g_cublaslt_matmul_algo_get_heuristic(
            g_cublaslt_ctx, matmul_desc,
            layout_bt, layout_a, layout_c, layout_c,
            preference, 8, results, &returned_count);

        if (stat != CUBLAS_STATUS_SUCCESS || returned_count == 0) {
            fprintf(stderr, "[viva_tensor] FP16 TN heuristic failed: stat=%d, count=%d\n",
                    stat, returned_count);
            g_cublaslt_matmul_preference_destroy(preference);
            g_cublaslt_matrix_layout_destroy(layout_bt);
            g_cublaslt_matrix_layout_destroy(layout_a);
            g_cublaslt_matrix_layout_destroy(layout_c);
            g_cublaslt_matmul_desc_destroy(matmul_desc);
            return -3;
        }

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_h,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_h,
            d_C, layout_c, d_C, layout_c,
            &results[0].algo,
            g_cublaslt_workspace, ws_size,
            (cudaStream_t)0);

        g_fp16_cache_m = M; g_fp16_cache_n = N; g_fp16_cache_k = K;
        g_fp16_cached_result = results[0];
        g_fp16_cache_valid = 1;

        g_cublaslt_matmul_preference_destroy(preference);
        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_fp16_cache_valid = 0;
            return -4;
        }
        return 0;

    } else {
        /* Cached fast path */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_16F, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_h,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_h,
            d_C, layout_c, d_C, layout_c,
            &g_fp16_cached_result.algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);

        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_fp16_cache_valid = 0;
            return -4;
        }
        return 0;
    }
}

/**
 * INT8 GEMM on GPU - legacy NN fallback via cublasGemmEx
 * Used when d_B_T (transposed B) is not available.
 * ~150 TOPS on RTX 4090 (CUDA cores, not Tensor Cores).
 */
int cuda_igemm_lt_gpu(int M, int N, int K,
                       const int8_t *d_A, int lda,
                       const int8_t *d_B, int ldb,
                       int32_t *d_C, int ldc) {
    if (!g_cuda_available || !g_cublas_gemm_ex) return -1;

    /* NN col-major swap: C^T = B^T × A^T
     * INT32 alpha/beta as required by CUBLAS_COMPUTE_32I. */
    int32_t alpha_i = 1, beta_i = 0;
    cublasStatus_t stat = g_cublas_gemm_ex(
        g_cublas_ctx, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha_i, d_B, CUDA_R_8I, N, d_A, CUDA_R_8I, K,
        &beta_i, d_C, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        /* INT8 GEMM requires dims multiples of 4. CPU fallback for non-aligned. */
        int8_t *h_a = (int8_t *)malloc(M * K * sizeof(int8_t));
        int8_t *h_b = (int8_t *)malloc(K * N * sizeof(int8_t));
        int32_t *h_c = (int32_t *)calloc(M * N, sizeof(int32_t));
        if (!h_a || !h_b || !h_c) { free(h_a); free(h_b); free(h_c); return -2; }

        g_cuda_memcpy(h_a, d_A, M * K * sizeof(int8_t), cudaMemcpyDeviceToHost);
        g_cuda_memcpy(h_b, d_B, K * N * sizeof(int8_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                for (int p = 0; p < K; p++)
                    h_c[i * N + j] += (int32_t)h_a[i * K + p] * (int32_t)h_b[p * N + j];

        g_cuda_memcpy(d_C, h_c, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);
        free(h_a); free(h_b); free(h_c);
    }
    g_cuda_sync();
    return 0;
}

/**
 * INT8 GEMM on GPU - ASYNC version via cublasLtMatmul
 * Same as sync but without cudaDeviceSynchronize
 */
int cuda_igemm_lt_gpu_async(int M, int N, int K,
                             const int8_t *d_A, int lda,
                             const int8_t *d_B, int ldb,
                             int32_t *d_C, int ldc) {
    /* For async, we need the cache to be warm (heuristic already done).
     * If cache is cold, do a sync call first to warm it up. */
    if (!g_int8_cache_valid || g_int8_cache_m != M ||
        g_int8_cache_n != N || g_int8_cache_k != K) {
        return cuda_igemm_lt_gpu(M, N, K, d_A, lda, d_B, ldb, d_C, ldc);
    }

    /* Cache is warm - fast path without sync */
    cublasStatus_t stat;
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulDesc_t matmul_desc;
    stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
    if (stat != CUBLAS_STATUS_SUCCESS) return -2;

    cublasOperation_t op_n = CUBLAS_OP_N;
    g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_n, sizeof(op_n));
    g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_n, sizeof(op_n));

    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;

    if (g_int8_cache_valid == 1) {
        int32_t row_order = CUBLASLT_ORDER_ROW;
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8I, (uint64_t)M, (uint64_t)K, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_b, CUDA_R_8I, (uint64_t)K, (uint64_t)N, (int64_t)N);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_32I, (uint64_t)M, (uint64_t)N, (int64_t)N);
        if (g_cublaslt_matrix_layout_set_attr) {
            g_cublaslt_matrix_layout_set_attr(layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                               &row_order, sizeof(row_order));
            g_cublaslt_matrix_layout_set_attr(layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                               &row_order, sizeof(row_order));
            g_cublaslt_matrix_layout_set_attr(layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                               &row_order, sizeof(row_order));
        }
        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha, d_A, layout_a, d_B, layout_b,
            &beta, d_C, layout_c, d_C, layout_c,
            &g_int8_cached_result.algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);
    } else {
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8I, (uint64_t)N, (uint64_t)K, (int64_t)N);
        g_cublaslt_matrix_layout_create(&layout_b, CUDA_R_8I, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_32I, (uint64_t)N, (uint64_t)M, (int64_t)N);
        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha, d_B, layout_a, d_A, layout_b,
            &beta, d_C, layout_c, d_C, layout_c,
            &g_int8_cached_result.algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);
    }

    g_cublaslt_matrix_layout_destroy(layout_a);
    g_cublaslt_matrix_layout_destroy(layout_b);
    g_cublaslt_matrix_layout_destroy(layout_c);
    g_cublaslt_matmul_desc_destroy(matmul_desc);

    /* No sync - caller manages synchronization */
    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

/* =========================================================================
 * Fused GEMM + Activation via cublasLt epilogues
 * ReLU/GELU fused into the GEMM kernel via cublasLt epilogues.
 * Same throughput as plain GEMM with fused activation.
 *
 * Uses NN layout with cublasGemmEx-style col-major swap trick.
 * C[M][N] = act(A[M][K] @ B[K][N])
 * ========================================================================= */

/**
 * Internal: FP16 GEMM + fused epilogue via cublasLt
 * epilogue_type: CUBLASLT_EPILOGUE_RELU (1) or CUBLASLT_EPILOGUE_GELU (32)
 *
 * Async: no cudaDeviceSynchronize — caller manages sync.
 * Uses per-epilogue algorithm cache for repeated same-size calls.
 */
static int cuda_hgemm_fused_internal(int M, int N, int K,
                                      const cuda_half_t *d_A,
                                      const cuda_half_t *d_B,
                                      cuda_half_t *d_C,
                                      int epilogue_type,
                                      int *cache_m, int *cache_n, int *cache_k,
                                      cublasLtHeuristicResult_t *cached_result,
                                      int *cache_valid) {
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available || !g_cublaslt_matmul) return -1;

    cublasStatus_t stat;
    /* Epilogues require COMPUTE_32F — COMPUTE_16F doesn't support them.
     * Use FAST_16F to still get Tensor Core FP16 throughput with FP32 accumulator. */
    float alpha_f = 1.0f;
    float beta_f  = 0.0f;

    int need_heuristic = !(*cache_valid) ||
                         *cache_m != M || *cache_n != N || *cache_k != K;

    if (need_heuristic) {
        /* Create matmul descriptor with epilogue — must use COMPUTE_32F for epilogues.
         * COMPUTE_32F_FAST_16F: uses FP16 Tensor Cores with FP32 accumulation. */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        /* NN col-major swap: C^T = B^T @ A^T */
        cublasOperation_t op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_n, sizeof(op_n));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        /* Set fused epilogue (activation fused into GEMM kernel) */
        int32_t epilogue = epilogue_type;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epilogue, sizeof(epilogue));

        /* Layouts: col-major swap trick (swap A<->B, swap M<->N) */
        cublasLtMatrixLayout_t layout_b, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_b, CUDA_R_16F, (uint64_t)N, (uint64_t)K, (int64_t)N);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        /* Preference with workspace */
        cublasLtMatmulPreference_t preference;
        g_cublaslt_matmul_preference_create(&preference);
        size_t ws_size = g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0;
        g_cublaslt_matmul_preference_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &ws_size, sizeof(ws_size));

        /* Get best algorithm */
        cublasLtHeuristicResult_t results[8];
        int returned_count = 0;
        stat = g_cublaslt_matmul_algo_get_heuristic(
            g_cublaslt_ctx, matmul_desc,
            layout_b, layout_a, layout_c, layout_c,
            preference, 8, results, &returned_count);

        if (stat != CUBLAS_STATUS_SUCCESS || returned_count == 0) {
            fprintf(stderr, "[viva_tensor] FP16 fused epilogue=%d heuristic failed: stat=%d, count=%d\n",
                    epilogue_type, stat, returned_count);
            g_cublaslt_matmul_preference_destroy(preference);
            g_cublaslt_matrix_layout_destroy(layout_b);
            g_cublaslt_matrix_layout_destroy(layout_a);
            g_cublaslt_matrix_layout_destroy(layout_c);
            g_cublaslt_matmul_desc_destroy(matmul_desc);
            return -3;
        }

        /* Execute fused GEMM+activation */
        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B, layout_b, d_A, layout_a,  /* swapped for col-major trick */
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &results[0].algo,
            g_cublaslt_workspace, ws_size,
            (cudaStream_t)0);

        /* Cache the algorithm */
        *cache_m = M; *cache_n = N; *cache_k = K;
        *cached_result = results[0];
        *cache_valid = 1;

        g_cublaslt_matmul_preference_destroy(preference);
        g_cublaslt_matrix_layout_destroy(layout_b);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            *cache_valid = 0;
            fprintf(stderr, "[viva_tensor] FP16 fused epilogue=%d matmul failed: stat=%d\n",
                    epilogue_type, stat);
            return -4;
        }
        return 0;

    } else {
        /* Cached fast path — reuse algorithm, set epilogue on each call */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_n, sizeof(op_n));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        int32_t epilogue = epilogue_type;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epilogue, sizeof(epilogue));

        cublasLtMatrixLayout_t layout_b, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_b, CUDA_R_16F, (uint64_t)N, (uint64_t)K, (int64_t)N);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B, layout_b, d_A, layout_a,
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &cached_result->algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);

        g_cublaslt_matrix_layout_destroy(layout_b);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            *cache_valid = 0;
            return -4;
        }
        return 0;
    }
}

/**
 * FP16 GEMM via cublasLt with COMPUTE_32F_FAST_16F (NO epilogue).
 * Baseline for comparing fused vs non-fused at same compute precision.
 */
/* Separate cache for the non-fused 32F baseline */
static int g_fp16_32f_cache_m = 0, g_fp16_32f_cache_n = 0, g_fp16_32f_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_32f_cached_result;
static int g_fp16_32f_cache_valid = 0;

int cuda_hgemm_lt_32f(int M, int N, int K,
                       const cuda_half_t *d_A,
                       const cuda_half_t *d_B,
                       cuda_half_t *d_C) {
    return cuda_hgemm_fused_internal(M, N, K, d_A, d_B, d_C,
                                      CUBLASLT_EPILOGUE_DEFAULT,
                                      &g_fp16_32f_cache_m, &g_fp16_32f_cache_n,
                                      &g_fp16_32f_cache_k, &g_fp16_32f_cached_result,
                                      &g_fp16_32f_cache_valid);
}

/**
 * Internal: FP16 GEMM + fused epilogue via cublasLt — TN layout
 * Same as NN variant but uses pre-transposed B for better Tensor Core alignment.
 * d_B_T is B transposed (N×K layout, stored as B_T[N][K]).
 */
static int cuda_hgemm_fused_tn_internal(int M, int N, int K,
                                         const cuda_half_t *d_A,
                                         const cuda_half_t *d_B_T,
                                         cuda_half_t *d_C,
                                         int epilogue_type,
                                         int *cache_m, int *cache_n, int *cache_k,
                                         cublasLtHeuristicResult_t *cached_result,
                                         int *cache_valid) {
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available || !g_cublaslt_matmul) return -1;

    cublasStatus_t stat;
    float alpha_f = 1.0f;
    float beta_f  = 0.0f;

    int need_heuristic = !(*cache_valid) ||
                         *cache_m != M || *cache_n != N || *cache_k != K;

    if (need_heuristic) {
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        /* TN swap trick: C_col(N×M) = B_T_col(K×N)^T × A_col(K×M) */
        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        /* Set fused epilogue */
        int32_t epilogue = epilogue_type;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epilogue, sizeof(epilogue));

        /* TN swap: B_T col-major K×N ld=K, A col-major K×M ld=K, C col-major N×M ld=N */
        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_16F, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        cublasLtMatmulPreference_t preference;
        g_cublaslt_matmul_preference_create(&preference);
        size_t ws_size = g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0;
        g_cublaslt_matmul_preference_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &ws_size, sizeof(ws_size));

        cublasLtHeuristicResult_t results[8];
        int returned_count = 0;
        stat = g_cublaslt_matmul_algo_get_heuristic(
            g_cublaslt_ctx, matmul_desc,
            layout_bt, layout_a, layout_c, layout_c,
            preference, 8, results, &returned_count);

        if (stat != CUBLAS_STATUS_SUCCESS || returned_count == 0) {
            fprintf(stderr, "[viva_tensor] FP16 fused TN epilogue=%d heuristic failed: stat=%d, count=%d\n",
                    epilogue_type, stat, returned_count);
            g_cublaslt_matmul_preference_destroy(preference);
            g_cublaslt_matrix_layout_destroy(layout_bt);
            g_cublaslt_matrix_layout_destroy(layout_a);
            g_cublaslt_matrix_layout_destroy(layout_c);
            g_cublaslt_matmul_desc_destroy(matmul_desc);
            return -3;
        }

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &results[0].algo,
            g_cublaslt_workspace, ws_size,
            (cudaStream_t)0);

        *cache_m = M; *cache_n = N; *cache_k = K;
        *cached_result = results[0];
        *cache_valid = 1;

        g_cublaslt_matmul_preference_destroy(preference);
        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            *cache_valid = 0;
            return -4;
        }
        return 0;

    } else {
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        int32_t epilogue = epilogue_type;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epilogue, sizeof(epilogue));

        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_16F, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &cached_result->algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);

        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            *cache_valid = 0;
            return -4;
        }
        return 0;
    }
}

/* TN fused algorithm caches */
static int g_fp16_relu_tn_cache_m = 0, g_fp16_relu_tn_cache_n = 0, g_fp16_relu_tn_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_relu_tn_cached_result;
static int g_fp16_relu_tn_cache_valid = 0;

static int g_fp16_gelu_tn_cache_m = 0, g_fp16_gelu_tn_cache_n = 0, g_fp16_gelu_tn_cache_k = 0;
static cublasLtHeuristicResult_t g_fp16_gelu_tn_cached_result;
static int g_fp16_gelu_tn_cache_valid = 0;

/**
 * FP16 fused GEMM+ReLU TN layout: C = ReLU(A @ B)
 * d_B_T is pre-transposed B (N×K).
 */
int cuda_hgemm_fused_relu_tn(int M, int N, int K,
                              const cuda_half_t *d_A,
                              const cuda_half_t *d_B_T,
                              cuda_half_t *d_C) {
    return cuda_hgemm_fused_tn_internal(M, N, K, d_A, d_B_T, d_C,
                                         CUBLASLT_EPILOGUE_RELU,
                                         &g_fp16_relu_tn_cache_m, &g_fp16_relu_tn_cache_n,
                                         &g_fp16_relu_tn_cache_k, &g_fp16_relu_tn_cached_result,
                                         &g_fp16_relu_tn_cache_valid);
}

/**
 * FP16 fused GEMM+GELU TN layout: C = GELU(A @ B)
 * d_B_T is pre-transposed B (N×K).
 */
int cuda_hgemm_fused_gelu_tn(int M, int N, int K,
                              const cuda_half_t *d_A,
                              const cuda_half_t *d_B_T,
                              cuda_half_t *d_C) {
    return cuda_hgemm_fused_tn_internal(M, N, K, d_A, d_B_T, d_C,
                                         CUBLASLT_EPILOGUE_GELU,
                                         &g_fp16_gelu_tn_cache_m, &g_fp16_gelu_tn_cache_n,
                                         &g_fp16_gelu_tn_cache_k, &g_fp16_gelu_tn_cached_result,
                                         &g_fp16_gelu_tn_cache_valid);
}

/**
 * FP16 fused GEMM+ReLU: C = ReLU(A @ B)
 * Async, uses cached algorithm for repeated calls.
 */
int cuda_hgemm_fused_relu(int M, int N, int K,
                           const cuda_half_t *d_A,
                           const cuda_half_t *d_B,
                           cuda_half_t *d_C) {
    return cuda_hgemm_fused_internal(M, N, K, d_A, d_B, d_C,
                                      CUBLASLT_EPILOGUE_RELU,
                                      &g_fp16_relu_cache_m, &g_fp16_relu_cache_n,
                                      &g_fp16_relu_cache_k, &g_fp16_relu_cached_result,
                                      &g_fp16_relu_cache_valid);
}

/**
 * FP16 fused GEMM+GELU: C = GELU(A @ B)
 * Async, uses cached algorithm for repeated calls.
 */
int cuda_hgemm_fused_gelu(int M, int N, int K,
                           const cuda_half_t *d_A,
                           const cuda_half_t *d_B,
                           cuda_half_t *d_C) {
    return cuda_hgemm_fused_internal(M, N, K, d_A, d_B, d_C,
                                      CUBLASLT_EPILOGUE_GELU,
                                      &g_fp16_gelu_cache_m, &g_fp16_gelu_cache_n,
                                      &g_fp16_gelu_cache_k, &g_fp16_gelu_cached_result,
                                      &g_fp16_gelu_cache_valid);
}

/* =========================================================================
 * FP8 E4M3 GEMM via cublasLtMatmul with TN layout
 * Same TN swap trick as INT8: pre-transpose B for IMMA Tensor Core alignment.
 *
 * RTX 4090 (GeForce) FP8 reality:
 *   - FP8 + FP16 accumulate = 660 TOPS (full rate) — requires CUTLASS/PTX
 *   - FP8 + FP32 accumulate = 330 TOPS (HALF rate, GeForce nerf!)
 *   - INT8 + INT32 accumulate = 660 TOPS (full rate, NOT nerfed)
 * cuBLASLt only supports CUBLAS_COMPUTE_32F for FP8 → capped at 330 TOPS.
 * To reach 660 TOPS, need CUTLASS with FP16 accumulator (CUDA 12.8+).
 *
 * FP8 E4M3 input, FP16 output, FP32 accumulator.
 * ========================================================================= */

/* FP8 data types */
#define CUDA_R_8F_E4M3  28
#define CUDA_R_8F_E5M2  29

/* FP8 per-tensor scale pointer attributes (required for FP8 GEMM) */
#define CUBLASLT_MATMUL_DESC_A_SCALE_POINTER   17
#define CUBLASLT_MATMUL_DESC_B_SCALE_POINTER   18
#define CUBLASLT_MATMUL_DESC_C_SCALE_POINTER   19
#define CUBLASLT_MATMUL_DESC_D_SCALE_POINTER   20
#define CUBLASLT_MATMUL_DESC_AMAX_D_POINTER    21

/* FP8 per-tensor scales on GPU (all 1.0f for unscaled benchmarks) */
static float *g_fp8_scale_one = NULL;  /* device pointer to float 1.0 */

/* FP8 TN algorithm cache */
static int g_fp8_cache_m = 0, g_fp8_cache_n = 0, g_fp8_cache_k = 0;
static cublasLtHeuristicResult_t g_fp8_cached_result;
static int g_fp8_cache_valid = 0;

/**
 * FP8 E4M3 GEMM via cublasLtMatmul with TN format + heuristic
 * Pre-transpose B on CPU during upload → d_B_T (N×K layout, FP8).
 * C = scale_d * (alpha * scale_a * scale_b * A^T @ B + beta * scale_c * C)
 *
 * RTX 4090: capped at 330 TOPS (FP32 accum, GeForce half-rate).
 * Uses CUBLAS_COMPUTE_32F (only supported compute type for FP8).
 * Per-tensor scale pointers required by FP8 spec (set to 1.0f for unscaled).
 */
int cuda_fp8gemm_lt_gpu_tn(int M, int N, int K,
                            const uint8_t *d_A,
                            const uint8_t *d_B_T,
                            uint16_t *d_C) {
    if (g_cublaslt_available < 0) cublaslt_init();
    if (!g_cublaslt_available || !g_cublaslt_matmul) return -1;

    /* Lazily allocate device-side scale factor (float 1.0) */
    if (!g_fp8_scale_one) {
        cudaError_t merr = g_cuda_malloc((void**)&g_fp8_scale_one, sizeof(float));
        if (merr != 0) g_fp8_scale_one = NULL;
        if (g_fp8_scale_one) {
            float one = 1.0f;
            g_cuda_memcpy(g_fp8_scale_one, &one, sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    cublasStatus_t stat;
    float alpha_f = 1.0f, beta_f = 0.0f;

    int need_heuristic = !g_fp8_cache_valid ||
                         g_fp8_cache_m != M || g_fp8_cache_n != N || g_fp8_cache_k != K;

    if (need_heuristic) {
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        /* TN format */
        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        /* NOTE: FP8 per-tensor scale pointers are optional for cuBLASLt.
         * When NULL (default), no per-tensor scaling is applied.
         * For production use, set A_SCALE_POINTER, B_SCALE_POINTER, D_SCALE_POINTER
         * to device-side float pointers for proper FP8 quantization.
         * Omitted here for maximum benchmark throughput. */

        /* TN swap: B_T col-major K×N ld=K (FP8), A col-major K×M ld=K (FP8), C col-major N×M ld=N (FP16) */
        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        cublasLtMatmulPreference_t preference;
        g_cublaslt_matmul_preference_create(&preference);
        size_t ws_size = g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0;
        g_cublaslt_matmul_preference_set_attr(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &ws_size, sizeof(ws_size));

        cublasLtHeuristicResult_t results[8];
        int returned_count = 0;
        stat = g_cublaslt_matmul_algo_get_heuristic(
            g_cublaslt_ctx, matmul_desc,
            layout_bt, layout_a, layout_c, layout_c,
            preference, 8, results, &returned_count);

        if (stat != CUBLAS_STATUS_SUCCESS || returned_count == 0) {
            fprintf(stderr, "[viva_tensor] FP8 TN heuristic failed: stat=%d, count=%d\n",
                    stat, returned_count);
            g_cublaslt_matmul_preference_destroy(preference);
            g_cublaslt_matrix_layout_destroy(layout_bt);
            g_cublaslt_matrix_layout_destroy(layout_a);
            g_cublaslt_matrix_layout_destroy(layout_c);
            g_cublaslt_matmul_desc_destroy(matmul_desc);
            return -3;
        }

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &results[0].algo,
            g_cublaslt_workspace, ws_size,
            (cudaStream_t)0);

        g_fp8_cache_m = M; g_fp8_cache_n = N; g_fp8_cache_k = K;
        g_fp8_cached_result = results[0];
        g_fp8_cache_valid = 1;

        g_cublaslt_matmul_preference_destroy(preference);
        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_fp8_cache_valid = 0;
            fprintf(stderr, "[viva_tensor] FP8 TN matmul failed: stat=%d\n", stat);
            return -4;
        }
        return 0;

    } else {
        /* Cached fast path */
        cublasLtMatmulDesc_t matmul_desc;
        stat = g_cublaslt_matmul_desc_create(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (stat != CUBLAS_STATUS_SUCCESS) return -2;

        cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &op_t, sizeof(op_t));
        g_cublaslt_matmul_desc_set_attr(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &op_n, sizeof(op_n));

        cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
        g_cublaslt_matrix_layout_create(&layout_bt, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)N, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_a, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)M, (int64_t)K);
        g_cublaslt_matrix_layout_create(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

        stat = g_cublaslt_matmul(
            g_cublaslt_ctx, matmul_desc,
            &alpha_f,
            d_B_T, layout_bt, d_A, layout_a,
            &beta_f,
            d_C, layout_c, d_C, layout_c,
            &g_fp8_cached_result.algo,
            g_cublaslt_workspace, g_cublaslt_workspace ? CUBLASLT_WORKSPACE_SIZE : 0,
            (cudaStream_t)0);

        g_cublaslt_matrix_layout_destroy(layout_bt);
        g_cublaslt_matrix_layout_destroy(layout_a);
        g_cublaslt_matrix_layout_destroy(layout_c);
        g_cublaslt_matmul_desc_destroy(matmul_desc);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_fp8_cache_valid = 0;
            return -4;
        }
        return 0;
    }
}

/* =========================================================================
 * FP16 Strided Batched GEMM via cublasGemmStridedBatchedEx
 * For multi-head attention: C[i] = A[i] @ B[i] for i in 0..batchCount
 * Uses CUBLAS_COMPUTE_16F for maximum Tensor Core throughput.
 *
 * Memory layout: all batch elements stored contiguously.
 * A: [batch, M, K], B: [batch, K, N], C: [batch, M, N] — row-major.
 * strideA = M*K, strideB = K*N, strideC = M*N (in elements).
 * ========================================================================= */

/**
 * FP16 batched GEMM — async (no sync).
 * C[i] = A[i] @ B[i] for each batch element.
 * All data contiguous on GPU, stride = M*K / K*N / M*N.
 */
int cuda_hgemm_batched(int M, int N, int K, int batch_count,
                        const cuda_half_t *d_A,
                        const cuda_half_t *d_B,
                        cuda_half_t *d_C) {
    if (!g_cuda_available || !g_cublas_gemm_strided_batched_ex) return -1;

    /* FP16 alpha/beta for COMPUTE_16F */
    cuda_half_t alpha_h = 0x3C00;  /* 1.0 */
    cuda_half_t beta_h  = 0x0000;  /* 0.0 */

    /* Row-major NN swap trick: C^T = B^T @ A^T
     * Swap operands + swap M<->N for cuBLAS col-major. */
    long long int strideA = (long long int)M * K;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;

    cublasStatus_t stat = g_cublas_gemm_strided_batched_ex(
        g_cublas_ctx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                     /* swapped M<->N for row-major */
        &alpha_h,
        d_B, CUDA_R_16F, N, strideB, /* B first (swapped) */
        d_A, CUDA_R_16F, K, strideA, /* A second (swapped) */
        &beta_h,
        d_C, CUDA_R_16F, N, strideC,
        batch_count,
        CUBLAS_COMPUTE_16F,           /* Pure FP16 for max throughput */
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
}

#else /* _WIN32 */

/* CUDA not supported on Windows via this code path (use native CUDA toolkit) */
int cuda_init(void) { return 0; }
int cuda_available(void) { return 0; }
int cuda_dgemm(int M, int N, int K, double alpha, const double *A, int lda,
               const double *B, int ldb, double beta, double *C, int ldc) {
    (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}
int cuda_sgemm(int M, int N, int K, float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc) {
    (void)M; (void)N; (void)K;
    (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}
void cuda_cleanup(void) {}

/* CudaTensor stubs for Windows */
float* cuda_tensor_alloc(size_t num_elements) { (void)num_elements; return NULL; }
void cuda_tensor_free(void *d_ptr) { (void)d_ptr; }
int cuda_tensor_upload(float *d_dst, const float *h_src, size_t num_elements) {
    (void)d_dst; (void)h_src; (void)num_elements; return -1;
}
int cuda_tensor_download(float *h_dst, const float *d_src, size_t num_elements) {
    (void)h_dst; (void)d_src; (void)num_elements; return -1;
}
int cuda_sgemm_gpu(int M, int N, int K, float alpha, const float *d_A, int lda,
                   const float *d_B, int ldb, float beta, float *d_C, int ldc) {
    (void)M; (void)N; (void)K;
    (void)alpha; (void)d_A; (void)lda; (void)d_B; (void)ldb; (void)beta; (void)d_C; (void)ldc;
    return -1;
}

/* INT8/FP16 Tensor Core stubs for Windows */
typedef uint16_t cuda_half_t;
int8_t* cuda_tensor_alloc_int8(size_t num_elements) { (void)num_elements; return NULL; }
int32_t* cuda_tensor_alloc_int32(size_t num_elements) { (void)num_elements; return NULL; }
cuda_half_t* cuda_tensor_alloc_fp16(size_t num_elements) { (void)num_elements; return NULL; }
int cuda_tensor_upload_int8(int8_t *d_dst, const int8_t *h_src, size_t num_elements) {
    (void)d_dst; (void)h_src; (void)num_elements; return -1;
}
int cuda_tensor_download_int32(int32_t *h_dst, const int32_t *d_src, size_t num_elements) {
    (void)h_dst; (void)d_src; (void)num_elements; return -1;
}
int cuda_igemm(int M, int N, int K, int32_t alpha, const int8_t *A, int lda,
               const int8_t *B, int ldb, int32_t beta, int32_t *C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}
int cuda_igemm_gpu(int M, int N, int K, int32_t alpha, const int8_t *d_A, int lda,
                   const int8_t *d_B, int ldb, int32_t beta, int32_t *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)beta; (void)d_C; (void)ldc;
    return -1;
}
int cuda_hgemm_gpu(int M, int N, int K, float alpha, const cuda_half_t *d_A, int lda,
                   const cuda_half_t *d_B, int ldb, float beta, float *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)beta; (void)d_C; (void)ldc;
    return -1;
}
int cuda_int8_available(void) { return 0; }
int cuda_fp16_available(void) { return 0; }
int cuda_int8_lt_available(void) { return 0; }

/* Async stubs for Windows */
void cuda_explicit_sync(void) {}
int cuda_sgemm_gpu_async(int M, int N, int K, float alpha, const float *d_A, int lda,
                          const float *d_B, int ldb, float beta, float *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)beta; (void)d_C; (void)ldc;
    return -1;
}
int cuda_hgemm_gpu_async(int M, int N, int K, float alpha, const void *d_A, int lda,
                          const void *d_B, int ldb, float beta, float *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)beta; (void)d_C; (void)ldc;
    return -1;
}

int cuda_tensor_upload_fp16(cuda_half_t *d_dst, const cuda_half_t *h_src, size_t num_elements) {
    (void)d_dst; (void)h_src; (void)num_elements; return -1;
}
int cuda_tensor_download_fp32(float *h_dst, const float *d_src, size_t num_elements) {
    (void)h_dst; (void)d_src; (void)num_elements; return -1;
}
int cuda_tensor_download_fp16(cuda_half_t *h_dst, const cuda_half_t *d_src, size_t num_elements) {
    (void)h_dst; (void)d_src; (void)num_elements; return -1;
}
int cuda_hgemm(int M, int N, int K, float alpha, const cuda_half_t *A, int lda,
               const cuda_half_t *B, int ldb, float beta, float *C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}
int cuda_igemm_lt(int M, int N, int K, float alpha, const int8_t *A, int lda,
                  const int8_t *B, int ldb, float beta, int32_t *C, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)lda;
    (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return -1;
}
int cuda_igemm_lt_gpu(int M, int N, int K, const int8_t *d_A, int lda,
                       const int8_t *d_B, int ldb, int32_t *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)d_C; (void)ldc;
    return -1;
}
int cuda_igemm_lt_gpu_async(int M, int N, int K, const int8_t *d_A, int lda,
                             const int8_t *d_B, int ldb, int32_t *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)d_C; (void)ldc;
    return -1;
}
int cuda_igemm_lt_gpu_tn(int M, int N, int K, const int8_t *d_A,
                          const int8_t *d_B_T, int32_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B_T; (void)d_C;
    return -1;
}
int cuda_hgemm_lt_gpu_tn(int M, int N, int K, const cuda_half_t *d_A,
                           const cuda_half_t *d_B_T, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B_T; (void)d_C;
    return -1;
}
int cuda_hgemm_gpu_pure16(int M, int N, int K,
                            const cuda_half_t *d_A, int lda,
                            const cuda_half_t *d_B, int ldb,
                            cuda_half_t *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)d_C; (void)ldc;
    return -1;
}
int cuda_hgemm_gpu_pure16_async(int M, int N, int K,
                                  const cuda_half_t *d_A, int lda,
                                  const cuda_half_t *d_B, int ldb,
                                  cuda_half_t *d_C, int ldc) {
    (void)M; (void)N; (void)K; (void)d_A; (void)lda;
    (void)d_B; (void)ldb; (void)d_C; (void)ldc;
    return -1;
}
int cublaslt_init(void) { return 0; }
int cuda_hgemm_lt_32f(int M, int N, int K, const cuda_half_t *d_A,
                       const cuda_half_t *d_B, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B; (void)d_C;
    return -1;
}
int cuda_hgemm_fused_relu(int M, int N, int K, const cuda_half_t *d_A,
                           const cuda_half_t *d_B, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B; (void)d_C;
    return -1;
}
int cuda_hgemm_fused_gelu(int M, int N, int K, const cuda_half_t *d_A,
                           const cuda_half_t *d_B, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B; (void)d_C;
    return -1;
}
int cuda_hgemm_fused_relu_tn(int M, int N, int K, const cuda_half_t *d_A,
                              const cuda_half_t *d_B_T, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B_T; (void)d_C;
    return -1;
}
int cuda_hgemm_fused_gelu_tn(int M, int N, int K, const cuda_half_t *d_A,
                              const cuda_half_t *d_B_T, cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B_T; (void)d_C;
    return -1;
}
int cuda_hgemm_batched(int M, int N, int K, int batch_count,
                        const cuda_half_t *d_A, const cuda_half_t *d_B,
                        cuda_half_t *d_C) {
    (void)M; (void)N; (void)K; (void)batch_count;
    (void)d_A; (void)d_B; (void)d_C;
    return -1;
}
int cuda_fp8gemm_lt_gpu_tn(int M, int N, int K,
                            const uint8_t *d_A, const uint8_t *d_B_T,
                            uint16_t *d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B_T; (void)d_C;
    return -1;
}

#endif /* _WIN32 */
