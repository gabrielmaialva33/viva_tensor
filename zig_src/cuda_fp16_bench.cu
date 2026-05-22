/**
 * Self-contained FP16 GEMM benchmark via cublasLt + COMPUTE_16F.
 *
 * The dense FP16 path in nif_cpu_ops.c goes through cuda_hgemm which uses
 * CUBLAS_COMPUTE_32F_FAST_16F — half-rate on GeForce (the GPU-vendor's nerf
 * for consumer cards). This bench calls cuda_hgemm_lt_gpu_tn which uses
 * CUBLAS_COMPUTE_16F (FP16 accum) + FP16 alpha/beta, unlocking full-rate
 * Tensor Core throughput (~165 TFLOPS theoretical on Ada / 4090).
 *
 * Allocates A[M,K], B^T[K,N], C[M,N] on the device, runs `iters` GEMMs,
 * returns elapsed microseconds (kernel-only via CUDA events).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

/* Forward decl — defined in cuda_gemm.c */
typedef __half cuda_half_t;
int cuda_hgemm_lt_gpu_tn(int M, int N, int K, const cuda_half_t *d_A, const cuda_half_t *d_B_T,
                         cuda_half_t *d_C);

int cublaslt_fp16_bench(int M, int N, int K, int iters) {
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0)
        return -10;

    size_t bytes_A = (size_t)M * K * 2; /* FP16 = 2 bytes */
    size_t bytes_BT = (size_t)K * N * 2;
    size_t bytes_C = (size_t)M * N * 2;

    cuda_half_t *d_A = nullptr, *d_BT = nullptr, *d_C = nullptr;
    if (cudaMalloc((void **)&d_A, bytes_A) != cudaSuccess)
        return -12;
    if (cudaMalloc((void **)&d_BT, bytes_BT) != cudaSuccess) {
        cudaFree(d_A);
        return -13;
    }
    if (cudaMalloc((void **)&d_C, bytes_C) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_BT);
        return -14;
    }

    /* Fill A/B with FP16 ~0.5 (0x3800) to avoid Inf/NaN. Leave C zero. */
    cudaMemset(d_A, 0x38, bytes_A);
    cudaMemset(d_BT, 0x38, bytes_BT);
    cudaMemset(d_C, 0, bytes_C);

    /* Warmup (also populates the per-shape heuristic cache in cuda_gemm.c). */
    int rc = cuda_hgemm_lt_gpu_tn(M, N, K, d_A, d_BT, d_C);
    if (rc != 0) {
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
        return rc - 100;
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; ++i) {
        cuda_hgemm_lt_gpu_tn(M, N, K, d_A, d_BT, d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_BT);
    cudaFree(d_C);

    return (int)(elapsed_ms * 1000.0f);
}

} /* extern "C" */
