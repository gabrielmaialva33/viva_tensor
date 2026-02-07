// Test INT8 IMMA vs DP4A on RTX 4090
// Compile: nvcc -o test_int8_imma test_int8_imma.cu -lcublas
// Profile: ncu --metrics smsp__sass_inst_executed_op_imma ./test_int8_imma

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int M = 4096, N = 4096, K = 4096;

    printf("Testing INT8 GEMM on RTX 4090\n");
    printf("Matrix size: %dx%d @ %dx%d\n", M, K, K, N);

    // Allocate
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    // Initialize with some data
    int8_t *h_A = (int8_t*)malloc(M * K);
    int8_t *h_B = (int8_t*)malloc(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = (i % 127) - 64;
    for (int i = 0; i < K * N; i++) h_B[i] = (i % 127) - 64;
    cudaMemcpy(d_A, h_A, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set Tensor Op math mode
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    int32_t alpha = 1, beta = 0;

    // Warmup
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_8I, N,
        d_A, CUDA_R_8I, K,
        &beta,
        d_C, CUDA_R_32I, N,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_8I, N,
            d_A, CUDA_R_8I, K,
            &beta,
            d_C, CUDA_R_32I, N,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N * K * iters;
    double tflops = flops / (ms / 1000.0) / 1e12;

    printf("Time: %.2f ms for %d iterations\n", ms, iters);
    printf("Performance: %.1f TFLOPS\n", tflops);
    printf("\nExpected for IMMA: 200-400 TFLOPS\n");
    printf("If < 100 TFLOPS, likely using DP4A fallback!\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    cublasDestroy(handle);

    return 0;
}
