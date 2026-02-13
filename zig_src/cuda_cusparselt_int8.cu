/**
 * cuSPARSELt 0.8.1 — Multi-datatype 2:4 Structured Sparse GEMM
 *
 * INT8 sparse:  1321 TOPS peak on RTX 4090
 * FP8 E4M3 sparse: 1321 TOPS peak (NEW in v0.8.0 for SM89!)
 * FP16 sparse:  660 TFLOPS peak
 *
 * Returns kernel-only elapsed time via CUDA events (microseconds).
 * Setup (init, prune, compress, search) is NOT included in timing.
 *
 * Optimizations:
 * - CUSPARSELT_MATMUL_SEARCH_ITERATIONS = 20
 * - Reports ALG_CONFIG_MAX_ID to verify algorithm candidate pool
 * - CUSPARSELT_MATMUL_SPARSE_MAT_POINTER set before search
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cusparseLt.h"

/* FP8 E4M3 data type ID */
#ifndef CUDA_R_8F_E4M3
#define CUDA_R_8F_E4M3 ((cudaDataType)28)
#endif

#define CHECK_CUDA(func) do {                                       \
    cudaError_t status = (func);                                    \
    if (status != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error at line %d: %s\n",             \
                __LINE__, cudaGetErrorString(status));              \
        return -100;                                                \
    }                                                               \
} while(0)

#define CHECK_CUSPARSE(func) do {                                   \
    cusparseStatus_t status = (func);                               \
    if (status != CUSPARSE_STATUS_SUCCESS) {                        \
        fprintf(stderr, "cuSPARSELt error at line %d: %s (%d)\n",  \
                __LINE__, cusparseLtGetErrorString(status), status);\
        return -1;                                                  \
    }                                                               \
} while(0)

extern "C" {

/**
 * cusparselt_int8_sparse_bench(M, N, K, iters, mode)
 *
 * mode: 0 = auto (MatmulSearch), 2 = split-K one kernel, 3 = split-K two kernels
 * Returns: elapsed microseconds (>0) or negative on error.
 */
int cusparselt_int8_sparse_bench(int M, int N, int K, int iters, int mode) {
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t   matmulDescr;
    cusparseLtMatmulAlgSelection_t algSel;
    cusparseLtMatmulPlan_t         plan;

    CHECK_CUSPARSE(cusparseLtInit(&handle));

    /* A (sparse): INT8, ROW-major */
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &handle, &matA, M, K, K,
        16, CUDA_R_8I, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    /* B (dense): INT8, COL-major → NN layout */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matB, K, N, K,
        16, CUDA_R_8I, CUSPARSE_ORDER_COL));

    /* C/D: INT32, ROW-major */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, N,
        16, CUDA_R_32I, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matD, M, N, N,
        16, CUDA_R_32I, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        &handle, &matmulDescr,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD,
        CUSPARSE_COMPUTE_32I));

    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
        &handle, &algSel, &matmulDescr,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    /* Increase search iterations for more stable tuning */
    int searchIters = 20;
    cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
        &searchIters, sizeof(searchIters));

    /* Report algorithm candidate pool size */
    int maxAlgId = 0;
    cusparseLtMatmulAlgGetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &maxAlgId, sizeof(maxAlgId));
    fprintf(stderr, "[cuSPARSELt INT8] %dx%dx%d: %d algo candidates\n",
            M, N, K, maxAlgId);

    /* Configure split-K if requested */
    if (mode == 2) {
        cusparseLtSplitKMode_t sk = CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
        cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
            CUSPARSELT_MATMUL_SPLIT_K_MODE, &sk, sizeof(sk));
        int skBufs = 8;
        cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
            CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &skBufs, sizeof(skBufs));
    } else if (mode == 3) {
        cusparseLtSplitKMode_t sk = CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
        cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
            CUSPARSELT_MATMUL_SPLIT_K_MODE, &sk, sizeof(sk));
        int skBufs = 8;
        cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
            CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &skBufs, sizeof(skBufs));
    }

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel));

    size_t workspaceSize = 0;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize));

    /* Allocate GPU memory */
    int8_t  *d_A = nullptr, *d_B = nullptr;
    int32_t *d_C = nullptr;
    void    *d_workspace = nullptr;
    void    *d_compressed = nullptr;
    void    *d_compressedBuffer = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(d_A, 1, (size_t)M * K));
    CHECK_CUDA(cudaMemset(d_B, 1, (size_t)K * N));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(int32_t)));

    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    /* Prune + compress */
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(
        &handle, &matmulDescr, d_A, d_A,
        CUSPARSELT_PRUNE_SPMMA_STRIP, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t compressedSize = 0, compressedBufferSize = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        &handle, &plan, &compressedSize, &compressedBufferSize));
    CHECK_CUDA(cudaMalloc(&d_compressed, compressedSize));
    CHECK_CUDA(cudaMalloc(&d_compressedBuffer, compressedBufferSize));
    CHECK_CUSPARSE(cusparseLtSpMMACompress(
        &handle, &plan, d_A, d_compressed, d_compressedBuffer, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Set SPARSE_MAT_POINTER hint for better algo search */
    cusparseLtMatmulDescSetAttribute(&handle, &matmulDescr,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
        &d_compressed, sizeof(d_compressed));

    float alpha = 1.0f, beta = 0.0f;

    /* Auto-tune: search for best algorithm */
    CHECK_CUSPARSE(cusparseLtMatmulSearch(
        &handle, &plan, &alpha,
        d_compressed, d_B, &beta, d_C, d_C,
        d_workspace, nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Warmup (10 iterations) */
    for (int w = 0; w < 10; w++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Bench loop with CUDA event timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iters; i++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    int elapsed_us = (int)(elapsed_ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Cleanup */
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    cusparseLtDestroy(&handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_compressed);
    cudaFree(d_compressedBuffer);

    return (elapsed_us > 0) ? elapsed_us : 1;
}

/**
 * cusparselt_fp8_sparse_bench(M, N, K, iters)
 *
 * FP8 E4M3 2:4 sparse — NEW in cuSPARSELt v0.8.0 for SM89!
 * Same 1321 TOPS peak as INT8 sparse (same 8-bit datapath).
 * FP8 input, FP16 output, FP32 accumulator.
 */
int cusparselt_fp8_sparse_bench(int M, int N, int K, int iters) {
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t   matmulDescr;
    cusparseLtMatmulAlgSelection_t algSel;
    cusparseLtMatmulPlan_t         plan;

    CHECK_CUSPARSE(cusparseLtInit(&handle));

    /* A (sparse): FP8 E4M3, ROW-major */
    cusparseStatus_t st = cusparseLtStructuredDescriptorInit(
        &handle, &matA, M, K, K,
        16, CUDA_R_8F_E4M3, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[cuSPARSELt FP8] StructuredDescriptorInit failed: %s (%d)\n",
                cusparseLtGetErrorString(st), st);
        cusparseLtDestroy(&handle);
        return -2;
    }

    /* B (dense): FP8 E4M3, COL-major */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matB, K, N, K,
        16, CUDA_R_8F_E4M3, CUSPARSE_ORDER_COL));

    /* C/D: FP16, ROW-major */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, N,
        16, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matD, M, N, N,
        16, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        &handle, &matmulDescr,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD,
        CUSPARSE_COMPUTE_32F));

    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
        &handle, &algSel, &matmulDescr,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    int searchIters = 20;
    cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
        &searchIters, sizeof(searchIters));

    int maxAlgId = 0;
    cusparseLtMatmulAlgGetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &maxAlgId, sizeof(maxAlgId));
    fprintf(stderr, "[cuSPARSELt FP8] %dx%dx%d: %d algo candidates\n",
            M, N, K, maxAlgId);

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel));

    size_t workspaceSize = 0;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize));

    /* Allocate GPU memory (FP8 = 1 byte per element) */
    void    *d_A = nullptr, *d_B = nullptr;
    void    *d_C = nullptr;  /* FP16 = 2 bytes */
    void    *d_workspace = nullptr;
    void    *d_compressed = nullptr;
    void    *d_compressedBuffer = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));  /* FP16 */
    CHECK_CUDA(cudaMemset(d_A, 0x38, (size_t)M * K));  /* ~0.5 in E4M3 */
    CHECK_CUDA(cudaMemset(d_B, 0x38, (size_t)K * N));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * 2));

    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    /* Prune + compress */
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(
        &handle, &matmulDescr, d_A, d_A,
        CUSPARSELT_PRUNE_SPMMA_STRIP, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t compressedSize = 0, compressedBufferSize = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        &handle, &plan, &compressedSize, &compressedBufferSize));
    CHECK_CUDA(cudaMalloc(&d_compressed, compressedSize));
    CHECK_CUDA(cudaMalloc(&d_compressedBuffer, compressedBufferSize));
    CHECK_CUSPARSE(cusparseLtSpMMACompress(
        &handle, &plan, d_A, d_compressed, d_compressedBuffer, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Set SPARSE_MAT_POINTER hint */
    cusparseLtMatmulDescSetAttribute(&handle, &matmulDescr,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
        &d_compressed, sizeof(d_compressed));

    float alpha = 1.0f, beta = 0.0f;

    /* Auto-tune */
    CHECK_CUSPARSE(cusparseLtMatmulSearch(
        &handle, &plan, &alpha,
        d_compressed, d_B, &beta, d_C, d_C,
        d_workspace, nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Warmup */
    for (int w = 0; w < 10; w++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Bench with CUDA event timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iters; i++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    int elapsed_us = (int)(elapsed_ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    cusparseLtDestroy(&handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_compressed);
    cudaFree(d_compressedBuffer);

    return (elapsed_us > 0) ? elapsed_us : 1;
}

/**
 * cusparselt_fp16_sparse_bench(M, N, K, iters)
 *
 * FP16 2:4 sparse — 660 TFLOPS peak on RTX 4090.
 * FP16 input/output, FP32 accumulator.
 */
int cusparselt_fp16_sparse_bench(int M, int N, int K, int iters) {
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t   matmulDescr;
    cusparseLtMatmulAlgSelection_t algSel;
    cusparseLtMatmulPlan_t         plan;

    CHECK_CUSPARSE(cusparseLtInit(&handle));

    /* A (sparse): FP16, ROW-major */
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &handle, &matA, M, K, K,
        16, CUDA_R_16F, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT));

    /* B (dense): FP16, COL-major */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matB, K, N, K,
        16, CUDA_R_16F, CUSPARSE_ORDER_COL));

    /* C/D: FP16, ROW-major */
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, N,
        16, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &handle, &matD, M, N, N,
        16, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        &handle, &matmulDescr,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matD,
        CUSPARSE_COMPUTE_32F));

    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
        &handle, &algSel, &matmulDescr,
        CUSPARSELT_MATMUL_ALG_DEFAULT));

    int searchIters = 20;
    cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
        &searchIters, sizeof(searchIters));

    int maxAlgId = 0;
    cusparseLtMatmulAlgGetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &maxAlgId, sizeof(maxAlgId));
    fprintf(stderr, "[cuSPARSELt FP16] %dx%dx%d: %d algo candidates\n",
            M, N, K, maxAlgId);

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel));

    size_t workspaceSize = 0;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize));

    void *d_A = nullptr, *d_B = nullptr;
    void *d_C = nullptr;
    void *d_workspace = nullptr;
    void *d_compressed = nullptr;
    void *d_compressedBuffer = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * 2));  /* FP16 */
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * 2));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * 2));
    CHECK_CUDA(cudaMemset(d_A, 0x3C, (size_t)M * K * 2));
    CHECK_CUDA(cudaMemset(d_B, 0x3C, (size_t)K * N * 2));
    CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * 2));

    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    CHECK_CUSPARSE(cusparseLtSpMMAPrune(
        &handle, &matmulDescr, d_A, d_A,
        CUSPARSELT_PRUNE_SPMMA_STRIP, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t compressedSize = 0, compressedBufferSize = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        &handle, &plan, &compressedSize, &compressedBufferSize));
    CHECK_CUDA(cudaMalloc(&d_compressed, compressedSize));
    CHECK_CUDA(cudaMalloc(&d_compressedBuffer, compressedBufferSize));
    CHECK_CUSPARSE(cusparseLtSpMMACompress(
        &handle, &plan, d_A, d_compressed, d_compressedBuffer, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Set SPARSE_MAT_POINTER hint */
    cusparseLtMatmulDescSetAttribute(&handle, &matmulDescr,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
        &d_compressed, sizeof(d_compressed));

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUSPARSE(cusparseLtMatmulSearch(
        &handle, &plan, &alpha,
        d_compressed, d_B, &beta, d_C, d_C,
        d_workspace, nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Warmup */
    for (int w = 0; w < 10; w++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Bench */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iters; i++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C,
            d_workspace, nullptr, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    int elapsed_us = (int)(elapsed_ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatDescriptorDestroy(&matD);
    cusparseLtDestroy(&handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_compressed);
    cudaFree(d_compressedBuffer);

    return (elapsed_us > 0) ? elapsed_us : 1;
}

}  /* extern "C" */
