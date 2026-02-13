/**
 * cuSPARSELt Layout test — find the layout with most algorithm candidates.
 * Tests 4 layout combinations for INT8 sparse GEMM.
 * Returns: algo candidate count (>0) or negative on error.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cusparseLt.h"

extern "C" {

/**
 * cusparselt_layout_test(M, N, K, iters, layout)
 *
 * layout: 0 = ROW/COL NN (current)
 *         1 = COL/COL TN
 *         2 = ROW/ROW NT
 *         3 = COL/ROW TT
 *
 * Returns: elapsed microseconds (>0) or negative on error.
 * Prints algo candidate count to stderr.
 */
int cusparselt_layout_test(int M, int N, int K, int iters, int layout) {
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t   matmulDescr;
    cusparseLtMatmulAlgSelection_t algSel;
    cusparseLtMatmulPlan_t         plan;

    cusparseStatus_t st;
    st = cusparseLtInit(&handle);
    if (st != CUSPARSE_STATUS_SUCCESS) return -1;

    cusparseOrder_t orderA, orderB, orderC;
    cusparseOperation_t opA, opB;
    int64_t ldA, ldB, ldC;

    switch (layout) {
    case 0: /* ROW/COL NN */
        orderA = CUSPARSE_ORDER_ROW; orderB = CUSPARSE_ORDER_COL; orderC = CUSPARSE_ORDER_ROW;
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE; opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        ldA = K; ldB = K; ldC = N;
        break;
    case 1: /* COL/COL TN */
        orderA = CUSPARSE_ORDER_COL; orderB = CUSPARSE_ORDER_COL; orderC = CUSPARSE_ORDER_COL;
        opA = CUSPARSE_OPERATION_TRANSPOSE; opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        ldA = K; ldB = K; ldC = M;
        break;
    case 2: /* ROW/ROW NT */
        orderA = CUSPARSE_ORDER_ROW; orderB = CUSPARSE_ORDER_ROW; orderC = CUSPARSE_ORDER_ROW;
        opA = CUSPARSE_OPERATION_NON_TRANSPOSE; opB = CUSPARSE_OPERATION_TRANSPOSE;
        ldA = K; ldB = N; ldC = N;
        break;
    case 3: /* COL/ROW TT */
        orderA = CUSPARSE_ORDER_COL; orderB = CUSPARSE_ORDER_ROW; orderC = CUSPARSE_ORDER_COL;
        opA = CUSPARSE_OPERATION_TRANSPOSE; opB = CUSPARSE_OPERATION_TRANSPOSE;
        ldA = K; ldB = N; ldC = M;
        break;
    default:
        return -2;
    }

    const char *layoutNames[] = {"ROW/COL NN", "COL/COL TN", "ROW/ROW NT", "COL/ROW TT"};

    st = cusparseLtStructuredDescriptorInit(
        &handle, &matA, (opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? M : K),
        (opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? K : M), ldA,
        16, CUDA_R_8I, orderA, CUSPARSELT_SPARSITY_50_PERCENT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] matA init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -3;
    }

    st = cusparseLtDenseDescriptorInit(
        &handle, &matB, (opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? K : N),
        (opB == CUSPARSE_OPERATION_NON_TRANSPOSE ? N : K), ldB,
        16, CUDA_R_8I, orderB);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] matB init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -4;
    }

    st = cusparseLtDenseDescriptorInit(
        &handle, &matC, M, N, ldC, 16, CUDA_R_32I, orderC);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] matC init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -5;
    }

    st = cusparseLtDenseDescriptorInit(
        &handle, &matD, M, N, ldC, 16, CUDA_R_32I, orderC);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] matD init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -6;
    }

    st = cusparseLtMatmulDescriptorInit(
        &handle, &matmulDescr, opA, opB,
        &matA, &matB, &matC, &matD, CUSPARSE_COMPUTE_32I);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] matmul descr init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -7;
    }

    st = cusparseLtMatmulAlgSelectionInit(
        &handle, &algSel, &matmulDescr, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] alg sel init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -8;
    }

    int maxAlgId = 0;
    cusparseLtMatmulAlgGetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &maxAlgId, sizeof(maxAlgId));

    int searchIters = 20;
    cusparseLtMatmulAlgSetAttribute(&handle, &algSel,
        CUSPARSELT_MATMUL_SEARCH_ITERATIONS, &searchIters, sizeof(searchIters));

    st = cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSel);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] plan init failed: %d\n", layout, layoutNames[layout], st);
        cusparseLtDestroy(&handle);
        return -9;
    }

    size_t workspaceSize = 0;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);

    /* Allocate */
    int8_t  *d_A = nullptr, *d_B = nullptr;
    int32_t *d_C = nullptr;
    void    *d_workspace = nullptr, *d_compressed = nullptr, *d_compressedBuffer = nullptr;

    cudaMalloc(&d_A, (size_t)M * K);
    cudaMalloc(&d_B, (size_t)K * N);
    cudaMalloc(&d_C, (size_t)M * N * sizeof(int32_t));
    cudaMemset(d_A, 1, (size_t)M * K);
    cudaMemset(d_B, 1, (size_t)K * N);
    cudaMemset(d_C, 0, (size_t)M * N * sizeof(int32_t));
    if (workspaceSize > 0) cudaMalloc(&d_workspace, workspaceSize);

    /* Prune + compress */
    cusparseLtSpMMAPrune(&handle, &matmulDescr, d_A, d_A, CUSPARSELT_PRUNE_SPMMA_STRIP, 0);
    cudaDeviceSynchronize();

    size_t compSz = 0, compBufSz = 0;
    cusparseLtSpMMACompressedSize(&handle, &plan, &compSz, &compBufSz);
    cudaMalloc(&d_compressed, compSz);
    cudaMalloc(&d_compressedBuffer, compBufSz);
    cusparseLtSpMMACompress(&handle, &plan, d_A, d_compressed, d_compressedBuffer, 0);
    cudaDeviceSynchronize();

    /* Set sparse pointer hint */
    cusparseLtMatmulDescSetAttribute(&handle, &matmulDescr,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &d_compressed, sizeof(d_compressed));

    float alpha = 1.0f, beta = 0.0f;

    /* Search */
    st = cusparseLtMatmulSearch(&handle, &plan, &alpha,
        d_compressed, d_B, &beta, d_C, d_C, d_workspace, nullptr, 0);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[layout %d %s] search failed: %d\n", layout, layoutNames[layout], st);
        goto cleanup;
    }
    cudaDeviceSynchronize();

    /* Warmup */
    for (int w = 0; w < 10; w++) {
        cusparseLtMatmul(&handle, &plan, &alpha,
            d_compressed, d_B, &beta, d_C, d_C, d_workspace, nullptr, 0);
    }
    cudaDeviceSynchronize();

    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; i++) {
            cusparseLtMatmul(&handle, &plan, &alpha,
                d_compressed, d_B, &beta, d_C, d_C, d_workspace, nullptr, 0);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        int us = (int)(ms * 1000.0f);

        double ops = 2.0 * M * N * K * (double)iters;
        double tops = ops / (ms / 1000.0) / 1.0e12;

        fprintf(stderr, "[layout %d %s] %d algos, %.2f ms, %.1f TOPS (%.1f%% of 1321T)\n",
                layout, layoutNames[layout], maxAlgId, ms, tops, tops/1321*100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        /* Cleanup */
cleanup:
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatDescriptorDestroy(&matD);
        cusparseLtDestroy(&handle);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        if (d_workspace) cudaFree(d_workspace);
        cudaFree(d_compressed); cudaFree(d_compressedBuffer);

        return (us > 0) ? us : 1;
    }
}

}  /* extern "C" */
