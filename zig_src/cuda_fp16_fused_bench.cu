/**
 * FP16 GEMM with cuBLASLt epilogue fusion benchmark.
 *
 * Measures kernel-only TFLOPS for matmul + bias / GELU / bias+GELU all
 * fused into a single kernel via cuBLASLt's epilogue infrastructure.
 * Uses CUBLAS_COMPUTE_16F + FP16 alpha/beta to keep the full-rate Tensor
 * Core path active (same as cublaslt_fp16_bench).
 *
 * epilogue codes (matching cublasLtEpilogue_t):
 *   1  = DEFAULT          (no fusion — baseline)
 *   2  = RELU
 *   4  = BIAS
 *   6  = BIAS + RELU
 *   32 = GELU
 *   36 = BIAS + GELU
 *
 * The win comes from saving HBM round-trips: a standalone bias+activation
 * kernel would re-read C from global memory, add bias, run GELU, and write
 * back. Fused, C never leaves the SMEM/registers of the GEMM kernel.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

extern "C" {

typedef __half cuda_half_t;

static cublasLtHandle_t g_ctx = nullptr;
static void *g_workspace = nullptr;
static size_t g_workspace_size = 32 * 1024 * 1024; /* 32 MiB — matches main NIF */

static int ensure_init() {
    if (g_ctx)
        return 0;
    if (cublasLtCreate(&g_ctx) != CUBLAS_STATUS_SUCCESS)
        return -1;
    if (cudaMalloc(&g_workspace, g_workspace_size) != cudaSuccess) {
        cublasLtDestroy(g_ctx);
        g_ctx = nullptr;
        return -2;
    }
    return 0;
}

int cublaslt_fp16_fused_bench(int M, int N, int K, int iters, int epilogue) {
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0)
        return -10;
    if (ensure_init() != 0)
        return -11;

    size_t bytes_A = (size_t)M * K * 2;
    size_t bytes_BT = (size_t)K * N * 2;
    size_t bytes_C = (size_t)M * N * 2;
    size_t bytes_bias = (size_t)M * 2; /* per-row bias when used */

    cuda_half_t *d_A = nullptr, *d_BT = nullptr, *d_C = nullptr, *d_bias = nullptr;
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
    if (cudaMalloc((void **)&d_bias, bytes_bias) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
        return -15;
    }

    cudaMemset(d_A, 0x38, bytes_A);
    cudaMemset(d_BT, 0x38, bytes_BT);
    cudaMemset(d_C, 0, bytes_C);
    cudaMemset(d_bias, 0x38, bytes_bias);

    /* Build matmul desc with COMPUTE_16F + requested epilogue */
    cublasLtMatmulDesc_t desc;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
        cudaFree(d_bias);
        return -16;
    }
    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    cublasLtEpilogue_t ep = (cublasLtEpilogue_t)epilogue;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    /* Bias pointer needed for any BIAS-bearing epilogue */
    if (epilogue == CUBLASLT_EPILOGUE_BIAS || epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
        epilogue == CUBLASLT_EPILOGUE_GELU_BIAS) {
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias,
                                       sizeof(d_bias));
    }

    /* TN swap for row-major: m_lt=N, n_lt=M, k_lt=K */
    cublasLtMatrixLayout_t layout_bt, layout_a, layout_c;
    cublasLtMatrixLayoutCreate(&layout_bt, CUDA_R_16F, (uint64_t)K, (uint64_t)N, (int64_t)K);
    cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F, (uint64_t)K, (uint64_t)M, (int64_t)K);
    cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16F, (uint64_t)N, (uint64_t)M, (int64_t)N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &g_workspace_size, sizeof(g_workspace_size));

    cublasLtMatmulHeuristicResult_t result;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_ctx, desc, layout_bt, layout_a, layout_c,
                                                       layout_c, pref, 1, &result, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layout_bt);
        cublasLtMatrixLayoutDestroy(layout_a);
        cublasLtMatrixLayoutDestroy(layout_c);
        cublasLtMatmulDescDestroy(desc);
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
        cudaFree(d_bias);
        return -17;
    }

    cuda_half_t alpha_h = 0x3C00; /* 1.0 */
    cuda_half_t beta_h = 0x0000;  /* 0.0 */

    /* Warmup */
    st = cublasLtMatmul(g_ctx, desc, &alpha_h, d_BT, layout_bt, d_A, layout_a, &beta_h, d_C,
                        layout_c, d_C, layout_c, &result.algo, g_workspace, g_workspace_size,
                        (cudaStream_t)0);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layout_bt);
        cublasLtMatrixLayoutDestroy(layout_a);
        cublasLtMatrixLayoutDestroy(layout_c);
        cublasLtMatmulDescDestroy(desc);
        cudaFree(d_A);
        cudaFree(d_BT);
        cudaFree(d_C);
        cudaFree(d_bias);
        return -1800 - (int)st;
    }
    cudaDeviceSynchronize();

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);

    for (int i = 0; i < iters; ++i) {
        cublasLtMatmul(g_ctx, desc, &alpha_h, d_BT, layout_bt, d_A, layout_a, &beta_h, d_C,
                       layout_c, d_C, layout_c, &result.algo, g_workspace, g_workspace_size,
                       (cudaStream_t)0);
    }

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_bt);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(d_A);
    cudaFree(d_BT);
    cudaFree(d_C);
    cudaFree(d_bias);

    return (int)(elapsed_ms * 1000.0f);
}

} /* extern "C" */
