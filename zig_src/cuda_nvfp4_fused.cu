/**
 * NVFP4 fused dequant + GEMM kernel — Ada SM89 PoC.
 *
 * Replaces the 2-step "dequant FP4 → write FP16 → run FP16 GEMM" pipeline
 * with a single kernel: each warp loads packed FP4 nibbles, dequantizes
 * in registers, accumulates a small tile in FP32. Crucially the FP16
 * intermediate never touches HBM — the 4-bit storage advantage is
 * preserved end-to-end.
 *
 * This is a *naïve* implementation (no Tensor Cores, no shared-mem
 * tiling) intended to validate the data path. Throughput will be modest
 * (~5-20 TFLOPS), well below the dense FP16 path's 165 TFLOPS, but it
 * proves the design: HBM read = N²·K/2 bytes vs FP16's N²·K·2 bytes
 * (4× savings).
 *
 * To match dense FP16 throughput would require porting this to a
 * CUTLASS EVT (epilogue visitor tree) that injects the dequant step
 * into the existing tensor-core MMA pipeline. That's the follow-up
 * post-CUTLASS-DSL migration.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

__constant__ uint16_t kFusedNVFP4_BITS[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};

__device__ __forceinline__ float fp4_decode_to_float(uint8_t nibble) {
    return __half2float(__ushort_as_half(kFusedNVFP4_BITS[nibble]));
}

/* One thread per output element. A is FP4 packed (M × K/2 bytes),
 * B is FP16 (K × N), C is FP16 (M × N). */
__global__ static void nvfp4_fused_gemm_kernel(
    const uint8_t *__restrict__ A_packed, /* M × K/2 bytes */
    const __half *__restrict__ B,         /* K × N */
    __half *__restrict__ C,               /* M × N */
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    int row_stride = K / 2; /* packed nibbles per row */

    /* Walk K in steps of 2 (one byte = 2 FP4 values). */
    for (int k = 0; k < K; k += 2) {
        uint8_t byte = A_packed[row * row_stride + k / 2];
        float a0 = fp4_decode_to_float(byte & 0xF);
        float a1 = fp4_decode_to_float((byte >> 4) & 0xF);
        acc += a0 * __half2float(B[(k + 0) * N + col]);
        acc += a1 * __half2float(B[(k + 1) * N + col]);
    }
    C[row * N + col] = __float2half(acc);
}

/** Fused dequant + GEMM bench. M, N, K all multiples of 16.
 * Returns elapsed µs (kernel-only). */
int nvfp4_fused_gemm_bench(int M, int N, int K, int iters) {
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || (K & 1))
        return -10;

    size_t bytes_A = (size_t)M * (K / 2); /* packed FP4 */
    size_t bytes_B = (size_t)K * N * 2;   /* FP16 */
    size_t bytes_C = (size_t)M * N * 2;   /* FP16 */

    uint8_t *d_A = nullptr;
    __half *d_B = nullptr, *d_C = nullptr;
    if (cudaMalloc((void **)&d_A, bytes_A) != cudaSuccess)
        return -11;
    if (cudaMalloc((void **)&d_B, bytes_B) != cudaSuccess) {
        cudaFree(d_A);
        return -12;
    }
    if (cudaMalloc((void **)&d_C, bytes_C) != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        return -13;
    }
    cudaMemset(d_A, 0x22, bytes_A); /* alternating FP4 codes ≈ ±1.0 */
    cudaMemset(d_B, 0x38, bytes_B); /* FP16 ≈ 0.5 */
    cudaMemset(d_C, 0, bytes_C);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    /* Warmup */
    nvfp4_fused_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);

    for (int i = 0; i < iters; ++i) {
        nvfp4_fused_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return (int)(elapsed_ms * 1000.0f);
}

} /* extern "C" */
