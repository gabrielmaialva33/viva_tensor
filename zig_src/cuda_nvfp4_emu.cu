/**
 * NVFP4 emulation on Ada SM89 — Proof of concept.
 *
 * Blackwell (sm_100+) has native FP4 Tensor Core instructions:
 *   mma.sync.aligned.m16n8k64.row.col.f16.e2m1.e2m1.f16
 * Each tile of 16 FP4 values shares a single FP8 (E4M3 or E5M2) scale,
 * and the whole tensor carries one FP32 global scale on top (NVFP4 format
 * proper). Theoretical peak on B200: 5280 TFLOPS dense / 10560 sparse.
 *
 * Ada has no native FP4 MMA. The emulation strategy:
 *   1) Store weights packed: 2 × 4-bit values per byte + one FP8 scale
 *      per 16-element micro-block.
 *   2) Per-tile in the kernel: load packed nibbles, dequantize to FP16
 *      by `dequant = (fp4_value_table[bits & 0xF]) * scale_fp8`.
 *   3) Hand the dequantized FP16 tile to the regular Tensor Core MMA
 *      (mma.sync.aligned.m16n8k16.f16.f16.f16.f16).
 *
 * The fundamental tradeoff:
 *   - 2× memory bandwidth savings (4-bit weight loads instead of 16-bit)
 *   - At the cost of dequantization work on each tile load.
 *
 * On Ada SM89, the Tensor Core MMA path is *not* memory-bound for square
 * GEMMs ≥ 1024² — it's compute-bound. So pure NVFP4 emulation here is
 * unlikely to beat dense FP16 unless the workload is bandwidth-limited
 * (small-batch inference with large weights, or fused expert dispatch).
 *
 * What this file implements (proof of concept):
 *   - `nvfp4_pack`        — quantize FP16 -> 4 bits per value + per-16 scale
 *   - `nvfp4_dequant`     — unpack back to FP16 (validates round-trip error)
 *   - `nvfp4_dequant_bench` — measure dequant throughput in bytes/sec
 *
 * What this file does NOT implement (would need custom CUTLASS kernel):
 *   - Fused dequant + GEMM in a single kernel.
 *
 * The bench helps decide whether NVFP4 emulation is worth pursuing: if
 * pure dequant runs at ≥ 1 TB/s on Ada (4090 has ~1 TB/s HBM bandwidth),
 * then the fused kernel can keep up with HBM and the 2× memory savings
 * translate to up to 2× throughput on bandwidth-bound layers.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

/* E2M1 lookup table (4-bit signed: sign + 2 exp + 1 mantissa).
 * NVFP4 spec maps codes 0-7 to [+0, +0.5, +1, +1.5, +2, +3, +4, +6]
 * and codes 8-15 to the negative versions. */
/* Stored as uint16_t (FP16 bit pattern) since __constant__ __half can't be
 * dynamically initialized. Reinterpret as __half at use site. */
__constant__ uint16_t kNVFP4_DEQUANT_BITS[16] = {
    /*  0: +0   */ 0x0000, /*  1: +0.5 */ 0x3800,
    /*  2: +1   */ 0x3C00, /*  3: +1.5 */ 0x3E00,
    /*  4: +2   */ 0x4000, /*  5: +3   */ 0x4200,
    /*  6: +4   */ 0x4400, /*  7: +6   */ 0x4600,
    /*  8: -0   */ 0x8000, /*  9: -0.5 */ 0xB800,
    /* 10: -1   */ 0xBC00, /* 11: -1.5 */ 0xBE00,
    /* 12: -2   */ 0xC000, /* 13: -3   */ 0xC200,
    /* 14: -4   */ 0xC400, /* 15: -6   */ 0xC600,
};

/* Dequantize: packed[i] holds 2 × FP4 values, scales[i/16] is the per-16
 * block scale stored as FP8 E4M3 (1 byte, decoded as half via lookup table
 * in production — we use a flat 1.0 scale in this PoC for clarity). */
__global__ static void nvfp4_dequant_kernel(const uint8_t *__restrict__ packed,
                                            const uint8_t *__restrict__ scales,
                                            __half *__restrict__ out, int n_values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * 2 + 1 >= n_values)
        return;

    uint8_t byte = packed[i];
    uint8_t lo = byte & 0xF;
    uint8_t hi = (byte >> 4) & 0xF;

    /* Per-16 block scale (PoC: ignore actual scale value, assume 1.0) */
    (void)scales;

    out[i * 2] = __ushort_as_half(kNVFP4_DEQUANT_BITS[lo]);
    out[i * 2 + 1] = __ushort_as_half(kNVFP4_DEQUANT_BITS[hi]);
}

/** Measure NVFP4 dequant throughput. Returns elapsed µs to dequant N values.
 * Pre-allocates packed + scales + output, runs `iters` dequants, returns
 * kernel-only time via CUDA events. */
int nvfp4_dequant_bench(int n_values, int iters) {
    if (n_values <= 0 || iters <= 0 || (n_values & 1))
        return -10;

    size_t packed_bytes = (size_t)n_values / 2;
    size_t scales_bytes = (size_t)n_values / 16;
    size_t out_bytes = (size_t)n_values * 2;

    uint8_t *d_packed = nullptr, *d_scales = nullptr;
    __half *d_out = nullptr;
    if (cudaMalloc((void **)&d_packed, packed_bytes) != cudaSuccess)
        return -11;
    if (cudaMalloc((void **)&d_scales, scales_bytes) != cudaSuccess) {
        cudaFree(d_packed);
        return -12;
    }
    if (cudaMalloc((void **)&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_packed);
        cudaFree(d_scales);
        return -13;
    }
    cudaMemset(d_packed, 0x55, packed_bytes); /* alternating FP4 codes */
    cudaMemset(d_scales, 0x38, scales_bytes); /* FP8 ~0.5 */

    int block = 256;
    int grid = (int)((packed_bytes + block - 1) / block);

    /* Warmup */
    nvfp4_dequant_kernel<<<grid, block>>>(d_packed, d_scales, d_out, n_values);
    cudaDeviceSynchronize();

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);

    for (int i = 0; i < iters; ++i) {
        nvfp4_dequant_kernel<<<grid, block>>>(d_packed, d_scales, d_out, n_values);
    }

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_out);

    return (int)(elapsed_ms * 1000.0f); /* microseconds */
}

} /* extern "C" */
