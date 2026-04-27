# CUDA GEMM Optimization Guide

## From 480 GFLOPS to 21 TFLOPS (44x speedup!)

Based on [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm) tutorial.

## Performance Progression (RTX 3080/3090)

| Version | Technique                          | GFLOPS     | Speedup |
|---------|------------------------------------|------------|---------|
| v2      | Naive (1 thread = 1 element)       | 480        | 1x      |
| v3      | Shared memory tiling               | ~2,000     | 4x      |
| v4      | Thread coarsening (2x2 per thread) | ~4,000     | 8x      |
| v5      | Larger tiles (4x4 per thread)      | ~7,000     | 15x     |
| v6      | Double buffering (2x tile load)    | ~8,000     | 17x     |
| v7      | 1D thread layout (256 threads)     | ~10,000    | 21x     |
| v10     | PTX asm loads (lds128/sts128)      | ~12,000    | 25x     |
| v11     | Ping-pong GMEM→SMEM + SMEM_LDA=132 | ~17,000    | 35x     |
| v12     | Ping-pong register panels          | **21,000** | **44x** |
| cuBLAS  | Reference                          | ~14,000    | 29x     |

**v12 BEATS cuBLAS by 50%!**

---

## Key Techniques

### 1. Shared Memory Tiling (v3)

```cuda
__shared__ float ashare[BLOCK][BLOCK];
__shared__ float bshare[BLOCK][BLOCK];

// Each thread loads one element
ashare[ty][tx] = a[...];
bshare[ty][tx] = b[...];
__syncthreads();

// Compute using shared memory
for (int k = 0; k < BLOCK; ++k) {
    sum += ashare[ty][k] * bshare[k][tx];
}
```

**Why it works:** SMEM is 10-100x faster than GMEM.

---

### 2. Thread Coarsening (v4-v5)

```cuda
// Each thread computes STRIDE x STRIDE elements
float sum[STRIDE][STRIDE] = {0};

for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
        for (int k = 0; k < STEP; ++k) {
            sum[i][j] += ashare[ty*STRIDE + i][k] * bshare[k][tx*STRIDE + j];
        }
    }
}
```

**Why it works:** More work per thread = better instruction-level parallelism (ILP).

---

### 3. Double Buffering (v6)

```cuda
// Load 2x STEP at once
__shared__ float ashare[STEP][2 * STEP];  // Double width

// Load both halves
ashare[ty][tx] = a[...];
ashare[ty][tx + STEP] = a[... + STEP];
```

**Why it works:** Hides memory latency by overlapping load with compute.

---

### 4. PTX Assembly Loads (v10+)

```cuda
// Vector load 4 floats at once (lds128)
__device__ void lds128(float &r0, float &r1, float &r2, float &r3, uint32_t addr) {
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
                 : "r"(addr));
}

// Non-coherent global load with L2 cache hint
__device__ void ldg32_nc_0(float &reg, const void *ptr) {
    asm volatile("ld.global.nc.L2::128B.f32 %0, [%1];\n"
                 : "=f"(reg) : "l"(ptr));
}
```

**Why it works:**

- Vector loads use full memory bus width
- `.nc` = non-coherent, bypasses L1 for streaming data
- `L2::128B` = L2 cache hint for 128-byte sector

---

### 5. Bank Conflict Avoidance (v11)

```cuda
#define SMEM_LDA (132)  // NOT 128!

// 132 = 128 + 4 = 32 banks + 1 element padding
// This ensures consecutive threads hit different banks
```

**Why it works:** SMEM has 32 banks. If threads access same bank = serialized.
Adding 4 floats (132 vs 128) staggers access patterns.

---

### 6. Ping-Pong Buffering (v11-v12)

#### GMEM → SMEM Ping-Pong (v11)

```cuda
// XOR to flip between two buffers
a_sts_addr ^= 0x2000;  // Flip 8KB offset
b_sts_addr ^= 0x1000;  // Flip 4KB offset

// While computing from buffer A, load into buffer B
```

#### Register Ping-Pong (v12)

```cuda
float panelA[2][8], panelB[2][8];  // TWO sets of registers

for (int subk = 0; subk < 8; ++subk) {
    const int pp = (subk + 1) % 2;  // Ping-pong index

    // Load NEXT panel into registers while computing CURRENT
    lds128(panelA[pp][...], ...);
    lds128(panelB[pp][...], ...);

    // Compute with CURRENT panel
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            sum[i][j] += panelA[subk % 2][i] * panelB[subk % 2][j];
}
```

**Why it works:** Overlaps SMEM loads with compute. Zero stalls!

---

### 7. Launch Bounds

```cuda
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(...)
```

- 256 threads per block
- 2 blocks per SM (for register allocation hints)

---

## Final Kernel Architecture (v12)

```
Block: 128x128 output tile
Threads: 256 (16x16 thread arrangement)
Each thread: 8x8 output elements (4 quadrants of 4x4)
K-tiling: 8 elements at a time

SMEM Layout:
├── ashare: 16KB (8 rows × 128 cols × sizeof(float) × 2 buffers)
└── bshare: 8KB (8 rows × 128 cols × sizeof(float) × 2 buffers)

Registers per thread:
├── sum[8][8] = 64 floats
├── panelA[2][8] = 16 floats
├── panelB[2][8] = 16 floats
├── a_ldg_reg[4] = 4 floats
└── b_ldg_reg[4] = 4 floats
Total: 104 registers/thread
```

---

## Application to viva_tensor

### Current State

- Using cuBLAS via `cuda_gemm.c`
- Async pipeline: 133 TFLOPS FP16, 78 TFLOPS FP32
- cuBLAS is good but not optimal

### Opportunities

1. **Custom FP16 kernel** - Could beat cuBLAS with these techniques
2. **Fused kernels** - GEMM + activation in one kernel (no GMEM round-trip)
3. **INT8 GEMM** - Custom kernel with dequant fusion
4. **Batched GEMM** - Better scheduling for small matrices

### Implementation Path

1. Port `sgemm_128x128x8` to our codebase
2. Adapt for FP16 (half precision)
3. Add Tensor Core wmma intrinsics for FP16
4. Profile and tune for RTX 4090

---

## References

- [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA's official template library
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
- [SageAttention](https://github.com/thu-ml/SageAttention) - 2-5x faster than FlashAttention
