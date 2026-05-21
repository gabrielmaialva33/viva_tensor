/**
 * FP8 E4M3 GEMM via CUTLASS with FP16 accumulation — 660 TOPS on RTX 4090!
 *
 * cuBLASLt is capped at 330 TOPS on GeForce (FP32 accum half-rate nerf).
 * CUTLASS with ElementAccumulator=half_t uses the FP16 MMA instruction:
 *   mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
 * This bypasses the GeForce nerf and runs at full rate: 660 TOPS.
 *
 * KEY INSIGHT: The MMA instruction is selected based on ElementAccumulator type:
 *   float    → f32.e4m3.e4m3.f32 (330 TOPS, half-rate on GeForce)
 *   half_t   → f16.e4m3.e4m3.f16 (660 TOPS, full-rate!)
 * See: cutlass/arch/mma_sm89.h lines 110-172 (FP32) vs 378-442 (FP16)
 *
 * BSD-3-Clause License (CUTLASS)
 */

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

/* =========================================================================
 * Configuration A: FP16 accumulation (660 TOPS target!)
 *
 * ElementAccumulator = half_t selects the FP16 MMA:
 *   mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
 * Tile: 128×256×64 (default SM89 FP8), warp 64×64×64, instr 16×8×32
 * ========================================================================= */

using ElementA_f16  = cutlass::float_e4m3_t;
using ElementB_f16  = cutlass::float_e4m3_t;
using ElementOut_f16 = cutlass::half_t;
using ElementAcc_f16 = cutlass::half_t;  /* KEY: half_t → FP16 MMA (660 TOPS!) */

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

/* Epilogue with FP16 accumulator — all FP16 */
using EpilogueOp_f16 = cutlass::epilogue::thread::LinearCombination<
    ElementOut_f16,     /* output type */
    128 / cutlass::sizeof_bits<ElementOut_f16>::value,  /* elements per access = 8 */
    ElementAcc_f16,     /* accumulator type = half_t */
    ElementAcc_f16      /* compute type for epilogue = half_t */
>;

/* FP16 accumulation GEMM — full-rate 660 TOPS on Ada! */
using GemmFP16Acc = cutlass::gemm::device::Gemm<
    ElementA_f16, LayoutA,
    ElementB_f16, LayoutB,
    ElementOut_f16, LayoutC,
    ElementAcc_f16,                                     /* half_t accumulator! */
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,             /* threadblock (SM89 default) */
    cutlass::gemm::GemmShape<64, 64, 64>,               /* warp (SM89 default) */
    cutlass::gemm::GemmShape<16, 8, 32>,                /* instruction */
    EpilogueOp_f16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,   /* stages */
    16,  /* alignmentA */
    16,  /* alignmentB */
    false,  /* SplitKSerial */
    cutlass::arch::OpMultiplyAddFastAccum               /* FP16 fast accum */
>;

/* =========================================================================
 * Configuration B: FP32 accumulation (330 TOPS, for comparison)
 *
 * ElementAccumulator = float selects the FP32 MMA:
 *   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * ========================================================================= */

using ElementAcc_f32 = float;
static constexpr int kStages = 3;

using EpilogueOp_f32 = cutlass::epilogue::thread::LinearCombination<
    ElementOut_f16,     /* output type = half_t */
    128 / cutlass::sizeof_bits<ElementOut_f16>::value,
    ElementAcc_f32,     /* accumulator type = float */
    ElementAcc_f32      /* compute type = float */
>;

using Gemm_FP8_F32_LargeKN = cutlass::gemm::device::Gemm<
    ElementA_f16, LayoutA,
    ElementB_f16, LayoutB,
    ElementOut_f16, LayoutC,
    ElementAcc_f32,                                     /* float accumulator */
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOp_f32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages, 16, 16,
    false,
    cutlass::arch::OpMultiplyAdd                        /* standard FP32 accum */
>;

using Gemm_FP8_F32_Default = cutlass::gemm::device::Gemm<
    ElementA_f16, LayoutA,
    ElementB_f16, LayoutB,
    ElementOut_f16, LayoutC,
    ElementAcc_f32,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOp_f32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages, 16, 16,
    false,
    cutlass::arch::OpMultiplyAdd
>;

/* GemmFP32AccOutF32 — same as GemmFP32Acc but stores FP32 output
 * instead of casting to FP16. Eliminates the cast saturation that
 * caps end-to-end FP8 precision at L2 ~13% on Llama-scale K. Trades
 * 2× output bandwidth for full FP32 dynamic range in the accumulator.
 */
using EpilogueOp_f32_out_f32 = cutlass::epilogue::thread::LinearCombination<
    float,                                              /* output = FP32 */
    128 / cutlass::sizeof_bits<float>::value,           /* 4 elems per access */
    ElementAcc_f32,                                     /* accumulator = float */
    ElementAcc_f32                                      /* compute = float */
>;

using Gemm_FP8_F32_out_f32_LargeKN = cutlass::gemm::device::Gemm<
    ElementA_f16, LayoutA,
    ElementB_f16, LayoutB,
    float, LayoutC,
    ElementAcc_f32,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOp_f32_out_f32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages, 16, 16,
    false,
    cutlass::arch::OpMultiplyAdd
>;

using Gemm_FP8_F32_out_f32_Default = cutlass::gemm::device::Gemm<
    ElementA_f16, LayoutA,
    ElementB_f16, LayoutB,
    float, LayoutC,
    ElementAcc_f32,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOp_f32_out_f32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages, 16, 16,
    false,
    cutlass::arch::OpMultiplyAdd
>;

/* =========================================================================
 * C-callable interface
 * ========================================================================= */
extern "C" {

static cudaStream_t g_fp8_dequant_stream = 0;

void cuda_fp8_dequant_set_stream(void *stream) {
    g_fp8_dequant_stream = (cudaStream_t)stream;
}

/* Dequant FP8 col-major weight back to FP16. Supports both layouts:
 *   block_size == 0  -> per-channel: scales[col]
 *   block_size  > 0  -> per-block along K: scales[col * (K/block_size) + (row/block_size)]
 */
__global__ void fp8_colmajor_dequant_to_fp16_kernel(
    const cutlass::float_e4m3_t *src,
    const float *scales,
    cutlass::half_t *dst,
    int K,
    int N,
    int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;
    int col = idx / K;
    int row = idx - col * K;
    float scale;
    if (block_size <= 0) {
        scale = scales ? scales[col] : 1.0f;
    } else {
        int num_blocks = K / block_size;
        int block_idx = row / block_size;
        scale = scales ? scales[col * num_blocks + block_idx] : 1.0f;
    }
    dst[idx] = cutlass::half_t(static_cast<float>(src[idx]) * scale);
}

int cuda_fp8_colmajor_dequant_to_fp16(const void *d_fp8,
                                       const float *d_scales,
                                       void *d_fp16,
                                       int K,
                                       int N) {
    /* Legacy per-channel entry — block_size = 0. */
    if (!d_fp8 || !d_fp16 || K <= 0 || N <= 0) return -10;
    int total = K * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp8_colmajor_dequant_to_fp16_kernel<<<blocks, threads, 0, g_fp8_dequant_stream>>>(
        static_cast<const cutlass::float_e4m3_t *>(d_fp8),
        d_scales,
        static_cast<cutlass::half_t *>(d_fp16),
        K,
        N,
        0);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -11;
}

/* Same but with per-block scales. block_size must divide K. */
int cuda_fp8_colmajor_dequant_to_fp16_blocked(const void *d_fp8,
                                                const float *d_scales,
                                                void *d_fp16,
                                                int K,
                                                int N,
                                                int block_size) {
    if (!d_fp8 || !d_fp16 || K <= 0 || N <= 0) return -10;
    if (block_size <= 0 || (K % block_size) != 0) return -12;
    int total = K * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp8_colmajor_dequant_to_fp16_kernel<<<blocks, threads, 0, g_fp8_dequant_stream>>>(
        static_cast<const cutlass::float_e4m3_t *>(d_fp8),
        d_scales,
        static_cast<cutlass::half_t *>(d_fp16),
        K,
        N,
        block_size);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -11;
}

/**
 * FP8 E4M3 GEMM with FP32 output — for the inference path where the
 * caller wants to apply per-row × per-channel dequant scales on FP32
 * accumulator values before the final FP16 cast on the host.
 * A[M,K] row-major FP8, B[K,N] col-major FP8, C[M,N] row-major FP32.
 * Returns 0 on success, negative on error.
 */
int cutlass_fp8_gemm_f32acc_out_f32(int M, int N, int K,
                                      const void *d_A, const void *d_B,
                                      float *d_C) {
    if (K == 4096 && N == 4096) {
        Gemm_FP8_F32_out_f32_LargeKN gemm_op;

        float alpha = 1.0f, beta = 0.0f;

        Gemm_FP8_F32_out_f32_LargeKN::Arguments args(
            {M, N, K},
            {static_cast<const ElementA_f16*>(d_A), K},
            {static_cast<const ElementB_f16*>(d_B), K},
            {d_C, N},
            {d_C, N},
            {alpha, beta}
        );

        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) return -1;

        size_t workspace_size = Gemm_FP8_F32_out_f32_LargeKN::get_workspace_size(args);
        void *workspace = nullptr;
        if (workspace_size > 0) {
            if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) return -2;
        }

        status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            if (workspace) cudaFree(workspace);
            return -3;
        }

        status = gemm_op();
        if (workspace) cudaFree(workspace);
        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    Gemm_FP8_F32_out_f32_Default gemm_op;

    float alpha = 1.0f, beta = 0.0f;

    Gemm_FP8_F32_out_f32_Default::Arguments args(
        {M, N, K},
        {static_cast<const ElementA_f16*>(d_A), K},
        {static_cast<const ElementB_f16*>(d_B), K},
        {d_C, N},
        {d_C, N},
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t workspace_size = Gemm_FP8_F32_out_f32_Default::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) return -2;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -3;
    }

    status = gemm_op();
    if (workspace) cudaFree(workspace);
    return (status == cutlass::Status::kSuccess) ? 0 : -4;
}


/**
 * FP8 E4M3 GEMM with FP16 accumulation — 660 TOPS target!
 * A[M,K] row-major FP8, B[K,N] col-major FP8, C[M,N] row-major FP16
 * Returns 0 on success, negative on error.
 */
int cutlass_fp8_gemm_f16acc(int M, int N, int K,
                             const void *d_A, const void *d_B, void *d_C) {
    GemmFP16Acc gemm_op;

    cutlass::half_t alpha(1.0f), beta(0.0f);

    GemmFP16Acc::Arguments args(
        {M, N, K},
        {static_cast<const ElementA_f16*>(d_A), K},    /* A: row-major, ld=K */
        {static_cast<const ElementB_f16*>(d_B), K},    /* B: col-major, ld=K */
        {static_cast<ElementOut_f16*>(d_C), N},         /* C: row-major, ld=N */
        {static_cast<ElementOut_f16*>(d_C), N},         /* D = C (in-place) */
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t workspace_size = GemmFP16Acc::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) return -2;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -3;
    }

    status = gemm_op();
    if (workspace) cudaFree(workspace);

    return (status == cutlass::Status::kSuccess) ? 0 : -4;
}

/**
 * FP8 E4M3 GEMM with FP32 accumulation — 330 TOPS (same as cuBLASLt).
 * For comparison only.
 */
int cutlass_fp8_gemm_f32acc(int M, int N, int K,
                             const void *d_A, const void *d_B, void *d_C) {
    if (K == 4096 && N == 4096) {
        Gemm_FP8_F32_LargeKN gemm_op;

        float alpha = 1.0f, beta = 0.0f;

        Gemm_FP8_F32_LargeKN::Arguments args(
            {M, N, K},
            {static_cast<const ElementA_f16*>(d_A), K},
            {static_cast<const ElementB_f16*>(d_B), K},
            {static_cast<ElementOut_f16*>(d_C), N},
            {static_cast<ElementOut_f16*>(d_C), N},
            {alpha, beta}
        );

        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) return -1;

        size_t workspace_size = Gemm_FP8_F32_LargeKN::get_workspace_size(args);
        void *workspace = nullptr;
        if (workspace_size > 0) {
            cudaError_t err = cudaMalloc(&workspace, workspace_size);
            if (err != cudaSuccess) return -2;
        }

        status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            if (workspace) cudaFree(workspace);
            return -3;
        }

        status = gemm_op();
        if (workspace) cudaFree(workspace);

        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    Gemm_FP8_F32_Default gemm_op;

    float alpha = 1.0f, beta = 0.0f;

    Gemm_FP8_F32_Default::Arguments args(
        {M, N, K},
        {static_cast<const ElementA_f16*>(d_A), K},
        {static_cast<const ElementB_f16*>(d_B), K},
        {static_cast<ElementOut_f16*>(d_C), N},
        {static_cast<ElementOut_f16*>(d_C), N},
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t workspace_size = Gemm_FP8_F32_Default::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) return -2;
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -3;
    }

    status = gemm_op();
    if (workspace) cudaFree(workspace);

    return (status == cutlass::Status::kSuccess) ? 0 : -4;
}

/**
 * Self-contained FP8 E4M3 GEMM benchmark.
 * Allocates A[M,K], B[K,N], C[M,N] on the device, runs `iters` GEMMs,
 * returns total elapsed microseconds (kernel-only via CUDA events).
 * Mode: 0 = FP16 accumulation (660 TOPS path), 1 = FP32 accumulation (330 TOPS).
 * Returns negative on error.
 */
int cutlass_fp8_bench(int M, int N, int K, int iters, int mode) {
    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0) return -10;

    size_t bytes_A = (size_t)M * K;  /* FP8 = 1 byte each */
    size_t bytes_B = (size_t)K * N;
    size_t bytes_C = (size_t)M * N * 2;  /* FP16 = 2 bytes */

    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (cudaMalloc(&d_A, bytes_A) != cudaSuccess) return -11;
    if (cudaMalloc(&d_B, bytes_B) != cudaSuccess) { cudaFree(d_A); return -12; }
    if (cudaMalloc(&d_C, bytes_C) != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); return -13;
    }

    /* Fill with non-zero pattern to avoid sparsity tricks skewing the bench. */
    cudaMemset(d_A, 0x3C, bytes_A);  /* E4M3 ~= 0.5 */
    cudaMemset(d_B, 0x3C, bytes_B);
    cudaMemset(d_C, 0, bytes_C);

    /* Warmup */
    int rc = (mode == 0)
        ? cutlass_fp8_gemm_f16acc(M, N, K, d_A, d_B, d_C)
        : cutlass_fp8_gemm_f32acc(M, N, K, d_A, d_B, d_C);
    if (rc != 0) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return rc - 100;
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; ++i) {
        if (mode == 0) cutlass_fp8_gemm_f16acc(M, N, K, d_A, d_B, d_C);
        else           cutlass_fp8_gemm_f32acc(M, N, K, d_A, d_B, d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (int)(elapsed_ms * 1000.0f);  /* microseconds */
}

}  /* extern "C" */
