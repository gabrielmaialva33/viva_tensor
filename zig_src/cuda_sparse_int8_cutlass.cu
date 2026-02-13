/**
 * INT8 2:4 Structured Sparse GEMM via CUTLASS 4.3.5 — 1321 TOPS target on RTX 4090!
 *
 * INT8 sparse uses the MMA instruction:
 *   mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite
 *
 * RTX 4090 peak: INT8 dense=660 TOPS, INT8 sparse=1321 TOPS (2x from 2:4 sparsity)
 *
 * Four kernel families:
 *   - Configs 0-5:   SparseGemm (basic, no split-K)
 *   - Configs 10-14: GemmSparseUniversal (split-K support)
 *   - Configs 20-24: GemmSparseUniversal + Swizzle<8>
 *   - Configs 25-28: GemmSparseUniversal + Swizzle<4> / no-swizzle
 *
 * Key optimizations:
 *   - GemmSparseUniversal instead of SparseGemm
 *   - AlignmentA=16 (128-bit loads, cp.async hardware max for INT8)
 *   - Swizzle<4>/<8> for L2-friendly CTA scheduling
 *   - Multiple tile configs for different matrix shapes
 *   - NOTE: A32 impossible for INT8 — cp.async max 16 bytes, INT8×32=32 bytes exceeds it
 *
 * BSD-3-Clause License (CUTLASS)
 */

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include <cstdio>
#include <cuda_runtime.h>

/* Common types for INT8 sparse */
using ElementA   = int8_t;
using ElementB   = int8_t;
using ElementC   = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,  /* = 4 elements per access */
    ElementAcc,
    ElementAcc
>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

/* =========================================================================
 * FAMILY 1: SparseGemm (basic, no split-K) — configs 0-5
 *   MMA instruction: m16n8k64, InstrK=64
 *   WarpK must be >= 128 (InstrK=64, kWarpGemmIterations >= 2 and even)
 *
 *   SMEM per stage for INT8 sparse (A compressed 50%):
 *     A: M_tile * (K_tile/2) * 1 byte
 *     B: K_tile * N_tile * 1 byte
 * ========================================================================= */

/* Config 0: 128×128×128, warp 64×64×128, 3 stages
 *   A: 128*64=8KB, B: 128*128=16KB → 24KB/stage → 72KB for 3 stages */
using SparseGemm_0 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 3, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 1: 128×128×256, warp 64×64×128, 2 stages
 *   A: 128*128=16KB, B: 256*128=32KB → 48KB/stage → 96KB for 2 stages */
using SparseGemm_1 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 2: 128×256×128, warp 64×64×128, 2 stages
 *   A: 128*64=8KB, B: 128*256=32KB → 40KB/stage → 80KB for 2 stages */
using SparseGemm_2 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 3: 256×128×128, warp 64×64×128, 2 stages
 *   A: 256*64=16KB, B: 128*128=16KB → 32KB/stage → 64KB for 2 stages */
using SparseGemm_3 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 4: 64×128×128, warp 32×64×128, 3 stages (more CTAs per SM)
 *   A: 64*64=4KB, B: 128*128=16KB → 20KB/stage → 60KB for 3 stages */
using SparseGemm_4 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 128>,
    cutlass::gemm::GemmShape<32, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 3, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 5: 128×128×128, warp 64×64×128, 4 stages (deeper pipeline)
 *   24KB/stage → 96KB for 4 stages */
using SparseGemm_5 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 4, 16, 16, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 2: GemmSparseUniversal — configs 10-15
 *   Same tiles as 0-5 but with Universal API (split-K support)
 *   Still uses Alignment=16, no swizzle
 * ========================================================================= */

/* Config 10: Universal 128×128×128, 3 stages */
using SparseUniv_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 3, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 11: Universal 128×128×256, 2 stages */
using SparseUniv_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 12: Universal 256×128×128, 2 stages */
using SparseUniv_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 13: Universal 128×256×128, 2 stages */
using SparseUniv_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 14: Universal 256×128×128, 3 stages
 *   A: 256*64=16KB, B: 128*128=16KB → 32KB/stage → 96KB for 3 stages */
using SparseUniv_4 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 3, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 15: REMOVED — 4-stage 128x128 has cp_async alignment issues */

/* =========================================================================
 * FAMILY 3: GemmSparseUniversal + Swizzle<8> — configs 20-24
 *   L2-aware rasterization for better cache utilization
 * ========================================================================= */
using Swizzle8 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

/* Config 20: Universal 256×128×128, 2stg, S8 */
using SparseUnivS8_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle8, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 21: Universal 128×128×256, 2stg, S8 */
using SparseUnivS8_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle8, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 22: Universal 128×256×128, 2stg, S8 */
using SparseUnivS8_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle8, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 23: Universal 128×128×128, 3stg, S8 */
using SparseUnivS8_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle8, 3, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 24: Universal 256×128×128, 3stg, S8 (deep pipeline + swizzle) */
using SparseUnivS8_4 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle8, 3, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * NOTE: FAMILY 4 (AlignmentA=32) REMOVED — cp.async max is 16 bytes.
 *   INT8 Alignment=16 already achieves 128-bit loads (16 bytes = hardware max).
 *   INT4 Alignment=32 = 32×0.5 = 16 bytes worked, but INT8 Alignment=32 = 32×1 = 32 bytes exceeds cp.async limit.
 *   The optimization path for INT8 sparse is different from INT4: focus on tile shape, swizzle, and pipeline depth.
 * ========================================================================= */
using Swizzle4 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;

/* Config 25: Universal 256×128×128, 2stg, A16, Swizzle<4> */
using SparseUnivS4_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle4, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 26: Universal 128×128×256, 2stg, A16, Swizzle<4> */
using SparseUnivS4_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle4, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 27: Universal 128×256×128, 2stg, A16, Swizzle<4> */
using SparseUnivS4_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle4, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 28: Universal 256×128×128, 2stg, A16, no swizzle — baseline */
using SparseUnivNS_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * Templated benchmark function — basic SparseGemm (legacy)
 * ========================================================================= */
template <typename SparseGemm>
static int run_sparse_bench(int M, int N, int K, int iters, int split_k_slices) {
    using ElementE = typename SparseGemm::ElementE;

    constexpr int kSparse = SparseGemm::kSparse;
    constexpr int kElemsPerE = SparseGemm::kElementsPerElementE;

    size_t sizeA = (size_t)M * (K / kSparse);
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;
    size_t metaK = K / kSparse / kElemsPerE;
    size_t sizeE = (size_t)M * metaK;

    int8_t   *d_A = nullptr;
    int8_t   *d_B = nullptr;
    int32_t  *d_C = nullptr;
    ElementE *d_E = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_A, sizeA); if (err != cudaSuccess) return -10;
    err = cudaMalloc(&d_B, sizeB); if (err != cudaSuccess) { cudaFree(d_A); return -11; }
    err = cudaMalloc(&d_C, sizeC * sizeof(int32_t)); if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -12; }
    err = cudaMalloc(&d_E, sizeE * sizeof(ElementE)); if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -13; }

    cudaMemset(d_E, 0, sizeE * sizeof(ElementE));
    cudaMemset(d_C, 0, sizeC * sizeof(int32_t));

    SparseGemm gemm_op;
    ElementAcc alpha = 1, beta = 0;

    int ldA = K / kSparse;
    int ldB = K;
    int ldC = N;
    int ldE = metaK;

    typename SparseGemm::Arguments args(
        {M, N, K},
        {d_A, ldA}, {d_B, ldB}, {d_C, ldC}, {d_C, ldC},
        {d_E, ldE}, {alpha, beta}, split_k_slices
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
        return -1;
    }

    size_t workspace_size = SparseGemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
            return -2;
        }
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -3;
    }

    /* Warmup */
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -4;
    }
    cudaDeviceSynchronize();

    /* Bench loop */
    for (int i = 0; i < iters; i++) {
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
            if (workspace) cudaFree(workspace);
            return -5;
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
    if (workspace) cudaFree(workspace);
    return 0;
}

/* =========================================================================
 * Templated benchmark function — GemmSparseUniversal (optimized!)
 *   Uses CUDA event GPU timing for accurate measurement
 *   Returns elapsed time in microseconds via pointer
 * ========================================================================= */
template <typename SparseUniv>
static int run_int8_sparse_universal_bench(int M, int N, int K, int iters, int split_k_slices) {
    using ElementE = typename SparseUniv::ElementE;
    using GemmKernel = typename SparseUniv::GemmKernel;

    constexpr int kSparse = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    size_t sizeA = (size_t)M * (K / kSparse);
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;
    size_t metaK = K / kSparse / kElemsPerE;
    size_t sizeE = (size_t)M * metaK;

    int8_t   *d_A = nullptr, *d_B = nullptr;
    int32_t  *d_C = nullptr, *d_D = nullptr;
    ElementE *d_E = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_A, sizeA);
    if (err != cudaSuccess) return -10;
    err = cudaMalloc(&d_B, sizeB);
    if (err != cudaSuccess) { cudaFree(d_A); return -11; }
    err = cudaMalloc(&d_C, sizeC * sizeof(int32_t));
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -12; }
    err = cudaMalloc(&d_D, sizeC * sizeof(int32_t));
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -12; }
    err = cudaMalloc(&d_E, sizeE * sizeof(ElementE));
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); return -13; }

    cudaMemset(d_A, 0x11, sizeA);
    cudaMemset(d_B, 0x11, sizeB);
    cudaMemset(d_C, 0, sizeC * sizeof(int32_t));
    cudaMemset(d_D, 0, sizeC * sizeof(int32_t));
    cudaMemset(d_E, 0, sizeE * sizeof(ElementE));

    SparseUniv gemm_op;
    ElementAcc alpha = 1, beta = 0;

    int ldA = K / kSparse;
    int ldB = K;
    int ldC = N;
    int ldD = N;
    int ldE = (int)metaK;

    typename SparseUniv::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        split_k_slices,
        {alpha, beta},
        (void const *)d_A,
        (void const *)d_B,
        (void const *)d_C,
        (void *)d_D,
        (void const *)d_E,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        ldA, ldB, ldC, ldD, ldE
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "INT8 univ: can_implement failed (%d) for %dx%dx%d sk=%d\n",
                (int)status, M, N, K, split_k_slices);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
        return -1;
    }

    size_t workspace_size = SparseUniv::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
            return -2;
        }
    }

    status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -3;
    }

    /* Warmup */
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -4;
    }
    cudaDeviceSynchronize();

    /* Bench loop */
    for (int i = 0; i < iters; i++) {
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
            if (workspace) cudaFree(workspace);
            return -5;
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
    if (workspace) cudaFree(workspace);
    return 0;
}

/* =========================================================================
 * C-callable interface
 * ========================================================================= */
extern "C" {

int cutlass_int8_sparse_gemm_bench_ex(int M, int N, int K, int iters, int config, int split_k) {
    switch (config) {
        /* FAMILY 1: Basic SparseGemm (legacy) */
        case 0: return run_sparse_bench<SparseGemm_0>(M, N, K, iters, split_k);
        case 1: return run_sparse_bench<SparseGemm_1>(M, N, K, iters, split_k);
        case 2: return run_sparse_bench<SparseGemm_2>(M, N, K, iters, split_k);
        case 3: return run_sparse_bench<SparseGemm_3>(M, N, K, iters, split_k);
        case 4: return run_sparse_bench<SparseGemm_4>(M, N, K, iters, split_k);
        case 5: return run_sparse_bench<SparseGemm_5>(M, N, K, iters, split_k);

        /* FAMILY 2: Universal (basic alignment) */
        case 10: return run_int8_sparse_universal_bench<SparseUniv_0>(M, N, K, iters, split_k);
        case 11: return run_int8_sparse_universal_bench<SparseUniv_1>(M, N, K, iters, split_k);
        case 12: return run_int8_sparse_universal_bench<SparseUniv_2>(M, N, K, iters, split_k);
        case 13: return run_int8_sparse_universal_bench<SparseUniv_3>(M, N, K, iters, split_k);
        case 14: return run_int8_sparse_universal_bench<SparseUniv_4>(M, N, K, iters, split_k);
        /* 15: removed — 4-stage cp_async alignment issue */

        /* FAMILY 3: Universal + Swizzle<8> */
        case 20: return run_int8_sparse_universal_bench<SparseUnivS8_0>(M, N, K, iters, split_k);
        case 21: return run_int8_sparse_universal_bench<SparseUnivS8_1>(M, N, K, iters, split_k);
        case 22: return run_int8_sparse_universal_bench<SparseUnivS8_2>(M, N, K, iters, split_k);
        case 23: return run_int8_sparse_universal_bench<SparseUnivS8_3>(M, N, K, iters, split_k);
        case 24: return run_int8_sparse_universal_bench<SparseUnivS8_4>(M, N, K, iters, split_k);

        /* FAMILY 4: Universal + Swizzle<4> and no-swizzle variants */
        case 25: return run_int8_sparse_universal_bench<SparseUnivS4_0>(M, N, K, iters, split_k);
        case 26: return run_int8_sparse_universal_bench<SparseUnivS4_1>(M, N, K, iters, split_k);
        case 27: return run_int8_sparse_universal_bench<SparseUnivS4_2>(M, N, K, iters, split_k);
        case 28: return run_int8_sparse_universal_bench<SparseUnivNS_0>(M, N, K, iters, split_k);

        default: return -100;
    }
}

/* Legacy 4-arg interfaces (backward compatibility) */
int cutlass_int8_sparse_gemm_bench(int M, int N, int K, int iters) {
    return run_sparse_bench<SparseGemm_0>(M, N, K, iters, 1);
}
int cutlass_int8_sparse_gemm_bench_b(int M, int N, int K, int iters) {
    return run_sparse_bench<SparseGemm_4>(M, N, K, iters, 1);
}
int cutlass_int8_sparse_gemm_bench_c(int M, int N, int K, int iters) {
    return run_sparse_bench<SparseGemm_4>(M, N, K, iters, 1);
}
int cutlass_int8_sparse_gemm_bench_d(int M, int N, int K, int iters) {
    return run_sparse_bench<SparseGemm_0>(M, N, K, iters, 1);
}

void cutlass_int8_sparse_info(int *out_sparse, int *out_elements_per_e, int *out_sizeof_e) {
    *out_sparse = SparseGemm_0::kSparse;
    *out_elements_per_e = SparseGemm_0::kElementsPerElementE;
    *out_sizeof_e = (int)sizeof(typename SparseGemm_0::ElementE);
}

}  /* extern "C" */
