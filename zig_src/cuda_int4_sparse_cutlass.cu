/**
 * INT4 2:4 Structured Sparse GEMM via CUTLASS 4.3.5 — 2642 TOPS target on RTX 4090!
 *
 * INT4 sparse uses the MMA instruction:
 *   mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite
 *
 * This processes 2x more elements per instruction than INT8 sparse (m16n8k64):
 *   INT8 sparse: 16*8*64 = 8192 ops/instruction → 1321 TOPS
 *   INT4 sparse: 16*8*128 = 16384 ops/instruction → 2642 TOPS
 *
 * RTX 4090 peaks: INT4 dense=1320T, INT4 sparse=2642T (theoretical)
 *
 * Two kernel families:
 *   - Configs 0-5: SparseGemm (basic, no split-K)
 *   - Configs 10-15: GemmSparseUniversal (split-K support!)
 *
 * Achieved: 1854 TOPS (70.2%) peak — Universal cfg=29 (256x128x256, A32+B32, Swizzle<8>) at 8192x4096x131072
 * Sustained: 1845 TOPS (69.9%) at iters=100, 4096x4096x131072
 *
 * Key optimization: AlignmentA=32 (128-bit loads, CUTLASS default) vs AlignmentA=16 (64-bit) = +50 TOPS
 *
 * BSD-3-Clause License (CUTLASS)
 */

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include <cstdio>
#include <cuda_runtime.h>

/* Common types for INT4 sparse */
using ElementA   = cutlass::int4b_t;
using ElementB   = cutlass::int4b_t;
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
 * ========================================================================= */

/* Config 0: 128×128×256, warp 64×64×256, 3 stages (~84KB SMEM) — proven */
using SparseGemm_0 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 1: 128×256×256, warp 64×64×256, 2 stages (~88KB) — BEST basic */
using SparseGemm_1 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 2: 256×128×256, warp 64×64×256, 2 stages (~88KB) — large M */
using SparseGemm_2 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 3: 64×128×256, warp 32×64×256, 3 stages (~60KB) — more CTAs */
using SparseGemm_3 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 256>,
    cutlass::gemm::GemmShape<32, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 4: 64×64×512, warp 32×32×256, 3 stages (~84KB) — large K-tile */
using SparseGemm_4 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<32, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 5: 128×64×256, warp 64×32×256, 3 stages (~60KB) — tall narrow */
using SparseGemm_5 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 256>,
    cutlass::gemm::GemmShape<64, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 2: GemmSparseUniversal (split-K support!) — configs 10-15
 *   Same tile configs as 0-5 but via GemmSparseUniversal device class
 *   which supports GemmUniversalMode::kGemm with serial split-K reduction
 * ========================================================================= */

/* Config 10: Universal 128×128×256, warp 64×64×256, 3 stages */
using SparseUniv_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 11: Universal 128×256×256, warp 64×64×256, 2 stages */
using SparseUniv_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 12: Universal 256×128×256, warp 64×64×256, 2 stages */
using SparseUniv_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 13: Universal 64×128×256, warp 32×64×256, 3 stages */
using SparseUniv_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 256>,
    cutlass::gemm::GemmShape<32, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 14: Universal 64×64×512, warp 32×32×256, 3 stages */
using SparseUniv_4 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<32, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 15: Universal 128×64×256, warp 64×32×256, 3 stages */
using SparseUniv_5 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 256>,
    cutlass::gemm::GemmShape<64, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 3, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 3: GemmSparseUniversal with Swizzle<4> — configs 20-22
 *   L2-aware rasterization: group 4 CTAs along N before advancing M
 * ========================================================================= */
using Swizzle4 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;

/* Config 20: Universal 256×128×256 + Swizzle<4>, 2 stages — best tile + L2 opt */
using SparseUnivS4_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 21: Universal 128×256×256 + Swizzle<4>, 2 stages */
using SparseUnivS4_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 22: Universal 128×128×256 + Swizzle<4>, 3 stages */
using SparseUnivS4_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 3, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 23: Universal 256×128×256 + Swizzle<8>, 2 stages — max L2 grouping */
using Swizzle8 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using SparseUnivS8_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 24: Universal 128×256×256 + Swizzle<2>, 2 stages — mild grouping */
using Swizzle2 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>;
using SparseUnivS2_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle2, 2, 16, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 4: Alignment-corrected configs — AlignmentA=32 (128-bit loads!)
 *   Default CUTLASS INT4 alignment is 32 elements = 128 bits per load.
 *   We were using AlignmentA=16 (64-bit loads) — potentially 2x bandwidth!
 * ========================================================================= */

/* Config 25: Universal 256×128×256, 2stg, AlignA=32 AlignB=32 + Swizzle<4> */
using SparseUnivA32_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 26: Universal 256×128×256, 2stg, AlignA=32 AlignB=32, no swizzle */
using SparseUnivA32_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 27: Universal 128×256×256, 2stg, AlignA=32 AlignB=32 + Swizzle<4> */
using SparseUnivA32_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 28: Universal 128×128×256, 3stg, AlignA=32 AlignB=32 + Swizzle<4> */
using SparseUnivA32_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle4, 3, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 29: Universal 256×128×256, 2stg, AlignA=32 AlignB=32 + Swizzle<8> */
using SparseUnivA32_4 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 30: Basic SparseGemm 128×256×256, 2stg, AlignA=32 AlignB=32 — for large K */
using SparseGemm_A32_0 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 32, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 31: Basic SparseGemm 256×128×256, 2stg, AlignA=32 AlignB=32 — for large K */
using SparseGemm_A32_1 = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 32, 32, false,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 32: Universal 64×128×256, 3stg, AlignA=32 AlignB=32 + Swizzle<8> — more CTAs */
using SparseUnivA32_5 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 256>,
    cutlass::gemm::GemmShape<32, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 3, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 33: Universal 64×64×512, 3stg, AlignA=32 AlignB=32 + Swizzle<8> — deep K */
using SparseUnivA32_6 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 512>,
    cutlass::gemm::GemmShape<32, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 3, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 5: LinearCombinationClamp epilogue (CUTLASS default for INT4)
 *   Uses float for alpha/beta computation instead of int32_t
 * ========================================================================= */
using EpilogueOpClamp = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,  /* = 4 elements per access */
    ElementAcc,
    float  /* CUTLASS default: float compute for epilogue */
>;

/* Config 34: Universal 256×128×256, 2stg, A32 B32, Clamp + Swizzle<8> */
using SparseUnivClamp_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOpClamp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 35: Universal 256×128×256, 2stg, A32 B32, Clamp + Swizzle<4> */
using SparseUnivClamp_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOpClamp, Swizzle4, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 36: Universal 256×128×256, 2stg, A32 B32, Clamp, no swizzle */
using SparseUnivClamp_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOpClamp, Swizzle, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 6: Warp shape variations — same 256×128×256 TB, different warp tiling
 *   Exploring whether different warp shapes reduce register pressure for 2 CTAs/SM
 * ========================================================================= */

/* Config 40: Universal 256×128×256, warp 128×32×256 (tall narrow), A32 S8 */
using SparseUnivWarp_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<128, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 41: Universal 256×128×256, warp 32×128×256 (wide short), A32 S8 */
using SparseUnivWarp_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<32, 128, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 42: Universal 128×128×256, 2stg, warp 64×64×256, A32 S8 — smaller TB for 2 CTAs */
using SparseUnivSmall_0 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 43: Universal 128×64×256, 2stg, warp 64×32×256, A32 S8 — compact for 2 CTAs */
using SparseUnivSmall_1 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 256>,
    cutlass::gemm::GemmShape<64, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 44: Universal 128×64×256, 3stg, warp 64×32×256, A32 S8 — 3 stages more prefetch */
using SparseUnivSmall_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 256>,
    cutlass::gemm::GemmShape<64, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 3, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 45: Universal 64×128×256, 2stg, warp 32×64×256, A32 S8 — wide for 2 CTAs */
using SparseUnivSmall_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 256>,
    cutlass::gemm::GemmShape<32, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* Config 46: Universal 128×128×256, 2stg, warp 64×32×256, A32 S8 — more warps in N */
using SparseUnivWarp_2 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 32, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * FAMILY 7: RESERVED (WarpK=128 configs removed — sparse requires WarpK≥256)
 *   kWarpGemmIterations = WarpK/InstrK must be even and ≥2
 *   InstrK=128 for INT4 sparse → WarpK≥256 is mandatory
 * ========================================================================= */

/* Config 47: Universal 128×128×256, 2stg, warp 32×64×256, A32 S8 — more warps in M */
using SparseUnivWarp_3 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<32, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle8, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate
>;

/* =========================================================================
 * Benchmark runner for SparseGemm (basic, no split-K)
 * ========================================================================= */
template <typename SparseGemm>
static int run_int4_sparse_bench(int M, int N, int K, int iters, int split_k_slices) {
    using ElementE = typename SparseGemm::ElementE;

    constexpr int kSparse = SparseGemm::kSparse;
    constexpr int kElemsPerE = SparseGemm::kElementsPerElementE;

    size_t sizeA_elems = (size_t)M * (K / kSparse);
    size_t sizeB_elems = (size_t)K * N;
    size_t sizeC_elems = (size_t)M * N;
    size_t metaK = K / kSparse / kElemsPerE;
    size_t sizeE = (size_t)M * metaK;

    size_t bytesA = (sizeA_elems + 1) / 2;
    size_t bytesB = (sizeB_elems + 1) / 2;
    size_t bytesC = sizeC_elems * sizeof(int32_t);
    size_t bytesE = sizeE * sizeof(ElementE);

    void *d_A = nullptr, *d_B = nullptr;
    int32_t  *d_C = nullptr;
    ElementE *d_E = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_A, bytesA);
    if (err != cudaSuccess) { return -10; }
    err = cudaMalloc(&d_B, bytesB);
    if (err != cudaSuccess) { cudaFree(d_A); return -11; }
    err = cudaMalloc(&d_C, bytesC);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -12; }
    err = cudaMalloc(&d_E, bytesE);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -13; }

    cudaMemset(d_A, 0x11, bytesA);
    cudaMemset(d_B, 0x11, bytesB);
    cudaMemset(d_C, 0, bytesC);
    cudaMemset(d_E, 0, bytesE);

    SparseGemm gemm_op;
    ElementAcc alpha = 1, beta = 0;

    int ldA = K / kSparse;
    int ldB = K;
    int ldC = N;
    int ldE = (int)metaK;

    typename SparseGemm::Arguments args(
        {M, N, K},
        {(ElementA*)d_A, ldA}, {(ElementB*)d_B, ldB},
        {d_C, ldC}, {d_C, ldC},
        {d_E, ldE},
        {alpha, beta}, split_k_slices
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

    for (int w = 0; w < 5; w++) { status = gemm_op(); }
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -4;
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++) { gemm_op(); }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    int elapsed_us = (int)(elapsed_ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_E);
    if (workspace) cudaFree(workspace);

    return (elapsed_us > 0) ? elapsed_us : 1;
}

/* =========================================================================
 * Benchmark runner for GemmSparseUniversal (split-K support!)
 *
 * Uses GemmUniversalMode::kGemm which does serial K-reduction:
 *   - split_k_slices > 1: partitions K among CTAs, accumulates with atomics
 *   - This allows more CTAs to be active for compute-bound problems
 * ========================================================================= */
template <typename SparseUniv>
static int run_int4_sparse_universal_bench(int M, int N, int K, int iters, int split_k_slices) {
    using ElementE = typename SparseUniv::ElementE;
    using GemmKernel = typename SparseUniv::GemmKernel;

    constexpr int kSparse = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    size_t sizeA_elems = (size_t)M * (K / kSparse);
    size_t sizeB_elems = (size_t)K * N;
    size_t sizeC_elems = (size_t)M * N;
    size_t metaK = K / kSparse / kElemsPerE;
    size_t sizeE = (size_t)M * metaK;

    size_t bytesA = (sizeA_elems + 1) / 2;
    size_t bytesB = (sizeB_elems + 1) / 2;
    size_t bytesC = sizeC_elems * sizeof(int32_t);
    size_t bytesE = sizeE * sizeof(ElementE);

    void *d_A = nullptr, *d_B = nullptr;
    int32_t  *d_C = nullptr, *d_D = nullptr;
    ElementE *d_E = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_A, bytesA);
    if (err != cudaSuccess) { return -10; }
    err = cudaMalloc(&d_B, bytesB);
    if (err != cudaSuccess) { cudaFree(d_A); return -11; }
    err = cudaMalloc(&d_C, bytesC);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return -12; }
    err = cudaMalloc(&d_D, bytesC);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return -12; }
    err = cudaMalloc(&d_E, bytesE);
    if (err != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); return -13; }

    cudaMemset(d_A, 0x11, bytesA);
    cudaMemset(d_B, 0x11, bytesB);
    cudaMemset(d_C, 0, bytesC);
    cudaMemset(d_D, 0, bytesC);
    cudaMemset(d_E, 0, bytesE);

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
        int64_t(0),   /* batch_stride_A */
        int64_t(0),   /* batch_stride_B */
        int64_t(0),   /* batch_stride_C */
        int64_t(0),   /* batch_stride_D */
        int64_t(0),   /* batch_stride_E */
        ldA, ldB, ldC, ldD, ldE
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "INT4 univ: can_implement failed (%d) for %dx%dx%d sk=%d\n",
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
        fprintf(stderr, "INT4 univ: initialize failed (%d)\n", (int)status);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -3;
    }

    for (int w = 0; w < 5; w++) { status = gemm_op(); }
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
        if (workspace) cudaFree(workspace);
        return -4;
    }
    cudaDeviceSynchronize();

    /* Try CUDA Graph capture for zero-overhead iteration replay */
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool useGraph = (iters >= 2);

    if (useGraph) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        /* Re-initialize with the stream */
        status = gemm_op.initialize(args, workspace, stream);
        if (status != cutlass::Status::kSuccess) useGraph = false;

        if (useGraph) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int i = 0; i < iters; i++) { gemm_op(stream); }
            cudaStreamEndCapture(stream, &graph);

            if (graph) {
                cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            }
            if (!graphExec) useGraph = false;
        }
        cudaStreamDestroy(stream);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (useGraph && graphExec) {
        /* Graph replay: zero launch overhead */
        cudaEventRecord(start, 0);
        cudaGraphLaunch(graphExec, 0);
        cudaEventRecord(stop, 0);
    } else {
        /* Fallback: regular iteration loop */
        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; i++) { gemm_op(); }
        cudaEventRecord(stop, 0);
    }
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    int elapsed_us = (int)(elapsed_ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (graphExec) cudaGraphExecDestroy(graphExec);
    if (graph) cudaGraphDestroy(graph);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D); cudaFree(d_E);
    if (workspace) cudaFree(workspace);

    return (elapsed_us > 0) ? elapsed_us : 1;
}

/* =========================================================================
 * C-callable interface
 * ========================================================================= */
extern "C" {

/**
 * cutlass_int4_sparse_gemm_bench(M, N, K, iters, config, split_k)
 *
 * config 0-5:   SparseGemm (basic, split_k must be 1)
 * config 10-15: GemmSparseUniversal (split_k >= 1 supported!)
 *
 * Returns elapsed microseconds or negative error code.
 */
int cutlass_int4_sparse_gemm_bench(int M, int N, int K, int iters, int config, int split_k) {
    switch (config) {
        /* Basic SparseGemm configs */
        case 0: return run_int4_sparse_bench<SparseGemm_0>(M, N, K, iters, split_k);
        case 1: return run_int4_sparse_bench<SparseGemm_1>(M, N, K, iters, split_k);
        case 2: return run_int4_sparse_bench<SparseGemm_2>(M, N, K, iters, split_k);
        case 3: return run_int4_sparse_bench<SparseGemm_3>(M, N, K, iters, split_k);
        case 4: return run_int4_sparse_bench<SparseGemm_4>(M, N, K, iters, split_k);
        case 5: return run_int4_sparse_bench<SparseGemm_5>(M, N, K, iters, split_k);

        /* Universal SparseGemm configs (split-K!) */
        case 10: return run_int4_sparse_universal_bench<SparseUniv_0>(M, N, K, iters, split_k);
        case 11: return run_int4_sparse_universal_bench<SparseUniv_1>(M, N, K, iters, split_k);
        case 12: return run_int4_sparse_universal_bench<SparseUniv_2>(M, N, K, iters, split_k);
        case 13: return run_int4_sparse_universal_bench<SparseUniv_3>(M, N, K, iters, split_k);
        case 14: return run_int4_sparse_universal_bench<SparseUniv_4>(M, N, K, iters, split_k);
        case 15: return run_int4_sparse_universal_bench<SparseUniv_5>(M, N, K, iters, split_k);

        /* Swizzled Universal configs */
        case 20: return run_int4_sparse_universal_bench<SparseUnivS4_0>(M, N, K, iters, split_k);
        case 21: return run_int4_sparse_universal_bench<SparseUnivS4_1>(M, N, K, iters, split_k);
        case 22: return run_int4_sparse_universal_bench<SparseUnivS4_2>(M, N, K, iters, split_k);
        case 23: return run_int4_sparse_universal_bench<SparseUnivS8_0>(M, N, K, iters, split_k);
        case 24: return run_int4_sparse_universal_bench<SparseUnivS2_0>(M, N, K, iters, split_k);
        /* Alignment-corrected configs (AlignA=32) */
        case 25: return run_int4_sparse_universal_bench<SparseUnivA32_0>(M, N, K, iters, split_k);
        case 26: return run_int4_sparse_universal_bench<SparseUnivA32_1>(M, N, K, iters, split_k);
        case 27: return run_int4_sparse_universal_bench<SparseUnivA32_2>(M, N, K, iters, split_k);
        case 28: return run_int4_sparse_universal_bench<SparseUnivA32_3>(M, N, K, iters, split_k);
        case 29: return run_int4_sparse_universal_bench<SparseUnivA32_4>(M, N, K, iters, split_k);
        case 30: return run_int4_sparse_bench<SparseGemm_A32_0>(M, N, K, iters, split_k);
        case 31: return run_int4_sparse_bench<SparseGemm_A32_1>(M, N, K, iters, split_k);
        case 32: return run_int4_sparse_universal_bench<SparseUnivA32_5>(M, N, K, iters, split_k);
        case 33: return run_int4_sparse_universal_bench<SparseUnivA32_6>(M, N, K, iters, split_k);
        /* Clamp epilogue configs */
        case 34: return run_int4_sparse_universal_bench<SparseUnivClamp_0>(M, N, K, iters, split_k);
        case 35: return run_int4_sparse_universal_bench<SparseUnivClamp_1>(M, N, K, iters, split_k);
        case 36: return run_int4_sparse_universal_bench<SparseUnivClamp_2>(M, N, K, iters, split_k);
        /* Warp shape variation configs */
        case 40: return run_int4_sparse_universal_bench<SparseUnivWarp_0>(M, N, K, iters, split_k);
        case 41: return run_int4_sparse_universal_bench<SparseUnivWarp_1>(M, N, K, iters, split_k);
        case 42: return run_int4_sparse_universal_bench<SparseUnivSmall_0>(M, N, K, iters, split_k);
        case 43: return run_int4_sparse_universal_bench<SparseUnivSmall_1>(M, N, K, iters, split_k);
        case 44: return run_int4_sparse_universal_bench<SparseUnivSmall_2>(M, N, K, iters, split_k);
        case 45: return run_int4_sparse_universal_bench<SparseUnivSmall_3>(M, N, K, iters, split_k);
        case 46: return run_int4_sparse_universal_bench<SparseUnivWarp_2>(M, N, K, iters, split_k);
        case 47: return run_int4_sparse_universal_bench<SparseUnivWarp_3>(M, N, K, iters, split_k);
        /* 50-55: reserved (WarpK=128 configs removed — sparse requires WarpK≥256) */
        default: return -100;
    }
}

void cutlass_int4_sparse_info(int *out_sparse, int *out_elements_per_e, int *out_sizeof_e) {
    *out_sparse = SparseGemm_0::kSparse;
    *out_elements_per_e = SparseGemm_0::kElementsPerElementE;
    *out_sizeof_e = (int)sizeof(typename SparseGemm_0::ElementE);
}

}  /* extern "C" */
