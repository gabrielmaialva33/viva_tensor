/**
 * cuda_int_sparse_run.cu — runtime launchers for INT8/INT4 2:4 sparse GEMM
 *
 * Companion to cuda_sparse_int8_cutlass.cu / cuda_int4_sparse_cutlass.cu
 * (which only expose self-contained *bench* functions that allocate and free
 * GEMM buffers internally).
 *
 * This file exposes "run" entrypoints that operate on caller-owned device
 * memory — suitable for inference where the packed weight lives across many
 * forward passes.
 *
 * Three entrypoints:
 *   int cutlass_int8_sparse_run(...)
 *   int cutlass_int4_sparse_run(...)
 *   int cusparselt_int8_build_and_compress(...)  — also lives in nif_prepack
 *
 * Configuration: cfg=28 (256x128x128, 2stg, A16, no swizzle) is the winner
 * for INT8 sparse @ ≤4096². cfg=29 (256x128x256, A32+B32) is the winner for
 * INT4 sparse.
 *
 * All functions return 0 on success or negative error code.
 */

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include <cuda_runtime.h>
#include <cstdio>

/* =========================================================================
 * INT8 sparse — config 28 (winner @ ≤4096²)
 * Layout: A row-major (sparse), B col-major, C row-major
 * MMA: m16n8k64.row.col.s32.s8.s8.s32.satfinite
 * ========================================================================= */
namespace int8_sparse_run {

using ElementA   = int8_t;
using ElementB   = int8_t;
using ElementC   = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAcc, ElementAcc>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

/* cfg 28: 256x128x128, 2 stages, A=16 — the winner */
using GemmCfg28 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    EpilogueOp, Swizzle, 2, 16, 16,
    cutlass::arch::OpMultiplyAddSaturate>;

}  /* namespace int8_sparse_run */

/* =========================================================================
 * INT4 sparse — config 29-ish (256x128x256, A32, swizzle<8>)
 * MMA: m16n8k128.row.col.s32.s4.s4.s32.satfinite
 * ========================================================================= */
namespace int4_sparse_run {

using ElementA   = cutlass::int4b_t;
using ElementB   = cutlass::int4b_t;
using ElementC   = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAcc, ElementAcc>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

/* Approximation of bench cfg=28: 256x128x256, 2stg, A=32 */
using GemmInt4Default = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 128>,
    EpilogueOp, Swizzle, 2, 32, 32,
    cutlass::arch::OpMultiplyAddSaturate>;

}  /* namespace int4_sparse_run */

/* =========================================================================
 * C-callable interface
 * ========================================================================= */
extern "C" {

/**
 * cutlass_int8_sparse_run — runs INT8 2:4 sparse GEMM on caller-owned device buffers.
 *
 *   M, N, K              GEMM dimensions (K must be multiple of 64).
 *   d_A_packed           device ptr to sparse A, size M * (K/2) bytes
 *   d_B                  device ptr to dense B (col-major), size K * N bytes
 *   d_C                  device ptr to output C (row-major INT32), size M * N * 4 bytes
 *   d_E                  device ptr to 2:4 metadata, size M * (K/2/16) bytes (ElementE = uint16)
 *   workspace            optional cudaMalloc'd workspace (or NULL)
 *   workspace_size       size of workspace
 *
 * Returns 0 on success, negative on error.
 */
int cutlass_int8_sparse_run(
    int M, int N, int K,
    const int8_t* d_A_packed,
    const int8_t* d_B,
    int32_t*      d_C,
    const void*   d_E,
    void*         workspace,
    size_t        workspace_size)
{
    using namespace int8_sparse_run;
    using GemmKernel = typename GemmCfg28::GemmKernel;
    constexpr int kSparse    = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    int ldA = K / kSparse;
    int ldB = K;
    int ldC = N;
    int ldE = (K / kSparse / kElemsPerE);

    ElementAcc alpha = 1, beta = 0;

    GemmCfg28 gemm_op;
    typename GemmCfg28::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,                    /* split-K slices */
        {alpha, beta},
        (void const*)d_A_packed,
        (void const*)d_B,
        (void const*)d_C,
        (void*)d_C,
        (void const*)d_E,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        ldA, ldB, ldC, ldC, ldE);

    cutlass::Status s = gemm_op.can_implement(args);
    if (s != cutlass::Status::kSuccess) return -1;

    size_t needed = GemmCfg28::get_workspace_size(args);
    if (needed > workspace_size) return -2;

    s = gemm_op.initialize(args, workspace);
    if (s != cutlass::Status::kSuccess) return -3;

    s = gemm_op();
    if (s != cutlass::Status::kSuccess) return -4;
    return 0;
}

/**
 * cutlass_int8_sparse_workspace_size — query needed workspace for given shape.
 * Returns size in bytes, or 0 if shape is invalid.
 */
size_t cutlass_int8_sparse_workspace_size(int M, int N, int K) {
    using namespace int8_sparse_run;
    ElementAcc alpha = 1, beta = 0;
    typename GemmCfg28::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, 1, {alpha, beta},
        nullptr, nullptr, nullptr, nullptr, nullptr,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        K / 2, K, N, N, (K / 2 / 16));
    GemmCfg28 op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return 0;
    return GemmCfg28::get_workspace_size(args);
}

/**
 * cutlass_int4_sparse_run — runs INT4 2:4 sparse GEMM on caller-owned device buffers.
 *
 *   M, N, K          GEMM dimensions (K must be multiple of 128).
 *   d_A_packed       sparse A, size M * (K/2/2) bytes (each int4 nibble = 0.5 byte; +50% sparse)
 *   d_B              dense B (col-major INT4), size (K * N) / 2 bytes
 *   d_C              output INT32, size M * N * 4 bytes
 *   d_E              2:4 metadata
 *   workspace        optional workspace
 *   workspace_size   size of workspace
 *
 * Returns 0 on success.
 */
int cutlass_int4_sparse_run(
    int M, int N, int K,
    const void*   d_A_packed,    /* packed int4 */
    const void*   d_B,           /* packed int4 */
    int32_t*      d_C,
    const void*   d_E,
    void*         workspace,
    size_t        workspace_size)
{
    using namespace int4_sparse_run;
    using GemmKernel = typename GemmInt4Default::GemmKernel;
    constexpr int kSparse    = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    int ldA = K / kSparse;            /* in INT4 elements */
    int ldB = K;                       /* in INT4 elements */
    int ldC = N;
    int ldE = (K / kSparse / kElemsPerE);

    ElementAcc alpha = 1, beta = 0;

    GemmInt4Default gemm_op;
    typename GemmInt4Default::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {alpha, beta},
        d_A_packed,
        d_B,
        (void const*)d_C,
        (void*)d_C,
        d_E,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        ldA, ldB, ldC, ldC, ldE);

    cutlass::Status s = gemm_op.can_implement(args);
    if (s != cutlass::Status::kSuccess) return -1;

    size_t needed = GemmInt4Default::get_workspace_size(args);
    if (needed > workspace_size) return -2;

    s = gemm_op.initialize(args, workspace);
    if (s != cutlass::Status::kSuccess) return -3;

    s = gemm_op();
    if (s != cutlass::Status::kSuccess) return -4;
    return 0;
}

size_t cutlass_int4_sparse_workspace_size(int M, int N, int K) {
    using namespace int4_sparse_run;
    ElementAcc alpha = 1, beta = 0;
    typename GemmInt4Default::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, 1, {alpha, beta},
        nullptr, nullptr, nullptr, nullptr, nullptr,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        K / 2, K, N, N, (K / 2 / 32));
    GemmInt4Default op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return 0;
    return GemmInt4Default::get_workspace_size(args);
}

/* Sparsity info accessors (so .c side can size buffers correctly) */
void cutlass_int8_sparse_run_info(int* sparse, int* elem_per_e, int* sizeof_e) {
    using namespace int8_sparse_run;
    *sparse = GemmCfg28::kSparse;
    *elem_per_e = GemmCfg28::kElementsPerElementE;
    *sizeof_e = (int)sizeof(typename GemmCfg28::ElementE);
}

void cutlass_int4_sparse_run_info(int* sparse, int* elem_per_e, int* sizeof_e) {
    using namespace int4_sparse_run;
    *sparse = GemmInt4Default::kSparse;
    *elem_per_e = GemmInt4Default::kElementsPerElementE;
    *sizeof_e = (int)sizeof(typename GemmInt4Default::ElementE);
}

}  /* extern "C" */
