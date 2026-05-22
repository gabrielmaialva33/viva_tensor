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
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_conversion.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

/* =========================================================================
 * INT8 sparse — config 28 (winner @ ≤4096²)
 * Layout: A row-major (sparse), B col-major, C row-major
 * MMA: m16n8k64.row.col.s32.s8.s8.s32.satfinite
 * ========================================================================= */
namespace int8_sparse_run {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, ElementAcc>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

/* cfg 28: 256x128x128, 2 stages, A=16 — the winner */
using GemmCfg28 = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>, EpilogueOp, Swizzle,
    2, 16, 16, cutlass::arch::OpMultiplyAddSaturate>;

} /* namespace int8_sparse_run */

/* =========================================================================
 * INT4 sparse — config 29-ish (256x128x256, A32, swizzle<8>)
 * MMA: m16n8k128.row.col.s32.s4.s4.s32.satfinite
 * ========================================================================= */
namespace int4_sparse_run {

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::int4b_t;
using ElementC = int32_t;
using ElementAcc = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAcc, ElementAcc>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

/* Approximation of bench cfg=28: 256x128x256, 2stg, A=32 */
using GemmInt4Default = cutlass::gemm::device::GemmSparseUniversal<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>, EpilogueOp,
    Swizzle, 2, 32, 32, cutlass::arch::OpMultiplyAddSaturate>;

} /* namespace int4_sparse_run */

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
int cutlass_int8_sparse_run(int M, int N, int K, const int8_t *d_A_packed, const int8_t *d_B,
                            int32_t *d_C, const void *d_E, void *workspace, size_t workspace_size) {
    using namespace int8_sparse_run;
    using GemmKernel = typename GemmCfg28::GemmKernel;
    constexpr int kSparse = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    int ldA = K / kSparse;
    int ldB = K;
    int ldC = N;
    /* LayoutE = ColumnMajorInterleaved<2>; ldE = M * kInterleave. */
    (void)kElemsPerE;
    int ldE = M * 2;

    ElementAcc alpha = 1, beta = 0;

    GemmCfg28 gemm_op;
    typename GemmCfg28::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
                                       1, /* split-K slices */
                                       {alpha, beta}, (void const *)d_A_packed, (void const *)d_B,
                                       (void const *)d_C, (void *)d_C, (void const *)d_E,
                                       int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
                                       ldA, ldB, ldC, ldC, ldE);

    cutlass::Status s = gemm_op.can_implement(args);
    if (s != cutlass::Status::kSuccess)
        return -1;

    size_t needed = GemmCfg28::get_workspace_size(args);
    if (needed > workspace_size)
        return -2;

    s = gemm_op.initialize(args, workspace);
    if (s != cutlass::Status::kSuccess)
        return -3;

    s = gemm_op();
    if (s != cutlass::Status::kSuccess)
        return -4;
    return 0;
}

/**
 * cutlass_int8_sparse_workspace_size — query needed workspace for given shape.
 * Returns size in bytes, or 0 if shape is invalid.
 */
size_t cutlass_int8_sparse_workspace_size(int M, int N, int K) {
    using namespace int8_sparse_run;
    ElementAcc alpha = 1, beta = 0;
    typename GemmCfg28::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1,
                                       {alpha, beta}, nullptr, nullptr, nullptr, nullptr, nullptr,
                                       int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
                                       K / 2, K, N, N,
                                       M * 2); /* ldE = M*kInterleave (ColumnMajorInterleaved<2>) */
    GemmCfg28 op;
    if (op.can_implement(args) != cutlass::Status::kSuccess)
        return 0;
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
int cutlass_int4_sparse_run(int M, int N, int K, const void *d_A_packed, /* packed int4 */
                            const void *d_B,                             /* packed int4 */
                            int32_t *d_C, const void *d_E, void *workspace, size_t workspace_size) {
    using namespace int4_sparse_run;
    using GemmKernel = typename GemmInt4Default::GemmKernel;
    constexpr int kSparse = GemmKernel::kSparse;
    constexpr int kElemsPerE = GemmKernel::kElementsPerElementE;

    int ldA = K / kSparse; /* in INT4 elements */
    int ldB = K;           /* in INT4 elements */
    int ldC = N;
    /* LayoutE = ColumnMajorInterleaved<2>; its packed stride is
     * extent.row() * kInterleave = M * 2, NOT K_words. The previous
     * value (K/kSparse/kElemsPerE) was the column count of E (= K_words)
     * and produced numerically incorrect output. */
    int ldE = M * 2;

    ElementAcc alpha = 1, beta = 0;

    GemmInt4Default gemm_op;
    typename GemmInt4Default::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1,
                                             {alpha, beta}, d_A_packed, d_B, (void const *)d_C,
                                             (void *)d_C, d_E, int64_t(0), int64_t(0), int64_t(0),
                                             int64_t(0), int64_t(0), ldA, ldB, ldC, ldC, ldE);

    cutlass::Status s = gemm_op.can_implement(args);
    if (s != cutlass::Status::kSuccess)
        return -1;

    size_t needed = GemmInt4Default::get_workspace_size(args);
    if (needed > workspace_size)
        return -2;

    s = gemm_op.initialize(args, workspace);
    if (s != cutlass::Status::kSuccess)
        return -3;

    s = gemm_op();
    if (s != cutlass::Status::kSuccess)
        return -4;
    return 0;
}

size_t cutlass_int4_sparse_workspace_size(int M, int N, int K) {
    using namespace int4_sparse_run;
    ElementAcc alpha = 1, beta = 0;
    typename GemmInt4Default::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {alpha, beta}, nullptr, nullptr,
        nullptr, nullptr, nullptr, int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        K / 2, K, N, N, M * 2); /* ldE = M*kInterleave (ColumnMajorInterleaved<2>) */
    GemmInt4Default op;
    if (op.can_implement(args) != cutlass::Status::kSuccess)
        return 0;
    return GemmInt4Default::get_workspace_size(args);
}

/* Sparsity info accessors (so .c side can size buffers correctly) */
void cutlass_int8_sparse_run_info(int *sparse, int *elem_per_e, int *sizeof_e) {
    using namespace int8_sparse_run;
    *sparse = GemmCfg28::kSparse;
    *elem_per_e = GemmCfg28::kElementsPerElementE;
    *sizeof_e = (int)sizeof(typename GemmCfg28::ElementE);
}

void cutlass_int4_sparse_run_info(int *sparse, int *elem_per_e, int *sizeof_e) {
    using namespace int4_sparse_run;
    *sparse = GemmInt4Default::kSparse;
    *elem_per_e = GemmInt4Default::kElementsPerElementE;
    *sizeof_e = (int)sizeof(typename GemmInt4Default::ElementE);
}

/* Reorder a logical row-major ElementE buffer into the layout that the
 * Sm80 sparse Tensor Op kernel actually consumes — exactly what
 * cutlass::reorder_meta() does in tools/util/include/cutlass/util/host_reorder.h.
 *
 * Inputs:
 *   src       row-major uint32_t[M][K_words], one ElementE per word
 *   M         number of A rows (out_features)
 *   K_words   = original_K / kSparse / kElementsPerElementE
 *
 * Outputs:
 *   dst       buffer sized to match LayoutE::packed({M, K_words}) — must
 *             be at least M*K_words*sizeof(uint32_t) bytes; caller-owned.
 *
 * Both src and dst expose the same logical (M, K_words) shape but with
 * different underlying memory layouts:
 *   - src: RowMajor (offset = m*K_words + k)
 *   - dst: ColumnMajorInterleaved<2> (offset = (k/2)*M*2 + m*2 + k%2)
 */
void cutlass_int4_sparse_reorder_meta_e(uint32_t *dst, const uint32_t *src, int M, int K_words) {
    using ElementE = uint32_t;
    using LayoutSrc = cutlass::layout::RowMajor;
    using LayoutDst = cutlass::layout::ColumnMajorInterleaved<2>;

    cutlass::TensorRef<ElementE, LayoutSrc> src_ref(const_cast<ElementE *>(src),
                                                    LayoutSrc::packed({M, K_words}));
    cutlass::TensorRef<ElementE, LayoutDst> dst_ref(dst, LayoutDst::packed({M, K_words}));

    cutlass::reorder_meta(dst_ref, src_ref, {M, /* N unused */ 0, K_words});
}

/* Round-trip validator: given the same compressed A and (non-reordered)
 * metadata that the prepack NIF built, reconstruct the dense INT4 A
 * (M x K, row-major) and write it to `dst`. Caller then runs a host
 * matmul against the input and compares with the kernel output — if
 * the round-trip matches the FP32 reference and the kernel disagrees,
 * the compress/meta encoding is right and the kernel-side path needs
 * fixing. Used only by debug/validation paths.
 *
 *   compressed_a    int8_t* with packed int4 — size M * (K/4) bytes
 *   meta            uint32_t* row-major logical ElementE
 *   K, M            shape
 *   dst             int8_t* output, one int4 value per int8 cell
 *                   (no nibble packing) so the caller can dot-product
 *                   without bit manipulation
 */
void cutlass_int4_sparse_uncompress_to_dense(int8_t *dst_int8_per_cell, const int8_t *compressed_a,
                                             const uint32_t *meta, int M, int K) {
    using ElementA = cutlass::int4b_t;
    using ElementE = uint32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutE = cutlass::layout::RowMajor;

    int Kp = K / 2;
    int Kw = K / 64;

    cutlass::TensorRef<ElementA, LayoutA> a_compressed_ref(reinterpret_cast<ElementA *>(
                                                               const_cast<int8_t *>(compressed_a)),
                                                           LayoutA::packed({M, Kp}));

    cutlass::TensorRef<ElementE, LayoutE> e_ref(const_cast<ElementE *>(meta),
                                                LayoutE::packed({M, Kw}));

    /* Need a uncompressed_a buffer of int4b_t with shape [M, K], packed. */
    size_t uncomp_bytes = (size_t)M * (size_t)(K / 2);
    int8_t *uncomp = (int8_t *)std::malloc(uncomp_bytes);
    if (!uncomp)
        return;
    std::memset(uncomp, 0, uncomp_bytes);

    cutlass::TensorRef<ElementA, LayoutA> uncompressed_ref(reinterpret_cast<ElementA *>(uncomp),
                                                           LayoutA::packed({M, K}));

    cutlass::uncompress<ElementA, LayoutA, ElementE, LayoutE>(uncompressed_ref, a_compressed_ref,
                                                              e_ref, M, K);

    /* Unpack nibble pairs into one int8 per cell (sign-extend int4). */
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            uint8_t byte = (uint8_t)uncomp[(size_t)m * (K / 2) + (k / 2)];
            int nib = (k % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            if (nib & 0x08)
                nib |= 0xFFFFFFF0;
            dst_int8_per_cell[(size_t)m * K + k] = (int8_t)nib;
        }
    }

    std::free(uncomp);
}

/* Self-test: run the kernel against a CUTLASS-built compressed_A + meta
 * (using exactly the same TensorFillRandomSparseMeta + uncompress flow
 * as example 15) and compare the kernel's INT32 output to a host-side
 * dense GEMM of uncompress(compressed_A, meta) × B.
 *
 *   Returns:
 *     0 on success (all output cells match)
 *     -1 if can_implement fails
 *     -2 if the kernel reports failure
 *     positive N = number of mismatched cells (also writes max_abs_diff
 *                  to *out_max_diff if non-null)
 */
int cutlass_int4_sparse_self_test(int M, int N, int K, int *out_max_diff) {
    using namespace int4_sparse_run;
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = int32_t;
    using ElementAcc = int32_t;
    using ElementE = uint32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutE = cutlass::layout::ColumnMajorInterleaved<2>;
    using LayoutE_RM = cutlass::layout::RowMajor;

    const int kSparse = 2;
    const int kElemsPerE = 32;
    const int kMetaSize = 4;

    cutlass::HostTensor<ElementA, LayoutA> tensor_A(cutlass::make_Coord(M, K / kSparse));
    cutlass::HostTensor<ElementB, LayoutB> tensor_B(cutlass::make_Coord(K, N));
    cutlass::HostTensor<ElementC, LayoutC> tensor_C(cutlass::make_Coord(M, N));
    cutlass::HostTensor<ElementC, LayoutC> tensor_D_kernel(cutlass::make_Coord(M, N));
    cutlass::HostTensor<ElementC, LayoutC> tensor_D_ref(cutlass::make_Coord(M, N));
    cutlass::HostTensor<ElementA, LayoutA> tensor_A_uncompressed(cutlass::make_Coord(M, K));
    cutlass::HostTensor<ElementE, LayoutE> tensor_E(
        cutlass::make_Coord(M, K / kSparse / kElemsPerE));
    cutlass::HostTensor<ElementE, LayoutE_RM> tensor_E_rm(
        cutlass::make_Coord(M, K / kSparse / kElemsPerE));

    cutlass::reference::host::TensorFillRandomUniform(tensor_A.host_view(), /*seed=*/1, /*max=*/4,
                                                      /*min=*/-4, /*bits=*/0);
    cutlass::reference::host::TensorFillRandomUniform(tensor_B.host_view(), /*seed=*/2, /*max=*/4,
                                                      /*min=*/-4, /*bits=*/0);
    cutlass::reference::host::TensorFill(tensor_C.host_view(), ElementC(0));
    cutlass::reference::host::TensorFill(tensor_D_kernel.host_view(), ElementC(0));
    cutlass::reference::host::TensorFill(tensor_D_ref.host_view(), ElementC(0));

    cutlass::reference::host::TensorFillRandomSparseMeta(tensor_E_rm.host_view(), /*seed=*/3,
                                                         kMetaSize);

    /* Reorder rm -> interleaved as the kernel expects. */
    cutlass::reorder_meta(tensor_E.host_ref(), tensor_E_rm.host_ref(),
                          {M, N, K / kSparse / kElemsPerE});

    /* Uncompress for host reference. */
    cutlass::uncompress(tensor_A_uncompressed.host_ref(), tensor_A.host_ref(),
                        tensor_E_rm.host_ref(), M, K);

    /* Host dense reference: D = A_uncompressed @ B. */
    cutlass::reference::host::compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                                           ElementAcc, ElementAcc>(
        {M, N, K}, ElementAcc(1), tensor_A_uncompressed.host_ref(), tensor_B.host_ref(),
        ElementAcc(0), tensor_C.host_ref(), tensor_D_ref.host_ref(), ElementAcc(0));

    /* Upload + run kernel. */
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_kernel.sync_device();
    tensor_E.sync_device();

    size_t ws = cutlass_int4_sparse_workspace_size(M, N, K);
    void *d_ws = nullptr;
    if (ws)
        cudaMalloc(&d_ws, ws);

    int rc =
        cutlass_int4_sparse_run(M, N, K, tensor_A.device_data(), tensor_B.device_data(),
                                tensor_D_kernel.device_data(), tensor_E.device_data(), d_ws, ws);

    if (d_ws)
        cudaFree(d_ws);
    if (rc != 0)
        return -2;

    tensor_D_kernel.sync_host();

    int diffs = 0, max_abs = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int k = tensor_D_kernel.host_view().at({m, n});
            int r = tensor_D_ref.host_view().at({m, n});
            int d = (k > r) ? (k - r) : (r - k);
            if (d > 0)
                ++diffs;
            if (d > max_abs)
                max_abs = d;
        }
    }
    if (out_max_diff)
        *out_max_diff = max_abs;
    return diffs;
}

} /* extern "C" */
