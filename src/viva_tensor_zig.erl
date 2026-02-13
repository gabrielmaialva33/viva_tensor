%% viva_tensor_zig.erl - Zig SIMD NIF wrapper
%%
%% Provides cross-platform SIMD-accelerated tensor operations via Zig.
%% Falls back to pure Erlang implementation if NIF is not available.

-module(viva_tensor_zig).
-export([
    simd_dot/2,
    simd_sum/1,
    simd_scale/2,
    simd_add/2,
    simd_mul/2,
    simd_matmul/5,
    simd_available/0,
    backend_info/0,
    is_loaded/0,
    cpu_topology/0
]).

%% NIF Resource API - zero-copy tensor operations
-export([
    nt_zeros/1, nt_ones/1, nt_fill/2, nt_from_list/2,
    nt_to_list/1, nt_shape/1, nt_size/1,
    nt_add/2, nt_sub/2, nt_mul/2, nt_scale/2, nt_negate/1,
    nt_dot/2, nt_sum/1, nt_max/1, nt_min/1,
    nt_matmul/5, nt_matmul_inplace/6, nt_matmul_blas/5, nt_matmul_cuda/5, nt_matmul_cuda_fp32/5,
    nt_matmul_int8_tc/5, nt_int8_tc_available/0, %% INT8 Tensor Cores (old DP4A)
    nt_matmul_fp16_tc/5, nt_fp16_tc_available/0, %% FP16 Tensor Cores - 330 TFLOPS!
    nt_matmul_int8_lt/5, nt_int8_lt_available/0, %% cublasLt IMMA Tensor Cores - 660 TFLOPS!
    nt_transpose/1,
    nt_relu/1, nt_sigmoid/1, nt_exp/1, nt_log/1,
    %% In-place mutation (zero allocation)
    nt_add_mut/2, nt_scale_mut/2, nt_negate_mut/1, nt_relu_mut/1,
    %% Retro / fused kernels
    nt_saturn_blend/3, nt_fused_linear_relu/6,
    %% Resonance kernels (Log-Number System) - f64
    nt_resonance_mul/2, nt_resonance_power/2
]).

%% LNS (True Log-Number System) - f32 via IADD, 8x throughput
-export([
    lns_from_f64/1, lns_to_f64/1,
    lns_mul/2, lns_mul_corrected/2, lns_div/2,
    lns_sqrt/1, lns_rsqrt/1
]).

%% Horde - SoA Physics, 10K+ entities at 60fps
-export([
    horde_create/2,
    horde_set_positions/2, horde_set_velocities/2,
    horde_integrate/2, horde_dampen/2, horde_wrap/2,
    horde_get_positions/1, horde_get_velocities/1,
    horde_count/1, horde_kinetic_energy/1
]).

%% HDC - Hyperdimensional Computing, one-shot learning
-export([
    hdc_create/1, hdc_random/2,
    hdc_bind/2, hdc_similarity/2, hdc_permute/2, hdc_dim/1
]).

%% CudaTensor - Persistent GPU memory (Linux only, RTX 4090: 40+ TFLOPS!)
-export([
    ct_from_list/2, ct_to_list/1, ct_shape/1, ct_matmul/5,
    ct_matmul_inplace/6   %% Zero-allocation FP32 GPU matmul
]).

%% CudaTensor16 - FP16 Tensor Cores with ZERO-COPY (Linux only, 330 TFLOPS!)
-export([
    ct16_from_list/2, ct16_to_list/1, ct16_shape/1, ct16_matmul/5, ct16_available/0,
    ct16_matmul_inplace/6   %% Pure FP16→FP16 zero-allocation (max throughput!)
]).

%% Async CUDA - No sync overhead for pipeline benchmarks (100+ TFLOPS!)
-export([
    cuda_sync/0,          %% Explicit GPU sync - call when you need results
    ct_matmul_async/5,    %% FP32 SGEMM async (no sync)
    ct16_matmul_async/5   %% FP16 Tensor Core async (no sync) - 100+ TFLOPS sustained!
]).

%% CudaInt8Tensor - INT8 Tensor Cores with ZERO-COPY (Linux only, 660 TFLOPS!)
%% Data stays on GPU as INT8, uses IMMA Tensor Cores for compute.
-export([
    ct_int8_available/0,      %% Check if INT8 Tensor Cores available
    ct_int8_from_list/2,      %% Upload list to GPU as INT8 (quantize once!)
    ct_int8_to_list/1,        %% Download INT32 accumulator from GPU
    ct_int8_shape/1,          %% Get tensor shape
    ct_int8_matmul/5,         %% INT8 GEMM with Tensor Cores (sync)
    ct_int8_matmul_inplace/6, %% INT8 zero-allocation matmul
    ct_int8_matmul_async/5,   %% INT8 GEMM async (NO sync!) - 300+ TFLOPS target!
    ct16_matmul_bench/7,      %% FP16 batch bench (loop in C, zero Erlang overhead)
    ct_int8_matmul_bench/7,   %% INT8 batch bench (loop in C, zero Erlang overhead)
    ct16_matmul_lt_32f_bench/7,     %% cublasLt FP16 COMPUTE_32F_FAST_16F baseline bench
    ct16_matmul_fused_relu/6,       %% FP16 GEMM+ReLU fused (activation FREE!)
    ct16_matmul_fused_gelu/6,       %% FP16 GEMM+GELU fused (activation FREE!)
    ct16_matmul_fused_relu_bench/7, %% FP16 GEMM+ReLU bench loop in C
    ct16_matmul_fused_gelu_bench/7, %% FP16 GEMM+GELU bench loop in C
    ct16_matmul_fused_relu_tn_bench/7, %% FP16 GEMM+ReLU TN layout bench
    ct16_matmul_fused_gelu_tn_bench/7, %% FP16 GEMM+GELU TN layout bench
    ct16_matmul_batched_bench/5,       %% FP16 batched GEMM bench (M,N,K,Batch,Iters)
    fp8_matmul_lt_tn_bench/4,          %% FP8 E4M3 cublasLt TN bench (M,N,K,Iters) -> 330 TOPS
    cutlass_fp8_f16acc_bench/4,        %% CUTLASS FP8 + FP16 accum bench -> 660 TOPS!
    cutlass_fp8_f32acc_bench/4,        %% CUTLASS FP8 + FP32 accum bench -> 330 TOPS
    cutlass_int8_sparse_bench/5,       %% CUTLASS INT8 2:4 sparse bench (M,N,K,Iters,Config) -> 1320 TOPS!
    cutlass_int8_sparse_bench_ex/6,    %% Extended: (M,N,K,Iters,Config,SplitK) with split-K
    cusparselt_int8_sparse_bench/5,    %% cuSPARSELt INT8 2:4 sparse (M,N,K,Iters,Mode)
    cusparselt_fp8_sparse_bench/4,    %% cuSPARSELt FP8 E4M3 2:4 sparse (M,N,K,Iters)
    cusparselt_fp16_sparse_bench/4,   %% cuSPARSELt FP16 2:4 sparse (M,N,K,Iters)
    cutlass_int4_sparse_bench/6      %% CUTLASS INT4 2:4 sparse (M,N,K,Iters,Config,SplitK)
]).

%% SparseTensor - 2:4 Sparsity with cuSPARSELt (Linux only, 660+ TFLOPS!)
%% Creates 2:4 sparse matrix (keeps 2 largest per 4 elements) for Tensor Core acceleration.
-export([
    sparse_from_ct16/1,         %% CudaTensor16 -> SparseTensor (prune + compress)
    sparse_shape/1,             %% SparseTensor -> [Rows, Cols]
    sparse_compression_ratio/1, %% SparseTensor -> float() (~2.0x)
    sparse_matmul/5,            %% SparseTensor @ CudaTensor16 -> CudaTensor16 (660 TFLOPS!)
    sparse_matmul_inplace/6,    %% In-place sparse GEMM (zero alloc, async)
    sparse_matmul_bench/7,      %% C-loop sparse GEMM bench (zero overhead)
    sparse_matmul_bench_tn/7,   %% C-loop TN layout bench (B transposed)
    sparse_available/0,         %% Is cuSPARSELt available?
    %% INT8 Sparse — 1320 TOPS (2x of 660T dense INT8!)
    sparse_from_int8/1,         %% CudaInt8Tensor -> SparseTensor (prune + compress)
    sparse_matmul_int8/5,       %% SparseTensor @ CudaInt8Tensor -> CudaInt8Tensor (1320 TOPS!)
    sparse_matmul_int8_inplace/6, %% In-place sparse INT8 GEMM (zero alloc, async)
    sparse_matmul_int8_bench/7  %% C-loop sparse INT8 GEMM bench (zero overhead)
]).

%% SageAttention - INT8 QK^T + FP8 (2-5x faster than FlashAttention!)
%% From: https://github.com/thu-ml/SageAttention (Apache 2.0)
-export([
    sage_available/0,       %% Is SageAttention CUDA backend available?
    sage_fp8_available/0,   %% Is FP8 (E4M3/E5M2) available? (Ada/Hopper only)
    sage_quant_int8/2,      %% Tensor + BlockSize -> {QuantTensor, Scales}
    sage_softmax/2,         %% Tensor + Dim -> Softmax result
    sage_attention/8,       %% Q, K, V, Batch, Heads, SeqQ, SeqK, HeadDim -> Output (CPU/MKL)
    sage_attention_ct/8     %% Q, K, V, Batch, Heads, SeqQ, SeqK, HeadDim -> Output (GPU/cuBLAS)
]).

%% Fused Quantized Matmul - Zero overhead dequantization!
%% 4x-8x memory compression with ZERO runtime overhead.
-export([
    nt_matmul_int8/6,    %% A @ (B_quant * scale) - 4x compression
    nt_matmul_nf4/7,     %% A @ dequant_nf4(B) - 8x compression
    nt_quantize_int8/1   %% Tensor -> {QuantList, Scale}
]).

%% Resource-Based Quantized Tensors - TRULY ZERO OVERHEAD!
%% Quantize ONCE, matmul MANY times without list conversion.
%% Expected: 600+ GFLOPS (same as dense baseline!)
-export([
    nt_to_qint8/1,       %% NativeTensor -> QuantInt8Tensor
    nt_matmul_qint8/5,   %% A @ B_qint8 = C (zero overhead!)
    nt_to_qnf4/2,        %% NativeTensor -> QuantNF4Tensor (with BlockSize)
    nt_matmul_qnf4/5,    %% A @ B_qnf4 = C (zero overhead!)
    qint8_scale/1,       %% Get INT8 scale factor
    qint8_shape/1,       %% Get INT8 tensor shape
    qnf4_info/1          %% Get NF4 info map
]).

-on_load(init/0).

%% NIF Loading
init() ->
    %% Try multiple locations for the NIF
    PrivDir = case code:priv_dir(viva_tensor) of
        {error, bad_name} ->
            case file:get_cwd() of
                {ok, Cwd} -> filename:join(Cwd, "priv");
                _ -> "priv"
            end;
        Dir -> Dir
    end,
    %% On Windows, try priv/ in project root first (for escript), then Gleam build dir
    NifPath = case os:type() of
        {win32, _} ->
            {ok, WinCwd} = file:get_cwd(),
            LocalPriv = filename:join(WinCwd, "priv"),
            case filelib:is_file(filename:join(LocalPriv, "viva_tensor_zig.dll")) of
                true -> filename:join(LocalPriv, "viva_tensor_zig");
                false -> filename:join(PrivDir, "viva_tensor_zig")
            end;
        _ ->
            filename:join(PrivDir, "viva_tensor_zig")
    end,
    io:format("[NIF] Loading from: ~s~n", [NifPath]),
    case erlang:load_nif(NifPath, 0) of
        ok ->
            io:format("[NIF] Loaded successfully!~n"),
            persistent_term:put(viva_tensor_zig_loaded, true),
            ok;
        {error, {load_failed, Msg}} ->
            io:format("[NIF] Load failed: ~s~n", [Msg]),
            persistent_term:put(viva_tensor_zig_loaded, false),
            ok;
        {error, {reload, _}} ->
            ok;
        {error, Reason} ->
            io:format("[NIF] Error: ~p~n", [Reason]),
            persistent_term:put(viva_tensor_zig_loaded, false),
            ok
    end.

%% Check if NIF is loaded
is_loaded() ->
    try persistent_term:get(viva_tensor_zig_loaded)
    catch error:badarg -> false
    end.

%% Backend info
backend_info() ->
    case is_loaded() of
        true -> nif_backend_info();
        false -> <<"Pure Erlang fallback">>
    end.

%% SIMD availability check
simd_available() ->
    case is_loaded() of
        true -> nif_simd_available();
        false -> false
    end.

%% ==========================================================================
%% SIMD Dot Product
%% ==========================================================================

simd_dot(A, B) ->
    case is_loaded() of
        true -> nif_simd_dot(A, B);
        false -> fallback_dot(A, B)
    end.

fallback_dot(A, B) ->
    {ok, lists:foldl(
        fun({X, Y}, Acc) -> Acc + X * Y end,
        0.0,
        lists:zip(A, B)
    )}.

%% ==========================================================================
%% SIMD Sum
%% ==========================================================================

simd_sum(Data) ->
    case is_loaded() of
        true -> nif_simd_sum(Data);
        false -> {ok, lists:sum(Data)}
    end.

%% ==========================================================================
%% SIMD Scale
%% ==========================================================================

simd_scale(Data, Scalar) ->
    case is_loaded() of
        true -> nif_simd_scale(Data, Scalar);
        false -> {ok, [X * Scalar || X <- Data]}
    end.

%% ==========================================================================
%% SIMD Add
%% ==========================================================================

simd_add(A, B) ->
    case is_loaded() of
        true -> nif_simd_add(A, B);
        false -> {ok, [X + Y || {X, Y} <- lists:zip(A, B)]}
    end.

%% ==========================================================================
%% SIMD Element-wise Multiply
%% ==========================================================================

simd_mul(A, B) ->
    case is_loaded() of
        true ->
            try nif_simd_mul(A, B)
            catch error:nif_not_loaded -> {ok, [X * Y || {X, Y} <- lists:zip(A, B)]}
            end;
        false -> {ok, [X * Y || {X, Y} <- lists:zip(A, B)]}
    end.

%% ==========================================================================
%% SIMD Matrix Multiplication
%% ==========================================================================

simd_matmul(A, B, M, N, K) ->
    case is_loaded() of
        true -> nif_simd_matmul(A, B, M, N, K);
        false -> fallback_matmul(A, B, M, N, K)
    end.

fallback_matmul(AList, BList, M, N, K) ->
    A = array:from_list(AList),
    B = array:from_list(BList),
    Result = [
        begin
            RowStart = I * K,
            lists:foldl(fun(KIdx, Acc) ->
                AVal = array:get(RowStart + KIdx, A),
                BVal = array:get(KIdx * N + J, B),
                Acc + AVal * BVal
            end, 0.0, lists:seq(0, K - 1))
        end
        || I <- lists:seq(0, M - 1),
           J <- lists:seq(0, N - 1)
    ],
    {ok, Result}.

%% ==========================================================================
%% NIF Stubs
%% ==========================================================================

nif_simd_dot(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_sum(_Data) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_scale(_Data, _Scalar) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_add(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_mul(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_matmul(_A, _B, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_available() ->
    erlang:nif_error(nif_not_loaded).

nif_backend_info() ->
    erlang:nif_error(nif_not_loaded).

cpu_topology() ->
    erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% NIF Resource API Stubs (replaced by C NIFs on load)
%% ==========================================================================

%% Constructors
nt_zeros(_Shape) -> erlang:nif_error(nif_not_loaded).
nt_ones(_Shape) -> erlang:nif_error(nif_not_loaded).
nt_fill(_Shape, _Value) -> erlang:nif_error(nif_not_loaded).
nt_from_list(_Data, _Shape) -> erlang:nif_error(nif_not_loaded).

%% Accessors
nt_to_list(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_shape(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_size(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Element-wise ops
nt_add(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_sub(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_scale(_Ref, _Scalar) -> erlang:nif_error(nif_not_loaded).
nt_negate(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Reductions
nt_dot(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_sum(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_max(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_min(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Matrix ops
nt_matmul(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_matmul_inplace(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_matmul_blas(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_matmul_cuda(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_matmul_cuda_fp32(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_matmul_int8_tc(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_int8_tc_available() -> erlang:nif_error(nif_not_loaded).
nt_matmul_fp16_tc(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_fp16_tc_available() -> erlang:nif_error(nif_not_loaded).
nt_matmul_int8_lt(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_int8_lt_available() -> erlang:nif_error(nif_not_loaded).
nt_transpose(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Activations
nt_relu(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_sigmoid(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_exp(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_log(_Ref) -> erlang:nif_error(nif_not_loaded).

%% In-place mutation (zero allocation)
nt_add_mut(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_scale_mut(_Ref, _Scalar) -> erlang:nif_error(nif_not_loaded).
nt_negate_mut(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_relu_mut(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Retro / fused kernels
nt_saturn_blend(_Texture, _Shade, _Bias) -> erlang:nif_error(nif_not_loaded).
nt_fused_linear_relu(_A, _B, _Bias, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).

%% Resonance kernels (Log-Number System) - f64
nt_resonance_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_resonance_power(_Ref, _Exponent) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% LNS (True Log-Number System) - f32 via IADD, 8x throughput
%% ==========================================================================

lns_from_f64(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_to_f64(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_mul_corrected(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_div(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_sqrt(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_rsqrt(_Ref) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Horde - SoA Physics, 10K+ entities at 60fps
%% ==========================================================================

horde_create(_EntityCount, _Dims) -> erlang:nif_error(nif_not_loaded).
horde_set_positions(_Horde, _Data) -> erlang:nif_error(nif_not_loaded).
horde_set_velocities(_Horde, _Data) -> erlang:nif_error(nif_not_loaded).
horde_integrate(_Horde, _Dt) -> erlang:nif_error(nif_not_loaded).
horde_dampen(_Horde, _Friction) -> erlang:nif_error(nif_not_loaded).
horde_wrap(_Horde, _MaxBound) -> erlang:nif_error(nif_not_loaded).
horde_get_positions(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_get_velocities(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_count(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_kinetic_energy(_Horde) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% HDC - Hyperdimensional Computing, one-shot learning
%% ==========================================================================

hdc_create(_Dim) -> erlang:nif_error(nif_not_loaded).
hdc_random(_Dim, _Seed) -> erlang:nif_error(nif_not_loaded).
hdc_bind(_A, _B) -> erlang:nif_error(nif_not_loaded).
hdc_similarity(_A, _B) -> erlang:nif_error(nif_not_loaded).
hdc_permute(_Vec, _Shift) -> erlang:nif_error(nif_not_loaded).
hdc_dim(_Vec) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% CudaTensor - Persistent GPU Memory (Linux only, RTX 4090: 40+ TFLOPS!)
%% Upload ONCE, compute MANY times without PCIe transfer overhead.
%% ==========================================================================

ct_from_list(_Data, _Shape) -> erlang:nif_error(nif_not_loaded).
ct_to_list(_Ref) -> erlang:nif_error(nif_not_loaded).
ct_shape(_Ref) -> erlang:nif_error(nif_not_loaded).
ct_matmul(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct_matmul_inplace(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% CudaTensor16 - FP16 Tensor Cores with ZERO-COPY (330 TFLOPS!)
%% Data stays on GPU as FP16, uses Tensor Cores for compute.
%% ==========================================================================

ct16_from_list(_Data, _Shape) -> erlang:nif_error(nif_not_loaded).
ct16_to_list(_Ref) -> erlang:nif_error(nif_not_loaded).
ct16_shape(_Ref) -> erlang:nif_error(nif_not_loaded).
ct16_matmul(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_inplace(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_available() -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Async CUDA - No sync overhead for pipeline benchmarks (100+ TFLOPS!)
%% ==========================================================================
cuda_sync() -> erlang:nif_error(nif_not_loaded).
ct_matmul_async(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_async(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% CudaInt8Tensor - INT8 Tensor Cores with ZERO-COPY (660 TFLOPS!)
%% Data stays on GPU as INT8, uses IMMA Tensor Cores for compute.
%% ==========================================================================
ct_int8_available() -> erlang:nif_error(nif_not_loaded).
ct_int8_from_list(_Data, _Shape) -> erlang:nif_error(nif_not_loaded).
ct_int8_to_list(_Ref) -> erlang:nif_error(nif_not_loaded).
ct_int8_shape(_Ref) -> erlang:nif_error(nif_not_loaded).
ct_int8_matmul(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct_int8_matmul_inplace(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct_int8_matmul_async(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct_int8_matmul_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Fused GEMM+Activation - cublasLt epilogues (ReLU/GELU FREE!)
%% Activation fused into GEMM kernel: same TFLOPS as plain GEMM.
%% ==========================================================================
ct16_matmul_lt_32f_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_relu(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_gelu(_A, _B, _C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_relu_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_gelu_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_relu_tn_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_fused_gelu_tn_bench(_A, _B, _C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
ct16_matmul_batched_bench(_M, _N, _K, _BatchCount, _Iters) -> erlang:nif_error(nif_not_loaded).
fp8_matmul_lt_tn_bench(_M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
cutlass_fp8_f16acc_bench(_M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
cutlass_fp8_f32acc_bench(_M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
cutlass_int8_sparse_bench(_M, _N, _K, _Iters, _Config) -> erlang:nif_error(nif_not_loaded).
cutlass_int8_sparse_bench_ex(_M, _N, _K, _Iters, _Config, _SplitK) -> erlang:nif_error(nif_not_loaded).
cusparselt_int8_sparse_bench(_M, _N, _K, _Iters, _Mode) -> erlang:nif_error(nif_not_loaded).
cusparselt_fp8_sparse_bench(_M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
cusparselt_fp16_sparse_bench(_M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
cutlass_int4_sparse_bench(_M, _N, _K, _Iters, _Config, _SplitK) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% SparseTensor - 2:4 Sparsity with cuSPARSELt (660+ TFLOPS!)
%% Prune dense matrix to 2:4 pattern, compress to ~50%, use Tensor Cores.
%% ==========================================================================

sparse_from_ct16(_CudaTensor16Ref) -> erlang:nif_error(nif_not_loaded).
sparse_shape(_SparseTensorRef) -> erlang:nif_error(nif_not_loaded).
sparse_compression_ratio(_SparseTensorRef) -> erlang:nif_error(nif_not_loaded).
sparse_matmul(_SparseTensor, _CudaTensor16B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_inplace(_SparseTensor, _CT16B, _CT16C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_bench(_SparseTensor, _CT16B, _CT16C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_bench_tn(_SparseTensor, _CT16B, _CT16C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).
sparse_available() -> erlang:nif_error(nif_not_loaded).

%% INT8 Sparse — 1320 TOPS (2x of 660T dense INT8!)
sparse_from_int8(_CudaInt8TensorRef) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_int8(_SparseTensor, _CudaInt8TensorB, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_int8_inplace(_SparseTensor, _Int8B, _Int8C, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
sparse_matmul_int8_bench(_SparseTensor, _Int8B, _Int8C, _M, _N, _K, _Iters) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% SageAttention - INT8 QK^T + FP8 (2-5x faster than FlashAttention!)
%% From: https://github.com/thu-ml/SageAttention (Apache 2.0)
%%
%% Uses INT8 Tensor Cores for QK^T (660 TFLOPS on RTX 4090)
%% Supports FP8 (E4M3/E5M2) for PV on Ada/Hopper GPUs.
%% ==========================================================================

%% Check if SageAttention CUDA backend is available
sage_available() -> erlang:nif_error(nif_not_loaded).

%% Check if FP8 (E4M3/E5M2) is available (requires Ada/Hopper GPU)
sage_fp8_available() -> erlang:nif_error(nif_not_loaded).

%% Quantize tensor to INT8 with per-block scaling
%% Returns: {ok, {QuantTensor, ScalesTensor}}
sage_quant_int8(_Tensor, _BlockSize) -> erlang:nif_error(nif_not_loaded).

%% Numerically stable softmax over last dimension
%% Returns: {ok, OutputTensor}
sage_softmax(_Tensor, _Dim) -> erlang:nif_error(nif_not_loaded).

%% SageAttention: INT8 QK^T + FP32 softmax + FP32 PV
%% Q, K, V: [Batch, Heads, Seq, HeadDim] tensors
%% Returns: {ok, OutputTensor}
sage_attention(_Q, _K, _V, _Batch, _Heads, _SeqQ, _SeqK, _HeadDim) ->
    erlang:nif_error(nif_not_loaded).

%% sage_attention_ct - GPU accelerated via cuBLAS!
%% Uses CudaTensors (data already on GPU) - ZERO PCIe overhead per call.
%% Expected: 10-20x faster than CPU version.
sage_attention_ct(_Q, _K, _V, _Batch, _Heads, _SeqQ, _SeqK, _HeadDim) ->
    erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Fused Quantized Matmul - Zero overhead dequantization!
%% Dequant happens on-the-fly during compute = no memory overhead.
%% ==========================================================================

%% INT8 Fused Matmul: C = A @ (B_quant * scale)
%% A: NativeTensor [M x K], B_quant: list(integer) [K x N], Scale: float
%% 4x memory compression with ZERO runtime overhead!
nt_matmul_int8(_A, _BQuantList, _BScale, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

%% NF4 Fused Matmul: C = A @ dequant_nf4(B_indices, B_scales)
%% A: NativeTensor [M x K], B_indices: list(byte) (packed nibbles),
%% B_scales: list(float) (one per block per column), BlockSize: int
%% 8x memory compression with ~0.1% error for Gaussian weights!
nt_matmul_nf4(_A, _BIndicesList, _BScalesList, _M, _N, _K, _BlockSize) ->
    erlang:nif_error(nif_not_loaded).

%% Quantize Tensor to INT8: returns {QuantList, Scale}
nt_quantize_int8(_Tensor) ->
    erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Resource-Based Quantized Tensors - TRULY ZERO OVERHEAD!
%% Quantize ONCE to a native resource, matmul MANY times without conversion.
%% This is 100x faster than list-based NIFs for large tensors!
%% ==========================================================================

%% Convert NativeTensor to QuantInt8Tensor resource (4x compression)
nt_to_qint8(_Tensor) ->
    erlang:nif_error(nif_not_loaded).

%% A @ B_qint8 = C with ZERO list conversion overhead!
%% Expected: 600+ GFLOPS (same as dense MKL baseline)
nt_matmul_qint8(_A, _BQint8, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

%% Convert NativeTensor to QuantNF4Tensor resource (8x compression)
%% BlockSize: typically 64 for optimal compression
nt_to_qnf4(_Tensor, _BlockSize) ->
    erlang:nif_error(nif_not_loaded).

%% A @ B_qnf4 = C with ZERO list conversion overhead!
%% 8x compression with ~0.1% error for Gaussian weights
nt_matmul_qnf4(_A, _BQnf4, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

%% Get the scale factor of a QuantInt8Tensor
qint8_scale(_QInt8Tensor) ->
    erlang:nif_error(nif_not_loaded).

%% Get the shape [Rows, Cols] of a QuantInt8Tensor
qint8_shape(_QInt8Tensor) ->
    erlang:nif_error(nif_not_loaded).

%% Get info about a QuantNF4Tensor: #{block_size, num_blocks, compression}
qnf4_info(_QNf4Tensor) ->
    erlang:nif_error(nif_not_loaded).
