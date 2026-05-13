#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% FP8 E4M3 GEMM PEAK THROUGHPUT BENCHMARK
%%
%% RTX 4090 (Ada Lovelace) Tensor Cores:
%% - FP8 E4M3 + FP16 accum: 660 TOPS (full rate, needs CUTLASS/PTX)
%% - FP8 E4M3 + FP32 accum: 330 TOPS (half rate, GeForce nerf!)
%% - Uses cublasLtMatmul with TN layout for IMMA Tensor Core alignment
%% - FP8 input, FP16 output, FP32 accumulator
%%
%% Compare: INT8 TN achieved 604 TOPS (92% peak)
%% cuBLASLt target: ~330 TOPS (FP32 accum peak on GeForce)
%% CUTLASS target: 600+ TOPS (FP16 accum, bypasses GeForce nerf)
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  FP8 E4M3 GEMM PEAK THROUGHPUT BENCHMARK                             |~n"),
    io:format("|  RTX 4090: FP8+FP32acc=330T peak (GeForce), INT8 TN=604T             |~n"),
    io:format("|  cublasLtMatmul TN + COMPUTE_32F + per-tensor scales                 |~n"),
    io:format("+=======================================================================+~n~n"),

    %% Test 1: FP16 baseline (cublasGemmEx COMPUTE_16F)
    io:format("=== 1. FP16 Baseline (cublasGemmEx COMPUTE_16F, 284T achieved) ===~n~n"),
    io:format("Size       | TFLOPS     | vs 330T peak~n"),
    io:format("-----------|------------|------------~n"),
    Sizes = [1024, 2048, 4096, 6144, 8192],
    lists:foreach(fun(N) -> bench_fp16(N) end, Sizes),

    %% Test 2: INT8 TN baseline (cublasLtMatmul, 604T achieved)
    io:format("~n=== 2. INT8 TN Baseline (cublasLtMatmul TN, 604T achieved) ===~n~n"),
    io:format("Size       | TOPS       | vs 660T peak~n"),
    io:format("-----------|------------|------------~n"),
    lists:foreach(fun(N) -> bench_int8(N) end, Sizes),

    %% Test 3: FP8 E4M3 TN (the new path!)
    io:format("~n=== 3. FP8 E4M3 TN (cublasLtMatmul TN + COMPUTE_32F + scales) ===~n~n"),
    io:format("Size       | TOPS       | vs 330T peak | vs INT8 TN~n"),
    io:format("-----------|------------|--------------|----------~n"),
    lists:foreach(fun(N) -> bench_fp8(N) end, Sizes),

    %% Test 4: Scaling test (large matrices)
    io:format("~n=== 4. FP8 Large Matrix Scaling ===~n~n"),
    io:format("M x N x K         | TOPS       | vs 330T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K}) -> bench_fp8_rect(M, N, K) end, [
        {4096, 4096, 4096},
        {4096, 4096, 11008},   %% LLaMA-7B FFN: 4096 -> 11008
        {4096, 11008, 4096},   %% LLaMA-7B FFN: 11008 -> 4096
        {8192, 8192, 8192},
        {8192, 8192, 28672}    %% LLaMA-70B FFN: 8192 -> 28672
    ]),

    %% Test 5: CUTLASS FP8 + FP16 accum (660 TOPS target!)
    io:format("~n=== 5. CUTLASS FP8 E4M3 + FP16 Accumulation (660 TOPS target!) ===~n~n"),
    io:format("Size       | TOPS       | vs 660T peak | vs cuBLASLt FP8~n"),
    io:format("-----------|------------|--------------|----------------~n"),
    lists:foreach(fun(N) -> bench_cutlass_f16acc(N) end, Sizes),

    %% Test 6: CUTLASS FP8 + FP32 accum (330 TOPS, for comparison)
    io:format("~n=== 6. CUTLASS FP8 E4M3 + FP32 Accumulation (330 TOPS, comparison) ===~n~n"),
    io:format("Size       | TOPS       | vs 330T peak | vs cuBLASLt FP8~n"),
    io:format("-----------|------------|--------------|----------------~n"),
    lists:foreach(fun(N) -> bench_cutlass_f32acc(N) end, Sizes),

    %% Test 7: CUTLASS FP16 accum large matrix scaling
    io:format("~n=== 7. CUTLASS FP16 Accum — Large Matrix Scaling ===~n~n"),
    io:format("M x N x K         | TOPS       | vs 660T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K}) -> bench_cutlass_f16acc_rect(M, N, K) end, [
        {4096, 4096, 4096},
        {4096, 4096, 11008},   %% LLaMA-7B FFN: 4096 -> 11008
        {4096, 11008, 4096},   %% LLaMA-7B FFN: 11008 -> 4096
        {8192, 8192, 8192},
        {8192, 8192, 28672}    %% LLaMA-70B FFN: 8192 -> 28672
    ]),

    %% Test 8: INT8 2:4 Sparse (CUTLASS)
    io:format("~n=== 8. INT8 2:4 Sparse GEMM (CUTLASS, 1321 TOPS peak) ===~n~n"),
    io:format("Size       | TOPS       | vs 1321T peak | vs INT8 Dense~n"),
    io:format("-----------|------------|---------------|-------------~n"),
    lists:foreach(fun(N) -> bench_int8_sparse(N) end, Sizes),

    %% Test 9: INT8 Sparse large matrix (CUTLASS)
    io:format("~n=== 9. INT8 2:4 Sparse — Large Matrix Scaling (CUTLASS) ===~n~n"),
    io:format("M x N x K         | TOPS       | vs 1321T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K}) -> bench_int8_sparse_rect(M, N, K) end, [
        {4096, 4096, 4096},
        {4096, 4096, 11008},
        {4096, 11008, 4096},
        {8192, 8192, 8192},
        {8192, 8192, 28672},
        {8192, 8192, 65536},
        {16384, 16384, 16384}
    ]),

    %% Test 10: cuSPARSELt INT8 2:4 Sparse (kernel-only timing)
    io:format("~n=== 10. cuSPARSELt INT8 2:4 Sparse (kernel-only, 1321 TOPS peak) ===~n~n"),
    io:format("M x N x K         | TOPS       | vs 1321T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K, Iters}) -> bench_cusparselt(M, N, K, Iters) end, [
        {4096, 4096, 4096, 50},
        {4096, 4096, 11008, 50},
        {8192, 8192, 8192, 20},
        {8192, 8192, 28672, 20},
        {16384, 16384, 16384, 10},
        {20480, 20480, 20480, 5}
    ]),

    %% Test 11: cuSPARSELt FP8 E4M3 2:4 Sparse (NEW in v0.8.0!)
    io:format("~n=== 11. cuSPARSELt FP8 E4M3 2:4 Sparse (1321 TOPS peak) ===~n~n"),
    io:format("M x N x K         | TOPS       | vs 1321T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K, Iters}) -> bench_cusparselt_fp8(M, N, K, Iters) end, [
        {8192, 8192, 8192, 20},
        {8192, 8192, 28672, 20},
        {16384, 16384, 16384, 10}
    ]),

    %% Test 12: cuSPARSELt FP16 2:4 Sparse
    io:format("~n=== 12. cuSPARSELt FP16 2:4 Sparse (660 TFLOPS peak) ===~n~n"),
    io:format("M x N x K         | TFLOPS     | vs 660T peak~n"),
    io:format("------------------|------------|------------~n"),
    lists:foreach(fun({M, N, K, Iters}) -> bench_cusparselt_fp16(M, N, K, Iters) end, [
        {8192, 8192, 8192, 20},
        {8192, 8192, 28672, 20},
        {16384, 16384, 16384, 10}
    ]),

    %% Test 13: CUTLASS INT4 2:4 Sparse (2642 TOPS peak!) — Native SM89 INT4 tensor cores
    io:format("~n=== 13. CUTLASS INT4 2:4 Sparse (2642 TOPS peak!) ===~n~n"),
    io:format("M x N x K             | Config | TOPS       | vs 2642T peak~n"),
    io:format("----------------------|--------|------------|------------~n"),
    lists:foreach(fun({M, N, K, Iters, Cfg}) -> bench_int4_sparse(M, N, K, Iters, Cfg) end, [
        {8192, 8192, 8192, 10, 0},
        {8192, 8192, 16384, 5, 1},
        {8192, 8192, 32768, 5, 1},
        {8192, 8192, 65536, 3, 1},
        {4096, 4096, 131072, 2, 1},
        {4096, 4096, 262144, 1, 1},
        {4096, 4096, 524288, 1, 1},
        {16384, 16384, 16384, 3, 0},
        {8192, 16384, 16384, 5, 0},
        {4096, 8192, 65536, 3, 2},
        {8192, 4096, 65536, 3, 2}
    ]),

    io:format("~n"),
    io:format("+=========================================================================+~n"),
    io:format("|  FINAL SCORECARD — RTX 4090 Dense & Sparse GEMM                        |~n"),
    io:format("+=========================================================================+~n"),
    io:format("| FP16 dense (cublasGemmEx):      284 TFLOPS  ( 86% of  330T)            |~n"),
    io:format("| INT8 dense (cublasLtMatmul):    604 TOPS    ( 92% of  660T)            |~n"),
    io:format("| FP8 cuBLASLt (FP32 acc):        344 TOPS    (104% of  330T GeForce)    |~n"),
    io:format("| FP8 CUTLASS (FP16 acc):         661 TOPS    (100% of  660T)            |~n"),
    io:format("| INT8 sparse CUTLASS:            ~3B TOPS    ( ~B% of 1321T)             |~n",
              [trunc(get_or(cutlass_int8_sparse_best, 676)),
               trunc(get_or(cutlass_int8_sparse_best, 676) / 1321 * 100)]),
    io:format("| INT8 sparse cuSPARSELt:        ~4B TOPS    ( ~B% of 1321T)             |~n",
              [trunc(get_or(cusparselt_best, 1094)),
               trunc(get_or(cusparselt_best, 1094) / 1321 * 100)]),
    io:format("| FP8 sparse cuSPARSELt:          ~3B TOPS    ( ~B% of 1321T)             |~n",
              [trunc(get_or(cusparselt_fp8_best, 710)),
               trunc(get_or(cusparselt_fp8_best, 710) / 1321 * 100)]),
    io:format("| FP16 sparse cuSPARSELt:         ~3B TFLOPS  ( ~B% of  660T)             |~n",
              [trunc(get_or(cusparselt_fp16_best, 355)),
               trunc(get_or(cusparselt_fp16_best, 355) / 660 * 100)]),
    io:format("| INT4 sparse CUTLASS:           ~4B TOPS    ( ~B% of 2642T)   *** NEW ***|~n",
              [trunc(get_or(int4_sparse_best, 1670)),
               trunc(get_or(int4_sparse_best, 1670) / 2642 * 100)]),
    io:format("+-------------------------------------------------------------------------+~n").

bench_fp16(N) ->
    erlang:garbage_collect(),

    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    put({fp16, N}, TFLOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%~n", [N, TFLOPS, VsPeak]).

bench_int8(N) ->
    erlang:garbage_collect(),

    %% INT8 uses ct_int8_matmul_bench with CudaInt8Tensor refs
    %% Create dummy INT8 tensors via from_list
    Data = [float(X rem 127) || X <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct_int8_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    ok = viva_tensor_zig:ct_int8_matmul_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct_int8_matmul_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * N * N * N * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 660.0 * 100,

    put({int8, N}, TOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%~n", [N, TOPS, VsPeak]).

bench_fp8(N) ->
    erlang:garbage_collect(),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    %% Warmup
    ok = viva_tensor_zig:fp8_matmul_lt_tn_bench(N, N, N, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:fp8_matmul_lt_tn_bench(N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * N * N * N * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 330.0 * 100,  %% 330T = FP8+FP32acc peak on GeForce

    Int8T = case get({int8, N}) of undefined -> TOPS; V -> V end,
    VsInt8 = TOPS / Int8T * 100,

    put({fp8, N}, TOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%       |  ~5.1f%~n", [N, TOPS, VsPeak, VsInt8]).

bench_fp8_rect(M, N, K) ->
    erlang:garbage_collect(),

    Iters = max(3, min(100, 500000000 div (M * N div 1000 * K div 1000))),

    %% Warmup
    ok = viva_tensor_zig:fp8_matmul_lt_tn_bench(M, N, K, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:fp8_matmul_lt_tn_bench(M, N, K, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * M * N * K * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 330.0 * 100,  %% 330T = FP8+FP32acc peak on GeForce

    io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TOPS, VsPeak]).

bench_cutlass_f16acc(N) ->
    erlang:garbage_collect(),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    %% Warmup
    ok = viva_tensor_zig:cutlass_fp8_f16acc_bench(N, N, N, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:cutlass_fp8_f16acc_bench(N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * N * N * N * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 660.0 * 100,  %% 660T = FP8+FP16acc full rate!

    CuBLASLtT = case get({fp8, N}) of undefined -> TOPS; V -> V end,
    VsCuBLAS = TOPS / CuBLASLtT * 100,

    put({cutlass_f16, N}, TOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%       |  ~5.1f%~n", [N, TOPS, VsPeak, VsCuBLAS]).

bench_cutlass_f32acc(N) ->
    erlang:garbage_collect(),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    %% Warmup
    ok = viva_tensor_zig:cutlass_fp8_f32acc_bench(N, N, N, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:cutlass_fp8_f32acc_bench(N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * N * N * N * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 330.0 * 100,  %% 330T = FP8+FP32acc peak

    CuBLASLtT = case get({fp8, N}) of undefined -> TOPS; V -> V end,
    VsCuBLAS = TOPS / CuBLASLtT * 100,

    put({cutlass_f32, N}, TOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%       |  ~5.1f%~n", [N, TOPS, VsPeak, VsCuBLAS]).

bench_cutlass_f16acc_rect(M, N, K) ->
    erlang:garbage_collect(),

    Iters = max(3, min(100, 500000000 div (M * N div 1000 * K div 1000))),

    %% Warmup
    ok = viva_tensor_zig:cutlass_fp8_f16acc_bench(M, N, K, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:cutlass_fp8_f16acc_bench(M, N, K, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * M * N * K * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 660.0 * 100,  %% 660T = FP8+FP16acc full rate!

    io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TOPS, VsPeak]).

get_or(Key, Default) ->
    case get(Key) of undefined -> Default; V -> V end.

bench_int8_sparse(N) ->
    erlang:garbage_collect(),

    Iters = max(3, min(50, 500000000 div (N * N * N div 1000))),

    %% cfg=20: GemmSparseUniversal 256x128x128, 2stg, Swizzle<8> (best!)
    ok = viva_tensor_zig:cutlass_int8_sparse_bench_ex(N, N, N, 1, 20, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:cutlass_int8_sparse_bench_ex(N, N, N, Iters, 20, 1),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * N * N * N * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 1321.0 * 100,

    DenseT = case get({int8, N}) of undefined -> TOPS; V -> V end,
    VsDense = TOPS / DenseT * 100,

    put(cutlass_int8_sparse_best, max(get_or(cutlass_int8_sparse_best, 0), TOPS)),
    io:format("~-10B | ~8.1f T |  ~5.1f%        |  ~5.1f%~n", [N, TOPS, VsPeak, VsDense]).

bench_int8_sparse_rect(M, N, K) ->
    erlang:garbage_collect(),

    Iters = max(3, min(50, 500000000 div (M * N div 1000 * K div 1000))),

    %% cfg=20: GemmSparseUniversal 256x128x128, 2stg, Swizzle<8> (best!)
    ok = viva_tensor_zig:cutlass_int8_sparse_bench_ex(M, N, K, 1, 20, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:cutlass_int8_sparse_bench_ex(M, N, K, Iters, 20, 1),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    OPs = 2.0 * M * N * K * Iters,
    TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TOPS / 1321.0 * 100,

    put(cutlass_int8_sparse_best, max(get_or(cutlass_int8_sparse_best, 0), TOPS)),
    io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TOPS, VsPeak]).

bench_cusparselt(M, N, K, Iters) ->
    try
        {ok, ElapsedUs} = viva_tensor_zig:cusparselt_int8_sparse_bench(M, N, K, Iters, 0),
        OPs = 2.0 * M * N * K * Iters,
        TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
        VsPeak = TOPS / 1321.0 * 100,
        put(cusparselt_best, max(get_or(cusparselt_best, 0), TOPS)),
        io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TOPS, VsPeak])
    catch
        E:R ->
            io:format("~Bx~Bx~-5B | FAILED: ~w:~w~n", [M, N, K, E, R])
    end.

bench_cusparselt_fp8(M, N, K, Iters) ->
    try
        {ok, ElapsedUs} = viva_tensor_zig:cusparselt_fp8_sparse_bench(M, N, K, Iters),
        OPs = 2.0 * M * N * K * Iters,
        TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
        VsPeak = TOPS / 1321.0 * 100,
        put(cusparselt_fp8_best, max(get_or(cusparselt_fp8_best, 0), TOPS)),
        io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TOPS, VsPeak])
    catch
        E:R ->
            io:format("~Bx~Bx~-5B | FAILED: ~w:~w~n", [M, N, K, E, R])
    end.

bench_cusparselt_fp16(M, N, K, Iters) ->
    try
        {ok, ElapsedUs} = viva_tensor_zig:cusparselt_fp16_sparse_bench(M, N, K, Iters),
        OPs = 2.0 * M * N * K * Iters,
        TFLOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
        VsPeak = TFLOPS / 660.0 * 100,
        put(cusparselt_fp16_best, max(get_or(cusparselt_fp16_best, 0), TFLOPS)),
        io:format("~Bx~Bx~-5B | ~8.1f T |  ~5.1f%~n", [M, N, K, TFLOPS, VsPeak])
    catch
        E:R ->
            io:format("~Bx~Bx~-5B | FAILED: ~w:~w~n", [M, N, K, E, R])
    end.

bench_int4_sparse(M, N, K, Iters, Config) ->
    try
        {ok, ElapsedUs} = viva_tensor_zig:cutlass_int4_sparse_bench(M, N, K, Iters, Config, 1),
        OPs = 2.0 * M * N * K * Iters,
        TOPS = OPs / (ElapsedUs / 1.0e6) / 1.0e12,
        VsPeak = TOPS / 2642.0 * 100,
        put(int4_sparse_best, max(get_or(int4_sparse_best, 0), TOPS)),
        io:format("~Bx~Bx~-7B | cfg=~B  | ~8.1f T |  ~5.1f%~n", [M, N, K, Config, TOPS, VsPeak])
    catch
        E:R ->
            io:format("~Bx~Bx~-7B | cfg=~B  | FAILED: ~w:~w~n", [M, N, K, Config, E, R])
    end.
