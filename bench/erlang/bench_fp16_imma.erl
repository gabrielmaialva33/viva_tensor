#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% FP16 TENSOR CORE - SUSTAINED THROUGHPUT BENCHMARK
%%
%% Uses ct16_matmul_inplace for ZERO allocation in the hot loop.
%% Pure FP16 compute (CUBLAS_COMPUTE_16F) with cublasLtMatmul TN + heuristic.
%%
%% RTX 4090: 330 TFLOPS theoretical FP16
%% Previous best: 276 TFLOPS with cublasGemmEx
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  FP16 TENSOR CORE - SUSTAINED THROUGHPUT BENCHMARK                    |~n"),
    io:format("|  RTX 4090: 330 TFLOPS theoretical | Previous: 276 TFLOPS              |~n"),
    io:format("|  Using ct16_matmul_inplace (zero alloc, TN cublasLtMatmul)            |~n"),
    io:format("+=======================================================================+~n~n"),

    case viva_tensor_zig:ct16_available() of
        true -> run_benchmarks();
        _ ->
            io:format("[FAIL] FP16 not available~n"),
            halt(1)
    end.

run_benchmarks() ->
    Sizes = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],

    io:format("=== Inplace Matmul (zero-alloc, TN cublasLtMatmul) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs PyTorch~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_inplace(N) end, Sizes),

    io:format("~n"),
    io:format("NOTE: PyTorch FP16 on RTX 4090 = 312 TFLOPS~n"),
    io:format("      Theoretical peak = 330 TFLOPS~n"),
    io:format("      Previous viva_tensor = 276 TFLOPS~n").

bench_inplace(N) ->
    erlang:garbage_collect(),

    %% Create FP16 tensors on GPU
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),

    %% Pre-allocate output tensor C (FP16)
    CData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(CData, [N, N]),

    %% Warmup: 3 runs to trigger heuristic caching
    ok = viva_tensor_zig:ct16_matmul_inplace(A, B, C, N, N, N),
    ok = viva_tensor_zig:ct16_matmul_inplace(A, B, C, N, N, N),
    ok = viva_tensor_zig:ct16_matmul_inplace(A, B, C, N, N, N),
    viva_tensor_zig:cuda_sync(),

    %% Determine iterations based on size
    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    %% Benchmark: inplace matmul (no allocation!) + sync once at end
    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    bench_inplace_loop(A, B, C, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    Efficiency = TFLOPS / 330.0 * 100,
    VsPyTorch = TFLOPS / 312.0,

    io:format("~-10B | ~8.1f T |  ~6.1f%  | ~5.2fx~n",
              [N, TFLOPS, Efficiency, VsPyTorch]),
    ok.

bench_inplace_loop(_A, _B, _C, _N, 0) -> ok;
bench_inplace_loop(A, B, C, N, Remaining) ->
    ok = viva_tensor_zig:ct16_matmul_inplace(A, B, C, N, N, N),
    bench_inplace_loop(A, B, C, N, Remaining - 1).
