#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% GPU PEAK THROUGHPUT BENCHMARK
%%
%% Loop runs entirely in C (zero Erlang overhead).
%% Measures raw GPU kernel throughput from BEAM.
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  GPU PEAK THROUGHPUT - C-LOOP BENCHMARK (zero Erlang overhead)        |~n"),
    io:format("|  RTX 4090: FP16=330T, INT8=660T theoretical                           |~n"),
    io:format("+=======================================================================+~n~n"),

    Sizes = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],

    %% FP16 Benchmark
    io:format("=== FP16 Tensor Core (cublasLtMatmul TN + heuristic) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs PyTorch~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_fp16(N) end, Sizes),
    io:format("~n"),

    %% INT8 Benchmark
    io:format("=== INT8 IMMA Tensor Core (cublasLtMatmul TN + heuristic) ===~n~n"),
    io:format("Size       | TOPS       | Efficiency | vs PyTorch~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_int8(N) end, Sizes),

    io:format("~n"),
    io:format("PyTorch: FP16=312T, INT8=160T | Peak: FP16=330T, INT8=660T~n").

bench_fp16(N) ->
    erlang:garbage_collect(),
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    CData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(CData, [N, N]),

    %% Warmup
    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    Efficiency = TFLOPS / 330.0 * 100,
    VsPyTorch = TFLOPS / 312.0,

    io:format("~-10B | ~8.1f T |  ~6.1f%  | ~5.2fx~n",
              [N, TFLOPS, Efficiency, VsPyTorch]).

bench_int8(N) ->
    erlang:garbage_collect(),
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    CData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct_int8_from_list(CData, [N, N]),

    %% Warmup
    ok = viva_tensor_zig:ct_int8_matmul_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct_int8_matmul_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    Efficiency = TOPS / 660.0 * 100,
    VsPyTorch = TOPS / 160.0,

    io:format("~-10B | ~8.1f T |  ~6.1f%  | ~5.2fx~n",
              [N, TOPS, Efficiency, VsPyTorch]).
