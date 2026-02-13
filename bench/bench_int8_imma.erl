#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% INT8 IMMA TENSOR CORE - SUSTAINED THROUGHPUT BENCHMARK
%%
%% Uses ct_int8_matmul_inplace for ZERO allocation in the hot loop.
%% This measures true GPU Tensor Core throughput without Erlang GC overhead.
%%
%% RTX 4090: 660 TFLOPS theoretical INT8
%% Standalone C benchmark: 664.5 TFLOPS at 8K with TN format
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  INT8 IMMA TENSOR CORE - SUSTAINED THROUGHPUT BENCHMARK              |~n"),
    io:format("|  RTX 4090: 660 TFLOPS theoretical | C standalone: 664.5 TFLOPS       |~n"),
    io:format("|  Using ct_int8_matmul_inplace (zero alloc in hot loop)               |~n"),
    io:format("+=======================================================================+~n~n"),

    case check_int8() of
        ok ->
            run_benchmarks();
        {error, Reason} ->
            io:format("[FAIL] INT8 not available: ~p~n", [Reason]),
            halt(1)
    end.

check_int8() ->
    try
        case viva_tensor_zig:ct_int8_available() of
            true -> ok;
            false -> {error, not_available}
        end
    catch
        _:Error -> {error, Error}
    end.

run_benchmarks() ->
    Sizes = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],

    io:format("=== Inplace Matmul (zero-alloc, TN IMMA) ===~n~n"),
    io:format("Size       | TOPS       | Efficiency | vs PyTorch~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_inplace(N) end, Sizes),

    io:format("~n"),
    io:format("NOTE: PyTorch INT8 on RTX 4090 = 160 TOPS~n"),
    io:format("      Theoretical peak = 660 TOPS~n"),
    io:format("      Standalone C benchmark = 664.5 TOPS at 8K~n").

bench_inplace(N) ->
    erlang:garbage_collect(),

    %% Create INT8 tensors on GPU
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),

    %% Pre-allocate output tensor C
    CData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct_int8_from_list(CData, [N, N]),

    %% Warmup: 3 runs to trigger heuristic caching
    ok = viva_tensor_zig:ct_int8_matmul_inplace(A, B, C, N, N, N),
    ok = viva_tensor_zig:ct_int8_matmul_inplace(A, B, C, N, N, N),
    ok = viva_tensor_zig:ct_int8_matmul_inplace(A, B, C, N, N, N),
    viva_tensor_zig:cuda_sync(),

    %% Determine iterations based on size for ~1 second of work
    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    %% Benchmark: inplace matmul (no allocation!) + sync once at end
    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    bench_inplace_loop(A, B, C, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    Efficiency = TOPS / 660.0 * 100,  %% vs 660 TOPS peak
    VsPyTorch = TOPS / 160.0,  %% vs PyTorch ~160 TOPS

    io:format("~-10B | ~8.1f T |  ~6.1f%  | ~5.2fx~n",
              [N, TOPS, Efficiency, VsPyTorch]),
    ok.

bench_inplace_loop(_A, _B, _C, _N, 0) -> ok;
bench_inplace_loop(A, B, C, N, Remaining) ->
    ok = viva_tensor_zig:ct_int8_matmul_inplace(A, B, C, N, N, N),
    bench_inplace_loop(A, B, C, N, Remaining - 1).
