#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% PIPELINED TENSOR CORE BENCHMARK - Sustained 100+ TFLOPS!
%%
%% The problem: cudaDeviceSynchronize() after each op kills throughput.
%% The solution: Run multiple matmuls, sync only at the end.
%%
%% RTX 4090: 82 TFLOPS FP32 | 330 TFLOPS FP16 | 660 TFLOPS INT8
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|     PIPELINED TENSOR CORE BENCHMARK - Sustained 100+ TFLOPS!          |~n"),
    io:format("|  No sync between ops, measure real GPU throughput!                    |~n"),
    io:format("+=======================================================================+~n~n"),

    case check_cuda() of
        ok ->
            run_benchmarks();
        {error, Reason} ->
            io:format("[FAIL] CUDA not available: ~p~n", [Reason]),
            halt(1)
    end.

check_cuda() ->
    try
        case viva_tensor_zig:ct16_available() of
            true -> ok;
            false -> {error, cuda_not_available}
        end
    catch
        _:Error -> {error, Error}
    end.

run_benchmarks() ->
    %% Test with larger sizes where GPU really shines
    Sizes = [2048, 3072, 4096, 5120],
    Iterations = 50,  %% Number of matmuls per batch - more = better saturation

    io:format("Strategy: ~B matmuls per batch, sync only at the end~n", [Iterations]),
    io:format("This shows SUSTAINED throughput, not single-op latency.~n~n"),

    io:format("=== FP32 Pipeline (Target: 50+ TFLOPS) ===~n"),
    io:format("~nSize       | Single-Op   | Pipelined   | Speedup~n"),
    io:format("-----------|-------------|-------------|--------~n"),
    lists:foreach(fun(N) -> bench_fp32_pipeline(N, Iterations) end, Sizes),

    io:format("~n=== FP16 Tensor Core Pipeline (Target: 100+ TFLOPS) ===~n"),
    io:format("~nSize       | Single-Op   | Pipelined   | Speedup~n"),
    io:format("-----------|-------------|-------------|--------~n"),
    lists:foreach(fun(N) -> bench_fp16_pipeline(N, Iterations) end, Sizes),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  If Pipelined >> Single-Op, sync overhead was killing performance!   |~n"),
    io:format("+=======================================================================+~n").

bench_fp32_pipeline(N, Iterations) ->
    erlang:garbage_collect(),

    %% Create FP32 tensors on GPU
    Data = [rand:uniform() - 0.5 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_from_list(Data, [N, N]),

    %% Warmup
    {ok, _} = viva_tensor_zig:ct_matmul(A, B, N, N, N),

    %% Single-op benchmark (with sync)
    T0 = erlang:monotonic_time(microsecond),
    lists:foreach(fun(_) ->
        {ok, _} = viva_tensor_zig:ct_matmul(A, B, N, N, N)
    end, lists:seq(1, Iterations)),
    T1 = erlang:monotonic_time(microsecond),
    SingleMs = (T1 - T0) / 1000,
    FLOPs = 2.0 * N * N * N * Iterations,
    SingleTFLOPS = FLOPs / (SingleMs / 1000.0) / 1.0e12,

    %% Pipelined benchmark (async + single sync)
    T2 = erlang:monotonic_time(microsecond),
    Results = lists:map(fun(_) ->
        {ok, C} = viva_tensor_zig:ct_matmul_async(A, B, N, N, N),
        C
    end, lists:seq(1, Iterations)),
    viva_tensor_zig:cuda_sync(),
    T3 = erlang:monotonic_time(microsecond),
    PipeMs = (T3 - T2) / 1000,
    PipeTFLOPS = FLOPs / (PipeMs / 1000.0) / 1.0e12,

    Speedup = PipeTFLOPS / max(SingleTFLOPS, 0.001),

    io:format("~-10B | ~9.1f T | ~9.1f T | ~5.2fx~n",
              [N, SingleTFLOPS, PipeTFLOPS, Speedup]),

    %% Resources cleaned up by GC
    _ = Results,
    ok.

bench_fp16_pipeline(N, Iterations) ->
    erlang:garbage_collect(),

    %% Create FP16 tensors on GPU
    Data = [rand:uniform() - 0.5 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),

    %% Warmup
    {ok, _} = viva_tensor_zig:ct16_matmul(A, B, N, N, N),

    %% Single-op benchmark (with sync)
    T0 = erlang:monotonic_time(microsecond),
    lists:foreach(fun(_) ->
        {ok, _} = viva_tensor_zig:ct16_matmul(A, B, N, N, N)
    end, lists:seq(1, Iterations)),
    T1 = erlang:monotonic_time(microsecond),
    SingleMs = (T1 - T0) / 1000,
    FLOPs = 2.0 * N * N * N * Iterations,
    SingleTFLOPS = FLOPs / (SingleMs / 1000.0) / 1.0e12,

    %% Pipelined benchmark (async + single sync)
    T2 = erlang:monotonic_time(microsecond),
    Results = lists:map(fun(_) ->
        {ok, C} = viva_tensor_zig:ct16_matmul_async(A, B, N, N, N),
        C
    end, lists:seq(1, Iterations)),
    viva_tensor_zig:cuda_sync(),
    T3 = erlang:monotonic_time(microsecond),
    PipeMs = (T3 - T2) / 1000,
    PipeTFLOPS = FLOPs / (PipeMs / 1000.0) / 1.0e12,

    Speedup = PipeTFLOPS / max(SingleTFLOPS, 0.001),

    io:format("~-10B | ~9.1f T | ~9.1f T | ~5.2fx~n",
              [N, SingleTFLOPS, PipeTFLOPS, Speedup]),

    %% Resources cleaned up by GC
    _ = Results,
    ok.
