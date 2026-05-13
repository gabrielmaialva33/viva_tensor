#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% INT8 TENSOR CORE PIPELINE BENCHMARK - Target 300+ TFLOPS!
%%
%% Uses cublasLt IMMA (Integer Matrix Multiply Accumulate) Tensor Cores.
%% RTX 4090: 660 TFLOPS theoretical for INT8!
%%
%% Strategy: Multiple async matmuls, sync only at the end = sustained throughput.
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|     INT8 TENSOR CORE PIPELINE - Target 300+ TFLOPS!                  |~n"),
    io:format("|  cublasLt IMMA Tensor Cores - RTX 4090: 660 TFLOPS theoretical       |~n"),
    io:format("+=======================================================================+~n~n"),

    case check_int8() of
        ok ->
            run_benchmarks();
        {error, Reason} ->
            io:format("[FAIL] INT8 Tensor Cores not available: ~p~n", [Reason]),
            halt(1)
    end.

check_int8() ->
    try
        case viva_tensor_zig:ct_int8_available() of
            true -> ok;
            false -> {error, int8_not_available}
        end
    catch
        _:Error -> {error, Error}
    end.

run_benchmarks() ->
    %% Test with sizes optimal for Tensor Cores (multiples of 16)
    Sizes = [2048, 3072, 4096, 5120],
    Iterations = 50,  %% Number of matmuls per batch - more = better saturation

    io:format("Strategy: ~B matmuls per batch, sync only at the end~n", [Iterations]),
    io:format("This shows SUSTAINED throughput, not single-op latency.~n~n"),

    io:format("=== INT8 IMMA Pipeline (Target: 300+ TFLOPS) ===~n"),
    io:format("~nSize       | Single-Op   | Pipelined   | Speedup~n"),
    io:format("-----------|-------------|-------------|--------~n"),
    lists:foreach(fun(N) -> bench_int8_pipeline(N, Iterations) end, Sizes),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  Comparison with FP16/FP32:                                          |~n"),
    io:format("|  FP32: 78.7 TFLOPS | FP16: 133 TFLOPS | INT8: ???                     |~n"),
    io:format("|  If INT8 > 200T, we're properly using IMMA Tensor Cores!             |~n"),
    io:format("+=======================================================================+~n").

bench_int8_pipeline(N, Iterations) ->
    erlang:garbage_collect(),

    %% Create INT8 tensors on GPU (quantize ONCE)
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_int8_from_list(Data, [N, N]),

    %% Warmup
    {ok, _} = viva_tensor_zig:ct_int8_matmul(A, B, N, N, N),
    viva_tensor_zig:cuda_sync(),
    erlang:garbage_collect(),

    %% Single-op benchmark (with sync)
    T0 = erlang:monotonic_time(microsecond),
    lists:foreach(fun(_) ->
        {ok, _} = viva_tensor_zig:ct_int8_matmul(A, B, N, N, N)
    end, lists:seq(1, Iterations)),
    T1 = erlang:monotonic_time(microsecond),
    SingleMs = (T1 - T0) / 1000,
    FLOPs = 2.0 * N * N * N * Iterations,
    SingleTFLOPS = FLOPs / (SingleMs / 1000.0) / 1.0e12,

    %% Pipelined benchmark (async + single sync)
    T2 = erlang:monotonic_time(microsecond),
    Results = lists:map(fun(_) ->
        {ok, C} = viva_tensor_zig:ct_int8_matmul_async(A, B, N, N, N),
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
