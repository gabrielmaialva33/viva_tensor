#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% INT8 Tensor Core Benchmark
%% RTX 4090: 660 TFLOPS INT8 vs 82 TFLOPS FP32 (8x faster!)
%%
%% Run: escript bench/bench_int8_tc.erl
%%

main(_) ->
    io:format("~n"),
    io:format("╔════════════════════════════════════════════════════════════╗~n"),
    io:format("║           INT8 TENSOR CORE BENCHMARK                      ║~n"),
    io:format("║        RTX 4090: Target 300+ TFLOPS (8x FP32)             ║~n"),
    io:format("╚════════════════════════════════════════════════════════════╝~n~n"),

    %% Check availability
    io:format("[CHECK] INT8 Tensor Cores: "),
    Int8Available = viva_tensor_zig:nt_int8_tc_available(),
    io:format("~s~n~n", [case Int8Available of true -> "AVAILABLE"; _ -> "NOT AVAILABLE" end]),

    case Int8Available of
        true -> run_benchmarks();
        _ -> io:format("ERROR: INT8 Tensor Cores not available. Need CUDA + cublasGemmEx.~n")
    end.

run_benchmarks() ->
    %% Sizes to test (must be multiples of 16 for best Tensor Core performance)
    Sizes = [1024, 2048, 3072, 4096],

    io:format("┌──────────┬──────────────┬──────────────┬──────────────┬─────────┐~n"),
    io:format("│ Size     │ FP32 GFLOPS  │ INT8 GFLOPS  │ Speedup      │ Status  │~n"),
    io:format("├──────────┼──────────────┼──────────────┼──────────────┼─────────┤~n"),

    lists:foreach(fun(N) ->
        {FP32_GFLOPS, INT8_GFLOPS} = bench_size(N),
        Speedup = INT8_GFLOPS / max(1.0, FP32_GFLOPS),
        Status = case INT8_GFLOPS > 300000 of
            true -> "GOAL!";
            _ -> case INT8_GFLOPS > 100000 of
                true -> "FAST";
                _ -> "OK"
            end
        end,
        io:format("│ ~-8B │ ~-12.1f │ ~-12.1f │ ~-12.2fx │ ~-7s │~n",
                  [N, FP32_GFLOPS, INT8_GFLOPS, Speedup, Status])
    end, Sizes),

    io:format("└──────────┴──────────────┴──────────────┴──────────────┴─────────┘~n"),
    io:format("~n"),
    io:format("NOTE: GFLOPS = Giga FLOPs (10^9 operations per second)~n"),
    io:format("      TFLOPS = 1000 GFLOPS~n"),
    io:format("      Target: 300 TFLOPS = 300,000 GFLOPS~n"),
    io:format("~n").

bench_size(N) ->
    %% Create random tensors
    erlang:garbage_collect(),
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:nt_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:nt_from_list(Data, [N, N]),

    %% Warmup
    catch viva_tensor_zig:nt_matmul_cuda_fp32(A, B, N, N, N),
    catch viva_tensor_zig:nt_matmul_int8_tc(A, B, N, N, N),

    erlang:garbage_collect(),

    %% Benchmark FP32
    FP32_Time = bench_loop(fun() ->
        viva_tensor_zig:nt_matmul_cuda_fp32(A, B, N, N, N)
    end, 5),

    %% Benchmark INT8 Tensor Cores
    INT8_Time = bench_loop(fun() ->
        viva_tensor_zig:nt_matmul_int8_tc(A, B, N, N, N)
    end, 5),

    %% Calculate GFLOPS: 2*N^3 FLOPs for matmul
    FLOPs = 2.0 * N * N * N,
    FP32_GFLOPS = FLOPs / FP32_Time / 1.0e6,  % us -> GFLOPS
    INT8_GFLOPS = FLOPs / INT8_Time / 1.0e6,

    {FP32_GFLOPS, INT8_GFLOPS}.

bench_loop(Fun, Iterations) ->
    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    bench_loop_inner(Fun, Iterations),
    T1 = erlang:monotonic_time(microsecond),
    (T1 - T0) / Iterations.

bench_loop_inner(_Fun, 0) -> ok;
bench_loop_inner(Fun, N) ->
    _ = Fun(),
    bench_loop_inner(Fun, N - 1).
