#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa _build/default/lib/*/ebin -pa build/dev/erlang/*/ebin

%% Benchmark: CUDA FP64 (DGEMM) vs FP32 (SGEMM)
%% RTX 4090: 1.3 TFLOPS FP64 vs 82 TFLOPS FP32 (60x theoretical!)

-mode(compile).

main(_) ->
    io:format("~n========================================~n"),
    io:format("  CUDA FP32 vs FP64 Benchmark~n"),
    io:format("  RTX 4090: 82 TFLOPS FP32, 1.3 TFLOPS FP64~n"),
    io:format("========================================~n~n"),

    code:add_pathz("_build/default/lib/viva_tensor/ebin"),
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),

    %% Check if NIF is loaded
    case viva_tensor_zig:is_loaded() of
        true -> io:format("[OK] NIF loaded~n");
        false ->
            io:format("[FAIL] NIF not loaded!~n"),
            halt(1)
    end,

    %% Test sizes
    Sizes = [1000, 2000, 3000, 4000, 5000],

    io:format("~n~-8s ~12s ~12s ~10s~n", ["Size", "FP64 GFLOPS", "FP32 GFLOPS", "Speedup"]),
    io:format("~s~n", [string:copies("-", 50)]),

    lists:foreach(fun(N) -> bench_size(N) end, Sizes),

    io:format("~n[Done]~n").

bench_size(N) ->
    M = N, K = N,
    TotalOps = 2 * M * N * K,

    %% Create random matrices
    Data = [rand:uniform() || _ <- lists:seq(1, N*N)],
    {ok, A} = viva_tensor_zig:nt_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:nt_from_list(Data, [N, N]),

    erlang:garbage_collect(),

    %% Warmup
    catch viva_tensor_zig:nt_matmul_cuda(A, B, M, N, K),
    catch viva_tensor_zig:nt_matmul_cuda_fp32(A, B, M, N, K),

    %% Benchmark FP64
    GflopsFP64 = bench_op(fun() -> viva_tensor_zig:nt_matmul_cuda(A, B, M, N, K) end, TotalOps, 5),

    %% Benchmark FP32
    GflopsFP32 = bench_op(fun() -> viva_tensor_zig:nt_matmul_cuda_fp32(A, B, M, N, K) end, TotalOps, 5),

    Speedup = case GflopsFP64 > 0 of
        true -> GflopsFP32 / GflopsFP64;
        false -> 0.0
    end,

    io:format("~-8B ~12.1f ~12.1f ~10.1fx~n", [N, GflopsFP64, GflopsFP32, Speedup]).

bench_op(Fun, TotalOps, Runs) ->
    Times = [begin
        T0 = erlang:monotonic_time(microsecond),
        case Fun() of
            {ok, _} -> ok;
            {error, _} -> error
        end,
        erlang:monotonic_time(microsecond) - T0
    end || _ <- lists:seq(1, Runs)],

    case lists:filter(fun(T) -> T > 0 end, Times) of
        [] -> 0.0;
        ValidTimes ->
            AvgUs = lists:sum(ValidTimes) / length(ValidTimes),
            AvgSec = AvgUs / 1_000_000,
            TotalOps / AvgSec / 1_000_000_000
    end.
