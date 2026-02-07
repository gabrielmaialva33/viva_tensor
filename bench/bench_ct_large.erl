#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin

-mode(compile).

main(_) ->
    io:format("~n========================================~n"),
    io:format("  CudaTensor Large Matrix Benchmark~n"),
    io:format("  RTX 4090: 82 TFLOPS FP32 Peak~n"),
    io:format("========================================~n~n"),

    %% Check NIF
    case viva_tensor_zig:is_loaded() of
        true -> io:format("[OK] NIF loaded~n~n");
        false ->
            io:format("[FAIL] NIF not loaded!~n"),
            halt(1)
    end,

    Sizes = [3000, 4000, 5000, 6000],

    io:format("~-10s ~15s ~10s ~12s~n",
              ["Size", "GFLOPS", "Time(ms)", "% Peak"]),
    io:format("~s~n", [string:copies("-", 52)]),

    lists:foreach(fun(N) -> test_size(N) end, Sizes),

    io:format("~n[Done]~n").

test_size(N) ->
    M = N, K = N,
    TotalOps = 2 * M * N * K,

    io:format("~-10s ", [io_lib:format("~Bx~B", [N, N])]),

    Data = [rand:uniform() || _ <- lists:seq(1, N*N)],
    {ok, A} = viva_tensor_zig:ct_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct_from_list(Data, [N, N]),

    erlang:garbage_collect(),
    timer:sleep(100),

    %% Warmup
    {ok, _} = viva_tensor_zig:ct_matmul(A, B, M, N, K),

    %% Benchmark (5 runs)
    Times = bench_loop(A, B, M, N, K, 5, []),

    AvgUs = lists:sum(Times) / length(Times),
    AvgSec = AvgUs / 1_000_000,
    Gflops = TotalOps / AvgSec / 1_000_000_000,
    PctPeak = Gflops / 82580 * 100,

    io:format("~12.1f ~10.2f ~10.1f%~n", [Gflops, AvgUs/1000, PctPeak]).

bench_loop(_A, _B, _M, _N, _K, 0, Acc) -> Acc;
bench_loop(A, B, M, N, K, Runs, Acc) ->
    T0 = erlang:monotonic_time(microsecond),
    {ok, _} = viva_tensor_zig:ct_matmul(A, B, M, N, K),
    T1 = erlang:monotonic_time(microsecond),
    bench_loop(A, B, M, N, K, Runs - 1, [T1 - T0 | Acc]).
