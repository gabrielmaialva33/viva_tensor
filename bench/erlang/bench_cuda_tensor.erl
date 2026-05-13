#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin

%% Benchmark: CudaTensor (zero-copy GPU) vs FP32 (with PCIe transfer)
%% Expected: 30x+ speedup for repeated operations!
%%
%% RTX 4090:
%% - PCIe 4.0 x16 = 28 GB/s
%% - FP32 Tensor Cores = 82 TFLOPS
%% - With transfer: ~1.2 TFLOPS (PCIe bound)
%% - CudaTensor: 40+ TFLOPS (compute bound)

-mode(compile).

main(_) ->
    io:format("~n========================================~n"),
    io:format("  CudaTensor Benchmark~n"),
    io:format("  Zero-Copy GPU vs PCIe Transfer~n"),
    io:format("========================================~n~n"),

    %% Check NIF
    case viva_tensor_zig:is_loaded() of
        true -> io:format("[OK] NIF loaded~n");
        false ->
            io:format("[FAIL] NIF not loaded!~n"),
            halt(1)
    end,

    %% Test sizes
    Sizes = [500, 1000, 2000, 3000],

    io:format("~n~-8s ~15s ~15s ~12s~n",
              ["Size", "FP32+PCIe", "CudaTensor", "Speedup"]),
    io:format("~s~n", [string:copies("-", 55)]),

    lists:foreach(fun(N) -> bench_size(N) end, Sizes),

    io:format("~n[Done]~n").

bench_size(N) ->
    M = N, K = N,
    TotalOps = 2 * M * N * K,

    %% Create random data
    Data = [rand:uniform() || _ <- lists:seq(1, N*N)],

    %% Create NativeTensor for FP32+PCIe baseline
    {ok, A_nt} = viva_tensor_zig:nt_from_list(Data, [N, N]),
    {ok, B_nt} = viva_tensor_zig:nt_from_list(Data, [N, N]),

    %% Create CudaTensor (upload ONCE)
    {ok, A_ct} = viva_tensor_zig:ct_from_list(Data, [N, N]),
    {ok, B_ct} = viva_tensor_zig:ct_from_list(Data, [N, N]),

    erlang:garbage_collect(),
    timer:sleep(100),

    %% Warmup
    catch viva_tensor_zig:nt_matmul_cuda_fp32(A_nt, B_nt, M, N, K),
    catch viva_tensor_zig:ct_matmul(A_ct, B_ct, M, N, K),

    %% Benchmark FP32+PCIe (each call transfers data)
    Runs = case N of
        N when N >= 3000 -> 3;
        N when N >= 2000 -> 5;
        _ -> 10
    end,

    GflopsFP32 = bench_op(
        fun() -> viva_tensor_zig:nt_matmul_cuda_fp32(A_nt, B_nt, M, N, K) end,
        TotalOps, Runs),

    %% Benchmark CudaTensor (NO transfer!)
    GflopsCT = bench_op(
        fun() -> viva_tensor_zig:ct_matmul(A_ct, B_ct, M, N, K) end,
        TotalOps, Runs),

    Speedup = case GflopsFP32 > 0 of
        true -> GflopsCT / GflopsFP32;
        false -> 0.0
    end,

    io:format("~-8B ~12.1f GFLOPS ~12.1f GFLOPS ~10.1fx~n",
              [N, GflopsFP32, GflopsCT, Speedup]).

bench_op(Fun, TotalOps, Runs) ->
    erlang:garbage_collect(),

    Times = bench_loop(Fun, Runs, []),

    case lists:filter(fun(T) -> T > 0 end, Times) of
        [] -> 0.0;
        ValidTimes ->
            AvgUs = lists:sum(ValidTimes) / length(ValidTimes),
            AvgSec = AvgUs / 1_000_000,
            TotalOps / AvgSec / 1_000_000_000
    end.

bench_loop(_Fun, 0, Acc) -> Acc;
bench_loop(Fun, N, Acc) ->
    T0 = erlang:monotonic_time(microsecond),
    case Fun() of
        {ok, _} -> ok;
        {error, _} -> error
    end,
    T1 = erlang:monotonic_time(microsecond),
    bench_loop(Fun, N - 1, [T1 - T0 | Acc]).
