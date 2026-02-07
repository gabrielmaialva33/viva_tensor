#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% CudaTensor16 BENCHMARK - FP16 Tensor Cores ZERO-COPY
%% RTX 4090: 330 TFLOPS theoretical (FP16 Tensor Cores)
%% This benchmark measures pure Tensor Core compute - NO PCIe overhead!
%%
%% Run: cd viva_tensor && escript bench/bench_ct16.erl
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("╔═══════════════════════════════════════════════════════════════════════╗~n"),
    io:format("║       CudaTensor16: FP16 TENSOR CORES ZERO-COPY BENCHMARK             ║~n"),
    io:format("║  RTX 4090: 330 TFLOPS theoretical | Target: 100+ TFLOPS               ║~n"),
    io:format("╚═══════════════════════════════════════════════════════════════════════╝~n~n"),

    case viva_tensor_zig:ct16_available() of
        true -> run_benchmarks();
        _ ->
            io:format("ERROR: CudaTensor16 not available. Need CUDA + cuBLAS.~n"),
            halt(1)
    end.

run_benchmarks() ->
    %% Sizes optimized for Tensor Cores (multiples of 16)
    Sizes = [1024, 2048, 3072, 4096, 5120],

    io:format("Step 1: Upload tensors to GPU (one-time cost)~n"),
    io:format("Step 2: Run matmul on GPU (zero PCIe!)~n~n"),

    io:format("┌──────────┬──────────────┬──────────────┬──────────────┬─────────────────┐~n"),
    io:format("│ Size     │ Upload (ms)  │ Compute (ms) │ TFLOPS       │ %% of 330T      │~n"),
    io:format("├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤~n"),

    Results = lists:map(fun(N) -> bench_size(N) end, Sizes),

    io:format("└──────────┴──────────────┴──────────────┴──────────────┴─────────────────┘~n"),

    %% Summary
    print_summary(Results),
    io:format("~n").

bench_size(N) ->
    erlang:garbage_collect(),

    %% Generate random data
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],

    %% Time upload (one-time cost)
    T0_upload = erlang:monotonic_time(microsecond),
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    T1_upload = erlang:monotonic_time(microsecond),
    UploadMs = (T1_upload - T0_upload) / 1000,

    %% Warmup
    {ok, _} = viva_tensor_zig:ct16_matmul(A, B, N, N, N),
    erlang:garbage_collect(),

    %% Benchmark compute only (zero PCIe!)
    Iterations = 10,
    T0_compute = erlang:monotonic_time(microsecond),
    bench_loop(fun() -> viva_tensor_zig:ct16_matmul(A, B, N, N, N) end, Iterations),
    T1_compute = erlang:monotonic_time(microsecond),
    ComputeMs = (T1_compute - T0_compute) / 1000 / Iterations,

    %% Calculate TFLOPS: 2*N^3 FLOPs for GEMM
    FLOPs = 2.0 * N * N * N,
    TFLOPS = FLOPs / (ComputeMs / 1000.0) / 1.0e12,  %% ms -> s -> TFLOPS
    Percent = TFLOPS / 330.0 * 100,

    io:format("│ ~-8B │ ~12.1f │ ~12.3f │ ~12.1f │ ~14.1f%% │~n",
              [N, UploadMs, ComputeMs, TFLOPS, Percent]),

    #{size => N, upload => UploadMs, compute => ComputeMs, tflops => TFLOPS}.

bench_loop(_Fun, 0) -> ok;
bench_loop(Fun, N) ->
    {ok, _} = Fun(),
    bench_loop(Fun, N - 1).

print_summary(Results) ->
    io:format("~n=== Summary ===~n"),

    PeakTFLOPS = lists:max([maps:get(tflops, R) || R <- Results]),
    BestSize = lists:foldl(fun(R, Acc) ->
        case maps:get(tflops, R) >= maps:get(tflops, Acc) of
            true -> R;
            false -> Acc
        end
    end, hd(Results), Results),

    io:format("Peak Performance: ~.1f TFLOPS (~.1f%% of 330 TFLOPS theoretical)~n",
              [PeakTFLOPS, PeakTFLOPS / 330.0 * 100]),
    io:format("Best at size: ~Bx~B~n", [maps:get(size, BestSize), maps:get(size, BestSize)]),
    io:format("~n"),

    %% Compare with CudaTensor FP32
    io:format("For comparison: CudaTensor FP32 peaks at 27.7 TFLOPS (33.5% of 82T)~n"),
    io:format("Expected FP16 TC speedup: ~.1fx over FP32~n", [PeakTFLOPS / 27.7]),
    io:format("~n"),

    case PeakTFLOPS >= 100.0 of
        true ->
            io:format(">>> SUCCESS! 100+ TFLOPS achieved! <<<~n");
        false ->
            io:format("~nNOTE: If performance is lower than expected, check:~n"),
            io:format("  1. Matrix dimensions should be multiples of 16~n"),
            io:format("  2. NVIDIA driver version (newest recommended)~n"),
            io:format("  3. GPU power/thermal limits~n")
    end.
