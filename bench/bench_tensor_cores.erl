#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% TENSOR CORE BENCHMARK - FP16 + INT8 IMMA
%% RTX 4090 Theoretical: FP32=82T, FP16=330T, INT8=660T
%%
%% Run: cd viva_tensor && escript bench/bench_tensor_cores.erl
%%

main(_) ->
    %% Ensure paths are loaded
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),
    io:format("~n"),
    io:format("╔═══════════════════════════════════════════════════════════════════════╗~n"),
    io:format("║              RTX 4090 TENSOR CORE BENCHMARK                           ║~n"),
    io:format("║  FP32: 82 TFLOPS | FP16 TC: 330 TFLOPS | INT8 IMMA: 660 TFLOPS        ║~n"),
    io:format("╚═══════════════════════════════════════════════════════════════════════╝~n~n"),

    %% Check backends
    io:format("=== Backend Availability ===~n"),
    FP16_OK = check_backend("FP16 Tensor Cores", fun() -> viva_tensor_zig:nt_fp16_tc_available() end),
    INT8_OK = check_backend("INT8 IMMA (cublasLt)", fun() -> viva_tensor_zig:nt_int8_lt_available() end),
    io:format("~n"),

    case FP16_OK orelse INT8_OK of
        true -> run_benchmarks(FP16_OK, INT8_OK);
        false ->
            io:format("ERROR: No Tensor Core backends available!~n"),
            io:format("Make sure CUDA and cuBLAS/cublasLt are installed.~n"),
            halt(1)
    end.

check_backend(Name, Fun) ->
    try Fun() of
        true ->
            io:format("[~s] ~s~n", [green("OK"), Name]),
            true;
        false ->
            io:format("[~s] ~s~n", [yellow("--"), Name]),
            false;
        _ ->
            io:format("[~s] ~s~n", [yellow("--"), Name]),
            false
    catch
        _:_ ->
            io:format("[~s] ~s (not loaded)~n", [red("FAIL"), Name]),
            false
    end.

run_benchmarks(FP16_OK, INT8_OK) ->
    %% Sizes optimized for Tensor Cores (multiples of 16)
    Sizes = [1024, 2048, 3072, 4096, 5120],

    io:format("=== Benchmark Results ===~n~n"),
    io:format("┌──────────┬──────────────┬──────────────┬──────────────┬─────────────────┐~n"),
    io:format("│ Size     │ FP32 cuBLAS  │ FP16 TC      │ INT8 IMMA    │ Best TFLOPS     │~n"),
    io:format("├──────────┼──────────────┼──────────────┼──────────────┼─────────────────┤~n"),

    Results = lists:map(fun(N) ->
        bench_size(N, FP16_OK, INT8_OK)
    end, Sizes),

    io:format("└──────────┴──────────────┴──────────────┴──────────────┴─────────────────┘~n"),
    io:format("~n"),

    %% Summary
    print_summary(Results),

    io:format("~n"),
    io:format("Legend: GFLOPS = 10^9 FLOPs/sec, TFLOPS = 10^12 FLOPs/sec~n"),
    io:format("        Tensor Core ops count as 2 FLOPs per multiply-accumulate~n"),
    io:format("~n").

bench_size(N, FP16_OK, INT8_OK) ->
    %% Create random tensors
    erlang:garbage_collect(),
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:nt_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:nt_from_list(Data, [N, N]),

    %% Warmup all backends
    catch viva_tensor_zig:nt_matmul_cuda_fp32(A, B, N, N, N),
    FP16_OK andalso catch viva_tensor_zig:nt_matmul_fp16_tc(A, B, N, N, N),
    INT8_OK andalso catch viva_tensor_zig:nt_matmul_int8_lt(A, B, N, N, N),

    erlang:garbage_collect(),

    %% Benchmark FP32 cuBLAS (baseline)
    FP32_GFLOPS = bench_op(fun() ->
        viva_tensor_zig:nt_matmul_cuda_fp32(A, B, N, N, N)
    end, N, 5),

    %% Benchmark FP16 Tensor Cores
    FP16_GFLOPS = case FP16_OK of
        true -> bench_op(fun() ->
            viva_tensor_zig:nt_matmul_fp16_tc(A, B, N, N, N)
        end, N, 5);
        false -> 0.0
    end,

    %% Benchmark INT8 IMMA (cublasLt)
    INT8_GFLOPS = case INT8_OK of
        true -> bench_op(fun() ->
            viva_tensor_zig:nt_matmul_int8_lt(A, B, N, N, N)
        end, N, 5);
        false -> 0.0
    end,

    %% Find best
    Best = lists:max([FP32_GFLOPS, FP16_GFLOPS, INT8_GFLOPS]),
    BestTFLOPS = Best / 1000.0,
    BestName = case Best of
        _ when Best == INT8_GFLOPS, INT8_GFLOPS > 0 -> "INT8";
        _ when Best == FP16_GFLOPS, FP16_GFLOPS > 0 -> "FP16";
        _ -> "FP32"
    end,

    %% Print row
    io:format("│ ~-8B │ ~-12.1f │ ~-12.1f │ ~-12.1f │ ~6.1f T (~s) │~n",
              [N, FP32_GFLOPS, FP16_GFLOPS, INT8_GFLOPS, BestTFLOPS, BestName]),

    #{size => N, fp32 => FP32_GFLOPS, fp16 => FP16_GFLOPS, int8 => INT8_GFLOPS}.

bench_op(Fun, N, Iterations) ->
    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    bench_loop(Fun, Iterations),
    T1 = erlang:monotonic_time(microsecond),
    AvgUs = (T1 - T0) / Iterations,

    %% FLOPs = 2 * M * N * K for GEMM
    FLOPs = 2.0 * N * N * N,
    FLOPs / AvgUs / 1.0e3.  %% us -> GFLOPS

bench_loop(_Fun, 0) -> ok;
bench_loop(Fun, N) ->
    case Fun() of
        {ok, _} -> ok;
        {error, Err} ->
            io:format("ERROR: ~p~n", [Err]),
            error
    end,
    bench_loop(Fun, N - 1).

print_summary(Results) ->
    io:format("~n=== Summary ===~n"),

    %% Find peak for each backend
    PeakFP32 = lists:max([maps:get(fp32, R) || R <- Results]),
    PeakFP16 = lists:max([maps:get(fp16, R) || R <- Results]),
    PeakINT8 = lists:max([maps:get(int8, R) || R <- Results]),

    io:format("Peak FP32 cuBLAS:  ~8.1f GFLOPS (~5.1f%% of 82 TFLOPS theoretical)~n",
              [PeakFP32, PeakFP32 / 820.0]),
    case PeakFP16 > 0.0 of
        true ->
            io:format("Peak FP16 TC:      ~8.1f GFLOPS (~5.1f%% of 330 TFLOPS theoretical)~n",
                      [PeakFP16, PeakFP16 / 3300.0]),
            io:format("  -> Speedup vs FP32: ~.1fx~n", [PeakFP16 / max(1.0, PeakFP32)]);
        false ->
            io:format("Peak FP16 TC:      NOT AVAILABLE~n")
    end,
    case PeakINT8 > 0.0 of
        true ->
            io:format("Peak INT8 IMMA:    ~8.1f GFLOPS (~5.1f%% of 660 TFLOPS theoretical)~n",
                      [PeakINT8, PeakINT8 / 6600.0]),
            io:format("  -> Speedup vs FP32: ~.1fx~n", [PeakINT8 / max(1.0, PeakFP32)]);
        false ->
            io:format("Peak INT8 IMMA:    NOT AVAILABLE~n")
    end,

    %% Overall best
    Overall = lists:max([PeakFP32, PeakFP16, PeakINT8]),
    io:format("~n>>> PEAK PERFORMANCE: ~.1f TFLOPS <<<~n", [Overall / 1000.0]).

%% ANSI color helpers
green(S) -> "\e[32m" ++ S ++ "\e[0m".
yellow(S) -> "\e[33m" ++ S ++ "\e[0m".
red(S) -> "\e[31m" ++ S ++ "\e[0m".
