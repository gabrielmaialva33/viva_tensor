#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% SparseTensor BENCHMARK - 2:4 Sparsity Tensor Cores
%% RTX 4090: 660 TFLOPS FP16 Sparse (2x of 330T dense!)
%%           1320 TFLOPS INT8 Sparse (2x of 660T dense!)
%%
%% Run: cd viva_tensor && escript bench/bench_sparse.erl
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("╔═══════════════════════════════════════════════════════════════════════╗~n"),
    io:format("║       SparseTensor: 2:4 SPARSITY TENSOR CORE BENCHMARK                ║~n"),
    io:format("║  RTX 4090: 660 TFLOPS FP16 Sparse (2x of 330T dense!)                 ║~n"),
    io:format("╚═══════════════════════════════════════════════════════════════════════╝~n~n"),

    io:format("=== Backend Availability ===~n"),
    case viva_tensor_zig:sparse_available() of
        true ->
            io:format("[OK] cuSPARSELt 2:4 Sparsity - AVAILABLE~n"),
            case viva_tensor_zig:ct16_available() of
                true ->
                    io:format("[OK] CudaTensor16 Dense - AVAILABLE~n~n"),
                    run_benchmarks();
                _ ->
                    io:format("[FAIL] CudaTensor16 not available~n"),
                    halt(1)
            end;
        _ ->
            io:format("[FAIL] cuSPARSELt not available.~n"),
            io:format("       Install CUDA Toolkit with cuSPARSELt library.~n"),
            io:format("       Path: /usr/local/cuda/lib64/libcusparseLt.so~n"),
            halt(1)
    end.

run_benchmarks() ->
    %% Sizes optimized for Tensor Cores (multiples of 16)
    Sizes = [1024, 2048, 3072, 4096],

    io:format("Workflow:~n"),
    io:format("  1. Create dense FP16 tensor on GPU (CudaTensor16)~n"),
    io:format("  2. Prune to 2:4 pattern + compress (SparseTensor)~n"),
    io:format("  3. Run sparse matmul (Tensor Cores)~n~n"),

    io:format("┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐~n"),
    io:format("│ Size     │ Dense TFLOPS │ Sparse TFLOPS│ Speedup      │ Compress     │~n"),
    io:format("├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤~n"),

    Results = lists:map(fun(N) -> bench_size(N) end, Sizes),

    io:format("└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘~n"),

    print_summary(Results),
    io:format("~n").

bench_size(N) ->
    erlang:garbage_collect(),

    %% Generate random data
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],

    %% Create dense CudaTensor16 (FP16 on GPU)
    {ok, A_dense} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),

    %% Create SparseTensor (prune A to 2:4 pattern)
    {ok, A_sparse} = viva_tensor_zig:sparse_from_ct16(A_dense),
    {ok, Ratio} = viva_tensor_zig:sparse_compression_ratio(A_sparse),

    %% Warmup
    {ok, _} = viva_tensor_zig:ct16_matmul(A_dense, B, N, N, N),
    {ok, _} = viva_tensor_zig:sparse_matmul(A_sparse, B, N, N, N),
    erlang:garbage_collect(),

    Iterations = 10,

    %% Benchmark dense matmul
    T0_dense = erlang:monotonic_time(microsecond),
    bench_loop(fun() -> viva_tensor_zig:ct16_matmul(A_dense, B, N, N, N) end, Iterations),
    T1_dense = erlang:monotonic_time(microsecond),
    DenseMs = (T1_dense - T0_dense) / 1000 / Iterations,

    %% Benchmark sparse matmul
    T0_sparse = erlang:monotonic_time(microsecond),
    bench_loop(fun() -> viva_tensor_zig:sparse_matmul(A_sparse, B, N, N, N) end, Iterations),
    T1_sparse = erlang:monotonic_time(microsecond),
    SparseMs = (T1_sparse - T0_sparse) / 1000 / Iterations,

    %% Calculate TFLOPS: 2*N^3 FLOPs for GEMM
    FLOPs = 2.0 * N * N * N,
    DenseTFLOPS = FLOPs / (DenseMs / 1000.0) / 1.0e12,
    SparseTFLOPS = FLOPs / (SparseMs / 1000.0) / 1.0e12,
    Speedup = SparseTFLOPS / DenseTFLOPS,

    io:format("│ ~-8B │ ~12.1f │ ~12.1f │ ~10.2fx │ ~10.1fx │~n",
              [N, DenseTFLOPS, SparseTFLOPS, Speedup, Ratio]),

    #{size => N, dense => DenseTFLOPS, sparse => SparseTFLOPS,
      speedup => Speedup, ratio => Ratio}.

bench_loop(_Fun, 0) -> ok;
bench_loop(Fun, N) ->
    {ok, _} = Fun(),
    bench_loop(Fun, N - 1).

print_summary(Results) ->
    io:format("~n=== Summary ===~n"),

    PeakDense = lists:max([maps:get(dense, R) || R <- Results]),
    PeakSparse = lists:max([maps:get(sparse, R) || R <- Results]),
    AvgRatio = lists:sum([maps:get(ratio, R) || R <- Results]) / length(Results),

    io:format("Peak Dense FP16:  ~.1f TFLOPS (~.1f%% of 330T theoretical)~n",
              [PeakDense, PeakDense / 330.0 * 100]),
    io:format("Peak Sparse FP16: ~.1f TFLOPS (~.1f%% of 660T theoretical)~n",
              [PeakSparse, PeakSparse / 660.0 * 100]),
    io:format("Avg Compression:  ~.2fx (target: 2.0x)~n", [AvgRatio]),
    io:format("~n"),

    case PeakSparse >= 100.0 of
        true ->
            io:format(">>> SUCCESS! 100+ TFLOPS achieved with 2:4 sparsity! <<<~n");
        false ->
            io:format("NOTE: If performance is lower than expected, check:~n"),
            io:format("  1. Matrix dimensions should be multiples of 16~n"),
            io:format("  2. cuSPARSELt library version~n"),
            io:format("  3. GPU power/thermal limits~n")
    end.
