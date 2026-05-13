#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% cuSPARSELt 2:4 SPARSE PEAK THROUGHPUT BENCHMARK
%%
%% RTX 4090 (Ada Lovelace) Tensor Core + 2:4 Sparsity:
%% - FP16 Dense: 330 TFLOPS
%% - FP16 Sparse (2:4): 660 TFLOPS (2x!)
%%
%% Tests both NN (standard) and TN (transposed B) layouts.
%% Loop runs entirely in C (zero Erlang overhead).
%% Measures raw GPU kernel throughput from BEAM.
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  cuSPARSELt 2:4 SPARSE PEAK THROUGHPUT - C-LOOP BENCHMARK            |~n"),
    io:format("|  RTX 4090: FP16 Sparse=660T theoretical (2x of 330T dense)            |~n"),
    io:format("+=======================================================================+~n~n"),

    Avail = viva_tensor_zig:sparse_available(),
    io:format("cuSPARSELt available: ~w~n~n", [Avail]),

    case Avail of
        true ->
            run_benchmarks();
        false ->
            io:format("cuSPARSELt not available, skipping tests~n")
    end.

run_benchmarks() ->
    %% Sizes must be multiples of 16. Use large sizes for peak perf.
    Sizes = [1024, 2048, 4096, 6144, 8192, 10240, 12288],

    io:format("=== FP16 Sparse 2:4 — NN Layout (standard) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs Dense   | vs PT Sparse~n"),
    io:format("-----------|------------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_sparse_fp16(N, nn) end, Sizes),

    io:format("~n=== FP16 Sparse 2:4 — TN Layout (B transposed) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs Dense   | vs PT Sparse~n"),
    io:format("-----------|------------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_sparse_fp16(N, tn) end, Sizes),

    io:format("~n"),
    io:format("Dense FP16: 284T (viva_tensor) | 312T (PyTorch)~n"),
    io:format("Sparse FP16 peak: 660T theoretical~n"),
    io:format("PyTorch sparse FP16: 294T @ 6K, 281T @ 8K~n").

bench_sparse_fp16(N, Layout) ->
    erlang:garbage_collect(),

    %% Create FP16 tensors on GPU
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),

    %% Create sparse tensor from A (prune + compress)
    {ok, SparseA} = viva_tensor_zig:sparse_from_ct16(A),

    %% Create output tensor (zeroed)
    CData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(CData, [N, N]),

    BenchFn = case Layout of
        nn -> fun(SA, Bb, Cc, M2, N2, K2, I) ->
                  viva_tensor_zig:sparse_matmul_bench(SA, Bb, Cc, M2, N2, K2, I)
              end;
        tn -> fun(SA, Bb, Cc, M2, N2, K2, I) ->
                  viva_tensor_zig:sparse_matmul_bench_tn(SA, Bb, Cc, M2, N2, K2, I)
              end
    end,

    %% Warmup (3 iters — also triggers plan creation + algorithm search)
    ok = BenchFn(SparseA, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    %% Determine iteration count (scale with problem size)
    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = BenchFn(SparseA, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    Efficiency = TFLOPS / 660.0 * 100,
    VsDense = TFLOPS / 284.0,      %% vs our best dense FP16

    %% PyTorch sparse FP16 reference (measured):
    %% 1024=10.3, 2048=78.1, 4096=270.7, 6144=294.2, 8192=281.2
    PytorchSparse = case N of
        1024  -> 10.3;
        2048  -> 78.1;
        4096  -> 270.7;
        6144  -> 294.2;
        8192  -> 281.2;
        10240 -> 211.4;
        12288 -> 194.3;
        _     -> 290.0  %% estimate for other sizes
    end,
    VsPytorchSparse = TFLOPS / PytorchSparse,

    io:format("~-10B | ~8.1f T |  ~6.1f%  | ~5.2fx     | ~5.2fx~n",
              [N, TFLOPS, Efficiency, VsDense, VsPytorchSparse]).
