#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% FP16 BATCHED GEMM PEAK THROUGHPUT BENCHMARK
%%
%% RTX 4090 (Ada Lovelace) Tensor Cores:
%% - FP16 Dense: 330 TFLOPS (theoretical), 284T achieved (single GEMM)
%% - Batched GEMM: tests multi-head attention workloads
%%
%% Transformer shapes: batch x heads x seq_len x head_dim
%% QK^T: [B*H, S, S] = [B*H, S, D] @ [B*H, D, S] (M=S, N=S, K=D)
%% AV:   [B*H, S, D] = [B*H, S, S] @ [B*H, S, D] (M=S, N=D, K=S)
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  FP16 BATCHED GEMM PEAK THROUGHPUT BENCHMARK                         |~n"),
    io:format("|  RTX 4090: FP16=330T peak, 284T single GEMM                          |~n"),
    io:format("|  cublasGemmStridedBatchedEx + COMPUTE_16F (Tensor Cores)              |~n"),
    io:format("+=======================================================================+~n~n"),

    %% Test 1: Single large GEMM baseline (for comparison)
    io:format("=== 1. Single GEMM Baseline (cublasGemmEx COMPUTE_16F) ===~n~n"),
    io:format("Size          | TFLOPS     | Efficiency~n"),
    io:format("--------------|------------|----------~n"),
    lists:foreach(fun({M,N,K}) -> bench_single(M, N, K) end,
                  [{1024,1024,1024}, {2048,2048,2048}, {4096,4096,4096}]),

    %% Test 2: Multi-head attention workloads (LLaMA-style)
    io:format("~n=== 2. Multi-Head Attention Batched GEMM ===~n~n"),
    io:format("Config                      | M    | N    | K    | Batch | TFLOPS     | Eff.~n"),
    io:format("----------------------------|------|------|------|-------|------------|-----~n"),

    %% LLaMA-7B: 32 heads, head_dim=128
    %% QK^T: seq_len x seq_len x head_dim, batch=32
    lists:foreach(fun({Label, M, N, K, Batch}) ->
        bench_batched(Label, M, N, K, Batch)
    end, [
        {"LLaMA-7B QK^T s=512",   512,  512,  128, 32},
        {"LLaMA-7B QK^T s=1024",  1024, 1024, 128, 32},
        {"LLaMA-7B QK^T s=2048",  2048, 2048, 128, 32},
        {"LLaMA-7B AV s=512",     512,  128,  512, 32},
        {"LLaMA-7B AV s=1024",    1024, 128,  1024, 32},
        {"LLaMA-7B AV s=2048",    2048, 128,  2048, 32}
    ]),

    io:format("~n"),

    %% Test 3: Varying batch sizes (GPU saturation test)
    io:format("=== 3. Batch Size Saturation (M=N=512, K=128) ===~n~n"),
    io:format("Batch | Total GFLOPs | TFLOPS     | Efficiency~n"),
    io:format("------|--------------|------------|----------~n"),
    lists:foreach(fun(B) -> bench_batched_size("", 512, 512, 128, B) end,
                  [1, 4, 8, 16, 32, 64, 128]),

    io:format("~n"),

    %% Test 4: Large batches with bigger matrices
    io:format("=== 4. Large Matrix Batched (GPU saturation) ===~n~n"),
    io:format("Config                      | M    | N    | K    | Batch | TFLOPS     | Eff.~n"),
    io:format("----------------------------|------|------|------|-------|------------|-----~n"),
    lists:foreach(fun({Label, M, N, K, Batch}) ->
        bench_batched(Label, M, N, K, Batch)
    end, [
        {"1Kx1K batch=8",     1024, 1024, 1024, 8},
        {"1Kx1K batch=16",    1024, 1024, 1024, 16},
        {"2Kx2K batch=4",     2048, 2048, 2048, 4},
        {"2Kx2K batch=8",     2048, 2048, 2048, 8}
    ]),

    io:format("~n"),
    io:format("Single GEMM peak: 284 TFLOPS (cublasGemmEx COMPUTE_16F)~n"),
    io:format("Batched target: same aggregate throughput for large total FLOPs~n").

bench_single(M, N, K) ->
    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, M * K)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [M, K]),
    DataB = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, K * N)],
    {ok, B} = viva_tensor_zig:ct16_from_list(DataB, [K, N]),
    ZData = [0.0 || _ <- lists:seq(1, M * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [M, N]),

    Iters = max(5, min(200, 500000000 div (M * N * K div 1000))),

    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, M, N, K, 3),
    viva_tensor_zig:cuda_sync(),

    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, M, N, K, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * M * N * K * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    io:format("~Bx~Bx~B     | ~8.1f T |  ~5.1f%~n", [M, N, K, TFLOPS, VsPeak]).

bench_batched(Label, M, N, K, Batch) ->
    Iters = max(3, min(100, 200000000 div (M * N * K * Batch div 1000))),

    %% Warmup
    ok = viva_tensor_zig:ct16_matmul_batched_bench(M, N, K, Batch, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_batched_bench(M, N, K, Batch, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * M * N * K * Batch * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    io:format("~-27s | ~-4B | ~-4B | ~-4B | ~-5B | ~8.1f T | ~5.1f%~n",
              [Label, M, N, K, Batch, TFLOPS, VsPeak]).

bench_batched_size(_Label, M, N, K, Batch) ->
    Iters = max(3, min(200, 200000000 div (M * N * K * Batch div 1000))),

    ok = viva_tensor_zig:ct16_matmul_batched_bench(M, N, K, Batch, 1),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_batched_bench(M, N, K, Batch, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    TotalFLOPs = 2.0 * M * N * K * Batch,
    TFLOPS = TotalFLOPs * Iters / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    TotalGFLOPs = TotalFLOPs / 1.0e9,
    io:format("~-5B | ~9.1f GF  | ~8.1f T |  ~5.1f%~n",
              [Batch, TotalGFLOPs, TFLOPS, VsPeak]).
