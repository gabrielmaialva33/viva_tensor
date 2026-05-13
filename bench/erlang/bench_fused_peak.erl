#!/usr/bin/env escript
%% -*- erlang -*-
%%! -pa build/dev/erlang/viva_tensor/ebin -pa priv

%%
%% cublasLt FUSED GEMM+ACTIVATION PEAK THROUGHPUT BENCHMARK
%%
%% RTX 4090 (Ada Lovelace) Tensor Cores:
%% - FP16 Dense: 330 TFLOPS (theoretical), 284T achieved (plain GEMM)
%% - Fused GEMM+ReLU/GELU: activation is FREE (same kernel!)
%%
%% Tests: Plain GEMM vs cublasLt 32F baseline vs GEMM+ReLU vs GEMM+GELU
%% Shows fused activation adds ZERO overhead to GEMM at same compute mode.
%% Loop runs entirely in C (zero Erlang overhead).
%%

main(_) ->
    code:add_pathz("build/dev/erlang/viva_tensor/ebin"),
    code:add_pathz("priv"),

    io:format("~n"),
    io:format("+=======================================================================+~n"),
    io:format("|  cublasLt FUSED GEMM+ACTIVATION BENCHMARK                            |~n"),
    io:format("|  RTX 4090: FP16=330T peak, 284T plain GEMM                           |~n"),
    io:format("+=======================================================================+~n~n"),

    Sizes = [1024, 2048, 4096, 6144, 8192],

    io:format("=== 1. FP16 Plain GEMM (cublasGemmEx COMPUTE_16F, fastest path) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency~n"),
    io:format("-----------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_plain(N) end, Sizes),

    io:format("~n=== 2. cublasLt COMPUTE_32F_FAST_16F (no epilogue, same-mode baseline) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs GemmEx~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_lt_32f(N) end, Sizes),

    io:format("~n=== 3. Fused GEMM+ReLU (cublasLt COMPUTE_32F_FAST_16F + epilogue) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs Lt32F   | Activation cost~n"),
    io:format("-----------|------------|------------|------------|---------------~n"),
    lists:foreach(fun(N) -> bench_fused(N, relu) end, Sizes),

    io:format("~n=== 4. Fused GEMM+GELU (cublasLt COMPUTE_32F_FAST_16F + epilogue) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs Lt32F   | Activation cost~n"),
    io:format("-----------|------------|------------|------------|---------------~n"),
    lists:foreach(fun(N) -> bench_fused(N, gelu) end, Sizes),

    io:format("~n=== 5. Fused GEMM+ReLU TN (pre-transposed B + epilogue) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs NN ReLU~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_fused_tn(N, relu) end, Sizes),

    io:format("~n=== 6. Fused GEMM+GELU TN (pre-transposed B + epilogue) ===~n~n"),
    io:format("Size       | TFLOPS     | Efficiency | vs NN GELU~n"),
    io:format("-----------|------------|------------|----------~n"),
    lists:foreach(fun(N) -> bench_fused_tn(N, gelu) end, Sizes),

    io:format("~n"),
    io:format("+-----------------------------------------------------------------------+~n"),
    io:format("| ANALYSIS                                                              |~n"),
    io:format("+-----------------------------------------------------------------------+~n"),
    io:format("| COMPUTE_16F (GemmEx): fastest raw GEMM (284T), no epilogue support    |~n"),
    io:format("| COMPUTE_32F_FAST_16F (Lt): FP32 accum, 160T, supports epilogues       |~n"),
    io:format("| Fused GEMM+ReLU/GELU: activation is FREE vs same-mode GEMM            |~n"),
    io:format("| Use case: training (needs FP32 accum) gets free ReLU/GELU!            |~n"),
    io:format("+-----------------------------------------------------------------------+~n").

bench_plain(N) ->
    erlang:garbage_collect(),

    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    put({plain, N}, TFLOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f%~n", [N, TFLOPS, VsPeak]).

bench_lt_32f(N) ->
    erlang:garbage_collect(),

    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    ok = viva_tensor_zig:ct16_matmul_lt_32f_bench(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = viva_tensor_zig:ct16_matmul_lt_32f_bench(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    PlainT = case get({plain, N}) of undefined -> TFLOPS; V -> V end,
    VsPlain = TFLOPS / PlainT * 100,

    put({lt32f, N}, TFLOPS),
    io:format("~-10B | ~8.1f T |  ~5.1f% |  ~5.1f%~n",
              [N, TFLOPS, VsPeak, VsPlain]).

bench_fused(N, Type) ->
    erlang:garbage_collect(),

    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    BenchFn = case Type of
        relu -> fun(AA, BB, CC, M, NN, K, I) ->
                    viva_tensor_zig:ct16_matmul_fused_relu_bench(AA, BB, CC, M, NN, K, I)
                end;
        gelu -> fun(AA, BB, CC, M, NN, K, I) ->
                    viva_tensor_zig:ct16_matmul_fused_gelu_bench(AA, BB, CC, M, NN, K, I)
                end
    end,

    ok = BenchFn(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = BenchFn(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    Lt32fT = case get({lt32f, N}) of undefined -> TFLOPS; V -> V end,
    VsLt32f = TFLOPS / Lt32fT * 100,

    %% Cost = how much slower than same-mode baseline (should be ~0%)
    Cost = case TFLOPS < Lt32fT of
        true -> io_lib:format("~.1f% slower", [(1.0 - TFLOPS / Lt32fT) * 100]);
        false -> "FREE"
    end,

    %% Store fused result for TN comparison
    put({fused, Type, N}, TFLOPS),

    io:format("~-10B | ~8.1f T |  ~5.1f% |  ~5.1f% | ~s~n",
              [N, TFLOPS, VsPeak, VsLt32f, Cost]).

bench_fused_tn(N, Type) ->
    erlang:garbage_collect(),

    Data = [rand:uniform() * 2.0 - 1.0 || _ <- lists:seq(1, N * N)],
    {ok, A} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    {ok, B} = viva_tensor_zig:ct16_from_list(Data, [N, N]),
    ZData = [0.0 || _ <- lists:seq(1, N * N)],
    {ok, C} = viva_tensor_zig:ct16_from_list(ZData, [N, N]),

    Iters = max(5, min(200, 500000000 div (N * N * N div 1000))),

    BenchFn = case Type of
        relu -> fun(AA, BB, CC, M, NN, K, I) ->
                    viva_tensor_zig:ct16_matmul_fused_relu_tn_bench(AA, BB, CC, M, NN, K, I)
                end;
        gelu -> fun(AA, BB, CC, M, NN, K, I) ->
                    viva_tensor_zig:ct16_matmul_fused_gelu_tn_bench(AA, BB, CC, M, NN, K, I)
                end
    end,

    ok = BenchFn(A, B, C, N, N, N, 3),
    viva_tensor_zig:cuda_sync(),

    erlang:garbage_collect(),
    T0 = erlang:monotonic_time(microsecond),
    ok = BenchFn(A, B, C, N, N, N, Iters),
    viva_tensor_zig:cuda_sync(),
    T1 = erlang:monotonic_time(microsecond),

    ElapsedUs = T1 - T0,
    FLOPs = 2.0 * N * N * N * Iters,
    TFLOPS = FLOPs / (ElapsedUs / 1.0e6) / 1.0e12,
    VsPeak = TFLOPS / 330.0 * 100,

    NnT = case get({fused, Type, N}) of undefined -> TFLOPS; V -> V end,
    VsNn = TFLOPS / NnT * 100,

    io:format("~-10B | ~8.1f T |  ~5.1f% |  ~5.1f%~n",
              [N, TFLOPS, VsPeak, VsNn]).
