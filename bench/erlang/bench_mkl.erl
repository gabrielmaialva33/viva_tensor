-module(bench_mkl).
-export([run/0]).

run() ->
    io:format("~n=== Intel MKL DGEMM Benchmark (i7-13700K) ===~n"),
    io:format("Target: 600+ GFLOPS~n~n"),

    Loaded = viva_tensor_zig:is_loaded(),
    io:format("NIF loaded: ~p~n", [Loaded]),

    case Loaded of
        false ->
            io:format("ERROR: NIF not loaded!~n"),
            halt(1);
        true ->
            io:format("Backend: ~s~n~n", [viva_tensor_zig:backend_info()]),
            run_benchmarks()
    end.

run_benchmarks() ->
    BenchLoop = fun BL(0, _) -> ok; BL(Cnt, F) -> F(), BL(Cnt-1, F) end,

    %% Test both Zig GEMM and MKL BLAS
    lists:foreach(fun(Size) ->
        io:format("~n~B x ~B matmul:~n", [Size, Size]),
        N = Size * Size,
        Data = [float(I rem 100) / 100.0 || I <- lists:seq(1, N)],
        {ok, A} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),
        {ok, B} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),

        %% Warmup
        {ok, _} = viva_tensor_zig:nt_matmul(A, B, Size, Size, Size),
        {ok, _} = viva_tensor_zig:nt_matmul_blas(A, B, Size, Size, Size),
        erlang:garbage_collect(),

        %% Benchmark iterations
        Iters = if Size >= 2000 -> 3; Size >= 1000 -> 5; true -> 20 end,
        Flops = 2.0 * Size * Size * Size,

        %% Zig GEMM (hand-rolled)
        T1Start = erlang:monotonic_time(microsecond),
        BenchLoop(Iters, fun() -> viva_tensor_zig:nt_matmul(A, B, Size, Size, Size) end),
        T1End = erlang:monotonic_time(microsecond),
        T1 = (T1End - T1Start) / Iters,
        G1 = Flops / T1 / 1000,
        io:format("  Zig GEMM:  ~8B us (~4B GFLOPS)~n", [round(T1), round(G1)]),

        %% Intel MKL cblas_dgemm
        T2Start = erlang:monotonic_time(microsecond),
        BenchLoop(Iters, fun() -> viva_tensor_zig:nt_matmul_blas(A, B, Size, Size, Size) end),
        T2End = erlang:monotonic_time(microsecond),
        T2 = (T2End - T2Start) / Iters,
        G2 = Flops / T2 / 1000,
        Speedup = T1 / T2,
        io:format("  Intel MKL: ~8B us (~4B GFLOPS) [~.1fx faster]~n", [round(T2), round(G2), Speedup])
    end, [200, 500, 1000, 2000, 4000]),

    io:format("~n=== Done! ===~n"),
    ok.
