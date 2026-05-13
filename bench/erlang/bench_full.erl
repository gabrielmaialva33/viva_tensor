-module(bench_full).
-export([run/0]).

run() ->
    io:format("~n=== VIVA TENSOR FULL BENCHMARK ===~n"),
    io:format("Backend: ~s~n", [viva_tensor_zig:backend_info()]),
    io:format("MKL_NUM_THREADS: ~s~n~n", [os:getenv("MKL_NUM_THREADS", "default")]),

    BenchLoop = fun BL(0, _) -> ok; BL(Cnt, F) -> F(), BL(Cnt-1, F) end,

    lists:foreach(fun(Size) ->
        N = Size * Size,
        Data = [float(I rem 100) / 100.0 || I <- lists:seq(1, N)],
        {ok, A} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),
        {ok, B} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),

        %% Warmup
        {ok, _} = viva_tensor_zig:nt_matmul_blas(A, B, Size, Size, Size),
        erlang:garbage_collect(),

        Iters = if Size >= 4000 -> 3; Size >= 2000 -> 5; true -> 10 end,
        Flops = 2.0 * Size * Size * Size,

        %% MKL
        T1 = erlang:monotonic_time(microsecond),
        BenchLoop(Iters, fun() -> viva_tensor_zig:nt_matmul_blas(A, B, Size, Size, Size) end),
        T2 = erlang:monotonic_time(microsecond),
        MklTime = (T2 - T1) / Iters,
        MklGflops = Flops / MklTime / 1000,

        io:format("~4B x ~4B: ~6B us (~4B GFLOPS)~n", [Size, Size, round(MklTime), round(MklGflops)])
    end, [500, 1000, 2000, 3000, 4000, 5000]),

    io:format("~n=== DONE ===~n"),
    ok.
