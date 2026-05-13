-module(bench_win).
-export([run/0]).

run() ->
    io:format("=== Windows Native Benchmark (i9-13900K) ===~n"),

    Loaded = viva_tensor_zig:is_loaded(),
    io:format("NIF loaded: ~p~n", [Loaded]),

    case Loaded of
        false ->
            io:format("NIF not loaded, cannot benchmark~n"),
            halt(1);
        true ->
            io:format("Backend: ~s~n~n", [viva_tensor_zig:backend_info()]),
            run_benchmarks()
    end.

run_benchmarks() ->
    BenchLoop = fun BL(0, _) -> ok; BL(Cnt, F) -> F(), BL(Cnt-1, F) end,

    lists:foreach(fun(Size) ->
        io:format("~B x ~B matmul: ", [Size, Size]),
        N = Size * Size,
        Data = [float(I rem 100) / 100.0 || I <- lists:seq(1, N)],
        {ok, A} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),
        {ok, B} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),

        %% Warmup
        {ok, _} = viva_tensor_zig:nt_matmul(A, B, Size, Size, Size),
        erlang:garbage_collect(),

        %% Benchmark
        Iters = if Size >= 1000 -> 5; true -> 20 end,
        T1Start = erlang:monotonic_time(microsecond),
        BenchLoop(Iters, fun() -> viva_tensor_zig:nt_matmul(A, B, Size, Size, Size) end),
        T1End = erlang:monotonic_time(microsecond),
        T1 = (T1End - T1Start) / Iters,

        Flops = 2.0 * Size * Size * Size,
        G1 = Flops / T1 / 1000,

        io:format("~B us (~B GFLOPS)~n", [round(T1), round(G1)])
    end, [200, 500, 1000, 2000]),

    io:format("~nDone!~n"),
    ok.
