-module(bench_big).
-export([run/0]).

run() ->
    io:format("~n=== BIG Matrix Test (5000x5000) ===~n"),
    io:format("MKL_NUM_THREADS: ~s~n~n", [os:getenv("MKL_NUM_THREADS", "default")]),

    Size = 5000,
    N = Size * Size,
    io:format("Creating ~Bx~B matrices (~.1f MB each)...~n", [Size, Size, N * 8 / 1024 / 1024]),

    Data = [float(I rem 100) / 100.0 || I <- lists:seq(1, N)],
    {ok, A} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),
    {ok, B} = viva_tensor_zig:nt_from_list(Data, [Size, Size]),
    erlang:garbage_collect(),

    io:format("Running MKL matmul...~n"),
    T1 = erlang:monotonic_time(microsecond),
    {ok, _} = viva_tensor_zig:nt_matmul_blas(A, B, Size, Size, Size),
    T2 = erlang:monotonic_time(microsecond),

    Time = T2 - T1,
    Flops = 2.0 * Size * Size * Size,
    GFLOPS = Flops / Time / 1000,

    io:format("~n5000x5000 MKL: ~B us (~B GFLOPS)~n", [Time, round(GFLOPS)]),

    %% Compare with Zig
    io:format("~nRunning Zig GEMM for comparison...~n"),
    T3 = erlang:monotonic_time(microsecond),
    {ok, _} = viva_tensor_zig:nt_matmul(A, B, Size, Size, Size),
    T4 = erlang:monotonic_time(microsecond),

    Time2 = T4 - T3,
    GFLOPS2 = Flops / Time2 / 1000,

    io:format("5000x5000 Zig: ~B us (~B GFLOPS)~n", [Time2, round(GFLOPS2)]),
    io:format("~nMKL is ~.1fx faster~n", [Time2 / Time]),
    ok.
