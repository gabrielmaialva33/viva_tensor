-module(bench_sage).
-export([run/0, run/4]).

run() ->
    run(1, 8, 512, 64).

run(Batch, Heads, Seq, HeadDim) ->
    io:format("=== SageAttention Benchmark (MKL) ===~n"),
    io:format("Config: B=~p, H=~p, Seq=~p, D=~p~n", [Batch, Heads, Seq, HeadDim]),

    N = Batch * Heads * Seq * HeadDim,
    io:format("Total elements: ~p (~.2f MB)~n", [N, N * 8 / 1024 / 1024]),

    %% Generate random data
    QData = [rand:uniform() || _ <- lists:seq(1, N)],
    KData = [rand:uniform() || _ <- lists:seq(1, N)],
    VData = [rand:uniform() || _ <- lists:seq(1, N)],

    {ok, Q} = viva_tensor_zig:nt_from_list(QData, [Batch, Heads, Seq, HeadDim]),
    {ok, K} = viva_tensor_zig:nt_from_list(KData, [Batch, Heads, Seq, HeadDim]),
    {ok, V} = viva_tensor_zig:nt_from_list(VData, [Batch, Heads, Seq, HeadDim]),

    io:format("Tensors created~n"),

    %% Warmup
    {ok, _} = viva_tensor_zig:sage_attention(Q, K, V, Batch, Heads, Seq, Seq, HeadDim),
    erlang:garbage_collect(),

    %% Benchmark
    Iters = 50,
    T0 = erlang:monotonic_time(microsecond),
    bench_loop(Q, K, V, Batch, Heads, Seq, HeadDim, Iters),
    T1 = erlang:monotonic_time(microsecond),

    AvgUs = (T1 - T0) / Iters,
    io:format("~nSageAttention: ~.1f us/iter (~.2f ms)~n", [AvgUs, AvgUs / 1000]),

    %% FLOPS = 4 * B * H * SeqQ * SeqK * D for attention
    %% Actually: 2 * M * N * K for each matmul
    %% QK^T: 2 * Seq * Seq * HeadDim * B * H
    %% attn@V: 2 * Seq * HeadDim * Seq * B * H
    Flops = 2 * 2 * Batch * Heads * Seq * Seq * HeadDim,
    GFlops = Flops / AvgUs / 1000,
    io:format("Estimated: ~.2f GFLOPS~n~n", [GFlops]),

    ok.

bench_loop(_Q, _K, _V, _B, _H, _Sq, _D, 0) -> ok;
bench_loop(Q, K, V, B, H, Sq, D, N) ->
    {ok, _} = viva_tensor_zig:sage_attention(Q, K, V, B, H, Sq, Sq, D),
    bench_loop(Q, K, V, B, H, Sq, D, N - 1).
