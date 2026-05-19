%%% Llama sampling utilities — pure Erlang, no NIFs.
%%%
%%% Exports:
%%%   - argmax/1: clean re-export of llama_forward semantics.
%%%   - softmax/1: numerically stable host softmax (max-subtraction).
%%%   - sample/2: multinomial sampling with temperature / top_k / top_p.
%%%   - test/0: self-contained correctness check.
-module(llama_sampling).

-export([argmax/1, softmax/1, sample/2, test/0]).

%% ---------------------------------------------------------------------------
%% argmax — same semantics as llama_forward:argmax/1.
%% Returns {ZeroBasedIndex, MaxValue}.
%% ---------------------------------------------------------------------------
argmax(L) ->
    {_, BestI, BestV} = lists:foldl(
        fun(X, {I, BI, BV}) ->
            case X > BV of
                true  -> {I + 1, I, X};
                false -> {I + 1, BI, BV}
            end
        end,
        {0, 0, -1.0e308},
        L),
    {BestI, BestV}.

%% ---------------------------------------------------------------------------
%% softmax — stable softmax with max-subtraction.
%% ---------------------------------------------------------------------------
softmax(Logits) ->
    Max = lists:max(Logits),
    Exps = [math:exp(L - Max) || L <- Logits],
    Sum = lists:sum(Exps),
    case Sum =< 0.0 of
        true ->
            %% Pathological: uniform fallback (shouldn't happen with finite logits).
            N = length(Logits),
            [1.0 / N || _ <- Logits];
        false ->
            [E / Sum || E <- Exps]
    end.

%% ---------------------------------------------------------------------------
%% sample/2 — multinomial sampling with temperature, top_k, top_p.
%%
%% Opts:
%%   temperature :: float()         (default 1.0; 0 => argmax)
%%   top_k       :: pos_integer() | nil (default nil)
%%   top_p       :: float() | nil    (default nil)
%%   seed        :: integer()       (default erlang:phash2(make_ref()))
%% ---------------------------------------------------------------------------
sample(Logits, Opts) ->
    Temperature = maps:get(temperature, Opts, 1.0),
    TopK        = maps:get(top_k, Opts, nil),
    TopP        = maps:get(top_p, Opts, nil),
    Seed        = maps:get(seed, Opts, erlang:phash2(make_ref())),

    %% Seed RNG deterministically.
    _ = rand:seed(exsss, {Seed, Seed bxor 16#5DEECE66D, Seed bxor 16#B16B00B5}),

    case Temperature =< 0.0 of
        true ->
            %% Temperature 0 => degenerate to argmax.
            {TokenId, _} = argmax(Logits),
            TokenId;
        false ->
            Scaled = [L / Temperature || L <- Logits],
            Probs0 = softmax(Scaled),
            Probs1 = maybe_top_k(Probs0, TopK),
            Probs2 = maybe_top_p(Probs1, TopP),
            multinomial(Probs2)
    end.

%% ---------------------------------------------------------------------------
%% maybe_top_k — keep K highest-prob entries, zero the rest, renormalize.
%% nil or K >= length(Probs) => no-op.
%% ---------------------------------------------------------------------------
maybe_top_k(Probs, nil) -> Probs;
maybe_top_k(Probs, K) when is_integer(K), K > 0 ->
    N = length(Probs),
    case K >= N of
        true  -> Probs;
        false ->
            %% Index-tag, sort desc by prob, take K, build keep-set.
            Indexed = lists:zip(lists:seq(0, N - 1), Probs),
            Sorted = lists:sort(fun({_, A}, {_, B}) -> A > B end, Indexed),
            {KeepTop, _Drop} = lists:split(K, Sorted),
            KeepIdx = lists:foldl(fun({I, _}, Acc) -> Acc#{I => true} end,
                                  #{}, KeepTop),
            Masked = [case maps:is_key(I, KeepIdx) of
                          true -> P;
                          false -> 0.0
                      end || {I, P} <- Indexed],
            renormalize(Masked)
    end.

%% ---------------------------------------------------------------------------
%% maybe_top_p — nucleus sampling. Keep smallest set whose cum prob >= P.
%% nil or P >= 1.0 => no-op.
%% ---------------------------------------------------------------------------
maybe_top_p(Probs, nil) -> Probs;
maybe_top_p(Probs, P) when is_float(P), P >= 1.0 -> Probs;
maybe_top_p(Probs, P) when is_float(P), P > 0.0 ->
    N = length(Probs),
    Indexed = lists:zip(lists:seq(0, N - 1), Probs),
    Sorted = lists:sort(fun({_, A}, {_, B}) -> A > B end, Indexed),
    %% Walk sorted, accumulating cumulative prob; mark idx kept until cum >= P.
    Keep = pick_nucleus(Sorted, P, 0.0, #{}),
    Masked = [case maps:is_key(I, Keep) of
                  true  -> Pr;
                  false -> 0.0
              end || {I, Pr} <- Indexed],
    renormalize(Masked).

pick_nucleus([], _P, _Cum, Keep) -> Keep;
pick_nucleus([{I, Pr} | Rest], P, Cum, Keep) ->
    Keep1 = Keep#{I => true},
    Cum1  = Cum + Pr,
    case Cum1 >= P of
        true  -> Keep1;
        false -> pick_nucleus(Rest, P, Cum1, Keep1)
    end.

%% ---------------------------------------------------------------------------
%% renormalize — divide each prob by sum so they total 1.0 again.
%% ---------------------------------------------------------------------------
renormalize(Probs) ->
    Sum = lists:sum(Probs),
    case Sum =< 0.0 of
        true ->
            N = length(Probs),
            [1.0 / N || _ <- Probs];
        false ->
            [P / Sum || P <- Probs]
    end.

%% ---------------------------------------------------------------------------
%% multinomial — draw one index from a probability distribution.
%% Uses inverse CDF: walk cumulative sum until we exceed U ~ Uniform(0,1).
%% Returns zero-based index.
%% ---------------------------------------------------------------------------
multinomial(Probs) ->
    U = rand:uniform_real(),
    pick_cdf(Probs, U, 0.0, 0, length(Probs) - 1).

pick_cdf([], _U, _Cum, _I, LastI) -> LastI;
pick_cdf([P | Rest], U, Cum, I, LastI) ->
    Cum1 = Cum + P,
    case U =< Cum1 of
        true  -> I;
        false -> pick_cdf(Rest, U, Cum1, I + 1, LastI)
    end.

%% ---------------------------------------------------------------------------
%% test/0 — self-contained correctness check.
%% Run: erl -pa /tmp -noshell -s llama_sampling test -s init stop
%% ---------------------------------------------------------------------------
test() ->
    io:format("=== llama_sampling tests ===~n"),
    Logits = [1.0, 2.0, 3.0, 4.0, 5.0],

    %% Test 1: argmax returns {4, 5.0}.
    {TokA, ValA} = argmax(Logits),
    io:format("  [1] argmax/1 -> {~p, ~p} ", [TokA, ValA]),
    case {TokA, ValA} of
        {4, 5.0} -> io:format("OK~n");
        _        -> io:format("FAIL (expected {4, 5.0})~n"), halt(1)
    end,

    %% Test 2: temperature 0.01 sharpens to argmax (token 4).
    TokB = sample(Logits, #{temperature => 0.01, seed => 1}),
    io:format("  [2] sample temp=0.01 -> ~p ", [TokB]),
    case TokB of
        4 -> io:format("OK~n");
        _ -> io:format("FAIL (expected 4)~n"), halt(1)
    end,

    %% Test 3: same seed => reproducible draw.
    TokC1 = sample(Logits, #{temperature => 100.0, seed => 42}),
    TokC2 = sample(Logits, #{temperature => 100.0, seed => 42}),
    io:format("  [3] sample temp=100 seed=42 -> ~p / ~p ", [TokC1, TokC2]),
    case TokC1 =:= TokC2 of
        true  -> io:format("OK (reproducible)~n");
        false -> io:format("FAIL (non-deterministic)~n"), halt(1)
    end,

    %% Test 4: top_k=1 always returns token 4.
    TokD1 = sample(Logits, #{top_k => 1, seed => 7}),
    TokD2 = sample(Logits, #{top_k => 1, temperature => 50.0, seed => 999}),
    io:format("  [4] sample top_k=1 -> ~p / ~p ", [TokD1, TokD2]),
    case {TokD1, TokD2} of
        {4, 4} -> io:format("OK~n");
        _      -> io:format("FAIL (expected {4, 4})~n"), halt(1)
    end,

    %% Test 5: softmax basic sanity — sums to ~1.0, max idx == 4.
    Probs = softmax(Logits),
    PSum = lists:sum(Probs),
    {MaxIdx, _} = argmax(Probs),
    io:format("  [5] softmax sum=~.6f max_idx=~p ", [PSum, MaxIdx]),
    case (abs(PSum - 1.0) < 1.0e-9) andalso MaxIdx =:= 4 of
        true  -> io:format("OK~n");
        false -> io:format("FAIL~n"), halt(1)
    end,

    %% Test 6: top_p=0.5 with seed sanity — should be deterministic.
    TokE1 = sample(Logits, #{top_p => 0.5, seed => 123}),
    TokE2 = sample(Logits, #{top_p => 0.5, seed => 123}),
    io:format("  [6] sample top_p=0.5 seed=123 -> ~p / ~p ", [TokE1, TokE2]),
    case TokE1 =:= TokE2 of
        true  -> io:format("OK (reproducible)~n");
        false -> io:format("FAIL~n"), halt(1)
    end,

    io:format("=== all tests passed ===~n"),
    ok.
