%%% Multi-block forward through TinyLlama-1.1B using viva_tensor's FP8
%%% linears. Extends the single-block smoke test to iterate over every
%%% transformer layer + final norm + LM head + argmax sampling.
%%%
%%% Scope of this iteration (task #61):
%%%   - Load all 22 transformer layers + final norm + embedding + lm_head.
%%%   - Forward a token through every block sequentially.
%%%   - Apply final RMSNorm and lm_head projection (hidden_size -> vocab).
%%%   - Argmax over the logits to pick the next token.
%%%
%%% Not yet (follow-up tasks):
%%%   - RoPE (task #62): rotary positional embedding for Q/K
%%%   - Real GQA + KV cache (task #63): currently uses the single-token
%%%     attention shortcut (V replicated across Q heads), which is correct
%%%     for the very first forward step but degenerate for longer contexts.
%%%
%%% Run: erlc -o /tmp dev/llama_forward.erl
%%%      erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
%%%          -s llama_forward run -s init stop

-module(llama_forward).
-export([run/0, run_n/1, run_seq/2, build_layer/2,
         precompute_rope_table/3, apply_rope/5]).

-define(PATH, <<"tmp/tinyllama/model.safetensors">>).
-define(HIDDEN, 2048).
-define(KV_DIM, 256).
-define(NUM_HEADS, 32).
-define(NUM_KV_HEADS, 4).
-define(HEAD_DIM, 64).
-define(FFN, 5632).
-define(NUM_LAYERS, 22).
-define(VOCAB, 32000).
-define(EPS, 1.0e-5).
-define(BOS_TOKEN, 1).        %% <s> in TinyLlama tokenizer
-define(ROPE_THETA, 10000.0). %% From config.json

run() -> run_n(?NUM_LAYERS).

%% Sequential multi-token decode: feed N hardcoded tokens through N forward
%% passes, threading the KV cache. Validates that the cache + GQA softmax
%% behave correctly when there's more than one position to attend to.
run_seq(NumLayers, NumTokens) ->
    io:format("~n=== TinyLlama-1.1B sequential decode (N_layers=~p, N_tokens=~p) ===~n~n",
              [NumLayers, NumTokens]),
    T0 = ms(),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layers = build_layers(Header, NumLayers),
    EmbedTbl = load_embed_table(Header),
    FinalNorm = load_rmsnorm(Header, <<"model.norm.weight">>),
    RopeTable = precompute_rope_table(64, ?HEAD_DIM, ?ROPE_THETA),
    io:format("[~5w ms] All ~p layers + helpers loaded~n", [ms() - T0, NumLayers]),

    %% Use BOS, then arbitrary token ids — for now we just want to exercise
    %% the cache; sampling lands in task #64.
    Tokens = [?BOS_TOKEN | lists:seq(100, 100 + NumTokens - 2)],
    EmptyCaches = [{[], []} || _ <- lists:seq(1, NumLayers)],

    {LastHidden, _Caches} = lists:foldl(
        fun({Pos, TokenId}, {_, Caches}) ->
            TF = ms(),
            HiddenIn = embed_row(EmbedTbl, TokenId),
            {HiddenOut, NewCaches} = lists:foldl(
                fun(LayerIdx, {H, CL}) ->
                    {KC, VC} = lists:nth(LayerIdx + 1, CL),
                    {HOut, KC2, VC2} = forward_block(
                        H, lists:nth(LayerIdx + 1, Layers),
                        LayerIdx, Pos, RopeTable, KC, VC),
                    {HOut, lists_replace_nth(LayerIdx + 1, {KC2, VC2}, CL)}
                end,
                {HiddenIn, Caches},
                lists:seq(0, NumLayers - 1)
            ),
            io:format("  pos=~p token=~p  forward=~p ms  hidden mean_abs=~p~n",
                      [Pos, TokenId, ms() - TF, mean_abs(HiddenOut)]),
            {HiddenOut, NewCaches}
        end,
        {[], EmptyCaches},
        lists:zip(lists:seq(0, length(Tokens) - 1), Tokens)
    ),

    NormFinal = rmsnorm(LastHidden, FinalNorm, ?EPS),
    io:format("~nFinal normed hidden  mean_abs=~p  finite=~p~n",
              [mean_abs(NormFinal),
               lists:foldl(fun(X, A) -> case is_float(X) of true -> A + 1; false -> A end end, 0, NormFinal)]),
    io:format("[~5w ms] Total~n", [ms() - T0]),
    ok.

%% Same as run/0 but with a configurable number of layers — handy for
%% incremental validation (start with 2 layers, then go up to 22).
run_n(NumLayers) ->
    io:format("~n=== TinyLlama-1.1B multi-block forward (N=~p) ===~n~n", [NumLayers]),
    T0 = ms(),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    io:format("[~5w ms] Header opened~n", [ms() - T0]),

    %% --- Build the requested number of layers (load + transpose + prepack). --
    TL0 = ms(),
    Layers = build_layers(Header, NumLayers),
    io:format("[~5w ms] All ~p layers prepacked~n", [ms() - TL0, NumLayers]),

    %% --- Final RMSNorm + LM head + embedding table. ----------------------
    TF0 = ms(),
    FinalNorm = load_rmsnorm(Header, <<"model.norm.weight">>),
    EmbedTbl  = load_embed_table(Header),     %% lazy: index -> row
    LmHead    = load_linear(Header, <<"lm_head.weight">>, ?VOCAB, ?HIDDEN),
    LmHeadPk  = prepack(LmHead, ?HIDDEN, ?VOCAB),
    RopeTable = precompute_rope_table(64, ?HEAD_DIM, ?ROPE_THETA),
    io:format("[~5w ms] Final norm + lm_head prepacked + embed + RoPE table~n",
              [ms() - TF0]),

    %% --- Initial hidden state: embedding row for BOS token. --------------
    Token = ?BOS_TOKEN,
    HiddenIn = embed_row(EmbedTbl, Token),
    io:format("~nToken in: ~p   embedding mean_abs=~p~n",
              [Token, mean_abs(HiddenIn)]),

    %% --- Forward through all transformer blocks. -------------------------
    TF1 = ms(),
    Pos = 0,    %% single-token forward: position 0
    %% Per-layer KV caches start empty; each layer accumulates its own.
    EmptyCaches = [{[], []} || _ <- lists:seq(1, NumLayers)],
    {HiddenOut, _FinalCaches} = lists:foldl(
        fun(LayerIdx, {H, Caches}) ->
            {KC, VC} = lists:nth(LayerIdx + 1, Caches),
            {HOut, KC2, VC2} = forward_block(
                H, lists:nth(LayerIdx + 1, Layers),
                LayerIdx, Pos, RopeTable, KC, VC),
            UpdatedCaches = lists_replace_nth(LayerIdx + 1, {KC2, VC2}, Caches),
            {HOut, UpdatedCaches}
        end,
        {HiddenIn, EmptyCaches},
        lists:seq(0, NumLayers - 1)
    ),
    io:format("[~5w ms] Forward over ~p blocks done. Final mean_abs=~p~n",
              [ms() - TF1, NumLayers, mean_abs(HiddenOut)]),

    %% --- Final norm + LM head + argmax sampling. ------------------------
    NormFinal = rmsnorm(HiddenOut, FinalNorm, ?EPS),
    NormFp16 = floats_to_fp16(NormFinal),
    Logits = linear_fp8(NormFp16, LmHeadPk, ?VOCAB),
    {Top1Token, Top1Logit} = argmax(Logits),
    io:format("~nFinal normed hidden mean_abs=~p~n", [mean_abs(NormFinal)]),
    io:format("Logits mean_abs=~p~n", [mean_abs(Logits)]),
    io:format("argmax token id = ~p (logit ~p)~n", [Top1Token, Top1Logit]),
    io:format("~n[~5w ms] Total elapsed~n", [ms() - T0]),
    ok.

%% ---------------------------------------------------------------------------
%% Layer construction
%% ---------------------------------------------------------------------------

%% Returns a list of #{...} layer maps, one per transformer block.
build_layers(Header, N) ->
    [build_layer(Header, I) || I <- lists:seq(0, N - 1)].

build_layer(Header, LayerIdx) ->
    T0 = ms(),
    Prefix = "model.layers." ++ integer_to_list(LayerIdx) ++ ".",
    P = fun(Suffix) -> list_to_binary(Prefix ++ Suffix) end,

    QProj    = load_linear(Header, P("self_attn.q_proj.weight"), ?HIDDEN, ?HIDDEN),
    KProj    = load_linear(Header, P("self_attn.k_proj.weight"), ?KV_DIM, ?HIDDEN),
    VProj    = load_linear(Header, P("self_attn.v_proj.weight"), ?KV_DIM, ?HIDDEN),
    OProj    = load_linear(Header, P("self_attn.o_proj.weight"), ?HIDDEN, ?HIDDEN),
    GateProj = load_linear(Header, P("mlp.gate_proj.weight"),    ?FFN,    ?HIDDEN),
    UpProj   = load_linear(Header, P("mlp.up_proj.weight"),      ?FFN,    ?HIDDEN),
    DownProj = load_linear(Header, P("mlp.down_proj.weight"),    ?HIDDEN, ?FFN),

    Layer = #{
        norm1 => load_rmsnorm(Header, P("input_layernorm.weight")),
        norm2 => load_rmsnorm(Header, P("post_attention_layernorm.weight")),
        q     => prepack(QProj,    ?HIDDEN, ?HIDDEN),
        k     => prepack(KProj,    ?HIDDEN, ?KV_DIM),
        v     => prepack(VProj,    ?HIDDEN, ?KV_DIM),
        o     => prepack(OProj,    ?HIDDEN, ?HIDDEN),
        gate  => prepack(GateProj, ?HIDDEN, ?FFN),
        up    => prepack(UpProj,   ?HIDDEN, ?FFN),
        down  => prepack(DownProj, ?FFN,    ?HIDDEN)
    },
    io:format("[~5w ms]   layer ~2.. p loaded~n", [ms() - T0, LayerIdx]),
    Layer.

%% ---------------------------------------------------------------------------
%% Single transformer block forward
%% ---------------------------------------------------------------------------
%% Input  : H = list(float)  [hidden_size]
%% Output : H' = list(float) [hidden_size]
%%
%% Steps (HuggingFace Llama convention):
%%   x1 = norm1(x)
%%   q, k, v = linear(x1, q_proj), linear(x1, k_proj), linear(x1, v_proj)
%%   attn_out = attention(q, k, v)        -- single-token shortcut here
%%   x = x + linear(attn_out, o_proj)
%%   x2 = norm2(x)
%%   ffn = down(silu(gate(x2)) * up(x2))
%%   x = x + ffn
forward_block(H, Layer, _LayerIdx, Pos, RopeTable, KCache, VCache) ->
    #{norm1 := N1, norm2 := N2,
      q := Q, k := K, v := V, o := O,
      gate := G, up := U, down := D} = Layer,

    %% Self-attention path
    X1 = rmsnorm(H, N1, ?EPS),
    X1Fp16 = floats_to_fp16(X1),
    QOutRaw = linear_fp8(X1Fp16, Q, ?HIDDEN),
    KOutRaw = linear_fp8(X1Fp16, K, ?KV_DIM),
    VOut    = linear_fp8(X1Fp16, V, ?KV_DIM),

    %% Apply RoPE to Q and K (NOT V) for positional encoding.
    QOut = apply_rope(QOutRaw, Pos, RopeTable, ?NUM_HEADS,    ?HEAD_DIM),
    KOut = apply_rope(KOutRaw, Pos, RopeTable, ?NUM_KV_HEADS, ?HEAD_DIM),

    %% GQA attention with KV cache (real softmax over all positions).
    {AttnOut, KCache2, VCache2} = attention_gqa(QOut, KOut, VOut, KCache, VCache),

    AttnOutFp16 = floats_to_fp16(AttnOut),
    OOut = linear_fp8(AttnOutFp16, O, ?HIDDEN),

    %% Residual 1
    H1 = list_add(H, OOut),

    %% FFN path
    X2 = rmsnorm(H1, N2, ?EPS),
    SwInter = linear_swiglu_intermediate(X2, 1, ?HIDDEN, G, U, ?FFN),
    SwInterFp16 = floats_to_fp16(SwInter),
    Ffn = linear_fp8(SwInterFp16, D, ?HIDDEN),

    %% Residual 2
    HOut = list_add(H1, Ffn),
    {HOut, KCache2, VCache2}.

%% ---------------------------------------------------------------------------
%% RoPE (rotary positional embedding)
%% ---------------------------------------------------------------------------
%% Llama applies RoPE to Q and K (NOT V) before attention. Per head_dim/2 pair
%% (i, i + head_dim/2) the value is rotated by angle = pos * theta_i where
%%   theta_i = rope_theta ** (-2i/head_dim) for i in [0, head_dim/2)
%%
%% precompute_rope_table(MaxPos) returns a tuple of MaxPos rotation tables,
%% each table being a list of {cos, sin} pairs of length head_dim/2.
%% We index by position to read once per forward.

precompute_rope_freqs(HeadDim, Theta) ->
    Half = HeadDim div 2,
    [math:pow(Theta, -2.0 * float(I) / float(HeadDim))
     || I <- lists:seq(0, Half - 1)].

precompute_rope_table(MaxPos, HeadDim, Theta) ->
    Freqs = precompute_rope_freqs(HeadDim, Theta),
    list_to_tuple(
        [[ {math:cos(float(Pos) * F), math:sin(float(Pos) * F)} || F <- Freqs ]
         || Pos <- lists:seq(0, MaxPos - 1)]
    ).

%% Apply RoPE to a flat list of [num_heads * head_dim] floats at the given
%% absolute position. Returns the rotated flat list with the same shape.
apply_rope(Flat, Pos, RopeTable, NumHeads, HeadDim) ->
    RotationsAtPos = element(Pos + 1, RopeTable),    %% list of {cos, sin}
    Heads = chunks(Flat, HeadDim),
    Rotated = [rotate_head(H, RotationsAtPos, HeadDim) || H <- Heads],
    _Count = NumHeads,
    lists:flatten(Rotated).

rotate_head(HeadVec, Rotations, HeadDim) ->
    Half = HeadDim div 2,
    HeadTup = list_to_tuple(HeadVec),
    [rotate_at(HeadTup, I, Half, Rotations) || I <- lists:seq(0, HeadDim - 1)].

rotate_at(HeadTup, I, Half, Rotations) ->
    case I < Half of
        true ->
            X1 = element(I + 1, HeadTup),
            X2 = element(I + Half + 1, HeadTup),
            {C, S} = lists:nth(I + 1, Rotations),
            X1 * C - X2 * S;
        false ->
            J = I - Half,
            X1 = element(J + 1, HeadTup),
            X2 = element(I + 1, HeadTup),
            {C, S} = lists:nth(J + 1, Rotations),
            X1 * S + X2 * C
    end.

%% ---------------------------------------------------------------------------
%% GQA attention with KV cache.
%%
%% Inputs:
%%   QFlat  : [num_heads * head_dim]      Q at current position (RoPE-applied)
%%   KFlat  : [num_kv_heads * head_dim]   K at current position (RoPE-applied)
%%   VFlat  : [num_kv_heads * head_dim]   V at current position
%%   KCache : list of K vectors, oldest first (positions 0..pos-1)
%%   VCache : list of V vectors, oldest first
%% Outputs:
%%   AttnOut    : [num_heads * head_dim] flat list
%%   NewKCache  : KCache ++ [KFlat]
%%   NewVCache  : VCache ++ [VFlat]
attention_gqa(QFlat, KFlat, VFlat, KCache, VCache) ->
    NewKCache = KCache ++ [KFlat],
    NewVCache = VCache ++ [VFlat],

    %% Each cached K/V is split into NumKVHeads chunks of HeadDim. Cache as
    %% tuple of tuples for O(1) head lookup.
    %% Layout: KByHeadByPos[kv_head_idx] = list of head-vectors across positions
    KByHead = transpose_cache(NewKCache, ?NUM_KV_HEADS, ?HEAD_DIM),
    VByHead = transpose_cache(NewVCache, ?NUM_KV_HEADS, ?HEAD_DIM),

    QHeads = chunks(QFlat, ?HEAD_DIM),
    Scale  = 1.0 / math:sqrt(float(?HEAD_DIM)),
    QPerKV = ?NUM_HEADS div ?NUM_KV_HEADS,    %% = 8

    HeadOuts = [
        begin
            KvIdx = QHeadIdx div QPerKV,
            Khead = lists:nth(KvIdx + 1, KByHead),    %% list of pos-len head_dim
            Vhead = lists:nth(KvIdx + 1, VByHead),
            QHead = lists:nth(QHeadIdx + 1, QHeads),
            attn_one_head(QHead, Khead, Vhead, Scale)
        end
        || QHeadIdx <- lists:seq(0, ?NUM_HEADS - 1)
    ],

    AttnOut = lists:flatten(HeadOuts),
    {AttnOut, NewKCache, NewVCache}.

%% transpose_cache([Flat]) -> [[HeadVecPos0, HeadVecPos1, ...] per head_idx]
%% Each Flat is [NumHeads * HeadDim]. Output is NumHeads lists, each of
%% length len(Cache), each element is HeadDim floats.
transpose_cache(Cache, NumHeads, HeadDim) ->
    %% First chunk each cached vector into per-head pieces.
    PerPosHeads = [chunks(Flat, HeadDim) || Flat <- Cache],
    %% Then pivot: HeadByPos[h] = [PerPosHeads[0][h], PerPosHeads[1][h], ...]
    [
        [lists:nth(H + 1, PerPos) || PerPos <- PerPosHeads]
        || H <- lists:seq(0, NumHeads - 1)
    ].

%% Single Q head against (Khead, Vhead) lists where Khead is a list of
%% per-position head_dim vectors (similarly Vhead). Returns a head_dim vector.
attn_one_head(QVec, Khead, Vhead, Scale) ->
    %% Scaled dot product scores
    Scores = [dot(QVec, K) * Scale || K <- Khead],
    Weights = softmax(Scores),
    %% Weighted sum of V vectors
    weighted_sum(Weights, Vhead).

dot(A, B) ->
    lists:foldl(
        fun({X, Y}, Acc) -> Acc + X * Y end,
        0.0,
        lists:zip(A, B)
    ).

softmax(Xs) ->
    M = lists:foldl(fun(X, Mx) -> max(X, Mx) end, -1.0e308, Xs),
    Exps = [math:exp(X - M) || X <- Xs],
    S = lists:sum(Exps),
    [E / S || E <- Exps].

%% weighted_sum([w0, w1, ...], [V0, V1, ...]) -> sum_t w_t * V_t
weighted_sum(Weights, Vectors) ->
    HeadDim = length(hd(Vectors)),
    Init = lists:duplicate(HeadDim, 0.0),
    lists:foldl(
        fun({W, V}, Acc) ->
            lists:zipwith(fun(A, Vi) -> A + W * Vi end, Acc, V)
        end,
        Init,
        lists:zip(Weights, Vectors)
    ).

chunks([], _) -> [];
chunks(L, N) ->
    {Head, Tail} = lists:split(N, L),
    [Head | chunks(Tail, N)].

%% ---------------------------------------------------------------------------
%% Loaders
%% ---------------------------------------------------------------------------

load_linear(Header, Name, OutF, InF) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
    Fp32 = viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16),
    %% HF stores [out, in] row-major. viva_tensor prepack expects
    %% [in, out] row-major — transpose.
    {ok, Trans} = viva_tensor_safetensors_ffi:transpose_fp32(Fp32, OutF, InF),
    Trans.

load_rmsnorm(Header, Name) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
    viva_tensor_safetensors_ffi:rmsnorm_weight_to_fp32_list(Bf16).

%% Embedding table loaded lazily: stash the bf16 binary + meta, look up
%% row by index on demand to avoid materializing all 32000×2048 floats.
load_embed_table(Header) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header,
        <<"model.embed_tokens.weight">>),
    {Bf16, ?HIDDEN}.

embed_row({Bf16, RowLen}, TokenId) ->
    ByteOff = TokenId * RowLen * 2,    %% bf16 = 2 bytes/elem
    RowBytes = binary:part(Bf16, ByteOff, RowLen * 2),
    viva_tensor_safetensors_ffi:rmsnorm_weight_to_fp32_list(RowBytes).

prepack(Bin, InF, OutF) when is_binary(Bin) ->
    case viva_tensor_zig:nt_prepack_fp8(Bin, [InF, OutF]) of
        {ok, {Resource, _, _, _}} -> Resource;
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({prepack_failed, Other})
    end.

%% ---------------------------------------------------------------------------
%% Linear NIF wrappers
%% ---------------------------------------------------------------------------

linear_fp8(InputFp16, Packed, OutF) when is_binary(InputFp16) ->
    case viva_tensor_zig:nt_linear_fp8(InputFp16, Packed, nil, 0) of
        {ok, OutBin} ->
            Vals = viva_tensor_inference_ffi:fp16_binary_to_floats(OutBin),
            verify_size(Vals, OutF),
            Vals;
        Error ->
            error({linear_fp8_failed, Error})
    end.

linear_swiglu_intermediate(InputFp32List, B, InF, Gate, Up, FfnDim) ->
    case viva_tensor_zig:nt_linear_swiglu_fp8(InputFp32List, [B, InF], Gate, Up, nil) of
        {ok, OutList} when is_list(OutList) ->
            verify_size(OutList, B * FfnDim),
            OutList;
        Error ->
            error({swiglu_failed, Error})
    end.

verify_size(L, Expected) ->
    case length(L) of
        Expected -> ok;
        N -> error({size_mismatch, got, N, expected, Expected})
    end.

%% ---------------------------------------------------------------------------
%% Math helpers
%% ---------------------------------------------------------------------------

rmsnorm(X, Gamma, Eps) ->
    SumSq = lists:foldl(fun(V, A) -> A + V * V end, 0.0, X),
    N = length(X),
    InvRms = 1.0 / math:sqrt(SumSq / N + Eps),
    lists:zipwith(fun(V, Gv) -> V * InvRms * Gv end, X, Gamma).

list_add(A, B) ->
    lists:zipwith(fun(X, Y) -> X + Y end, A, B).

mean_abs(L) ->
    case L of
        [] -> 0.0;
        _ ->
            {Sum, _Inf} = lists:foldl(
                fun(X, {S, Inf}) ->
                    A = abs(X),
                    case A > 1.0e30 of
                        true  -> {S, Inf + 1};
                        false -> {S + A, Inf}
                    end
                end, {0.0, 0}, L),
            Sum / length(L)
    end.

floats_to_fp16(L) when is_list(L) ->
    viva_tensor_inference_ffi:floats_to_fp16_binary(L).

%% Returns {index_zero_based, value} of the max-value entry.
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

ms() -> erlang:monotonic_time(millisecond).

%% lists:nth-style 1-based replacement.
lists_replace_nth(N, NewVal, List) ->
    {Before, [_Old | After]} = lists:split(N - 1, List),
    Before ++ [NewVal | After].
