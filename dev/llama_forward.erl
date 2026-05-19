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
-export([run/0, run_n/1, run_seq/2, run_generate/4, bisect/0, bisect_w8a16/0,
          bisect_calibrated/0, run_n_calibrated/1, bisect_batch16/0,
          build_layer/2, build_layer_calibrated/3,
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

%% Bisect: walk layer 0 stage-by-stage on the BOS token, printing the
%% same mean_abs + first-5 values that dev/hf_bisect.py prints for the
%% HuggingFace fp32 reference. Compare side-by-side to find where the
%% two pipelines first diverge.
bisect() ->
    io:format("~n=== viva_tensor bisect — layer 0, BOS (pos=0) ===~n~n"),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layer = build_layer(Header, 0),
    EmbedTbl = load_embed_table(Header),
    RopeTable = precompute_rope_table(16, ?HEAD_DIM, ?ROPE_THETA),

    %% 0: embedding
    X = embed_row(EmbedTbl, ?BOS_TOKEN),
    dump("embed[BOS]", X),

    %% Need access to the *raw* RMSNorm weights (not just packed). Reload.
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),

    %% 1: input_layernorm
    XNorm1 = rmsnorm(X, Norm1, ?EPS),
    dump("after input_layernorm", XNorm1),

    #{q := Q, k := K, v := V, o := O,
      gate := G, up := U, down := D} = Layer,

    %% 2: Q/K/V projections (raw, pre-RoPE)
    XNorm1Fp16 = floats_to_fp16(XNorm1),
    QRaw = linear_fp8(XNorm1Fp16, Q, ?HIDDEN),
    KRaw = linear_fp8(XNorm1Fp16, K, ?KV_DIM),
    VRaw = linear_fp8(XNorm1Fp16, V, ?KV_DIM),
    dump("Q proj raw", QRaw),
    dump("K proj raw", KRaw),
    dump("V proj raw", VRaw),

    %% 3: RoPE at pos=0
    QRot = apply_rope(QRaw, 0, RopeTable, ?NUM_HEADS, ?HEAD_DIM),
    KRot = apply_rope(KRaw, 0, RopeTable, ?NUM_KV_HEADS, ?HEAD_DIM),
    dump("Q after RoPE", QRot),
    dump("K after RoPE", KRot),

    %% 4: single-token attention with KV cache empty (just appends and reads back)
    {AttnOut, _, _} = attention_gqa(QRot, KRot, VRaw, [], []),
    dump("attention output", AttnOut),

    %% 5: O proj + residual 1
    AttnOutFp16 = floats_to_fp16(AttnOut),
    OOut = linear_fp8(AttnOutFp16, O, ?HIDDEN),
    dump("O proj", OOut),
    H1 = list_add(X, OOut),
    dump("residual 1", H1),

    %% 6: post_attention_layernorm
    XNorm2 = rmsnorm(H1, Norm2, ?EPS),
    dump("after post_attention_layernorm", XNorm2),

    %% 7: SwiGLU FFN — we use the fused NIF which returns silu(gate)*up.
    %% Dump it directly; the agent's path is gate+up+silu*mul internally.
    SwInter = linear_swiglu_intermediate(XNorm2, 1, ?HIDDEN, G, U, ?FFN),
    dump("silu(gate) * up (fused)", SwInter),

    SwInterFp16 = floats_to_fp16(SwInter),
    Ffn = linear_fp8(SwInterFp16, D, ?HIDDEN),
    dump("down proj (FFN out)", Ffn),

    H2 = list_add(H1, Ffn),
    dump("residual 2 (block 0 hidden)", H2),
    ok.

bisect_w8a16() ->
    io:format("~n=== viva_tensor W8A16 bisect — layer 0, BOS (pos=0) ===~n~n"),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layer = build_layer(Header, 0),
    EmbedTbl = load_embed_table(Header),
    RopeTable = precompute_rope_table(16, ?HEAD_DIM, ?ROPE_THETA),

    X = embed_row(EmbedTbl, ?BOS_TOKEN),
    dump("embed[BOS]", X),

    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),
    XNorm1 = rmsnorm(X, Norm1, ?EPS),
    dump("after input_layernorm", XNorm1),

    #{q := Q, k := K, v := V, o := O,
      gate := G, up := U, down := D} = Layer,

    XNorm1Fp16 = floats_to_fp16(XNorm1),
    QRaw = linear_fp8_w8a16(XNorm1Fp16, Q, ?HIDDEN),
    KRaw = linear_fp8_w8a16(XNorm1Fp16, K, ?KV_DIM),
    VRaw = linear_fp8_w8a16(XNorm1Fp16, V, ?KV_DIM),
    dump("Q proj raw", QRaw),
    dump("K proj raw", KRaw),
    dump("V proj raw", VRaw),

    QRot = apply_rope(QRaw, 0, RopeTable, ?NUM_HEADS, ?HEAD_DIM),
    KRot = apply_rope(KRaw, 0, RopeTable, ?NUM_KV_HEADS, ?HEAD_DIM),
    dump("Q after RoPE", QRot),
    dump("K after RoPE", KRot),
    {AttnOut, _, _} = attention_gqa(QRot, KRot, VRaw, [], []),
    dump("attention output", AttnOut),

    AttnOutFp16 = floats_to_fp16(AttnOut),
    OOut = linear_fp8_w8a16(AttnOutFp16, O, ?HIDDEN),
    dump("O proj", OOut),
    H1 = list_add(X, OOut),
    dump("residual 1", H1),

    XNorm2 = rmsnorm(H1, Norm2, ?EPS),
    dump("after post_attention_layernorm", XNorm2),
    SwInter = linear_swiglu_intermediate(XNorm2, 1, ?HIDDEN, G, U, ?FFN),
    dump("silu(gate) * up (fused)", SwInter),

    SwInterFp16 = floats_to_fp16(SwInter),
    Ffn = linear_fp8_w8a16(SwInterFp16, D, ?HIDDEN),
    dump("down proj (FFN out)", Ffn),
    H2 = list_add(H1, Ffn),
    dump("residual 2 (block 0 hidden)", H2),
    ok.

%% Same setup as bisect/0 but run Q proj with batch=16 (replicate XNorm1).
%% If zeros vanish at M=16, the CUTLASS M=1 padding path has a bug.
%% If zeros remain at M=16, it's accumulator numerical cancellation.
bisect_batch16() ->
    io:format("~n=== viva_tensor bisect batch=16 — layer 0, BOS ===~n~n"),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layer = build_layer(Header, 0),
    EmbedTbl = load_embed_table(Header),
    X = embed_row(EmbedTbl, ?BOS_TOKEN),
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    XNorm1 = rmsnorm(X, Norm1, ?EPS),
    Replicated = lists:flatten(lists:duplicate(16, XNorm1)),
    InBin = viva_tensor_inference_ffi:floats_to_fp16_binary(Replicated),
    #{q := Q} = Layer,
    {ok, QBin} = viva_tensor_zig:nt_linear_fp8(InBin, Q, nil, 0),
    QOut = viva_tensor_inference_ffi:fp16_binary_to_floats(QBin),
    Row0 = lists:sublist(QOut, ?HIDDEN),
    Row1 = lists:sublist(QOut, ?HIDDEN + 1, ?HIDDEN),
    Row15 = lists:sublist(QOut, 15 * ?HIDDEN + 1, ?HIDDEN),
    Z0 = length([X1 || X1 <- Row0, X1 == 0.0]),
    Z1 = length([X1 || X1 <- Row1, X1 == 0.0]),
    Z15 = length([X1 || X1 <- Row15, X1 == 0.0]),
    io:format("Row 0 zeros=~p/2048  first5=~p~n", [Z0, lists:sublist(Row0, 5)]),
    io:format("Row 1 zeros=~p/2048  first5=~p~n", [Z1, lists:sublist(Row1, 5)]),
    io:format("Row 15 zeros=~p/2048 first5=~p~n", [Z15, lists:sublist(Row15, 5)]),
    %% Compare row 0 vs row 1 — same input, should give identical output.
    Diff = lists:any(fun({A, B}) -> abs(A - B) > 1.0e-6 end, lists:zip(Row0, Row1)),
    io:format("Row 0 differs from Row 1? ~p~n", [Diff]),
    ok.

dump(Label, L) ->
    First5 = [io_lib:format("~.6e", [X]) || X <- lists:sublist(L, 5)],
    Zeros = length([X || X <- L, X == 0.0]),
    io:format("  ~-40s mean_abs=~.6f  zeros=~p/~p  [~s, ...]~n",
              [Label, mean_abs(L), Zeros, length(L),
               string:join(First5, ", ")]).

%% End-to-end text generation: encode prompt → forward each token, threading
%% the KV cache → sample next token via dev/llama_sampling → decode.
%%
%% Args:
%%   NumLayers : 1..22 — how many transformer blocks to use (22 = full model).
%%   Prompt    : binary or string — text to feed in.
%%   MaxNew    : integer — number of tokens to generate after the prompt.
%%   SampOpts  : map — passed as-is to llama_sampling:sample/2 (e.g.
%%               #{temperature => 0.8, top_k => 40}). Use #{} for argmax-style.
run_generate(NumLayers, Prompt, MaxNew, SampOpts) ->
    io:format("~n=== TinyLlama-1.1B text generation (N_layers=~p) ===~n~n",
              [NumLayers]),
    T0 = ms(),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    {ok, Tokenizer} = viva_tensor_tokenizer_ffi:load(
        <<"tmp/tinyllama/tokenizer.json">>),
    Layers = build_layers(Header, NumLayers),
    EmbedTbl = load_embed_table(Header),
    FinalNorm = load_rmsnorm(Header, <<"model.norm.weight">>),
    LmHead    = load_linear(Header, <<"lm_head.weight">>, ?VOCAB, ?HIDDEN),
    LmHeadPk  = prepack(LmHead, ?HIDDEN, ?VOCAB),
    RopeTable = precompute_rope_table(256, ?HEAD_DIM, ?ROPE_THETA),
    io:format("[~5w ms] Model + tokenizer ready~n", [ms() - T0]),

    %% Encode the prompt; prepend BOS (Llama convention).
    BOS = viva_tensor_tokenizer_ffi:bos_id(Tokenizer),
    EOS = viva_tensor_tokenizer_ffi:eos_id(Tokenizer),
    PromptTokens = [BOS | viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt)],
    io:format("Prompt: ~ts~n", [Prompt]),
    io:format("Encoded ~p tokens: ~p~n", [length(PromptTokens), PromptTokens]),

    %% Initial pass: process the prompt tokens to fill the KV cache.
    EmptyCaches = [{[], []} || _ <- lists:seq(1, NumLayers)],
    TPrompt = ms(),
    {LastHidden, Caches} = prefill(
        PromptTokens, Layers, EmbedTbl, RopeTable, EmptyCaches, NumLayers),
    io:format("[~5w ms] Prefill of ~p prompt tokens done~n",
              [ms() - TPrompt, length(PromptTokens)]),

    %% Iteratively decode MaxNew tokens.
    TGen = ms(),
    GeneratedIds = decode_loop(LastHidden, Caches, Layers,
                                EmbedTbl, FinalNorm, LmHeadPk,
                                RopeTable, NumLayers,
                                length(PromptTokens), MaxNew, SampOpts,
                                EOS, []),
    io:format("[~5w ms] Generated ~p tokens~n",
              [ms() - TGen, length(GeneratedIds)]),

    %% Decode back to text.
    AllText = viva_tensor_tokenizer_ffi:decode(Tokenizer, GeneratedIds),
    io:format("~n--- generated ---~n~ts~n--- end ---~n", [AllText]),
    io:format("[~5w ms] Total~n", [ms() - T0]),
    {ok, GeneratedIds, AllText}.

%% Run the prompt through the model to populate KV caches. Returns the
%% hidden state after the LAST prompt token + the populated caches.
prefill(Tokens, Layers, EmbedTbl, RopeTable, Caches, NumLayers) ->
    lists:foldl(
        fun({Pos, TokenId}, {_, CL}) ->
            HiddenIn = embed_row(EmbedTbl, TokenId),
            forward_all_blocks(HiddenIn, Layers, RopeTable,
                               CL, NumLayers, Pos)
        end,
        {[], Caches},
        lists:zip(lists:seq(0, length(Tokens) - 1), Tokens)
    ).

forward_all_blocks(HiddenIn, Layers, RopeTable, Caches, NumLayers, Pos) ->
    lists:foldl(
        fun(LayerIdx, {H, CL}) ->
            {KC, VC} = lists:nth(LayerIdx + 1, CL),
            {HOut, KC2, VC2} = forward_block(
                H, lists:nth(LayerIdx + 1, Layers),
                LayerIdx, Pos, RopeTable, KC, VC),
            {HOut, lists_replace_nth(LayerIdx + 1, {KC2, VC2}, CL)}
        end,
        {HiddenIn, Caches},
        lists:seq(0, NumLayers - 1)
    ).

decode_loop(_Hidden, _Caches, _Layers, _EmbedTbl, _FinalNorm, _LmHeadPk,
            _RopeTable, _NumLayers, _Pos, 0, _SampOpts, _EOS, Acc) ->
    lists:reverse(Acc);
decode_loop(Hidden, Caches, Layers, EmbedTbl, FinalNorm, LmHeadPk,
            RopeTable, NumLayers, Pos, Remaining, SampOpts, EOS, Acc) ->
    NormFinal = rmsnorm(Hidden, FinalNorm, ?EPS),
    NormFp16  = floats_to_fp16(NormFinal),
    Logits    = linear_fp8(NormFp16, LmHeadPk, ?VOCAB),
    NextTok   = case maps:size(SampOpts) of
        0 -> {Id, _} = argmax(Logits), Id;
        _ -> llama_sampling:sample(Logits, SampOpts)
    end,
    io:format("  pos=~p -> token ~p~n", [Pos, NextTok]),
    case NextTok of
        EOS -> lists:reverse([NextTok | Acc]);
        _ ->
            NextHidden = embed_row(EmbedTbl, NextTok),
            {HiddenOut, NewCaches} = forward_all_blocks(
                NextHidden, Layers, RopeTable, Caches, NumLayers, Pos),
            decode_loop(HiddenOut, NewCaches, Layers, EmbedTbl, FinalNorm,
                        LmHeadPk, RopeTable, NumLayers, Pos + 1,
                        Remaining - 1, SampOpts, EOS, [NextTok | Acc])
    end.

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

linear_fp8_w8a16(InputFp16, Packed, OutF) when is_binary(InputFp16) ->
    case viva_tensor_zig:nt_linear_fp8_w8a16(InputFp16, Packed, nil) of
        {ok, OutBin} ->
            Vals = viva_tensor_inference_ffi:fp16_binary_to_floats(OutBin),
            verify_size(Vals, OutF),
            Vals;
        Error ->
            error({linear_fp8_w8a16_failed, Error})
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

%% ---------------------------------------------------------------------------
%% SmoothQuant calibration (host-side, pure Erlang)
%% ---------------------------------------------------------------------------
%%
%% Walks ~10 synthetic calibration tokens through layer 0's pre-projection
%% stages (input_layernorm output -> Q/K/V; attn output -> O; post-attn
%% layernorm output -> gate/up; silu*up -> down), accumulating per-input-
%% channel max-abs vectors. Feeds llama_calibration:smoothquant_scale/5 to
%% produce per-input-channel scales s[c] for each layer-0 linear.
%%
%% Weights are scaled by s before prepack. At forward time, inputs are
%% divided by s (multiplied by 1/s) before linear_fp8 so the matmul stays
%% mathematically identical to the unscaled version.

-define(CAL_TOKEN_IDS, [1, 2, 3, 5, 10, 25, 50, 100, 200, 400]).

%% Load layer N raw FP32 weights ([InFeatures, OutFeatures] row-major).
load_layer_raw(Header, LayerIdx) ->
    Prefix = "model.layers." ++ integer_to_list(LayerIdx) ++ ".",
    P = fun(Suffix) -> list_to_binary(Prefix ++ Suffix) end,
    #{
        q_raw    => load_linear(Header, P("self_attn.q_proj.weight"), ?HIDDEN, ?HIDDEN),
        k_raw    => load_linear(Header, P("self_attn.k_proj.weight"), ?KV_DIM, ?HIDDEN),
        v_raw    => load_linear(Header, P("self_attn.v_proj.weight"), ?KV_DIM, ?HIDDEN),
        o_raw    => load_linear(Header, P("self_attn.o_proj.weight"), ?HIDDEN, ?HIDDEN),
        gate_raw => load_linear(Header, P("mlp.gate_proj.weight"),    ?FFN,    ?HIDDEN),
        up_raw   => load_linear(Header, P("mlp.up_proj.weight"),      ?FFN,    ?HIDDEN),
        down_raw => load_linear(Header, P("mlp.down_proj.weight"),    ?HIDDEN, ?FFN)
    }.

%% Collect per-input-channel max-abs activations for each linear's input,
%% across the CAL_TOKEN_IDS set, using the *unscaled* layer 0 weights.
collect_calibration_stats(Header, Layer0Raw) ->
    EmbedTbl = load_embed_table(Header),
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),
    #{q_raw := QRaw, k_raw := KRaw, v_raw := VRaw,
      gate_raw := GRaw, up_raw := URaw} = Layer0Raw,
    QPk = prepack(QRaw, ?HIDDEN, ?HIDDEN),
    KPk = prepack(KRaw, ?HIDDEN, ?KV_DIM),
    VPk = prepack(VRaw, ?HIDDEN, ?KV_DIM),
    GPk = prepack(GRaw, ?HIDDEN, ?FFN),
    UPk = prepack(URaw, ?HIDDEN, ?FFN),

    Init1 = lists:duplicate(?HIDDEN, 0.0),
    Init2 = lists:duplicate(?HIDDEN, 0.0),
    Init3 = lists:duplicate(?HIDDEN, 0.0),
    Init4 = lists:duplicate(?FFN, 0.0),
    lists:foldl(
        fun(TokId, {M1, M2, M3, M4}) ->
            X = embed_row(EmbedTbl, TokId),
            XN1 = rmsnorm(X, Norm1, ?EPS),
            M1New = list_max_abs_elem(M1, XN1),
            X1Fp16 = floats_to_fp16(XN1),
            %% Q/K not needed for stats; V drives attn output stats.
            VOut = linear_fp8(X1Fp16, VPk, ?KV_DIM),
            _ = linear_fp8(X1Fp16, QPk, ?HIDDEN),
            _ = linear_fp8(X1Fp16, KPk, ?KV_DIM),
            AttnOut = expand_v_to_q_heads(VOut),
            M2New = list_max_abs_elem(M2, AttnOut),
            %% For Norm2 input we'd need real attention out + residual.
            %% Approximation: use X (residual without O contribution).
            H1 = X,
            XN2 = rmsnorm(H1, Norm2, ?EPS),
            M3New = list_max_abs_elem(M3, XN2),
            SwInter = linear_swiglu_intermediate(
                XN2, 1, ?HIDDEN, GPk, UPk, ?FFN),
            M4New = list_max_abs_elem(M4, SwInter),
            {M1New, M2New, M3New, M4New}
        end,
        {Init1, Init2, Init3, Init4},
        ?CAL_TOKEN_IDS).

%% Build layer 0 with SmoothQuant scaling, returns same-shape map as
%% build_layer/2 plus the seven inverse-scale lists (one per linear).
build_layer_calibrated(Header, 0, Stats) ->
    Layer0Raw = load_layer_raw(Header, 0),
    #{q_raw := QRaw, k_raw := KRaw, v_raw := VRaw, o_raw := ORaw,
      gate_raw := GRaw, up_raw := URaw, down_raw := DRaw} = Layer0Raw,
    {ANorm1, AAttn, ANorm2, ASw} = Stats,
    Alpha = 0.5,
    SQ = llama_calibration:smoothquant_scale(QRaw, ANorm1, ?HIDDEN, ?HIDDEN, Alpha),
    SK = llama_calibration:smoothquant_scale(KRaw, ANorm1, ?HIDDEN, ?KV_DIM, Alpha),
    SV = llama_calibration:smoothquant_scale(VRaw, ANorm1, ?HIDDEN, ?KV_DIM, Alpha),
    SO = llama_calibration:smoothquant_scale(ORaw, AAttn,  ?HIDDEN, ?HIDDEN, Alpha),
    SG = llama_calibration:smoothquant_scale(GRaw, ANorm2, ?HIDDEN, ?FFN,    Alpha),
    SU = llama_calibration:smoothquant_scale(URaw, ANorm2, ?HIDDEN, ?FFN,    Alpha),
    SD = llama_calibration:smoothquant_scale(DRaw, ASw,    ?FFN,    ?HIDDEN, Alpha),
    {QAdj, _} = llama_calibration:apply_smoothquant(QRaw, SQ, ?HIDDEN, ?HIDDEN),
    {KAdj, _} = llama_calibration:apply_smoothquant(KRaw, SK, ?HIDDEN, ?KV_DIM),
    {VAdj, _} = llama_calibration:apply_smoothquant(VRaw, SV, ?HIDDEN, ?KV_DIM),
    {OAdj, _} = llama_calibration:apply_smoothquant(ORaw, SO, ?HIDDEN, ?HIDDEN),
    {GAdj, _} = llama_calibration:apply_smoothquant(GRaw, SG, ?HIDDEN, ?FFN),
    {UAdj, _} = llama_calibration:apply_smoothquant(URaw, SU, ?HIDDEN, ?FFN),
    {DAdj, _} = llama_calibration:apply_smoothquant(DRaw, SD, ?FFN, ?HIDDEN),
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),
    #{
        norm1 => Norm1, norm2 => Norm2,
        q => prepack(QAdj, ?HIDDEN, ?HIDDEN),
        k => prepack(KAdj, ?HIDDEN, ?KV_DIM),
        v => prepack(VAdj, ?HIDDEN, ?KV_DIM),
        o => prepack(OAdj, ?HIDDEN, ?HIDDEN),
        gate => prepack(GAdj, ?HIDDEN, ?FFN),
        up   => prepack(UAdj, ?HIDDEN, ?FFN),
        down => prepack(DAdj, ?FFN, ?HIDDEN),
        sq_inv => list_reciprocal(SQ),
        sk_inv => list_reciprocal(SK),
        sv_inv => list_reciprocal(SV),
        so_inv => list_reciprocal(SO),
        sg_inv => list_reciprocal(SG),
        su_inv => list_reciprocal(SU),
        sd_inv => list_reciprocal(SD)
    };
build_layer_calibrated(Header, LayerIdx, _Stats) ->
    build_layer(Header, LayerIdx).

list_reciprocal(L) ->
    [case S > 0.0 of true -> 1.0 / S; false -> 1.0 end || S <- L].

list_max_abs_elem([], []) -> [];
list_max_abs_elem([M | RM], [X | RX]) ->
    A = abs(X),
    [case A > M of true -> A; false -> M end | list_max_abs_elem(RM, RX)].

list_max_abs(L) ->
    lists:foldl(fun(X, M) ->
        A = abs(X),
        case A > M of true -> A; false -> M end
    end, 0.0, L).

%% Single-token attention shortcut: V replicated NumHeads/NumKVHeads times.
expand_v_to_q_heads(VFlat) ->
    QPerKV = ?NUM_HEADS div ?NUM_KV_HEADS,
    KvChunks = chunks(VFlat, ?HEAD_DIM),
    lists:flatten([lists:duplicate(QPerKV, Chunk) || Chunk <- KvChunks]).

list_mul(A, B) ->
    lists:zipwith(fun(X, Y) -> X * Y end, A, B).

avg_lists(A, B) ->
    lists:zipwith(fun(X, Y) -> 0.5 * (X + Y) end, A, B).

%% Calibrated bisect — same shape as bisect/0 but layer 0 is SmoothQuant-
%% prepacked and inputs are divided by per-linear scale before FP8.
bisect_calibrated() ->
    io:format("~n=== viva_tensor bisect [CALIBRATED] — layer 0, BOS (pos=0) ===~n~n"),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layer0Raw = load_layer_raw(Header, 0),
    io:format("[cal] collecting activation stats over ~p tokens...~n",
              [length(?CAL_TOKEN_IDS)]),
    Stats = collect_calibration_stats(Header, Layer0Raw),
    {ANorm1, _, _, _} = Stats,
    io:format("[cal] act_max(norm1) mean=~.6f max=~.6f~n",
              [lists:sum(ANorm1) / length(ANorm1), lists:max(ANorm1)]),
    Layer = build_layer_calibrated(Header, 0, Stats),
    EmbedTbl = load_embed_table(Header),
    RopeTable = precompute_rope_table(16, ?HEAD_DIM, ?ROPE_THETA),

    X = embed_row(EmbedTbl, ?BOS_TOKEN),
    dump("embed[BOS]", X),
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),
    XNorm1 = rmsnorm(X, Norm1, ?EPS),
    dump("after input_layernorm", XNorm1),

    #{q := Q, k := K, v := V, o := O,
      gate := G, up := U, down := D,
      sq_inv := SQinv, sk_inv := SKinv, sv_inv := SVinv,
      so_inv := SOinv, sg_inv := SGinv, su_inv := SUinv,
      sd_inv := SDinv} = Layer,

    QInput = list_mul(XNorm1, SQinv),
    KInput = list_mul(XNorm1, SKinv),
    VInput = list_mul(XNorm1, SVinv),
    io:format("  [diag] XNorm1 absmax=~.6f V-input absmax=~.6f~n",
              [list_max_abs(XNorm1), list_max_abs(VInput)]),
    QRaw = linear_fp8(floats_to_fp16(QInput), Q, ?HIDDEN),
    KRaw = linear_fp8(floats_to_fp16(KInput), K, ?KV_DIM),
    VRaw = linear_fp8(floats_to_fp16(VInput), V, ?KV_DIM),
    dump("Q proj raw", QRaw),
    dump("K proj raw", KRaw),
    dump("V proj raw", VRaw),

    QRot = apply_rope(QRaw, 0, RopeTable, ?NUM_HEADS, ?HEAD_DIM),
    KRot = apply_rope(KRaw, 0, RopeTable, ?NUM_KV_HEADS, ?HEAD_DIM),
    dump("Q after RoPE", QRot),
    dump("K after RoPE", KRot),
    {AttnOut, _, _} = attention_gqa(QRot, KRot, VRaw, [], []),
    dump("attention output", AttnOut),

    OInput = list_mul(AttnOut, SOinv),
    OOut = linear_fp8(floats_to_fp16(OInput), O, ?HIDDEN),
    dump("O proj", OOut),
    H1 = list_add(X, OOut),
    dump("residual 1", H1),
    XNorm2 = rmsnorm(H1, Norm2, ?EPS),
    dump("after post_attention_layernorm", XNorm2),

    %% Fused swiglu can only accept one input — average gate/up scales.
    AvgGU = avg_lists(SGinv, SUinv),
    XNorm2Gu = list_mul(XNorm2, AvgGU),
    SwInter = linear_swiglu_intermediate(XNorm2Gu, 1, ?HIDDEN, G, U, ?FFN),
    dump("silu(gate) * up (fused)", SwInter),
    SwInterIn = list_mul(SwInter, SDinv),
    Ffn = linear_fp8(floats_to_fp16(SwInterIn), D, ?HIDDEN),
    dump("down proj (FFN out)", Ffn),
    H2 = list_add(H1, Ffn),
    dump("residual 2 (block 0 hidden)", H2),
    ok.

%% Run a single calibrated layer-0 forward (does not chain other layers).
run_n_calibrated(NumLayers) ->
    io:format("~n=== TinyLlama-1.1B [CALIBRATED layer 0] (N=~p) ===~n~n",
              [NumLayers]),
    T0 = ms(),
    {ok, Header} = viva_tensor_safetensors_ffi:open_header(?PATH),
    Layer0Raw = load_layer_raw(Header, 0),
    Stats = collect_calibration_stats(Header, Layer0Raw),
    Layer0 = build_layer_calibrated(Header, 0, Stats),
    case NumLayers of
        1 -> ok;
        _ -> io:format("[warn] only layer 0 is calibrated; layers 1.. are vanilla~n", [])
    end,
    EmbedTbl = load_embed_table(Header),
    HiddenIn = embed_row(EmbedTbl, ?BOS_TOKEN),
    RopeTable = precompute_rope_table(16, ?HEAD_DIM, ?ROPE_THETA),
    {HiddenOut, _, _} = forward_block_calibrated(
        HiddenIn, Layer0, 0, 0, RopeTable, [], []),
    io:format("[~5w ms] Layer 0 calibrated forward done. mean_abs=~p~n",
              [ms() - T0, mean_abs(HiddenOut)]),
    ok.

forward_block_calibrated(H, Layer, _LayerIdx, Pos, RopeTable, KCache, VCache) ->
    #{norm1 := N1, norm2 := N2,
      q := Q, k := K, v := V, o := O,
      gate := G, up := U, down := D,
      sq_inv := SQinv, sk_inv := SKinv, sv_inv := SVinv,
      so_inv := SOinv, sg_inv := SGinv, su_inv := SUinv,
      sd_inv := SDinv} = Layer,
    X1 = rmsnorm(H, N1, ?EPS),
    QInput = list_mul(X1, SQinv),
    KInput = list_mul(X1, SKinv),
    VInput = list_mul(X1, SVinv),
    QOutRaw = linear_fp8(floats_to_fp16(QInput), Q, ?HIDDEN),
    KOutRaw = linear_fp8(floats_to_fp16(KInput), K, ?KV_DIM),
    VOut    = linear_fp8(floats_to_fp16(VInput), V, ?KV_DIM),
    QOut = apply_rope(QOutRaw, Pos, RopeTable, ?NUM_HEADS, ?HEAD_DIM),
    KOut = apply_rope(KOutRaw, Pos, RopeTable, ?NUM_KV_HEADS, ?HEAD_DIM),
    {AttnOut, KCache2, VCache2} = attention_gqa(QOut, KOut, VOut, KCache, VCache),
    OInput = list_mul(AttnOut, SOinv),
    OOut = linear_fp8(floats_to_fp16(OInput), O, ?HIDDEN),
    H1 = list_add(H, OOut),
    X2 = rmsnorm(H1, N2, ?EPS),
    AvgGU = avg_lists(SGinv, SUinv),
    X2Gu = list_mul(X2, AvgGU),
    SwInter = linear_swiglu_intermediate(X2Gu, 1, ?HIDDEN, G, U, ?FFN),
    SwInterIn = list_mul(SwInter, SDinv),
    Ffn = linear_fp8(floats_to_fp16(SwInterIn), D, ?HIDDEN),
    HOut = list_add(H1, Ffn),
    {HOut, KCache2, VCache2}.
