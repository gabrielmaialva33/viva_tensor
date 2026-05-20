%%% @doc Public LLM loading and generation API.
%%%
%%% This module packages the TinyLlama/Llama decode-step path used by
%%% dev/llama_forward.erl behind an opaque model handle. The hot generation
%%% loop still calls viva_tensor_zig:nt_forward_decode_step/8.
-module(viva_tensor_llm).

-export([
    load/2,
    generate/3,
    load_for_gleam/1,
    generate_for_gleam/8,
    path_exists/1
]).

-define(DEFAULT_BLOCK_SIZE, 16).
-define(DEFAULT_MAX_SEQ, 2048).
-define(DEFAULT_HEAD_DIM, 64).
-define(DEFAULT_EPS, 1.0e-5).
-define(DEFAULT_ROPE_THETA, 10000.0).

load(SafetensorsPath0, Opts0) when is_map(Opts0) ->
    try
        SafetensorsPath = to_binary(SafetensorsPath0),
        T0 = us(),
        {ok, Header} = viva_tensor_safetensors_ffi:open(SafetensorsPath),
        Config = model_config(Header, SafetensorsPath, Opts0),
        TokenizerPath = tokenizer_path(SafetensorsPath, Opts0),
        {ok, Tokenizer} = viva_tensor_tokenizer_ffi:load(TokenizerPath),
        Layers = [build_layer_blocked(Header, I, Config)
                  || I <- lists:seq(0, maps:get(num_layers, Config) - 1)],
        EmbedTable = load_embed_table_resource(Header, Config),
        FinalNorm = load_rmsnorm_bin(Header, <<"model.norm.weight">>),
        LmHeadName = case maps:get(tie_word_embeddings, Config, false) of
            true -> <<"model.embed_tokens.weight">>;
            false -> <<"lm_head.weight">>
        end,
        LmHead = load_linear(Header, LmHeadName,
                             maps:get(vocab_size, Config), maps:get(hidden_size, Config)),
        LmHeadPacked = prepack_blocked(
            LmHead,
            maps:get(hidden_size, Config),
            maps:get(vocab_size, Config),
            maps:get(block_size, Config)
        ),
        RopeFreqs = precompute_rope_freqs_bin(
            maps:get(head_dim, Config),
            maps:get(rope_theta, Config)
        ),
        InitialCaches = new_kv_caches(Config),
        Handle = #{
            viva_tensor_llm_handle => true,
            safetensors_path => SafetensorsPath,
            tokenizer_path => TokenizerPath,
            tokenizer => Tokenizer,
            layers => Layers,
            embed_table_ref => EmbedTable,
            kv_caches => InitialCaches,
            lm_head => LmHeadPacked,
            final_norm => FinalNorm,
            rope_freqs => RopeFreqs,
            config => Config,
            load_us => us() - T0
        },
        {ok, Handle}
    catch
        Class:Reason:Stack ->
            {error, {Class, Reason, Stack}}
    end.

generate(Handle, Prompt0, GenOpts0)
        when is_map(Handle), is_map(GenOpts0) ->
    try
        true = maps:get(viva_tensor_llm_handle, Handle, false),
        Prompt = to_binary(Prompt0),
        GenOpts = generation_options(GenOpts0),
        case maps:get(temperature, GenOpts) of
            Temp when Temp =< 0.0 ->
                generate_argmax(Handle, Prompt, GenOpts);
            _ ->
                generate_sampling(Handle, Prompt, GenOpts)
        end
    catch
        Class:Reason:Stack ->
            {error, {Class, Reason, Stack}}
    end.

load_for_gleam(Path) ->
    case load(Path, #{}) of
        {ok, Handle} -> {ok, Handle};
        {error, Reason} -> {error, reason_to_binary(Reason)}
    end.

generate_for_gleam(Handle, Prompt, MaxNewTokens, Temperature, TopK, TopP, Seed, StopOnEos) ->
    Opts = #{
        max_new_tokens => MaxNewTokens,
        temperature => Temperature,
        top_k => case TopK of -1 -> infinity; _ -> TopK end,
        top_p => TopP,
        seed => Seed,
        stop_on_eos => StopOnEos
    },
    case generate(Handle, Prompt, Opts) of
        {ok, #{tokens := Tokens, text := Text, ms_per_token := Ms, total_tokens := Total}} ->
            {ok, {Tokens, Text, Ms, Total}};
        {error, Reason} ->
            {error, reason_to_binary(Reason)}
    end.

path_exists(Path) ->
    PathList = binary_to_list(to_binary(Path)),
    filelib:is_file(PathList) orelse filelib:is_dir(PathList).

generate_argmax(Handle, Prompt, Opts) ->
    Config = maps:get(config, Handle),
    Tokenizer = maps:get(tokenizer, Handle),
    Layers = maps:get(layers, Handle),
    EmbedTable = maps:get(embed_table_ref, Handle),
    FinalNorm = maps:get(final_norm, Handle),
    LmHead = maps:get(lm_head, Handle),
    RopeFreqs = maps:get(rope_freqs, Handle),
    MaxNew = maps:get(max_new_tokens, Opts),
    StopOnEos = maps:get(stop_on_eos, Opts),
    BOS = viva_tensor_tokenizer_ffi:bos_id(Tokenizer),
    EOS = viva_tensor_tokenizer_ffi:eos_id(Tokenizer),
    PromptTokens = [BOS | viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt)],
    MaxSeq = maps:get(max_seq, Config),
    case length(PromptTokens) + MaxNew >= MaxSeq of
        true ->
            {error, {max_sequence_exceeded, length(PromptTokens), MaxNew, MaxSeq}};
        false ->
            Caches = new_kv_caches(Config),
            {FirstNext, _} = lists:foldl(
                fun({Pos, TokenId}, {_, CL}) ->
                    Next = forward_decode_step(TokenId, EmbedTable, Layers, FinalNorm,
                                               LmHead, CL, Pos, RopeFreqs),
                    {Next, CL}
                end,
                {undefined, Caches},
                lists:zip(lists:seq(0, length(PromptTokens) - 1), PromptTokens)
            ),
            TGen = us(),
            GeneratedIds = decode_loop_decode_fused(
                FirstNext, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                RopeFreqs, length(PromptTokens), MaxNew, EOS, StopOnEos, []
            ),
            GenUs = us() - TGen,
            TokCount = length(GeneratedIds),
            MsPerToken = case TokCount of
                0 -> 0.0;
                _ -> float(GenUs) / 1000.0 / float(TokCount)
            end,
            Text = viva_tensor_tokenizer_ffi:decode(Tokenizer, GeneratedIds),
            {ok, #{
                tokens => GeneratedIds,
                text => Text,
                ms_per_token => MsPerToken,
                total_tokens => TokCount
            }}
    end.

decode_loop_decode_fused(_NextTok, _C, _L, _E, _FN, _LH, _R, _P, 0,
                         _EOS, _StopOnEos, Acc) ->
    lists:reverse(Acc);
decode_loop_decode_fused(NextTok, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                         RopeFreqs, Pos, Remaining, EOS, StopOnEos, Acc) ->
    case StopOnEos andalso NextTok =:= EOS of
        true ->
            lists:reverse([NextTok | Acc]);
        false ->
            Following = forward_decode_step(NextTok, EmbedTable, Layers, FinalNorm,
                                            LmHead, Caches, Pos, RopeFreqs),
            decode_loop_decode_fused(Following, Caches, Layers, EmbedTable,
                                     FinalNorm, LmHead, RopeFreqs, Pos + 1,
                                     Remaining - 1, EOS, StopOnEos,
                                     [NextTok | Acc])
    end.

generate_sampling(Handle, Prompt, Opts) ->
    Config = maps:get(config, Handle),
    Tokenizer = maps:get(tokenizer, Handle),
    Layers = maps:get(layers, Handle),
    EmbedTable = maps:get(embed_table_ref, Handle),
    FinalNorm = maps:get(final_norm, Handle),
    LmHead = maps:get(lm_head, Handle),
    RopeFreqs = maps:get(rope_freqs, Handle),
    MaxNew = maps:get(max_new_tokens, Opts),
    StopOnEos = maps:get(stop_on_eos, Opts),
    BOS = viva_tensor_tokenizer_ffi:bos_id(Tokenizer),
    EOS = viva_tensor_tokenizer_ffi:eos_id(Tokenizer),
    PromptTokens = [BOS | viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt)],
    MaxSeq = maps:get(max_seq, Config),
    case length(PromptTokens) + MaxNew >= MaxSeq of
        true ->
            {error, {max_sequence_exceeded, length(PromptTokens), MaxNew, MaxSeq}};
        false ->
            Caches = new_kv_caches(Config),
            TopK = sampling_top_k(Opts, maps:get(vocab_size, Config)),
            FirstNext = prefill_sampling(
                PromptTokens, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                RopeFreqs, TopK, Opts
            ),
            TGen = us(),
            GeneratedIds = decode_loop_decode_sampled(
                FirstNext, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                RopeFreqs, length(PromptTokens), MaxNew, EOS, StopOnEos,
                TopK, Opts, []
            ),
            GenUs = us() - TGen,
            TokCount = length(GeneratedIds),
            MsPerToken = case TokCount of
                0 -> 0.0;
                _ -> float(GenUs) / 1000.0 / float(TokCount)
            end,
            Text = viva_tensor_tokenizer_ffi:decode(Tokenizer, GeneratedIds),
            {ok, #{
                tokens => GeneratedIds,
                text => Text,
                ms_per_token => MsPerToken,
                total_tokens => TokCount
            }}
    end.

prefill_sampling(PromptTokens, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                 RopeFreqs, TopK, Opts) ->
    LastPos = length(PromptTokens) - 1,
    {Next, _} = lists:foldl(
        fun({Pos, TokenId}, {_, CL}) ->
            Sampled = case Pos =:= LastPos of
                true ->
                    forward_decode_step_sample(TokenId, EmbedTable, Layers, FinalNorm,
                                               LmHead, CL, Pos, RopeFreqs, TopK, Opts);
                false ->
                    forward_decode_step(TokenId, EmbedTable, Layers, FinalNorm,
                                        LmHead, CL, Pos, RopeFreqs)
            end,
            {Sampled, CL}
        end,
        {undefined, Caches},
        lists:zip(lists:seq(0, LastPos), PromptTokens)
    ),
    Next.

decode_loop_decode_sampled(_NextTok, _C, _L, _E, _FN, _LH, _R, _P, 0,
                           _EOS, _StopOnEos, _TopK, _Opts, Acc) ->
    lists:reverse(Acc);
decode_loop_decode_sampled(NextTok, Caches, Layers, EmbedTable, FinalNorm, LmHead,
                           RopeFreqs, Pos, Remaining, EOS, StopOnEos, TopK, Opts,
                           Acc) ->
    case StopOnEos andalso NextTok =:= EOS of
        true ->
            lists:reverse([NextTok | Acc]);
        false ->
            Following = forward_decode_step_sample(
                NextTok, EmbedTable, Layers, FinalNorm, LmHead, Caches, Pos,
                RopeFreqs, TopK, Opts
            ),
            decode_loop_decode_sampled(Following, Caches, Layers, EmbedTable,
                                       FinalNorm, LmHead, RopeFreqs, Pos + 1,
                                       Remaining - 1, EOS, StopOnEos, TopK, Opts,
                                       [NextTok | Acc])
    end.

forward_decode_step(TokenId, EmbedTable, Layers, FinalNorm, LmHead,
                    Caches, Pos, RopeFreqs) ->
    case viva_tensor_zig:nt_forward_decode_step(
             TokenId, EmbedTable, Layers, FinalNorm, LmHead, Caches, Pos, RopeFreqs) of
        {ok, NextToken} when is_integer(NextToken) ->
            NextToken;
        Error ->
            error({forward_decode_step_failed, Error})
    end.

forward_decode_step_sample(TokenId, EmbedTable, Layers, FinalNorm, LmHead,
                           Caches, Pos, RopeFreqs, TopK, Opts) ->
    case viva_tensor_zig:nt_forward_decode_step_topk(
             TokenId, EmbedTable, Layers, FinalNorm, LmHead, Caches, Pos,
             RopeFreqs, TopK) of
        {ok, {IndicesBin, ValuesBin}} when is_binary(IndicesBin), is_binary(ValuesBin) ->
            Indices = decode_int32_le(IndicesBin),
            Logits = decode_float32_le(ValuesBin),
            Pick = llama_sampling:sample(Logits, sampling_opts_for_pos(Opts, Pos)),
            lists:nth(Pick + 1, Indices);
        Error ->
            error({forward_decode_step_topk_failed, Error})
    end.

sampling_top_k(Opts, VocabSize) ->
    Requested = case maps:get(top_k, Opts) of
        infinity -> 256;
        K when is_integer(K), K > 0 -> K;
        _ -> 256
    end,
    min(VocabSize, min(256, Requested)).

sampling_opts_for_pos(Opts, Pos) ->
    Opts#{seed => maps:get(seed, Opts) + Pos}.

decode_int32_le(Bin) ->
    [I || <<I:32/signed-little>> <= Bin].

decode_float32_le(Bin) ->
    [F || <<F:32/float-little>> <= Bin].

model_config(Header, SafetensorsPath, Opts) ->
    FileConfig = read_hf_config(SafetensorsPath),
    NumLayers = opt(Opts, num_layers, detect_num_layers(Header)),
    BlockSize = opt(Opts, block_size, ?DEFAULT_BLOCK_SIZE),
    {VocabSize0, HiddenSize0} = shape2(Header, <<"model.embed_tokens.weight">>),
    Tied = case maps:get(<<"tie_word_embeddings">>, FileConfig, false) of
        true -> true;
        _ -> false
    end,
    LmHidden = case Tied of
        true -> HiddenSize0;
        false ->
            {_, LH} = shape2(Header, <<"lm_head.weight">>),
            LH
    end,
    HiddenSize = int_config(FileConfig, <<"hidden_size">>, HiddenSize0),
    VocabSize = int_config(FileConfig, <<"vocab_size">>, VocabSize0),
    NumHeads = int_config(FileConfig, <<"num_attention_heads">>,
                          max(1, HiddenSize div ?DEFAULT_HEAD_DIM)),
    NumKvHeads = int_config(FileConfig, <<"num_key_value_heads">>, NumHeads),
    HeadDim = case NumHeads of
        0 -> ?DEFAULT_HEAD_DIM;
        _ -> HiddenSize div NumHeads
    end,
    KvDim = NumKvHeads * HeadDim,
    FfnSize = int_config_lazy(FileConfig, <<"intermediate_size">>,
                              fun() -> first_layer_ffn(Header) end),
    #{
        num_layers => NumLayers,
        block_size => BlockSize,
        vocab_size => VocabSize,
        hidden_size => HiddenSize,
        lm_hidden_size => LmHidden,
        kv_dim => KvDim,
        ffn_size => FfnSize,
        num_heads => NumHeads,
        num_kv_heads => NumKvHeads,
        head_dim => HeadDim,
        eps => float_config(FileConfig, <<"rms_norm_eps">>, ?DEFAULT_EPS),
        rope_theta => float_config(FileConfig, <<"rope_theta">>, ?DEFAULT_ROPE_THETA),
        max_seq => opt(Opts, max_seq, ?DEFAULT_MAX_SEQ),
        tie_word_embeddings => Tied
    }.

read_hf_config(SafetensorsPath) ->
    ConfigPath = filename:join(model_dir(SafetensorsPath), "config.json"),
    case file:read_file(ConfigPath) of
        {ok, Bin} ->
            try json:decode(Bin)
            catch _:_ -> #{}
            end;
        _ ->
            #{}
    end.

detect_num_layers(Header) ->
    detect_num_layers(Header, 0).

detect_num_layers(Header, I) ->
    Name = list_to_binary("model.layers." ++ integer_to_list(I) ++
                          ".input_layernorm.weight"),
    case viva_tensor_safetensors_ffi:tensor_info(Header, Name) of
        {ok, _} -> detect_num_layers(Header, I + 1);
        {error, _} when I > 0 -> I;
        {error, _} -> 22
    end.

first_layer_ffn(Header) ->
    {Ffn, _Hidden} = shape2(Header, <<"model.layers.0.mlp.gate_proj.weight">>),
    Ffn.

shape2(Header, Name) ->
    {ok, #{shape := [A, B]}} = viva_tensor_safetensors_ffi:tensor_info(Header, Name),
    {A, B}.

build_layer_blocked(Header, LayerIdx, Config) ->
    Prefix = "model.layers." ++ integer_to_list(LayerIdx) ++ ".",
    P = fun(Suffix) -> list_to_binary(Prefix ++ Suffix) end,
    require_tensors(Header, [
        P("self_attn.q_proj.weight"),
        P("self_attn.k_proj.weight"),
        P("self_attn.v_proj.weight"),
        P("self_attn.o_proj.weight"),
        P("mlp.gate_proj.weight"),
        P("mlp.up_proj.weight"),
        P("mlp.down_proj.weight"),
        P("input_layernorm.weight"),
        P("post_attention_layernorm.weight")
    ], LayerIdx),
    Hidden = maps:get(hidden_size, Config),
    KvDim = maps:get(kv_dim, Config),
    Ffn = maps:get(ffn_size, Config),
    BlockSize = maps:get(block_size, Config),
    QProj = load_linear(Header, P("self_attn.q_proj.weight"), Hidden, Hidden),
    KProj = load_linear(Header, P("self_attn.k_proj.weight"), KvDim, Hidden),
    VProj = load_linear(Header, P("self_attn.v_proj.weight"), KvDim, Hidden),
    OProj = load_linear(Header, P("self_attn.o_proj.weight"), Hidden, Hidden),
    GateProj = load_linear(Header, P("mlp.gate_proj.weight"), Ffn, Hidden),
    UpProj = load_linear(Header, P("mlp.up_proj.weight"), Ffn, Hidden),
    DownProj = load_linear(Header, P("mlp.down_proj.weight"), Hidden, Ffn),
    QKVProj = concat_linear_columns([{QProj, Hidden}, {KProj, KvDim}, {VProj, KvDim}], Hidden),
    GateUpProj = concat_linear_columns([{GateProj, Ffn}, {UpProj, Ffn}], Hidden),
    #{
        norm1_bin => load_rmsnorm_bin(Header, P("input_layernorm.weight")),
        norm2_bin => load_rmsnorm_bin(Header, P("post_attention_layernorm.weight")),
        hidden_size => Hidden,
        kv_size => KvDim,
        ffn_size => Ffn,
        num_heads => maps:get(num_heads, Config),
        num_kv_heads => maps:get(num_kv_heads, Config),
        head_dim => maps:get(head_dim, Config),
        eps => maps:get(eps, Config),
        rope_theta => maps:get(rope_theta, Config),
        q => prepack_blocked(QProj, Hidden, Hidden, BlockSize),
        k => prepack_blocked(KProj, Hidden, KvDim, BlockSize),
        v => prepack_blocked(VProj, Hidden, KvDim, BlockSize),
        o => prepack_blocked(OProj, Hidden, Hidden, BlockSize),
        gate => prepack_blocked(GateProj, Hidden, Ffn, BlockSize),
        up => prepack_blocked(UpProj, Hidden, Ffn, BlockSize),
        qkv => prepack_blocked(QKVProj, Hidden, Hidden + KvDim + KvDim, BlockSize),
        gate_up => prepack_blocked(GateUpProj, Hidden, Ffn + Ffn, BlockSize),
        down => prepack_blocked(DownProj, Ffn, Hidden, BlockSize)
    }.

require_tensors(Header, Names, LayerIdx) ->
    lists:foreach(
        fun(Name) ->
            case viva_tensor_safetensors_ffi:tensor_info(Header, Name) of
                {ok, _} -> ok;
                {error, _} -> error({missing_llama_tensor, LayerIdx, Name})
            end
        end,
        Names
    ).

load_linear(Header, Name, OutF, InF) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
    Fp32 = viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16),
    {ok, Transposed} = viva_tensor_safetensors_ffi:transpose_fp32(Fp32, OutF, InF),
    Transposed.

concat_linear_columns(Parts, InF) ->
    BytesPerFloat = 4,
    list_to_binary([
        [binary:part(Bin, Row * OutF * BytesPerFloat, OutF * BytesPerFloat)
         || {Bin, OutF} <- Parts]
        || Row <- lists:seq(0, InF - 1)
    ]).

load_rmsnorm_bin(Header, Name) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
    viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16).

load_embed_table_resource(Header, Config) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(
        Header, <<"model.embed_tokens.weight">>),
    case viva_tensor_zig:nt_embedding_table_new(
             Bf16, maps:get(vocab_size, Config), maps:get(hidden_size, Config)) of
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({embedding_table_resource_failed, Other})
    end.

prepack_blocked(Bin, InF, OutF, BlockSize) when is_binary(Bin) ->
    case viva_tensor_zig:nt_prepack_fp8_blocked(Bin, [InF, OutF], BlockSize) of
        {ok, {Resource, _, _, _}} -> Resource;
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({prepack_blocked_failed, Other})
    end.

new_kv_caches(Config) ->
    [begin
         {ok, Cache} = viva_tensor_zig:nt_kv_cache_new(
             maps:get(max_seq, Config), maps:get(kv_dim, Config)),
         Cache
     end || _ <- lists:seq(1, maps:get(num_layers, Config))].

precompute_rope_freqs_bin(HeadDim, Theta) ->
    Half = HeadDim div 2,
    Freqs = [math:pow(Theta, -2.0 * float(I) / float(HeadDim))
             || I <- lists:seq(0, Half - 1)],
    << <<F:32/float-little>> || F <- Freqs >>.

generation_options(Opts) ->
    #{
        max_new_tokens => opt(Opts, max_new_tokens, 50),
        temperature => float_opt(Opts, temperature, 0.0),
        top_k => opt(Opts, top_k, infinity),
        top_p => float_opt(Opts, top_p, 1.0),
        seed => opt(Opts, seed, 42),
        stop_on_eos => opt(Opts, stop_on_eos, true)
    }.

tokenizer_path(SafetensorsPath, Opts) ->
    case opt(Opts, tokenizer_path, undefined) of
        undefined ->
            default_tokenizer_path(SafetensorsPath);
        Path ->
            to_binary(Path)
    end.

default_tokenizer_path(SafetensorsPath) ->
    Sibling = sibling_tokenizer_path(SafetensorsPath),
    case filelib:is_dir(binary_to_list(SafetensorsPath)) of
        true ->
            Sibling;
        false ->
            Inferred = inferred_tokenizer_path(SafetensorsPath),
            case filelib:is_file(binary_to_list(Inferred)) of
                true -> Inferred;
                false ->
                    case filelib:is_file(binary_to_list(Sibling)) of
                        true -> Sibling;
                        false -> Inferred
                    end
            end
    end.

inferred_tokenizer_path(SafetensorsPath) ->
    Root = filename:rootname(binary_to_list(SafetensorsPath)),
    list_to_binary(Root ++ "_tokenizer.json").

sibling_tokenizer_path(SafetensorsPath) ->
    list_to_binary(filename:join(model_dir(SafetensorsPath), "tokenizer.json")).

model_dir(Path0) ->
    Path = binary_to_list(Path0),
    case filelib:is_dir(Path) of
        true -> Path;
        false -> filename:dirname(Path)
    end.

opt(Map, Key, Default) ->
    case maps:find(Key, Map) of
        {ok, Value} -> Value;
        error ->
            BinKey = atom_to_binary(Key, utf8),
            maps:get(BinKey, Map, Default)
    end.

float_opt(Map, Key, Default) ->
    to_float(opt(Map, Key, Default)).

int_config(Config, Key, Default) ->
    case maps:get(Key, Config, Default) of
        V when is_integer(V) -> V;
        V when is_float(V) -> trunc(V);
        _ -> Default
    end.

int_config_lazy(Config, Key, DefaultFun) ->
    case maps:get(Key, Config, undefined) of
        V when is_integer(V) -> V;
        V when is_float(V) -> trunc(V);
        _ -> DefaultFun()
    end.

float_config(Config, Key, Default) ->
    to_float(maps:get(Key, Config, Default)).

to_float(V) when is_float(V) -> V;
to_float(V) when is_integer(V) -> float(V);
to_float(_) -> 0.0.

to_binary(V) when is_binary(V) -> V;
to_binary(V) when is_list(V) -> unicode:characters_to_binary(V);
to_binary(V) -> unicode:characters_to_binary(io_lib:format("~p", [V])).

reason_to_binary(Reason) ->
    unicode:characters_to_binary(io_lib:format("~p", [Reason])).

us() ->
    erlang:monotonic_time(microsecond).
