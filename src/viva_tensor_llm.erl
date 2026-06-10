%%% @doc Public LLM loading and generation API.
%%%
%%% This module packages the TinyLlama/Llama decode-step path used by
%%% dev/llama_forward.erl behind an opaque model handle. The hot generation
%%% loop still calls viva_tensor_zig:nt_forward_decode_step/10.
-module(viva_tensor_llm).

-export([
    load/2,
    generate/3,
    generate_batch/3,
    load_for_gleam/1,
    load_marlin_for_gleam/1,
    fp8_w8a16_atom/0,
    marlin_w4a16_atom/0,
    generate_for_gleam/9,
    generate_for_gleam/8,
    generate_batch_for_gleam/9,
    generate_batch_for_gleam/8,
    path_exists/1
]).

-define(DEFAULT_BLOCK_SIZE, 16).
-define(DEFAULT_MAX_SEQ, 2048).
-define(DEFAULT_HEAD_DIM, 64).
-define(DEFAULT_EPS, 1.0e-5).
-define(DEFAULT_ROPE_THETA, 10000.0).
-define(MARLIN_GROUPSIZE, 128).
-define(MARLIN_MIN_SCALE, 1.0e-4).
-define(MARLIN_EXACT_SCALE_MAX_ELEMS, 1048576).
-define(MARLIN_SCALE_SAMPLE_ELEMS, 65536).

load(SafetensorsPath0, Opts0) when is_map(Opts0) ->
    try
        SafetensorsPath = to_binary(SafetensorsPath0),
        T0 = us(),
        {ok, Header} = viva_tensor_safetensors_ffi:open(SafetensorsPath),
        Config = model_config(Header, SafetensorsPath, Opts0),
        TokenizerPath = tokenizer_path(SafetensorsPath, Opts0),
        %% The tokenizer (a large JSON vocab), embedding table, and lm_head are
        %% each as heavy as several layers. Build them — and the layers —
        %% concurrently so none dominates wall time.
        TokRef = async_build(fun() ->
            {ok, Tok} = viva_tensor_tokenizer_ffi:load(TokenizerPath),
            Tok
        end),
        EmbedRef = async_build(fun() -> load_embed_table_resource(Header, Config) end),
        LmHeadRef = async_build(fun() -> build_lm_head_packed(Header, Config) end),
        Layers = parallel_build_layers(Header, Config, maps:get(num_layers, Config)),
        Tokenizer = await_build(TokRef),
        EmbedTable = await_build(EmbedRef),
        LmHeadPacked = await_build(LmHeadRef),
        FinalNorm = load_rmsnorm_bin(Header, <<"model.norm.weight">>),
        RopeFreqs = precompute_rope_freqs_bin(
            maps:get(head_dim, Config),
            maps:get(rope_theta, Config),
            maps:get(rope_scaling, Config, none)
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

generate(Handle, Prompt0, GenOpts0) when
    is_map(Handle), is_map(GenOpts0)
->
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

generate_batch(Handle, Prompts, GenOpts0) when is_list(Prompts) ->
    try
        true = maps:get(viva_tensor_llm_handle, Handle, false),
        GenOpts = generation_options(GenOpts0),
        case maps:get(temperature, GenOpts) of
            Temp when Temp =< 0.0 ->
                generate_batch_native(Handle, Prompts, GenOpts);
            _ ->
                with_block_state(fun(BlockState) ->
                    lists:map(
                        fun(Prompt0) ->
                            try
                                Prompt = to_binary(Prompt0),
                                generate_sampling_with_state(
                                    Handle, Prompt, GenOpts, BlockState
                                )
                            catch
                                Class:Reason:Stack ->
                                    {error, {Class, Reason, Stack}}
                            end
                        end,
                        Prompts
                    )
                end)
        end
    catch
        Class:Reason:Stack ->
            [{error, {Class, Reason, Stack}} || _ <- Prompts]
    end.

generate_batch_native(_Handle, [], _GenOpts) ->
    [];
generate_batch_native(Handle, Prompts, GenOpts) ->
    with_block_state(fun(BlockState) ->
        Prepared = [
            prepare_batch_prompt(BlockState, Handle, Prompt0, GenOpts, Idx)
         || {Idx, Prompt0} <- lists:zip(lists:seq(1, length(Prompts)), Prompts)
        ],
        Errors = maps:from_list([
            {Idx, Error}
         || {error, Idx, Error} <- Prepared
        ]),
        States0 = [State || {ok, State} <- Prepared],
        TGen = us(),
        States = batch_decode_loop(BlockState, Handle, GenOpts, States0),
        ResultMap = maps:from_list([
            {Idx, batch_state_result(Handle, State, TGen)}
         || State = #{idx := Idx} <- States
        ]),
        [
            case maps:find(Idx, ResultMap) of
                {ok, Result} -> Result;
                error -> maps:get(Idx, Errors)
            end
         || Idx <- lists:seq(1, length(Prompts))
        ]
    end).

load_for_gleam(Path) ->
    case load(Path, #{}) of
        {ok, Handle} -> {ok, Handle};
        {error, Reason} -> {error, reason_to_binary(Reason)}
    end.

load_marlin_for_gleam(Path) ->
    case load(Path, #{}) of
        {ok, Handle0} ->
            case prepack_all_linears_marlin(Handle0) of
                {ok, MarlinHandles} ->
                    {ok, Handle0#{marlin_handles => MarlinHandles, weight_format => marlin_w4a16}};
                {error, R} ->
                    {error, reason_to_binary(R)}
            end;
        {error, Reason} ->
            {error, reason_to_binary(Reason)}
    end.

fp8_w8a16_atom() -> fp8_w8a16.

marlin_w4a16_atom() -> marlin_w4a16.

generate_for_gleam(Handle, Prompt, MaxNewTokens, Temperature, TopK, TopP, Seed, StopOnEos) ->
    generate_for_gleam(
        Handle,
        Prompt,
        MaxNewTokens,
        Temperature,
        TopK,
        TopP,
        Seed,
        StopOnEos,
        fp8_w8a16
    ).

generate_for_gleam(
    Handle,
    Prompt,
    MaxNewTokens,
    Temperature,
    TopK,
    TopP,
    Seed,
    StopOnEos,
    WeightFormat
) ->
    Opts = #{
        max_new_tokens => MaxNewTokens,
        temperature => Temperature,
        top_k =>
            case TopK of
                -1 -> infinity;
                _ -> TopK
            end,
        top_p => TopP,
        seed => Seed,
        stop_on_eos => StopOnEos,
        weight_format => weight_format_atom(WeightFormat)
    },
    case generate(Handle, Prompt, Opts) of
        {ok, #{tokens := Tokens, text := Text, ms_per_token := Ms, total_tokens := Total}} ->
            {ok, {Tokens, Text, Ms, Total}};
        {error, Reason} ->
            {error, reason_to_binary(Reason)}
    end.

generate_batch_for_gleam(Handle, Prompts, MaxNewTokens, Temperature, TopK, TopP, Seed, StopOnEos) ->
    generate_batch_for_gleam(
        Handle,
        Prompts,
        MaxNewTokens,
        Temperature,
        TopK,
        TopP,
        Seed,
        StopOnEos,
        fp8_w8a16
    ).

generate_batch_for_gleam(
    Handle,
    Prompts,
    MaxNewTokens,
    Temperature,
    TopK,
    TopP,
    Seed,
    StopOnEos,
    WeightFormat
) ->
    Opts = #{
        max_new_tokens => MaxNewTokens,
        temperature => Temperature,
        top_k =>
            case TopK of
                -1 -> infinity;
                _ -> TopK
            end,
        top_p => TopP,
        seed => Seed,
        stop_on_eos => StopOnEos,
        weight_format => weight_format_atom(WeightFormat)
    },
    [generate_result_for_gleam(Result) || Result <- generate_batch(Handle, Prompts, Opts)].

path_exists(Path) ->
    PathList = binary_to_list(to_binary(Path)),
    filelib:is_file(PathList) orelse filelib:is_dir(PathList).

collect_generate_result({Pid, Ref}, Timeout) ->
    receive
        {Pid, Result} ->
            receive
                {'DOWN', Ref, process, Pid, _Reason} -> ok
            after 0 ->
                ok
            end,
            Result;
        {'DOWN', Ref, process, Pid, normal} ->
            receive
                {Pid, Result} -> Result
            after 0 ->
                {error, {process_crashed, normal}}
            end;
        {'DOWN', Ref, process, Pid, Reason} ->
            {error, {process_crashed, Reason}}
    after Timeout ->
        exit(Pid, kill),
        receive
            {'DOWN', Ref, process, Pid, _Reason} -> ok
        after 0 ->
            ok
        end,
        {error, {process_timeout, Timeout}}
    end.

generate_result_for_gleam(
    {ok, #{tokens := Tokens, text := Text, ms_per_token := Ms, total_tokens := Total}}
) ->
    {ok, {Tokens, Text, Ms, Total}};
generate_result_for_gleam({error, Reason}) ->
    {error, reason_to_binary(Reason)}.

with_block_state(Fun) ->
    case viva_tensor_zig:block_state_new() of
        {ok, BlockState} ->
            try
                Fun(BlockState)
            after
                viva_tensor_zig:block_state_free(BlockState)
            end;
        Error ->
            error({block_state_new_failed, Error})
    end.

generate_argmax(Handle, Prompt, Opts) ->
    with_block_state(fun(BlockState) ->
        generate_argmax_with_state(Handle, Prompt, Opts, BlockState)
    end).

prepare_batch_prompt(BlockState, Handle, Prompt0, Opts, Idx) ->
    try
        Config = maps:get(config, Handle),
        Tokenizer = maps:get(tokenizer, Handle),
        Layers0 = maps:get(layers, Handle),
        EmbedTable = maps:get(embed_table_ref, Handle),
        FinalNorm = maps:get(final_norm, Handle),
        LmHead = maps:get(lm_head, Handle),
        RopeFreqs = maps:get(rope_freqs, Handle),
        MaxNew = maps:get(max_new_tokens, Opts),
        WeightFormat = maps:get(weight_format, Opts),
        Layers = enrich_layers_with_marlin(Layers0, Handle, WeightFormat),
        Prompt = to_binary(Prompt0),
        Enc0 = viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt),
        PromptTokens =
            case viva_tensor_tokenizer_ffi:prepends_bos(Tokenizer) of
                true -> [viva_tensor_tokenizer_ffi:bos_id(Tokenizer) | Enc0];
                false -> Enc0
            end,
        MaxSeq = maps:get(max_seq, Config),
        case length(PromptTokens) + MaxNew >= MaxSeq of
            true ->
                {error, Idx,
                    {error, {max_sequence_exceeded, length(PromptTokens), MaxNew, MaxSeq}}};
            false ->
                Caches = new_kv_caches(Config),
                {FirstNext, _} = lists:foldl(
                    fun({Pos, TokenId}, {_, CL}) ->
                        Next = forward_decode_step(
                            BlockState,
                            TokenId,
                            EmbedTable,
                            Layers,
                            FinalNorm,
                            LmHead,
                            CL,
                            Pos,
                            RopeFreqs,
                            WeightFormat
                        ),
                        {Next, CL}
                    end,
                    {undefined, Caches},
                    lists:zip(lists:seq(0, length(PromptTokens) - 1), PromptTokens)
                ),
                {ok, #{
                    idx => Idx,
                    caches => Caches,
                    next => FirstNext,
                    pos => length(PromptTokens),
                    remaining => MaxNew,
                    rev_tokens => [],
                    done => false,
                    layers => Layers
                }}
        end
    catch
        Class:Reason:Stack ->
            {error, Idx, {error, {Class, Reason, Stack}}}
    end.

batch_decode_loop(_BlockState, _Handle, _Opts, []) ->
    [];
batch_decode_loop(BlockState, Handle, Opts, States0) ->
    case lists:all(fun(#{done := Done}) -> Done end, States0) of
        true ->
            States0;
        false ->
            Tokenizer = maps:get(tokenizer, Handle),
            EosIds = stop_token_ids(Handle, Tokenizer),
            StopOnEos = maps:get(stop_on_eos, Opts),
            Active = [
                State
             || State <- States0,
                batch_state_needs_forward(State, EosIds, StopOnEos)
            ],
            NextByIdx =
                case Active of
                    [] ->
                        #{};
                    [Only] ->
                        #{
                            idx := Idx,
                            next := TokenId,
                            caches := Caches,
                            pos := Pos,
                            layers := Layers
                        } =
                            Only,
                        Next = forward_decode_step(
                            BlockState,
                            TokenId,
                            maps:get(embed_table_ref, Handle),
                            Layers,
                            maps:get(final_norm, Handle),
                            maps:get(lm_head, Handle),
                            Caches,
                            Pos,
                            maps:get(rope_freqs, Handle),
                            maps:get(weight_format, Opts)
                        ),
                        #{Idx => Next};
                    _ ->
                        TokenIds = [maps:get(next, State) || State <- Active],
                        KvCaches = [maps:get(caches, State) || State <- Active],
                        Positions = [maps:get(pos, State) || State <- Active],
                        [#{layers := Layers} | _] = Active,
                        case
                            viva_tensor_zig:nt_forward_decode_step_batched(
                                BlockState,
                                TokenIds,
                                maps:get(embed_table_ref, Handle),
                                Layers,
                                maps:get(final_norm, Handle),
                                maps:get(lm_head, Handle),
                                KvCaches,
                                Positions,
                                maps:get(rope_freqs, Handle),
                                maps:get(weight_format, Opts)
                            )
                        of
                            {ok, NextTokens} when is_list(NextTokens) ->
                                maps:from_list([
                                    {maps:get(idx, State), Next}
                                 || {State, Next} <- lists:zip(Active, NextTokens)
                                ]);
                            Error ->
                                error({forward_decode_step_batched_failed, Error})
                        end
                end,
            States = [
                advance_batch_state(State, NextByIdx, EosIds, StopOnEos)
             || State <- States0
            ],
            batch_decode_loop(BlockState, Handle, Opts, States)
    end.

batch_state_needs_forward(
    #{
        done := false, remaining := Remaining, next := Next
    },
    EosIds,
    StopOnEos
) ->
    Remaining > 1 andalso not (StopOnEos andalso lists:member(Next, EosIds));
batch_state_needs_forward(_State, _EosIds, _StopOnEos) ->
    false.

advance_batch_state(State = #{done := true}, _NextByIdx, _EosIds, _StopOnEos) ->
    State;
advance_batch_state(
    State = #{idx := Idx, next := Next, pos := Pos, remaining := Remaining, rev_tokens := Rev},
    NextByIdx,
    EosIds,
    StopOnEos
) ->
    case Remaining =< 0 of
        true ->
            State#{done => true};
        false ->
            Rev1 = [Next | Rev],
            case Remaining =:= 1 orelse (StopOnEos andalso lists:member(Next, EosIds)) of
                true ->
                    State#{remaining => Remaining - 1, rev_tokens => Rev1, done => true};
                false ->
                    Following = maps:get(Idx, NextByIdx),
                    State#{
                        next => Following,
                        pos => Pos + 1,
                        remaining => Remaining - 1,
                        rev_tokens => Rev1
                    }
            end
    end.

batch_state_result(Handle, #{rev_tokens := RevTokens}, TGen) ->
    Tokenizer = maps:get(tokenizer, Handle),
    GeneratedIds = lists:reverse(RevTokens),
    TokCount = length(GeneratedIds),
    GenUs = us() - TGen,
    MsPerToken =
        case TokCount of
            0 -> 0.0;
            _ -> float(GenUs) / 1000.0 / float(TokCount)
        end,
    Text = viva_tensor_tokenizer_ffi:decode(Tokenizer, GeneratedIds),
    {ok, #{
        tokens => GeneratedIds,
        text => Text,
        ms_per_token => MsPerToken,
        total_tokens => TokCount
    }}.

%% Internal: same as generate_argmax, but skips the with_block_state wrap so
%% the caller (e.g. generate_batch) can reuse a single BlockState across calls.
generate_argmax_with_state(Handle, Prompt, Opts, BlockState) ->
    Config = maps:get(config, Handle),
    Tokenizer = maps:get(tokenizer, Handle),
    Layers0 = maps:get(layers, Handle),
    EmbedTable = maps:get(embed_table_ref, Handle),
    FinalNorm = maps:get(final_norm, Handle),
    LmHead = maps:get(lm_head, Handle),
    RopeFreqs = maps:get(rope_freqs, Handle),
    MaxNew = maps:get(max_new_tokens, Opts),
    StopOnEos = maps:get(stop_on_eos, Opts),
    WeightFormat = maps:get(weight_format, Opts),
    Layers = enrich_layers_with_marlin(Layers0, Handle, WeightFormat),
    EosIds = stop_token_ids(Handle, Tokenizer),
    Enc0 = viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt),
    PromptTokens =
        case viva_tensor_tokenizer_ffi:prepends_bos(Tokenizer) of
            true -> [viva_tensor_tokenizer_ffi:bos_id(Tokenizer) | Enc0];
            false -> Enc0
        end,
    MaxSeq = maps:get(max_seq, Config),
    case length(PromptTokens) + MaxNew >= MaxSeq of
        true ->
            {error, {max_sequence_exceeded, length(PromptTokens), MaxNew, MaxSeq}};
        false ->
            generate_argmax_decode_with_state(
                BlockState,
                Handle,
                Tokenizer,
                Layers,
                EmbedTable,
                FinalNorm,
                LmHead,
                RopeFreqs,
                MaxNew,
                StopOnEos,
                WeightFormat,
                PromptTokens,
                EosIds
            )
    end.

generate_argmax_decode_with_state(
    BlockState,
    Handle,
    Tokenizer,
    Layers,
    EmbedTable,
    FinalNorm,
    LmHead,
    RopeFreqs,
    MaxNew,
    StopOnEos,
    WeightFormat,
    PromptTokens,
    EosIds
) ->
    Config = maps:get(config, Handle),
    Caches = new_kv_caches(Config),
    {FirstNext, _} = lists:foldl(
        fun({Pos, TokenId}, {_, CL}) ->
            Next = forward_decode_step(
                BlockState,
                TokenId,
                EmbedTable,
                Layers,
                FinalNorm,
                LmHead,
                CL,
                Pos,
                RopeFreqs,
                WeightFormat
            ),
            {Next, CL}
        end,
        {undefined, Caches},
        lists:zip(lists:seq(0, length(PromptTokens) - 1), PromptTokens)
    ),
    TGen = us(),
    GeneratedIds = decode_loop_decode_fused(
        BlockState,
        FirstNext,
        Caches,
        Layers,
        EmbedTable,
        FinalNorm,
        LmHead,
        RopeFreqs,
        length(PromptTokens),
        MaxNew,
        EosIds,
        StopOnEos,
        WeightFormat,
        []
    ),
    GenUs = us() - TGen,
    TokCount = length(GeneratedIds),
    MsPerToken =
        case TokCount of
            0 -> 0.0;
            _ -> float(GenUs) / 1000.0 / float(TokCount)
        end,
    Text = viva_tensor_tokenizer_ffi:decode(Tokenizer, GeneratedIds),
    {ok, #{
        tokens => GeneratedIds,
        text => Text,
        ms_per_token => MsPerToken,
        total_tokens => TokCount
    }}.

decode_loop_decode_fused(
    _BlockState,
    _NextTok,
    _C,
    _L,
    _E,
    _FN,
    _LH,
    _R,
    _P,
    0,
    _EosIds,
    _StopOnEos,
    _WeightFormat,
    Acc
) ->
    lists:reverse(Acc);
decode_loop_decode_fused(
    BlockState,
    NextTok,
    Caches,
    Layers,
    EmbedTable,
    FinalNorm,
    LmHead,
    RopeFreqs,
    Pos,
    Remaining,
    EosIds,
    StopOnEos,
    WeightFormat,
    Acc
) ->
    case StopOnEos andalso lists:member(NextTok, EosIds) of
        true ->
            lists:reverse([NextTok | Acc]);
        false ->
            Following = forward_decode_step(
                BlockState,
                NextTok,
                EmbedTable,
                Layers,
                FinalNorm,
                LmHead,
                Caches,
                Pos,
                RopeFreqs,
                WeightFormat
            ),
            decode_loop_decode_fused(
                BlockState,
                Following,
                Caches,
                Layers,
                EmbedTable,
                FinalNorm,
                LmHead,
                RopeFreqs,
                Pos + 1,
                Remaining - 1,
                EosIds,
                StopOnEos,
                WeightFormat,
                [NextTok | Acc]
            )
    end.

generate_sampling(Handle, Prompt, Opts) ->
    with_block_state(fun(BlockState) ->
        generate_sampling_with_state(Handle, Prompt, Opts, BlockState)
    end).

generate_sampling_with_state(Handle, Prompt, Opts, BlockState) ->
    Config = maps:get(config, Handle),
    Tokenizer = maps:get(tokenizer, Handle),
    Layers0 = maps:get(layers, Handle),
    WeightFormat = maps:get(weight_format, Opts),
    Layers = enrich_layers_with_marlin(Layers0, Handle, WeightFormat),
    EmbedTable = maps:get(embed_table_ref, Handle),
    FinalNorm = maps:get(final_norm, Handle),
    LmHead = maps:get(lm_head, Handle),
    RopeFreqs = maps:get(rope_freqs, Handle),
    MaxNew = maps:get(max_new_tokens, Opts),
    StopOnEos = maps:get(stop_on_eos, Opts),
    EosIds = stop_token_ids(Handle, Tokenizer),
    Enc0 = viva_tensor_tokenizer_ffi:encode(Tokenizer, Prompt),
    PromptTokens =
        case viva_tensor_tokenizer_ffi:prepends_bos(Tokenizer) of
            true -> [viva_tensor_tokenizer_ffi:bos_id(Tokenizer) | Enc0];
            false -> Enc0
        end,
    MaxSeq = maps:get(max_seq, Config),
    case length(PromptTokens) + MaxNew >= MaxSeq of
        true ->
            {error, {max_sequence_exceeded, length(PromptTokens), MaxNew, MaxSeq}};
        false ->
            Caches = new_kv_caches(Config),
            TopK = sampling_top_k(Opts, maps:get(vocab_size, Config)),
            FirstNext = prefill_sampling(
                BlockState,
                PromptTokens,
                Caches,
                Layers,
                EmbedTable,
                FinalNorm,
                LmHead,
                RopeFreqs,
                TopK,
                Opts
            ),
            TGen = us(),
            GeneratedIds = decode_loop_decode_sampled(
                BlockState,
                FirstNext,
                Caches,
                Layers,
                EmbedTable,
                FinalNorm,
                LmHead,
                RopeFreqs,
                length(PromptTokens),
                MaxNew,
                EosIds,
                StopOnEos,
                TopK,
                Opts,
                []
            ),
            GenUs = us() - TGen,
            TokCount = length(GeneratedIds),
            MsPerToken =
                case TokCount of
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

prefill_sampling(
    BlockState,
    PromptTokens,
    Caches,
    Layers,
    EmbedTable,
    FinalNorm,
    LmHead,
    RopeFreqs,
    TopK,
    Opts
) ->
    LastPos = length(PromptTokens) - 1,
    {Next, _} = lists:foldl(
        fun({Pos, TokenId}, {_, CL}) ->
            Sampled =
                case Pos =:= LastPos of
                    true ->
                        forward_decode_step_sample(
                            BlockState,
                            TokenId,
                            EmbedTable,
                            Layers,
                            FinalNorm,
                            LmHead,
                            CL,
                            Pos,
                            RopeFreqs,
                            TopK,
                            Opts
                        );
                    false ->
                        forward_decode_step(
                            BlockState,
                            TokenId,
                            EmbedTable,
                            Layers,
                            FinalNorm,
                            LmHead,
                            CL,
                            Pos,
                            RopeFreqs,
                            maps:get(weight_format, Opts)
                        )
                end,
            {Sampled, CL}
        end,
        {undefined, Caches},
        lists:zip(lists:seq(0, LastPos), PromptTokens)
    ),
    Next.

decode_loop_decode_sampled(
    _BlockState,
    _NextTok,
    _C,
    _L,
    _E,
    _FN,
    _LH,
    _R,
    _P,
    0,
    _EosIds,
    _StopOnEos,
    _TopK,
    _Opts,
    Acc
) ->
    lists:reverse(Acc);
decode_loop_decode_sampled(
    BlockState,
    NextTok,
    Caches,
    Layers,
    EmbedTable,
    FinalNorm,
    LmHead,
    RopeFreqs,
    Pos,
    Remaining,
    EosIds,
    StopOnEos,
    TopK,
    Opts,
    Acc
) ->
    case StopOnEos andalso lists:member(NextTok, EosIds) of
        true ->
            lists:reverse([NextTok | Acc]);
        false ->
            Following = forward_decode_step_sample(
                BlockState,
                NextTok,
                EmbedTable,
                Layers,
                FinalNorm,
                LmHead,
                Caches,
                Pos,
                RopeFreqs,
                TopK,
                Opts
            ),
            decode_loop_decode_sampled(
                BlockState,
                Following,
                Caches,
                Layers,
                EmbedTable,
                FinalNorm,
                LmHead,
                RopeFreqs,
                Pos + 1,
                Remaining - 1,
                EosIds,
                StopOnEos,
                TopK,
                Opts,
                [NextTok | Acc]
            )
    end.

forward_decode_step(
    BlockState,
    TokenId,
    EmbedTable,
    Layers,
    FinalNorm,
    LmHead,
    Caches,
    Pos,
    RopeFreqs,
    WeightFormat
) ->
    case
        viva_tensor_zig:nt_forward_decode_step(
            BlockState,
            TokenId,
            EmbedTable,
            Layers,
            FinalNorm,
            LmHead,
            Caches,
            Pos,
            RopeFreqs,
            WeightFormat
        )
    of
        {ok, NextToken} when is_integer(NextToken) ->
            NextToken;
        Error ->
            error({forward_decode_step_failed, Error})
    end.

forward_decode_step_sample(
    BlockState,
    TokenId,
    EmbedTable,
    Layers,
    FinalNorm,
    LmHead,
    Caches,
    Pos,
    RopeFreqs,
    TopK,
    Opts
) ->
    case
        viva_tensor_zig:nt_forward_decode_step_topk(
            BlockState,
            TokenId,
            EmbedTable,
            Layers,
            FinalNorm,
            LmHead,
            Caches,
            Pos,
            RopeFreqs,
            TopK,
            maps:get(weight_format, Opts)
        )
    of
        {ok, {IndicesBin, ValuesBin}} when is_binary(IndicesBin), is_binary(ValuesBin) ->
            Indices = decode_int32_le(IndicesBin),
            Logits = decode_float32_le(ValuesBin),
            Pick = llama_sampling:sample(Logits, sampling_opts_for_pos(Opts, Pos)),
            lists:nth(Pick + 1, Indices);
        Error ->
            error({forward_decode_step_topk_failed, Error})
    end.

sampling_top_k(Opts, VocabSize) ->
    Requested =
        case maps:get(top_k, Opts) of
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
    Tied =
        case maps:get(<<"tie_word_embeddings">>, FileConfig, false) of
            true -> true;
            _ -> false
        end,
    LmHidden =
        case Tied of
            true ->
                HiddenSize0;
            false ->
                {_, LH} = shape2(Header, <<"lm_head.weight">>),
                LH
        end,
    HiddenSize = int_config(FileConfig, <<"hidden_size">>, HiddenSize0),
    VocabSize = int_config(FileConfig, <<"vocab_size">>, VocabSize0),
    NumHeads = int_config(
        FileConfig,
        <<"num_attention_heads">>,
        max(1, HiddenSize div ?DEFAULT_HEAD_DIM)
    ),
    NumKvHeads = int_config(FileConfig, <<"num_key_value_heads">>, NumHeads),
    %% Prefer the explicit head_dim from config.json (some models set
    %% head_dim != hidden_size / num_heads); fall back to the derived value.
    HeadDim = int_config(
        FileConfig,
        <<"head_dim">>,
        case NumHeads of
            0 -> ?DEFAULT_HEAD_DIM;
            _ -> max(1, HiddenSize div NumHeads)
        end
    ),
    KvDim = NumKvHeads * HeadDim,
    FfnSize = int_config_lazy(
        FileConfig,
        <<"intermediate_size">>,
        fun() -> first_layer_ffn(Header) end
    ),
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
        rope_scaling => rope_scaling_config(FileConfig),
        eos_ids => resolve_eos_ids(FileConfig, SafetensorsPath),
        max_seq => opt(Opts, max_seq, ?DEFAULT_MAX_SEQ),
        tie_word_embeddings => Tied
    }.

%% Stop-token ids. Llama-3 instruct lists several (<|end_of_text|>, <|eom_id|>,
%% <|eot_id|>) in generation_config.json / config.json's eos_token_id, which may
%% be a single int or a list. Returns a deduped list (possibly empty); the
%% tokenizer's primary eos is added at generate time.
resolve_eos_ids(FileConfig, SafetensorsPath) ->
    GenConfig = read_generation_config(SafetensorsPath),
    FromConfig = eos_to_list(maps:get(<<"eos_token_id">>, FileConfig, undefined)),
    FromGen = eos_to_list(maps:get(<<"eos_token_id">>, GenConfig, undefined)),
    lists:usort(FromConfig ++ FromGen).

eos_to_list(I) when is_integer(I) -> [I];
eos_to_list(L) when is_list(L) -> [X || X <- L, is_integer(X)];
eos_to_list(_) -> [].

read_generation_config(SafetensorsPath) ->
    Path = filename:join(model_dir(SafetensorsPath), "generation_config.json"),
    case file:read_file(Path) of
        {ok, Bin} ->
            try
                json:decode(Bin)
            catch
                _:_ -> #{}
            end;
        _ ->
            #{}
    end.

%% All token ids that stop generation: the tokenizer's primary eos plus any
%% extra ids resolved from (generation_)config.json.
stop_token_ids(Handle, Tokenizer) ->
    Primary = viva_tensor_tokenizer_ffi:eos_id(Tokenizer),
    Extra = maps:get(eos_ids, maps:get(config, Handle, #{}), []),
    lists:usort([Primary | Extra]).

read_hf_config(SafetensorsPath) ->
    ConfigPath = filename:join(model_dir(SafetensorsPath), "config.json"),
    case file:read_file(ConfigPath) of
        {ok, Bin} ->
            try
                json:decode(Bin)
            catch
                Class:Reason ->
                    %% A present-but-malformed config.json must not be silently
                    %% swallowed: every dimension would silently fall back to a
                    %% (possibly wrong) default. Surface it, then degrade.
                    logger:warning(
                        "viva_tensor: failed to parse ~ts (~p:~p); "
                        "falling back to header-derived config",
                        [ConfigPath, Class, Reason]
                    ),
                    #{}
            end;
        {error, enoent} ->
            %% No config.json: legitimate, dimensions are derived from the
            %% safetensors header.
            #{};
        {error, Reason} ->
            logger:warning(
                "viva_tensor: cannot read ~ts (~p); "
                "falling back to header-derived config",
                [ConfigPath, Reason]
            ),
            #{}
    end.

detect_num_layers(Header) ->
    detect_num_layers(Header, 0).

detect_num_layers(Header, I) ->
    Name = list_to_binary(
        "model.layers." ++ integer_to_list(I) ++
            ".input_layernorm.weight"
    ),
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

%% Concurrency for the parallel layer build, capped by BOTH the dirty-CPU
%% scheduler count AND available memory. Each concurrent layer holds its seven
%% linear weights in FP32 simultaneously (until concat+prepack frees them), so
%% peak host scratch is roughly Limit * per-layer-fp32-bytes. At full scheduler
%% concurrency a 7B (hidden 4096, ffn 11008 -> ~0.8 GB/layer) would need ~26 GB
%% of scratch and thrash/OOM; a 1B (~0.24 GB/layer) is unaffected.
load_concurrency_limit(Config, NumLayers) ->
    Schedulers = erlang:system_info(dirty_cpu_schedulers),
    PerLayer = layer_fp32_scratch_bytes(Config),
    Budget = trunc(available_mem_bytes() * 0.40),
    ByMem = max(1, Budget div max(1, PerLayer)),
    max(1, min(NumLayers, min(Schedulers, ByMem))).

%% Peak FP32 host scratch for one layer: q,o (H*H) + k,v (H*Kv) + gate,up
%% (H*Ffn) + down (Ffn*H), 4 bytes each, plus the same again for the bf16->fp32
%% staging buffer that briefly coexists.
layer_fp32_scratch_bytes(Config) ->
    H = maps:get(hidden_size, Config),
    Kv = maps:get(kv_dim, Config),
    Ffn = maps:get(ffn_size, Config),
    Elems = 2 * H * H + 2 * H * Kv + 3 * H * Ffn,
    Elems * 4 * 2.

%% MemAvailable from /proc/meminfo (Linux); 8 GB fallback elsewhere.
available_mem_bytes() ->
    case file:read_file("/proc/meminfo") of
        {ok, Bin} ->
            case re:run(Bin, "MemAvailable:\\s+([0-9]+) kB", [{capture, [1], list}]) of
                {match, [Kb]} -> list_to_integer(Kb) * 1024;
                _ -> 8 * 1024 * 1024 * 1024
            end;
        _ ->
            8 * 1024 * 1024 * 1024
    end.

%% Build transformer layers concurrently. The per-tensor bf16->fp32 +
%% transpose + FP8 quantize is CPU-bound and dominates load time; running
%% layers across dirty CPU schedulers cuts wall time roughly by the core
%% count. Concurrency is capped at the dirty-CPU scheduler count so peak host
%% scratch (the FP32 staging buffers) stays bounded for large models.
parallel_build_layers(Header, Config, NumLayers) ->
    Limit = load_concurrency_limit(Config, NumLayers),
    Indices = lists:seq(0, NumLayers - 1),
    lists:append([
        pmap_ordered(fun(I) -> build_layer_blocked(Header, I, Config) end, Chunk)
     || Chunk <- chunk_list(Indices, Limit)
    ]).

%% Run Fun over Items concurrently (one process each), returning results in
%% input order and re-raising the first worker error in the caller.
pmap_ordered(Fun, Items) ->
    Parent = self(),
    Tagged = [{erlang:make_ref(), I} || I <- Items],
    lists:foreach(
        fun({Ref, I}) ->
            spawn_link(fun() ->
                R =
                    try
                        {ok, Fun(I)}
                    catch
                        Class:Reason:Stack -> {error, {Class, Reason, Stack}}
                    end,
                Parent ! {Ref, R}
            end)
        end,
        Tagged
    ),
    [
        receive
            {Ref, {ok, V}} -> V;
            {Ref, {error, {C, E, S}}} -> erlang:raise(C, E, S)
        end
     || {Ref, _} <- Tagged
    ].

chunk_list([], _N) ->
    [];
chunk_list(List, N) ->
    {Head, Tail} = lists:split(min(N, length(List)), List),
    [Head | chunk_list(Tail, N)].

%% Spawn Fun in a linked worker; await_build/1 collects its result, re-raising
%% any error in the caller.
async_build(Fun) ->
    Parent = self(),
    Ref = erlang:make_ref(),
    spawn_link(fun() ->
        R =
            try
                {ok, Fun()}
            catch
                Class:Reason:Stack -> {error, {Class, Reason, Stack}}
            end,
        Parent ! {Ref, R}
    end),
    Ref.

await_build(Ref) ->
    receive
        {Ref, {ok, V}} -> V;
        {Ref, {error, {C, E, S}}} -> erlang:raise(C, E, S)
    end.

%% Load + FP8-prepack the lm_head (tied to embed_tokens, or a dedicated
%% lm_head.weight).
build_lm_head_packed(Header, Config) ->
    LmHeadName =
        case maps:get(tie_word_embeddings, Config, false) of
            true -> <<"model.embed_tokens.weight">>;
            false -> <<"lm_head.weight">>
        end,
    LmHead = load_linear(
        Header, LmHeadName, maps:get(vocab_size, Config), maps:get(hidden_size, Config)
    ),
    prepack_blocked(
        LmHead,
        maps:get(hidden_size, Config),
        maps:get(vocab_size, Config),
        maps:get(block_size, Config)
    ).

build_layer_blocked(Header, LayerIdx, Config) ->
    Prefix = "model.layers." ++ integer_to_list(LayerIdx) ++ ".",
    P = fun(Suffix) -> list_to_binary(Prefix ++ Suffix) end,
    require_tensors(
        Header,
        [
            P("self_attn.q_proj.weight"),
            P("self_attn.k_proj.weight"),
            P("self_attn.v_proj.weight"),
            P("self_attn.o_proj.weight"),
            P("mlp.gate_proj.weight"),
            P("mlp.up_proj.weight"),
            P("mlp.down_proj.weight"),
            P("input_layernorm.weight"),
            P("post_attention_layernorm.weight")
        ],
        LayerIdx
    ),
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
    QkvBias = concat_bias([
        load_bias_opt(Header, P("self_attn.q_proj.bias")),
        load_bias_opt(Header, P("self_attn.k_proj.bias")),
        load_bias_opt(Header, P("self_attn.v_proj.bias"))
    ]),
    QKVProj = concat_linear_rows([{QProj, Hidden}, {KProj, KvDim}, {VProj, KvDim}]),
    GateUpProj = concat_linear_rows([{GateProj, Ffn}, {UpProj, Ffn}]),
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
        qkv => prepack_blocked_bias(QKVProj, Hidden, Hidden + KvDim + KvDim, BlockSize, QkvBias),
        gate_up => prepack_blocked(GateUpProj, Hidden, Ffn + Ffn, BlockSize),
        down => prepack_blocked(DownProj, Ffn, Hidden, BlockSize)
    }.

prepack_all_linears_marlin(Handle) ->
    try
        SafetensorsPath = maps:get(safetensors_path, Handle),
        Config = maps:get(config, Handle),
        {ok, Header} = viva_tensor_safetensors_ffi:open(SafetensorsPath),
        LayerIds = lists:seq(0, maps:get(num_layers, Config) - 1),
        lists:foldl(
            fun
                (LayerIdx, {ok, Acc}) ->
                    case prepack_layer_linears_marlin(Header, LayerIdx, Config) of
                        {ok, LayerHandles} -> {ok, maps:put(LayerIdx, LayerHandles, Acc)};
                        {error, _} = Error -> Error
                    end;
                (_LayerIdx, Error) ->
                    Error
            end,
            {ok, #{}},
            LayerIds
        )
    catch
        Class:Reason:Stack ->
            {error, {Class, Reason, Stack}}
    end.

prepack_layer_linears_marlin(Header, LayerIdx, Config) ->
    Prefix = "model.layers." ++ integer_to_list(LayerIdx) ++ ".",
    P = fun(Suffix) -> list_to_binary(Prefix ++ Suffix) end,
    Hidden = maps:get(hidden_size, Config),
    KvDim = maps:get(kv_dim, Config),
    Ffn = maps:get(ffn_size, Config),
    with_raw_marlin_linear(Header, P("self_attn.q_proj.weight"), fun(Q) ->
        with_raw_marlin_linear(Header, P("self_attn.k_proj.weight"), fun(K) ->
            with_raw_marlin_linear(Header, P("self_attn.v_proj.weight"), fun(V) ->
                QKVWeight = iolist_to_binary([Q, K, V]),
                with_marlin_pack(
                    QKVWeight,
                    compute_marlin_scales(
                        QKVWeight, Hidden + KvDim + KvDim, Hidden, ?MARLIN_GROUPSIZE
                    ),
                    Hidden + KvDim + KvDim,
                    Hidden,
                    fun(QKVHandle) ->
                        with_raw_marlin_pack(
                            Header, P("self_attn.o_proj.weight"), Hidden, Hidden, fun(OHandle) ->
                                with_raw_marlin_pack(
                                    Header, P("mlp.gate_proj.weight"), Ffn, Hidden, fun(GateHandle) ->
                                        with_raw_marlin_pack(
                                            Header, P("mlp.up_proj.weight"), Ffn, Hidden, fun(
                                                UpHandle
                                            ) ->
                                                with_raw_marlin_pack(
                                                    Header,
                                                    P("mlp.down_proj.weight"),
                                                    Hidden,
                                                    Ffn,
                                                    fun(DownHandle) ->
                                                        {ok, #{
                                                            qkv => QKVHandle,
                                                            o => OHandle,
                                                            gate => GateHandle,
                                                            up => UpHandle,
                                                            down => DownHandle
                                                        }}
                                                    end
                                                )
                                            end
                                        )
                                    end
                                )
                            end
                        )
                    end
                )
            end)
        end)
    end).

with_raw_marlin_pack(Header, Name, K, N, Fun) ->
    with_raw_marlin_linear(Header, Name, fun(WeightFp16) ->
        with_marlin_pack(
            WeightFp16,
            compute_marlin_scales(WeightFp16, K, N, ?MARLIN_GROUPSIZE),
            K,
            N,
            Fun
        )
    end).

with_raw_marlin_linear(Header, Name, Fun) ->
    case load_raw_fp16_linear(Header, Name) of
        {ok, WeightFp16} -> Fun(WeightFp16);
        {error, _} = Error -> Error
    end.

load_raw_fp16_linear(Header, Name) ->
    case viva_tensor_safetensors_ffi:read_tensor_raw(Header, Name) of
        {ok, <<"F16">>, Bin} -> {ok, Bin};
        {ok, <<"BF16">>, Bin} -> {ok, Bin};
        {ok, Dtype, _Bin} -> {error, {unsupported_dtype, Dtype}};
        Error -> Error
    end.

bf16_binary_to_fp16_binary(Bin) ->
    <<<<(fp16_encode(bf16_to_float(U))):16/unsigned-little>> || <<U:16/unsigned-little>> <= Bin>>.

with_marlin_pack_from_safetensors(Header, Name, OutF, InF, Fun) ->
    case load_marlin_linear(Header, Name, OutF, InF) of
        {ok, Linear} ->
            with_marlin_pack(
                maps:get(weight, Linear),
                maps:get(scales, Linear),
                InF,
                OutF,
                Fun
            );
        {error, _} = Error ->
            Error
    end.

with_marlin_linear(Header, Name, OutF, InF, Fun) ->
    case load_marlin_linear(Header, Name, OutF, InF) of
        {ok, Linear} -> Fun(Linear);
        {error, _} = Error -> Error
    end.

load_marlin_linear(Header, Name, OutF, InF) ->
    case load_linear_fp16_transposed(Header, Name, OutF, InF) of
        {ok, WeightFp16} ->
            {ok, #{
                weight => WeightFp16,
                scales => compute_marlin_scales(WeightFp16, InF, OutF, ?MARLIN_GROUPSIZE)
            }};
        {error, _} = Error ->
            Error
    end.

with_marlin_pack(WeightFp16, ScalesFp16, K, N, Fun) ->
    case quantize_and_pack(WeightFp16, ScalesFp16, K, N) of
        {ok, Handle} -> Fun(Handle);
        {error, _} = Error -> Error
    end.

quantize_and_pack(WeightFp16, ScalesFp16, K, N) ->
    case K rem ?MARLIN_GROUPSIZE =:= 0 andalso N rem 256 =:= 0 of
        true ->
            try
                normalize_marlin_pack_result(
                    viva_tensor_zig:marlin_w4a16_prepack(
                        WeightFp16,
                        ScalesFp16,
                        K,
                        N,
                        ?MARLIN_GROUPSIZE
                    )
                )
            catch
                Class:Reason:Stack -> {error, {marlin_prepack_failed, Class, Reason, Stack}}
            end;
        false ->
            {error, {invalid_marlin_shape, K, N, ?MARLIN_GROUPSIZE}}
    end.

normalize_marlin_pack_result({ok, Resource}) when is_reference(Resource) ->
    {ok, Resource};
normalize_marlin_pack_result({error, Reason, Code}) when is_integer(Code) ->
    {error, {Reason, Code}};
normalize_marlin_pack_result({error, Reason}) ->
    {error, Reason};
normalize_marlin_pack_result(Resource) when is_reference(Resource) ->
    {ok, Resource};
normalize_marlin_pack_result(Other) ->
    {error, {unexpected_marlin_prepack_result, Other}}.

load_linear_fp16_transposed(Header, Name, OutF, InF) ->
    case viva_tensor_safetensors_ffi:read_tensor_raw(Header, Name) of
        {ok, Dtype, Bin} ->
            transpose_raw_to_fp16(Dtype, Bin, OutF, InF);
        Error ->
            Error
    end.

transpose_raw_to_fp16(Dtype, Bin, OutF, InF) when is_binary(Bin) ->
    ExpectedBytes = OutF * InF * 2,
    case byte_size(Bin) of
        ExpectedBytes ->
            {ok,
                iolist_to_binary([
                    transpose_fp16_row(Dtype, Bin, K, OutF, InF)
                 || K <- lists:seq(0, InF - 1)
                ])};
        _ ->
            {error, {size_mismatch, Dtype, byte_size(Bin), ExpectedBytes}}
    end.

transpose_fp16_row(<<"F16">>, Bin, K, OutF, InF) ->
    [binary_part(Bin, (N * InF + K) * 2, 2) || N <- lists:seq(0, OutF - 1)];
transpose_fp16_row(<<"BF16">>, Bin, K, OutF, InF) ->
    [
        begin
            <<U:16/unsigned-little>> = binary_part(Bin, (N * InF + K) * 2, 2),
            <<(fp16_encode(bf16_to_float(U))):16/unsigned-little>>
        end
     || N <- lists:seq(0, OutF - 1)
    ];
transpose_fp16_row(Dtype, _Bin, _K, _OutF, _InF) ->
    error({unsupported_dtype, Dtype}).

concat_fp16_columns(Parts, Rows) ->
    BytesPerHalf = 2,
    iolist_to_binary([
        [
            binary_part(Bin, Row * Cols * BytesPerHalf, Cols * BytesPerHalf)
         || {Bin, Cols} <- Parts
        ]
     || Row <- lists:seq(0, Rows - 1)
    ]).

compute_marlin_scales(WeightFp16, K, N, Groupsize) ->
    case K * N =< ?MARLIN_EXACT_SCALE_MAX_ELEMS of
        true -> compute_marlin_scales_exact(WeightFp16, K, N, Groupsize);
        false -> compute_marlin_scales_sampled(WeightFp16, K, N, Groupsize)
    end.

compute_marlin_scales_exact(WeightFp16, K, N, Groupsize) ->
    Groups = K div Groupsize,
    iolist_to_binary([
        [
            <<
                (fp16_encode(marlin_scale_for_column(WeightFp16, G, Col, N, Groupsize))):16/unsigned-little
            >>
         || Col <- lists:seq(0, N - 1)
        ]
     || G <- lists:seq(0, Groups - 1)
    ]).

compute_marlin_scales_sampled(WeightFp16, K, N, Groupsize) ->
    Groups = K div Groupsize,
    SampleElems = min(?MARLIN_SCALE_SAMPLE_ELEMS, K * N),
    MaxAbs = sampled_fp16_max_abs(WeightFp16, SampleElems, 0.0),
    Scale0 = MaxAbs / 7.0,
    Scale =
        case Scale0 < ?MARLIN_MIN_SCALE of
            true -> ?MARLIN_MIN_SCALE;
            false -> Scale0
        end,
    binary:copy(<<(fp16_encode(Scale)):16/unsigned-little>>, Groups * N).

sampled_fp16_max_abs(_WeightFp16, 0, Acc) ->
    Acc;
sampled_fp16_max_abs(WeightFp16, Remaining, Acc) ->
    Offset = (?MARLIN_SCALE_SAMPLE_ELEMS - Remaining) * 2,
    <<H:16/unsigned-little>> = binary_part(WeightFp16, Offset, 2),
    sampled_fp16_max_abs(WeightFp16, Remaining - 1, max(Acc, abs_float(fp16_to_float(H)))).

marlin_scale_for_column(WeightFp16, Group, Col, N, Groupsize) ->
    MaxAbs = marlin_group_col_max_abs(WeightFp16, Group * Groupsize, Groupsize, Col, N, 0.0),
    Scale0 = MaxAbs / 7.0,
    case Scale0 < ?MARLIN_MIN_SCALE of
        true -> ?MARLIN_MIN_SCALE;
        false -> Scale0
    end.

marlin_group_col_max_abs(_WeightFp16, _Row, 0, _Col, _N, Acc) ->
    Acc;
marlin_group_col_max_abs(WeightFp16, Row, Remaining, Col, N, Acc) ->
    <<H:16/unsigned-little>> = binary_part(WeightFp16, (Row * N + Col) * 2, 2),
    V = abs_float(fp16_to_float(H)),
    marlin_group_col_max_abs(WeightFp16, Row + 1, Remaining - 1, Col, N, max(Acc, V)).

bf16_to_float(U) ->
    <<F:32/float-little>> = <<0:16/unsigned-little, U:16/unsigned-little>>,
    F.

fp16_to_float(H) ->
    Sign = (H bsr 15) band 1,
    Exp = (H bsr 10) band 16#1F,
    Frac = H band 16#3FF,
    V =
        case Exp of
            0 when Frac =:= 0 -> 0.0;
            0 -> (Frac / 1024.0) * math:pow(2.0, -14.0);
            16#1F when Frac =:= 0 -> 1.7976931348623157e308;
            16#1F -> 0.0;
            _ -> (1.0 + Frac / 1024.0) * math:pow(2.0, float(Exp - 15))
        end,
    case Sign of
        0 -> V;
        1 -> -V
    end.

fp16_encode(F) when is_float(F) ->
    <<S:1, E:8, M:23>> = <<F:32/float-big>>,
    case E of
        0 ->
            S bsl 15;
        255 ->
            (S bsl 15) bor (16#1F bsl 10) bor
                (case M of
                    0 -> 0;
                    _ -> 1
                end);
        _ ->
            UnbiasedE = E - 127,
            case UnbiasedE of
                X when X < -24 ->
                    S bsl 15;
                X when X < -14 ->
                    MantFull = M bor 16#800000,
                    Shift = -1 - X,
                    (S bsl 15) bor (MantFull bsr Shift);
                X when X > 15 ->
                    (S bsl 15) bor (16#1F bsl 10);
                X ->
                    Eh = X + 15,
                    Mh = M bsr 13,
                    Round = (M bsr 12) band 1,
                    Sticky =
                        case M band 16#FFF of
                            0 -> 0;
                            _ -> 1
                        end,
                    {Mh2, Eh2} =
                        case Round of
                            0 ->
                                {Mh, Eh};
                            1 when Sticky =:= 1 ->
                                fp16_round_mantissa(Mh, Eh);
                            1 when (Mh band 1) =:= 1 ->
                                fp16_round_mantissa(Mh, Eh);
                            1 ->
                                {Mh, Eh}
                        end,
                    (S bsl 15) bor (Eh2 bsl 10) bor Mh2
            end
    end;
fp16_encode(0) ->
    0;
fp16_encode(F) when is_integer(F) ->
    fp16_encode(float(F)).

fp16_round_mantissa(Mh, Eh) ->
    Mh1 = Mh + 1,
    case Mh1 of
        1024 -> {0, Eh + 1};
        _ -> {Mh1, Eh}
    end.

abs_float(V) when V < 0.0 -> -V;
abs_float(V) -> V.

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

%% Load a linear weight in HF-native [out_features, in_features] row-major
%% layout (no transpose). The FP8 prepack consumes this directly via layout=1,
%% which both skips the transpose and makes the quantize loop contiguous.
load_linear(Header, Name, _OutF, _InF) ->
    {ok, Fp32} = viva_tensor_safetensors_ffi:read_tensor_fp32(Header, Name),
    Fp32.

%% Concatenate per-output-channel weights along the output (row) axis. With
%% native [out, in] row-major layout each part's rows are already contiguous,
%% so this is a plain append (the old column-concat reorganized row by row).
concat_linear_rows(Parts) ->
    list_to_binary([Bin || {Bin, _OutF} <- Parts]).

load_rmsnorm_bin(Header, Name) ->
    {ok, Fp32} = viva_tensor_safetensors_ffi:read_tensor_fp32(Header, Name),
    Fp32.

load_embed_table_resource(Header, Config) ->
    {ok, Dtype, Bin} = viva_tensor_safetensors_ffi:read_tensor_raw(
        Header, <<"model.embed_tokens.weight">>
    ),
    NewTable =
        case Dtype of
            <<"BF16">> -> fun viva_tensor_zig:nt_embedding_table_new/3;
            <<"F16">> -> fun viva_tensor_zig:nt_embedding_table_new_fp16/3;
            Unsupported -> error({unsupported_dtype, Unsupported})
        end,
    case NewTable(Bin, maps:get(vocab_size, Config), maps:get(hidden_size, Config)) of
        {ok, Resource} when is_reference(Resource) -> Resource;
        LoadError -> error({embedding_table_resource_failed, LoadError})
    end.

prepack_blocked(Bin, InF, OutF, BlockSize) when is_binary(Bin) ->
    case viva_tensor_zig:nt_prepack_fp8_blocked(Bin, [InF, OutF], BlockSize, 1) of
        {ok, {Resource, _, _, _}} -> Resource;
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({prepack_blocked_failed, Other})
    end.

%% Like prepack_blocked but with an optional FP32 per-output-channel bias
%% (Qwen-style QKV bias). nil falls back to the biasless path.
prepack_blocked_bias(Bin, InF, OutF, BlockSize, nil) ->
    prepack_blocked(Bin, InF, OutF, BlockSize);
prepack_blocked_bias(Bin, InF, OutF, BlockSize, Bias) when is_binary(Bias) ->
    case viva_tensor_zig:nt_prepack_fp8_blocked(Bin, [InF, OutF], BlockSize, 1, Bias) of
        {ok, {Resource, _, _, _}} -> Resource;
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({prepack_blocked_bias_failed, Other})
    end.

%% Read an optional FP32 bias tensor; returns nil when the tensor is absent
%% (e.g. Llama has no QKV bias, Qwen does).
load_bias_opt(Header, Name) ->
    case viva_tensor_safetensors_ffi:read_tensor_fp32(Header, Name) of
        {ok, Bin} -> Bin;
        _ -> nil
    end.

%% Concatenate q/k/v biases into one [out_q + out_k + out_v] FP32 binary, or
%% nil when none are present (matching the fused QKV weight layout).
concat_bias(Biases) ->
    case lists:any(fun(B) -> is_binary(B) end, Biases) of
        false -> nil;
        true -> list_to_binary([B || B <- Biases, is_binary(B)])
    end.

new_kv_caches(Config) ->
    [
        begin
            {ok, Cache} = viva_tensor_zig:nt_kv_cache_new(
                maps:get(max_seq, Config), maps:get(kv_dim, Config)
            ),
            Cache
        end
     || _ <- lists:seq(1, maps:get(num_layers, Config))
    ].

%% Read rope_scaling from HF config.json. Returns `none` for classic RoPE
%% (theta only), or `{llama3, Factor, LowFreqF, HighFreqF, OrigMax}` for the
%% Llama-3 "NTK-by-parts" frequency rescaling that Llama-3.1/3.2 require.
rope_scaling_config(FileConfig) ->
    case maps:get(<<"rope_scaling">>, FileConfig, undefined) of
        Scaling when is_map(Scaling) ->
            RopeType = maps:get(
                <<"rope_type">>, Scaling, maps:get(<<"type">>, Scaling, undefined)
            ),
            case RopeType of
                <<"llama3">> ->
                    {llama3, float_config(Scaling, <<"factor">>, 8.0),
                        float_config(Scaling, <<"low_freq_factor">>, 1.0),
                        float_config(Scaling, <<"high_freq_factor">>, 4.0),
                        float_config(
                            Scaling, <<"original_max_position_embeddings">>, 8192.0
                        )};
                _ ->
                    none
            end;
        _ ->
            none
    end.

precompute_rope_freqs_bin(HeadDim, Theta) ->
    precompute_rope_freqs_bin(HeadDim, Theta, none).

precompute_rope_freqs_bin(HeadDim, Theta, Scaling) ->
    Half = HeadDim div 2,
    Freqs = [
        scale_rope_freq(math:pow(Theta, -2.0 * float(I) / float(HeadDim)), Scaling)
     || I <- lists:seq(0, Half - 1)
    ],
    <<<<F:32/float-little>> || F <- Freqs>>.

%% Llama-3 "NTK-by-parts" rope scaling, faithful to the HuggingFace reference:
%% high-frequency components are kept, low-frequency ones are divided by Factor,
%% and a smooth interpolation bridges the band in between.
scale_rope_freq(Freq, none) ->
    Freq;
scale_rope_freq(Freq, {llama3, Factor, LowF, HighF, OrigMax}) ->
    LowWavelen = OrigMax / LowF,
    HighWavelen = OrigMax / HighF,
    Wavelen = 2.0 * math:pi() / Freq,
    if
        Wavelen < HighWavelen ->
            Freq;
        Wavelen > LowWavelen ->
            Freq / Factor;
        true ->
            Smooth = (OrigMax / Wavelen - LowF) / (HighF - LowF),
            (1.0 - Smooth) * Freq / Factor + Smooth * Freq
    end.

generation_options(Opts) ->
    #{
        max_new_tokens => opt(Opts, max_new_tokens, 50),
        temperature => float_opt(Opts, temperature, 0.0),
        top_k => opt(Opts, top_k, infinity),
        top_p => float_opt(Opts, top_p, 1.0),
        seed => opt(Opts, seed, 42),
        stop_on_eos => opt(Opts, stop_on_eos, true),
        weight_format => weight_format_atom(opt(Opts, weight_format, fp8_w8a16))
    }.

%% Inject top-level marlin_handles map (indexed by LayerIdx) into each layer map
%% so the NIF can find handles via decode_layer_has_marlin_weights + get_layer_marlin.
enrich_layers_with_marlin(Layers, _Handle, fp8_w8a16) ->
    Layers;
enrich_layers_with_marlin(Layers, Handle, marlin_w4a16) ->
    MarlinHandles = maps:get(marlin_handles, Handle, #{}),
    {Enriched, _} = lists:mapfoldl(
        fun(LayerMap, Idx) ->
            PerLayer = maps:get(Idx, MarlinHandles, #{}),
            {LayerMap#{marlin_handles => PerLayer}, Idx + 1}
        end,
        0,
        Layers
    ),
    Enriched.

weight_format_atom(fp8_w8a16) -> fp8_w8a16;
weight_format_atom(marlin_w4a16) -> marlin_w4a16;
weight_format_atom(<<"fp8_w8a16">>) -> fp8_w8a16;
weight_format_atom(<<"marlin_w4a16">>) -> marlin_w4a16;
weight_format_atom("fp8_w8a16") -> fp8_w8a16;
weight_format_atom("marlin_w4a16") -> marlin_w4a16;
weight_format_atom(_) -> invalid_weight_format.

generate_batch_timeout(Opts) when is_map(Opts) ->
    opt(Opts, timeout, 60000);
generate_batch_timeout(_Opts) ->
    60000.

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
                true ->
                    Inferred;
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
        {ok, Value} ->
            Value;
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
