%% viva_tensor_tokenizer_ffi.erl
%% ------------------------------------------------------------------
%% LLaMA / TinyLlama SentencePiece-style BPE tokenizer in pure Erlang.
%%
%% Implements the byte-level-with-▁-prefix BPE used by TinyLlama-1.1B
%% (HuggingFace `tokenizer.json`, model.type = "BPE", byte_fallback=true).
%%
%% Normalizer pipeline (from tokenizer.json):
%%   1. Prepend "▁" to the text.
%%   2. Replace every ASCII space with "▁".
%% Decoder pipeline:
%%   1. Replace every "▁" with " ".
%%   2. ByteFallback: collapse runs of <0xNN> tokens into raw bytes.
%%   3. Fuse: concatenate.
%%   4. Strip(start=1): drop one leading space.
%%
%% Reference (transformers / sentencepiece):
%%   "Hello world"          -> [1, 15043, 3186]
%%   "Olá mundo"            -> [1, 7137, 29976, 13864]
%%   "The quick brown fox"  -> [1, 450, 4996, 17354, 1701, 29916]
%%   "Como vai você?"       -> [1, 17295, 325, 1794, 7931, 30037, 29973]
%%   BOS=1, EOS=2, UNK=0
%%
%% Note: this module returns IDs WITHOUT the BOS prefix. Callers that need
%% the LLaMA `<s>` token must prepend `bos_id(State)` themselves (matches
%% the contract requested by `llama_forward:run_n/1`).
%% ------------------------------------------------------------------
-module(viva_tensor_tokenizer_ffi).

-export([
    load/1,
    encode/2,
    decode/2,
    bos_id/1,
    eos_id/1,
    unk_id/1
]).

%% State shape — opaque to callers.
%% #{
%%    vocab       => #{binary() => integer()},   %% token -> id
%%    id_to_token => #{integer() => binary()},   %% id    -> token
%%    merges      => #{{binary(), binary()} => integer()}, %% pair -> rank
%%    bos         => integer(),
%%    eos         => integer(),
%%    unk         => integer(),
%%    byte_ids    => tuple()  %% 256-tuple: byte value -> id of "<0xNN>"
%% }

%% ---------------------------------------------------------------- load/1
load(Path) ->
    case file:read_file(Path) of
        {error, Reason} ->
            {error, {read_file, Reason}};
        {ok, Bin} ->
            try
                Json = json:decode(Bin),
                Hints = read_tokenizer_config(Path),
                build_state(Json, Hints)
            catch
                Class:Err:Stack ->
                    {error, {parse, Class, Err, Stack}}
            end
    end.

%% Best-effort read of tokenizer_config.json (same dir) for the canonical
%% bos_token / eos_token names. Returns #{} on any failure so loading a bare
%% tokenizer.json keeps working.
read_tokenizer_config(TokenizerPath) ->
    ConfigPath = filename:join(
        filename:dirname(TokenizerPath), "tokenizer_config.json"
    ),
    case file:read_file(ConfigPath) of
        {ok, Bin} ->
            try
                C = json:decode(Bin),
                #{
                    bos_name => special_token_name(
                        maps:get(<<"bos_token">>, C, undefined)
                    ),
                    eos_name => special_token_name(
                        maps:get(<<"eos_token">>, C, undefined)
                    )
                }
            catch
                _:_ -> #{}
            end;
        _ ->
            #{}
    end.

%% A HF special token is either a plain string or an object with "content".
special_token_name(N) when is_binary(N) -> N;
special_token_name(#{<<"content">> := C}) when is_binary(C) -> C;
special_token_name(_) -> undefined.

build_state(Json, Hints) ->
    Model = maps:get(<<"model">>, Json),
    VocabMap = maps:get(<<"vocab">>, Model),
    MergesList = maps:get(<<"merges">>, Model),

    %% Build reverse lookup id -> token.
    IdToToken = maps:fold(
        fun(Tok, Id, Acc) -> maps:put(Id, Tok, Acc) end,
        #{},
        VocabMap
    ),

    %% Build merge ranks: {<<"a">>, <<"b">>} -> Rank (lower = higher priority).
    {_, Merges} = lists:foldl(
        fun(Merge, {Rank, Acc}) ->
            {A, B} =
                case Merge of
                    [A0, B0] when is_binary(A0), is_binary(B0) -> {A0, B0};
                    MergeBin when is_binary(MergeBin) ->
                        [A0, B0] = binary:split(MergeBin, <<" ">>),
                        {A0, B0}
                end,
            {Rank + 1, maps:put({A, B}, Rank, Acc)}
        end,
        {0, #{}},
        MergesList
    ),

    %% Resolve special tokens. Prefer the canonical name from
    %% tokenizer_config.json, then known per-family candidates (Llama-3
    %% byte-level `<|begin_of_text|>`/`<|eot_id|>` vs Llama-2 SentencePiece
    %% `<s>`/`</s>`), falling back to the legacy defaults. Looks up both the
    %% added_tokens table and the main vocab.
    Added = maps:get(<<"added_tokens">>, Json, []),
    Bos = resolve_special(
        maps:get(bos_name, Hints, undefined),
        [<<"<|begin_of_text|>">>, <<"<s>">>],
        Added,
        VocabMap,
        1
    ),
    Eos = resolve_special(
        maps:get(eos_name, Hints, undefined),
        [<<"<|eot_id|>">>, <<"<|end_of_text|>">>, <<"</s>">>],
        Added,
        VocabMap,
        2
    ),
    Unk = find_special(Added, <<"<unk>">>, 0),

    %% Special tokens (e.g. <|begin_of_text|>, <|eot_id|>, <|start_header_id|>)
    %% must be matched atomically inside the text — chat templates embed them
    %% inline, and BPE would otherwise shatter them into raw bytes.
    SpecialMap = build_special_map(Added),
    SpecialPattern =
        case maps:keys(SpecialMap) of
            [] -> none;
            Names -> binary:compile_pattern(Names)
        end,

    %% Build byte-fallback table: byte N -> id of "<0xNN>".
    ByteIds = build_byte_table(VocabMap),

    %% Detect byte-level BPE (GPT-2/Llama-3 style: vocab contains 'Ġ' marker).
    ByteLevel = detect_byte_level(VocabMap),
    ByteDecoder =
        case ByteLevel of
            true -> build_byte_decoder();
            false -> #{}
        end,

    {ok, #{
        vocab => VocabMap,
        id_to_token => IdToToken,
        merges => Merges,
        bos => Bos,
        eos => Eos,
        unk => Unk,
        byte_ids => ByteIds,
        byte_level => ByteLevel,
        byte_decoder => ByteDecoder,
        special_tokens => SpecialMap,
        special_pattern => SpecialPattern
    }}.

%% Build content -> id map of special tokens (those flagged `special: true`
%% in added_tokens), longest content first so e.g. `<|eot_id|>` wins over a
%% shorter overlapping marker during matching.
build_special_map(Added) ->
    lists:foldl(
        fun(Tok, Acc) ->
            case
                {
                    maps:get(<<"special">>, Tok, false),
                    maps:get(<<"content">>, Tok, undefined),
                    maps:get(<<"id">>, Tok, undefined)
                }
            of
                {true, C, I} when is_binary(C), is_integer(I) -> maps:put(C, I, Acc);
                _ -> Acc
            end
        end,
        #{},
        Added
    ).

%% Detect byte-level BPE (GPT-2/Llama-3): many tokens start with Ġ AND
%% no SentencePiece ▁ markers present. Threshold of 100 avoids false positives
%% from vocabularies that include a stray Ġ as a single token.
detect_byte_level(VocabMap) ->
    {SpCount, ByteCount} = maps:fold(
        fun(Tok, _, {Sp, By}) ->
            Sp1 =
                case binary:match(Tok, <<"▁"/utf8>>) of
                    nomatch -> Sp;
                    _ -> Sp + 1
                end,
            By1 =
                case binary:match(Tok, <<"Ġ"/utf8>>) of
                    nomatch -> By;
                    _ -> By + 1
                end,
            {Sp1, By1}
        end,
        {0, 0},
        VocabMap
    ),
    SpCount < 10 andalso ByteCount > 100.

%% Build GPT-2 style byte-level decoder map: unicode codepoint -> byte (0..255).
%% Printable bytes [33-126, 161-172, 174-255] map to themselves.
%% Non-printable bytes get codepoints starting at 256.
build_byte_decoder() ->
    Printable = lists:seq(33, 126) ++ lists:seq(161, 172) ++ lists:seq(174, 255),
    PrintableSet = sets:from_list(Printable),
    PrintableMap = maps:from_list([{B, B} || B <- Printable]),
    NonPrintable = [B || B <- lists:seq(0, 255), not sets:is_element(B, PrintableSet)],
    {_, NonPrintableMap} = lists:foldl(
        fun(B, {N, Acc}) -> {N + 1, maps:put(256 + N, B, Acc)} end,
        {0, #{}},
        NonPrintable
    ),
    maps:merge(PrintableMap, NonPrintableMap).

find_special([], _Tok, Default) ->
    Default;
find_special([H | T], Tok, Default) ->
    case maps:get(<<"content">>, H, undefined) of
        Tok -> maps:get(<<"id">>, H);
        _ -> find_special(T, Tok, Default)
    end.

%% Resolve a special-token id: prefer the canonical name from
%% tokenizer_config.json, then the per-family candidate names, then Default.
resolve_special(undefined, Candidates, Added, Vocab, Default) ->
    resolve_candidates(Candidates, Added, Vocab, Default);
resolve_special(Name, Candidates, Added, Vocab, Default) ->
    case lookup_token_id(Name, Added, Vocab) of
        undefined -> resolve_candidates(Candidates, Added, Vocab, Default);
        Id -> Id
    end.

resolve_candidates([], _Added, _Vocab, Default) ->
    Default;
resolve_candidates([Name | Rest], Added, Vocab, Default) ->
    case lookup_token_id(Name, Added, Vocab) of
        undefined -> resolve_candidates(Rest, Added, Vocab, Default);
        Id -> Id
    end.

%% Look up a token id by name in the added_tokens table first, then the vocab.
lookup_token_id(Name, Added, Vocab) ->
    case find_special(Added, Name, undefined) of
        undefined -> maps:get(Name, Vocab, undefined);
        Id -> Id
    end.

build_byte_table(Vocab) ->
    list_to_tuple([
        maps:get(byte_token_name(B), Vocab, undefined)
     || B <- lists:seq(0, 255)
    ]).

byte_token_name(B) ->
    Hex = string:uppercase(integer_to_binary(B, 16)),
    Pad =
        case byte_size(Hex) of
            1 -> <<"0", Hex/binary>>;
            _ -> Hex
        end,
    <<"<0x", Pad/binary, ">">>.

%% ---------------------------------------------------------------- bos_id/eos_id/unk_id
bos_id(#{bos := B}) -> B.
eos_id(#{eos := E}) -> E.
unk_id(#{unk := U}) -> U.

%% ---------------------------------------------------------------- encode/2
encode(State, Text) when is_list(Text) ->
    encode(State, unicode:characters_to_binary(Text));
encode(State, Text) when is_binary(Text) ->
    case maps:get(special_pattern, State, none) of
        none ->
            encode_segment(State, Text);
        Pattern ->
            Specials = maps:get(special_tokens, State, #{}),
            lists:flatmap(
                fun
                    ({text, T}) -> encode_segment(State, T);
                    ({special, Id}) -> [Id]
                end,
                split_on_specials(Text, Pattern, Specials)
            )
    end.

%% Encode a span guaranteed to contain no special tokens.
encode_segment(State, Text) ->
    case maps:get(byte_level, State, false) of
        true -> encode_byte_level(State, Text);
        false -> encode_sentencepiece(State, Text)
    end.

%% Split text on special-token occurrences, keeping each special token atomic.
%% binary:match with a compiled pattern returns the leftmost (and, among ties,
%% longest) match — exactly HF's "special tokens are atomic" rule.
split_on_specials(<<>>, _Pattern, _Specials) ->
    [];
split_on_specials(Text, Pattern, Specials) ->
    case binary:match(Text, Pattern) of
        nomatch ->
            [{text, Text}];
        {Start, Len} ->
            <<Before:Start/binary, Matched:Len/binary, After/binary>> = Text,
            Id = maps:get(Matched, Specials),
            Rest = [{special, Id} | split_on_specials(After, Pattern, Specials)],
            case Before of
                <<>> -> Rest;
                _ -> [{text, Before} | Rest]
            end
    end.

%% SentencePiece (Llama-2 / TinyLlama): prepend ▁, spaces -> ▁, BPE over
%% codepoints. NOTE: Erlang treats <<"▁">> as Latin-1 by default — use /utf8.
encode_sentencepiece(State, Text) ->
    Up = <<"▁"/utf8>>,
    Prepended = <<Up/binary, Text/binary>>,
    Normalized = binary:replace(Prepended, <<" ">>, Up, [global]),
    Pieces = split_chars(Normalized),
    Merged = bpe_loop(Pieces, maps:get(merges, State)),
    tokens_to_ids(Merged, State).

%% Byte-level BPE (GPT-2 / Llama-3): GPT-2 regex pre-tokenization, map each
%% raw UTF-8 byte to its byte-level codepoint (e.g. space 0x20 -> "Ġ"), then
%% BPE-merge each piece independently. Mirrors the reference HF pipeline.
encode_byte_level(State, Text) ->
    Merges = maps:get(merges, State),
    Encoder = byte_encoder(),
    lists:flatmap(
        fun(Piece) ->
            Chars = byte_encode_piece(Piece, Encoder),
            Merged = bpe_loop(Chars, Merges),
            tokens_to_ids(Merged, State)
        end,
        pretokenize_gpt2(Text)
    ).

%% GPT-2 / Llama-3 pre-tokenization regex (PCRE via Erlang re; `ucp` enables
%% \p{L} / \p{N} unicode properties). `Isolated` behavior == every match is a
%% piece, and the alternation covers whitespace, so the matches tile the input.
pretokenize_gpt2(Text) ->
    Pattern =
        <<"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"/utf8>>,
    case re:run(Text, Pattern, [unicode, ucp, global, {capture, first, binary}]) of
        {match, Matches} -> [M || [M | _] <- Matches];
        nomatch -> []
    end.

%% Map a raw piece's UTF-8 bytes to single-codepoint binaries via the GPT-2
%% byte->unicode table.
byte_encode_piece(Piece, Encoder) ->
    [unicode:characters_to_binary([maps:get(B, Encoder)], utf8) || <<B>> <= Piece].

%% GPT-2 byte-level encoder: byte (0..255) -> printable unicode codepoint.
%% Inverse of build_byte_decoder/0; printable bytes map to themselves, the rest
%% to 256+N in byte order.
byte_encoder() ->
    Printable = lists:seq(33, 126) ++ lists:seq(161, 172) ++ lists:seq(174, 255),
    PrintableSet = sets:from_list(Printable),
    PrintablePairs = [{B, B} || B <- Printable],
    NonPrintable = [B || B <- lists:seq(0, 255), not sets:is_element(B, PrintableSet)],
    {_, NonPrintablePairs} = lists:foldl(
        fun(B, {N, Acc}) -> {N + 1, [{B, 256 + N} | Acc]} end,
        {0, []},
        NonPrintable
    ),
    maps:from_list(PrintablePairs ++ NonPrintablePairs).

%% Split a UTF-8 binary into a list of single-codepoint binaries.
split_chars(Bin) ->
    split_chars(Bin, []).

split_chars(<<>>, Acc) ->
    lists:reverse(Acc);
split_chars(Bin, Acc) ->
    case unicode:characters_to_list(Bin, utf8) of
        [C | _Rest] when is_integer(C) ->
            CharBin = unicode:characters_to_binary([C], utf8),
            CharSize = byte_size(CharBin),
            <<_:CharSize/binary, Tail/binary>> = Bin,
            split_chars(Tail, [CharBin | Acc]);
        _ ->
            %% Malformed UTF-8 — peel one byte and continue.
            <<B, Tail/binary>> = Bin,
            split_chars(Tail, [<<B>> | Acc])
    end.

%% Greedy lowest-rank BPE merge loop.
%% This is the classical SentencePiece/HF reference algorithm:
%% repeatedly find the adjacent pair with the smallest merge rank
%% and replace it with its concatenation. Stop when no pair has a rank.
bpe_loop([], _Merges) ->
    [];
bpe_loop([Single], _Merges) ->
    [Single];
bpe_loop(Pieces, Merges) ->
    case find_best_pair(Pieces, Merges) of
        none ->
            Pieces;
        {BestRank, _BestIdx} ->
            {A, B} = pair_at_rank(Pieces, Merges, BestRank),
            Merged = merge_all_pairs(Pieces, A, B),
            bpe_loop(Merged, Merges)
    end.

%% Find the pair with the minimum rank. Returns {Rank, Index} or none.
find_best_pair(Pieces, Merges) ->
    find_best_pair(Pieces, Merges, 0, none).

find_best_pair([_], _Merges, _Idx, Best) ->
    Best;
find_best_pair([], _Merges, _Idx, Best) ->
    Best;
find_best_pair([A, B | Rest], Merges, Idx, Best) ->
    NewBest =
        case maps:find({A, B}, Merges) of
            {ok, Rank} ->
                case Best of
                    none -> {Rank, Idx};
                    {BR, _} when Rank < BR -> {Rank, Idx};
                    _ -> Best
                end;
            error ->
                Best
        end,
    find_best_pair([B | Rest], Merges, Idx + 1, NewBest).

%% Find the actual pair (A, B) that has the minimum rank in the current
%% sequence. We re-scan to recover it as binaries because find_best_pair
%% only kept Rank+Idx.
pair_at_rank(Pieces, Merges, TargetRank) ->
    pair_at_rank_loop(Pieces, Merges, TargetRank).

pair_at_rank_loop([A, B | Rest], Merges, TargetRank) ->
    case maps:find({A, B}, Merges) of
        {ok, TargetRank} -> {A, B};
        _ -> pair_at_rank_loop([B | Rest], Merges, TargetRank)
    end.

%% Walk the sequence left-to-right, merging every non-overlapping occurrence
%% of (A, B) into AB. Matches HF's tokenizers behavior for this step.
merge_all_pairs([], _A, _B) -> [];
merge_all_pairs([X], _A, _B) -> [X];
merge_all_pairs([A, B | Rest], A, B) -> [<<A/binary, B/binary>> | merge_all_pairs(Rest, A, B)];
merge_all_pairs([X | Rest], A, B) -> [X | merge_all_pairs(Rest, A, B)].

%% ---------------------------------------------------------------- tokens -> ids
tokens_to_ids(Tokens, State) ->
    Vocab = maps:get(vocab, State),
    ByteIds = maps:get(byte_ids, State),
    Unk = maps:get(unk, State),
    lists:flatmap(
        fun(Tok) ->
            case maps:find(Tok, Vocab) of
                {ok, Id} -> [Id];
                error -> byte_fallback(Tok, ByteIds, Unk)
            end
        end,
        Tokens
    ).

byte_fallback(Tok, ByteIds, Unk) ->
    [
        case element(B + 1, ByteIds) of
            undefined -> Unk;
            Id -> Id
        end
     || <<B>> <= Tok
    ].

%% ---------------------------------------------------------------- decode/2
decode(State, Ids) ->
    IdToToken = maps:get(id_to_token, State),
    %% Convert ids -> token binaries, dropping unknown ids silently.
    Tokens = [maps:get(Id, IdToToken, <<>>) || Id <- Ids],
    case maps:get(byte_level, State, false) of
        true ->
            %% Byte-level BPE (GPT-2/Llama-3): concat tokens then map each
            %% unicode codepoint back to its raw byte via byte_decoder.
            Joined = iolist_to_binary(Tokens),
            ByteDecoder = maps:get(byte_decoder, State),
            byte_level_decode(Joined, ByteDecoder);
        false ->
            %% SentencePiece + byte-fallback (TinyLlama style).
            Joined = fuse_tokens(Tokens, []),
            Spaced = binary:replace(Joined, <<"▁"/utf8>>, <<" ">>, [global]),
            case Spaced of
                <<$\s, Rest/binary>> -> Rest;
                _ -> Spaced
            end
    end.

byte_level_decode(Joined, ByteDecoder) ->
    <<<<(maps:get(CP, ByteDecoder, $?))>> || <<CP/utf8>> <= Joined>>.

%% Collapse adjacent <0xNN> tokens into the raw bytes they encode.
%% Non-byte-fallback tokens are appended as-is.
fuse_tokens([], Acc) ->
    iolist_to_binary(lists:reverse(Acc));
fuse_tokens([Tok | Rest], Acc) ->
    case byte_token_value(Tok) of
        {ok, B} ->
            {Bytes, Rest2} = collect_bytes(Rest, [B]),
            fuse_tokens(Rest2, [Bytes | Acc]);
        error ->
            fuse_tokens(Rest, [Tok | Acc])
    end.

collect_bytes([Tok | Rest], Acc) ->
    case byte_token_value(Tok) of
        {ok, B} -> collect_bytes(Rest, [B | Acc]);
        error -> {list_to_binary(lists:reverse(Acc)), [Tok | Rest]}
    end;
collect_bytes([], Acc) ->
    {list_to_binary(lists:reverse(Acc)), []}.

%% If Tok looks like "<0xNN>", return {ok, ByteValue}, else error.
byte_token_value(<<"<0x", Hex:2/binary, ">">>) ->
    try
        {ok, binary_to_integer(Hex, 16)}
    catch
        _:_ -> error
    end;
byte_token_value(_) ->
    error.
