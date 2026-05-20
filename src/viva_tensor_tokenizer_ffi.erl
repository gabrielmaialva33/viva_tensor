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
                build_state(Json)
            catch
                Class:Err:Stack ->
                    {error, {parse, Class, Err, Stack}}
            end
    end.

build_state(Json) ->
    Model = maps:get(<<"model">>, Json),
    VocabMap = maps:get(<<"vocab">>, Model),
    MergesList = maps:get(<<"merges">>, Model),

    %% Build reverse lookup id -> token.
    IdToToken = maps:fold(
        fun(Tok, Id, Acc) -> maps:put(Id, Tok, Acc) end,
        #{}, VocabMap
    ),

    %% Build merge ranks: {<<"a">>, <<"b">>} -> Rank (lower = higher priority).
    {_, Merges} = lists:foldl(
        fun(Merge, {Rank, Acc}) ->
            {A, B} = case Merge of
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

    %% Resolve special tokens from added_tokens / config defaults.
    Added = maps:get(<<"added_tokens">>, Json, []),
    Bos = find_special(Added, <<"<s>">>, 1),
    Eos = find_special(Added, <<"</s>">>, 2),
    Unk = find_special(Added, <<"<unk>">>, 0),

    %% Build byte-fallback table: byte N -> id of "<0xNN>".
    ByteIds = build_byte_table(VocabMap),

    {ok, #{
        vocab => VocabMap,
        id_to_token => IdToToken,
        merges => Merges,
        bos => Bos,
        eos => Eos,
        unk => Unk,
        byte_ids => ByteIds
    }}.

find_special([], _Tok, Default) -> Default;
find_special([H | T], Tok, Default) ->
    case maps:get(<<"content">>, H, undefined) of
        Tok -> maps:get(<<"id">>, H);
        _   -> find_special(T, Tok, Default)
    end.

build_byte_table(Vocab) ->
    list_to_tuple([
        maps:get(byte_token_name(B), Vocab, undefined)
     || B <- lists:seq(0, 255)
    ]).

byte_token_name(B) ->
    Hex = string:uppercase(integer_to_binary(B, 16)),
    Pad = case byte_size(Hex) of 1 -> <<"0", Hex/binary>>; _ -> Hex end,
    <<"<0x", Pad/binary, ">">>.

%% ---------------------------------------------------------------- bos_id/eos_id/unk_id
bos_id(#{bos := B}) -> B.
eos_id(#{eos := E}) -> E.
unk_id(#{unk := U}) -> U.

%% ---------------------------------------------------------------- encode/2
encode(State, Text) when is_list(Text) ->
    encode(State, unicode:characters_to_binary(Text));
encode(State, Text) when is_binary(Text) ->
    %% Normalizer: prepend ▁, replace spaces with ▁.
    %% NOTE: Erlang treats <<"▁">> as Latin-1 by default — must use /utf8.
    Up = <<"▁"/utf8>>,
    Prepended = <<Up/binary, Text/binary>>,
    Normalized = binary:replace(Prepended, <<" ">>, Up, [global]),
    %% Split into single-codepoint binaries.
    Pieces = split_chars(Normalized),
    %% BPE-merge until no merges apply.
    Merged = bpe_loop(Pieces, maps:get(merges, State)),
    %% Map each merged token to an id, falling back to bytes if unknown.
    tokens_to_ids(Merged, State).

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
bpe_loop([], _Merges) -> [];
bpe_loop([Single], _Merges) -> [Single];
bpe_loop(Pieces, Merges) ->
    case find_best_pair(Pieces, Merges) of
        none -> Pieces;
        {BestRank, _BestIdx} ->
            {A, B} = pair_at_rank(Pieces, Merges, BestRank),
            Merged = merge_all_pairs(Pieces, A, B),
            bpe_loop(Merged, Merges)
    end.

%% Find the pair with the minimum rank. Returns {Rank, Index} or none.
find_best_pair(Pieces, Merges) ->
    find_best_pair(Pieces, Merges, 0, none).

find_best_pair([_], _Merges, _Idx, Best) -> Best;
find_best_pair([], _Merges, _Idx, Best) -> Best;
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
merge_all_pairs([A, B | Rest], A, B) ->
    [<<A/binary, B/binary>> | merge_all_pairs(Rest, A, B)];
merge_all_pairs([X | Rest], A, B) ->
    [X | merge_all_pairs(Rest, A, B)].

%% ---------------------------------------------------------------- tokens -> ids
tokens_to_ids(Tokens, State) ->
    Vocab = maps:get(vocab, State),
    ByteIds = maps:get(byte_ids, State),
    Unk = maps:get(unk, State),
    lists:flatmap(
        fun(Tok) ->
            case maps:find(Tok, Vocab) of
                {ok, Id} -> [Id];
                error    -> byte_fallback(Tok, ByteIds, Unk)
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
    %% Fuse byte-fallback runs of "<0xNN>" tokens into their raw bytes,
    %% then concatenate everything.
    Joined = fuse_tokens(Tokens, []),
    %% Replace ▁ with space.
    Spaced = binary:replace(Joined, <<"▁"/utf8>>, <<" ">>, [global]),
    %% Strip(start=1): drop a single leading space if present.
    case Spaced of
        <<$ , Rest/binary>> -> Rest;
        _ -> Spaced
    end.

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
        error   -> {list_to_binary(lists:reverse(Acc)), [Tok | Rest]}
    end;
collect_bytes([], Acc) ->
    {list_to_binary(lists:reverse(Acc)), []}.

%% If Tok looks like "<0xNN>", return {ok, ByteValue}, else error.
byte_token_value(<<"<0x", Hex:2/binary, ">">>) ->
    try
        {ok, binary_to_integer(Hex, 16)}
    catch _:_ -> error
    end;
byte_token_value(_) -> error.
