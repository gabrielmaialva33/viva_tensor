%%% @doc SafeTensors loader for end-to-end Llama / transformer inference
%%% smoke tests. Parses the SafeTensors header, exposes per-tensor
%%% metadata, reads the underlying tensor bytes, and converts bfloat16
%%% (the canonical Llama weight dtype) into IEEE-754 fp32 binaries that
%%% the prepack NIFs accept.
%%%
%%% File format (https://github.com/huggingface/safetensors):
%%%   bytes 0..7        : uint64 little-endian header size H
%%%   bytes 8..8+H-1    : JSON header
%%%   bytes 8+H..EOF    : raw tensor bytes referenced via per-tensor
%%%                       {data_offsets: [start, end]} (relative to the
%%%                       data segment, i.e. after the 8-byte size and
%%%                       the H header bytes).

-module(viva_tensor_safetensors_ffi).

-export([
    open_header/1,
    list_tensor_names/1,
    tensor_info/2,
    read_tensor_bf16/2,
    bf16_to_fp32_binary/1,
    transpose_fp32/3,
    rmsnorm_weight_to_fp32_list/1
]).

%% Returns {ok, #{header := map(), data_start := integer(), path := binary()}}.
%% header is the parsed JSON metadata: #{TensorName => #{dtype, shape, offsets}}.
open_header(Path) ->
    case file:open(Path, [read, raw, binary]) of
        {ok, F} ->
            {ok, <<HdrSize:64/little>>} = file:read(F, 8),
            {ok, HdrBin} = file:read(F, HdrSize),
            file:close(F),
            Header = json:decode(HdrBin),
            %% Strip metadata pseudo-key emitted by HF.
            Tensors = maps:filter(
                fun(K, _) -> K =/= <<"__metadata__">> end,
                Header
            ),
            Parsed = maps:map(
                fun(_K, V) ->
                    Dtype = maps:get(<<"dtype">>, V),
                    Shape = maps:get(<<"shape">>, V),
                    [Start, End] = maps:get(<<"data_offsets">>, V),
                    #{dtype => Dtype, shape => Shape, offsets => {Start, End}}
                end,
                Tensors
            ),
            {ok, #{header => Parsed, data_start => 8 + HdrSize, path => Path}};
        Error ->
            Error
    end.

list_tensor_names(#{header := H}) ->
    lists:sort(maps:keys(H)).

tensor_info(#{header := H}, Name) when is_binary(Name) ->
    case maps:find(Name, H) of
        {ok, Info} -> {ok, Info};
        error -> {error, not_found}
    end.

%% Read raw bf16 bytes for a tensor. Returns {ok, binary()}.
read_tensor_bf16(#{header := H, data_start := DS, path := Path}, Name) ->
    case maps:find(Name, H) of
        {ok, #{dtype := <<"BF16">>, offsets := {Start, End}}} ->
            ByteCount = End - Start,
            {ok, F} = file:open(Path, [read, raw, binary]),
            {ok, _} = file:position(F, DS + Start),
            {ok, Bin} = file:read(F, ByteCount),
            file:close(F),
            {ok, Bin};
        {ok, #{dtype := Other}} ->
            {error, {unsupported_dtype, Other}};
        error ->
            {error, not_found}
    end.

%% bf16 → fp32: bf16 is the upper 16 bits of an fp32 with the same value.
%% So conversion is "append 16 zero bits" — element-by-element little-endian
%% bf16 (uint16) becomes little-endian fp32 (uint32) with low 16 bits zero.
bf16_to_fp32_binary(BF16) when is_binary(BF16) ->
    bf16_to_fp32(BF16, <<>>).

bf16_to_fp32(<<>>, Acc) -> Acc;
bf16_to_fp32(<<U:16/unsigned-little, Rest/binary>>, Acc) ->
    bf16_to_fp32(Rest, <<Acc/binary, 0:16/unsigned-little, U:16/unsigned-little>>).

%% Transpose an fp32 row-major matrix of shape {Rows, Cols} into a new
%% binary with shape {Cols, Rows} (still row-major). Used to flip
%% HuggingFace [out, in] convention to viva_tensor's [in, out] prepack
%% convention.
%%
%% Fast path: calls the C NIF `viva_tensor_zig:nt_transpose_fp32/3`
%% (single-threaded tiled transpose; ~50× faster than pure Erlang for
%% the 32000×2048 lm_head). Falls back to the pure-Erlang implementation
%% if the NIF is not loaded (e.g. test runs without the .so).
transpose_fp32(Bin, Rows, Cols) when is_binary(Bin) ->
    Elems = Rows * Cols,
    case byte_size(Bin) of
        Sz when Sz =:= Elems * 4 ->
            try
                Out = viva_tensor_zig:nt_transpose_fp32(Bin, Rows, Cols),
                {ok, Out}
            catch
                error:nif_not_loaded -> transpose_fp32_erlang(Bin, Rows, Cols);
                error:undef          -> transpose_fp32_erlang(Bin, Rows, Cols);
                error:badarg         -> transpose_fp32_erlang(Bin, Rows, Cols)
            end;
        _ ->
            {error, size_mismatch}
    end.

%% Pure-Erlang fallback. Iterates columns of the input (becomes rows of the
%% output). For each output row c, build a contiguous Rows×4-byte chunk by
%% reading `binary_part(Bin, (r*Cols+c)*4, 4)` for r=0..Rows-1. Avoids the
%% O(Rows*Cols) tuple allocation that blew up the Erlang scheduler for
%% 32000×2048 matrices.
transpose_fp32_erlang(Bin, Rows, Cols) ->
    Out = iolist_to_binary(
        [transpose_col(Bin, C, Rows, Cols)
         || C <- lists:seq(0, Cols - 1)]
    ),
    {ok, Out}.

transpose_col(Bin, C, Rows, Cols) ->
    [binary_part(Bin, (R * Cols + C) * 4, 4) || R <- lists:seq(0, Rows - 1)].

%% RMSNorm weights are 1-D vectors of length hidden_size, typically only
%% 2-8 KB. Convert directly to a Gleam-friendly List(Float).
rmsnorm_weight_to_fp32_list(BF16) when is_binary(BF16) ->
    [bf16_to_float(U) || <<U:16/unsigned-little>> <= BF16].

bf16_to_float(U) ->
    <<F:32/float-little>> = <<0:16, U:16/unsigned-little>>,
    F.
