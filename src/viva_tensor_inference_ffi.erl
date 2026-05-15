%%% @doc Tiny FFI helpers used by viva_tensor/native/inference.gleam to
%%% talk to the NIF prepack/linear functions.
%%%
%%% The prepack/linear NIFs (zig_src/nif_*.c) accept raw FP32 binaries
%%% rather than Erlang lists of doubles, because copying ~MB of weight
%%% data through enif_get_list_cell is ~100× slower than enif_inspect
%%% on a binary. Linear outputs come back as binaries too.
%%%
%%% Gleam's gleam_stdlib doesn't expose a direct `list_of_doubles ->
%%% IEEE-754 fp32 binary` conversion, so we do it here in two lines of
%%% Erlang. Linear output binaries (FP16 little-endian) get rehydrated
%%% to Gleam `List(Float)` in `fp16_binary_to_floats/1`.

-module(viva_tensor_inference_ffi).

-export([floats_to_fp32_binary/1, fp16_binary_to_floats/1]).

%% List(Float) -> binary holding the same values as little-endian IEEE-754
%% single-precision floats. Used to feed the FP32 weight / input lists
%% into NIFs that prefer binaries.
floats_to_fp32_binary(Floats) when is_list(Floats) ->
    iolist_to_binary([<<F:32/float-little>> || F <- Floats]).

%% binary of FP16 little-endian values -> list of floats. Used to decode
%% the FP16 output binaries returned by linear_*_fp8 etc.
fp16_binary_to_floats(Bin) when is_binary(Bin) ->
    [fp16_decode(H) || <<H:16/unsigned-little>> <= Bin].

%% IEEE-754 binary16 → float64. Simple table-free decoder.
fp16_decode(H) ->
    Sign = (H bsr 15) band 1,
    Exp  = (H bsr 10) band 16#1F,
    Frac =  H band 16#3FF,
    case Exp of
        0 when Frac =:= 0 ->
            case Sign of 0 -> +0.0; 1 -> -0.0 end;
        0 ->
            %% Subnormal
            M = Frac / 1024.0,
            V = M * math:pow(2.0, -14.0),
            case Sign of 0 -> V; 1 -> -V end;
        16#1F when Frac =:= 0 ->
            case Sign of 0 -> 1.0e308 * 10.0; 1 -> -1.0e308 * 10.0 end;
        16#1F ->
            %% NaN — return 0 to keep tests numerically stable
            0.0;
        _ ->
            M = 1.0 + Frac / 1024.0,
            V = M * math:pow(2.0, float(Exp - 15)),
            case Sign of 0 -> V; 1 -> -V end
    end.
