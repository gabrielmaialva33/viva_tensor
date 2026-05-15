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

-export([
    floats_to_fp32_binary/1,
    floats_to_fp16_binary/1,
    fp16_binary_to_floats/1
]).

%% List(Float) -> binary holding the same values as little-endian IEEE-754
%% single-precision floats. Used to feed the FP32 weight / input lists
%% into NIFs that prefer binaries.
floats_to_fp32_binary(Floats) when is_list(Floats) ->
    iolist_to_binary([<<F:32/float-little>> || F <- Floats]).

%% List(Float) -> binary holding IEEE-754 half-precision values, little-endian.
%% Round-to-nearest-even on the mantissa, clamp infinities to FP16-max.
%% Used to feed the cuBLASLt FP8 NIFs which expect FP16 input binaries.
floats_to_fp16_binary(Floats) when is_list(Floats) ->
    iolist_to_binary([<<(fp16_encode(F)):16/unsigned-little>> || F <- Floats]).

%% float64 -> uint16 IEEE-754 binary16
fp16_encode(F) when is_float(F) ->
    <<S:1, E:8, M:23>> = <<F:32/float-big>>,
    case E of
        0 ->
            %% zero or subnormal float32 — both flush to ±0 fp16
            (S bsl 15);
        255 ->
            %% Inf / NaN
            (S bsl 15) bor (16#1F bsl 10) bor (case M of 0 -> 0; _ -> 1 end);
        _ ->
            UnbiasedE = E - 127,
            case UnbiasedE of
                X when X < -14 ->
                    (S bsl 15);  %% underflow → ±0
                X when X > 15 ->
                    (S bsl 15) bor (16#1F bsl 10);  %% overflow → ±Inf
                X ->
                    Eh = X + 15,
                    %% Round-to-nearest-even on the mantissa: keep top 10 bits
                    Mh = (M bsr 13),
                    Round = (M bsr 12) band 1,
                    Sticky = case M band 16#FFF of 0 -> 0; _ -> 1 end,
                    {Mh2, Eh2} = case Round of
                        0 -> {Mh, Eh};
                        1 when Sticky =:= 1 ->
                            Mh1 = Mh + 1,
                            case Mh1 of
                                1024 -> {0, Eh + 1};
                                _ -> {Mh1, Eh}
                            end;
                        1 when (Mh band 1) =:= 1 ->
                            Mh1 = Mh + 1,
                            case Mh1 of
                                1024 -> {0, Eh + 1};
                                _ -> {Mh1, Eh}
                            end;
                        1 -> {Mh, Eh}
                    end,
                    (S bsl 15) bor (Eh2 bsl 10) bor Mh2
            end
    end;
fp16_encode(0) -> 0;
fp16_encode(F) when is_integer(F) -> fp16_encode(float(F)).

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
            %% Inf: Erlang doesn't have a native Inf literal — return the
            %% largest finite double (1.79e308). Callers comparing for
            %% magnitude will still see "huge" without crashing.
            case Sign of 0 -> 1.7976931348623157e308; 1 -> -1.7976931348623157e308 end;
        16#1F ->
            %% NaN — return 0 to keep tests numerically stable
            0.0;
        _ ->
            M = 1.0 + Frac / 1024.0,
            V = M * math:pow(2.0, float(Exp - 15)),
            case Sign of 0 -> V; 1 -> -V end
    end.
