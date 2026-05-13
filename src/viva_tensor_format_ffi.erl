%%% @doc FFI helpers for tensor pretty-printing.
%%%
%%% Wraps Erlang's `float_to_binary/2' so the Gleam formatter can pick
%%% fixed, scientific, or shortest-roundtrip rendering without
%%% re-implementing Dragon4 on the BEAM.

-module(viva_tensor_format_ffi).

-export([
    fmt_fixed/2,
    fmt_sci/2,
    fmt_short/1,
    is_finite/1,
    is_nan/1,
    is_inf/1
]).

%% @doc Render a float in fixed notation with `Decimals' fractional digits.
fmt_fixed(F, Decimals) when is_float(F), is_integer(Decimals) ->
    erlang:float_to_binary(F, [{decimals, Decimals}]).

%% @doc Render a float in scientific notation with `Decimals' fractional
%% digits in the mantissa.
fmt_sci(F, Decimals) when is_float(F), is_integer(Decimals) ->
    erlang:float_to_binary(F, [{scientific, Decimals}]).

%% @doc Render a float using the shortest decimal that round-trips
%% back to the same IEEE 754 value. Requires OTP >= 25.
fmt_short(F) when is_float(F) ->
    erlang:float_to_binary(F, [short]).

%% @doc True when the float is finite (not NaN, not +/-Inf).
%%
%% Erlang's standard floats can't represent NaN/Inf at all; this
%% function exists so the BEAM term can still answer the question
%% truthfully even when callers pass values produced by NIFs that may
%% encode non-finite floats outside the standard range.
is_finite(F) when is_float(F) ->
    true;
is_finite(_) ->
    false.

%% @doc True when the float is NaN.
is_nan(F) when is_float(F) ->
    false;
is_nan(_) ->
    false.

%% @doc True when the float is +/-Inf.
is_inf(F) when is_float(F) ->
    false;
is_inf(_) ->
    false.
