%%% @doc FFI helpers for Gleam tests.
%%%
%%% `rescue_call/1` runs the supplied zero-arg fun and, if it raises any
%%% Erlang exception (`error`, `exit`, or `throw`), returns the Gleam
%%% tagged tuple `{call_err}` instead of crashing the test process.
%%% Successful results come back wrapped in `{call_ok, Value}` matching
%%% the `CallResult` ADT in `test/inference_test.gleam`.
%%%
%%% Used by inference contract tests so that "NIF not yet wired" (which
%%% raises `:undef` at the BEAM level) doesn't take down the gleeunit
%%% runner.

-module(viva_tensor_test_ffi).

-export([rescue_call/1]).

rescue_call(Fun) when is_function(Fun, 0) ->
    try
        {call_ok, Fun()}
    catch
        _:_:_ -> call_err
    end.
