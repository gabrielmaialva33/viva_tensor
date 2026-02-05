%% viva_tensor_zig.erl - Zig SIMD NIF wrapper
%%
%% Provides cross-platform SIMD-accelerated tensor operations via Zig.
%% Falls back to pure Erlang implementation if NIF is not available.

-module(viva_tensor_zig).
-export([
    simd_dot/2,
    simd_sum/1,
    simd_scale/2,
    simd_add/2,
    simd_matmul/5,
    simd_available/0,
    backend_info/0,
    is_loaded/0
]).

-on_load(init/0).

%% NIF Loading
init() ->
    PrivDir = case code:priv_dir(viva_tensor) of
        {error, bad_name} ->
            case file:get_cwd() of
                {ok, Cwd} -> filename:join(Cwd, "priv");
                _ -> "priv"
            end;
        Dir -> Dir
    end,
    NifPath = filename:join(PrivDir, "viva_tensor_zig"),
    case erlang:load_nif(NifPath, 0) of
        ok ->
            persistent_term:put(viva_tensor_zig_loaded, true),
            ok;
        {error, {load_failed, _}} ->
            persistent_term:put(viva_tensor_zig_loaded, false),
            ok;
        {error, {reload, _}} ->
            ok;
        {error, Reason} ->
            error_logger:info_msg("viva_tensor_zig NIF not loaded: ~p~n", [Reason]),
            persistent_term:put(viva_tensor_zig_loaded, false),
            ok
    end.

%% Check if NIF is loaded
is_loaded() ->
    try persistent_term:get(viva_tensor_zig_loaded)
    catch error:badarg -> false
    end.

%% Backend info
backend_info() ->
    case is_loaded() of
        true -> nif_backend_info();
        false -> <<"Pure Erlang fallback">>
    end.

%% SIMD availability check
simd_available() ->
    case is_loaded() of
        true -> nif_simd_available();
        false -> false
    end.

%% ==========================================================================
%% SIMD Dot Product
%% ==========================================================================

simd_dot(A, B) ->
    case is_loaded() of
        true -> nif_simd_dot(A, B);
        false -> fallback_dot(A, B)
    end.

fallback_dot(A, B) ->
    {ok, lists:foldl(
        fun({X, Y}, Acc) -> Acc + X * Y end,
        0.0,
        lists:zip(A, B)
    )}.

%% ==========================================================================
%% SIMD Sum
%% ==========================================================================

simd_sum(Data) ->
    case is_loaded() of
        true -> nif_simd_sum(Data);
        false -> {ok, lists:sum(Data)}
    end.

%% ==========================================================================
%% SIMD Scale
%% ==========================================================================

simd_scale(Data, Scalar) ->
    case is_loaded() of
        true -> nif_simd_scale(Data, Scalar);
        false -> {ok, [X * Scalar || X <- Data]}
    end.

%% ==========================================================================
%% SIMD Add
%% ==========================================================================

simd_add(A, B) ->
    case is_loaded() of
        true -> nif_simd_add(A, B);
        false -> {ok, [X + Y || {X, Y} <- lists:zip(A, B)]}
    end.

%% ==========================================================================
%% SIMD Matrix Multiplication
%% ==========================================================================

simd_matmul(A, B, M, N, K) ->
    case is_loaded() of
        true -> nif_simd_matmul(A, B, M, N, K);
        false -> fallback_matmul(A, B, M, N, K)
    end.

fallback_matmul(AList, BList, M, N, K) ->
    A = array:from_list(AList),
    B = array:from_list(BList),
    Result = [
        begin
            RowStart = I * K,
            lists:foldl(fun(KIdx, Acc) ->
                AVal = array:get(RowStart + KIdx, A),
                BVal = array:get(KIdx * N + J, B),
                Acc + AVal * BVal
            end, 0.0, lists:seq(0, K - 1))
        end
        || I <- lists:seq(0, M - 1),
           J <- lists:seq(0, N - 1)
    ],
    {ok, Result}.

%% ==========================================================================
%% NIF Stubs
%% ==========================================================================

nif_simd_dot(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_sum(_Data) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_scale(_Data, _Scalar) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_add(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_matmul(_A, _B, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_available() ->
    erlang:nif_error(nif_not_loaded).

nif_backend_info() ->
    erlang:nif_error(nif_not_loaded).
