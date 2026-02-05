%% viva_tensor_nif.erl - NIF wrapper with automatic fallback
%%
%% Provides Apple Accelerate-optimized tensor operations on macOS.
%% Falls back to pure Erlang implementation if NIF is not available.
%%
%% Usage from Gleam:
%%   @external(erlang, "viva_tensor_nif", "matmul")
%%   fn nif_matmul(a: List(Float), b: List(Float), m: Int, n: Int, k: Int)
%%     -> Result(List(Float), String)

-module(viva_tensor_nif).
-export([
    matmul/5,
    dot/2,
    sum/1,
    scale/2,
    is_nif_loaded/0,
    backend_info/0
]).

-on_load(init/0).

%% NIF Loading
init() ->
    %% Try to find the NIF in priv directory
    PrivDir = case code:priv_dir(viva_tensor) of
        {error, bad_name} ->
            %% Not installed as app, try relative path
            case file:get_cwd() of
                {ok, Cwd} -> filename:join(Cwd, "priv");
                _ -> "priv"
            end;
        Dir -> Dir
    end,
    NifPath = filename:join(PrivDir, "viva_tensor_nif"),
    case erlang:load_nif(NifPath, 0) of
        ok ->
            %% Mark NIF as loaded using persistent_term
            persistent_term:put(viva_tensor_nif_loaded, true),
            ok;
        {error, {load_failed, _}} ->
            %% NIF not built - silently fall back to Erlang
            persistent_term:put(viva_tensor_nif_loaded, false),
            ok;
        {error, {reload, _}} ->
            %% Already loaded
            ok;
        {error, Reason} ->
            error_logger:info_msg("viva_tensor NIF not loaded: ~p~n", [Reason]),
            persistent_term:put(viva_tensor_nif_loaded, false),
            ok
    end.

%% Check if NIF is loaded
is_nif_loaded() ->
    %% Check persistent_term flag set during init
    try persistent_term:get(viva_tensor_nif_loaded)
    catch error:badarg -> false
    end.

%% Get backend info string
backend_info() ->
    case is_nif_loaded() of
        true -> <<"Apple Accelerate (cblas_dgemm, vDSP)">>;
        false -> <<"Pure Erlang (O(1) array)">>
    end.

%% ==========================================================================
%% Matrix Multiplication
%% A[m,k] @ B[k,n] -> C[m,n]
%% ==========================================================================

matmul(AList, BList, M, N, K) ->
    case is_nif_loaded() of
        true ->
            nif_matmul(AList, BList, M, N, K);
        false ->
            %% Fallback to pure Erlang
            A = array:from_list(AList),
            B = array:from_list(BList),
            Result = viva_tensor_ffi:array_matmul(A, B, M, N, K),
            {ok, array:to_list(Result)}
    end.

%% ==========================================================================
%% Dot Product
%% ==========================================================================

dot(AList, BList) ->
    case is_nif_loaded() of
        true ->
            nif_dot(AList, BList);
        false ->
            %% Fallback to pure Erlang
            A = array:from_list(AList),
            B = array:from_list(BList),
            {ok, viva_tensor_ffi:array_dot(A, B)}
    end.

%% ==========================================================================
%% Sum
%% ==========================================================================

sum(List) ->
    case is_nif_loaded() of
        true ->
            nif_sum(List);
        false ->
            {ok, lists:sum(List)}
    end.

%% ==========================================================================
%% Scale
%% ==========================================================================

scale(List, Scalar) ->
    case is_nif_loaded() of
        true ->
            nif_scale(List, Scalar);
        false ->
            {ok, [X * Scalar || X <- List]}
    end.

%% ==========================================================================
%% NIF Stubs (replaced when NIF loads)
%% ==========================================================================

nif_matmul(_A, _B, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

nif_dot(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_sum(_List) ->
    erlang:nif_error(nif_not_loaded).

nif_scale(_List, _Scalar) ->
    erlang:nif_error(nif_not_loaded).
