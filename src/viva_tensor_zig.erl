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
    simd_mul/2,
    simd_matmul/5,
    simd_available/0,
    backend_info/0,
    is_loaded/0
]).

%% NIF Resource API - zero-copy tensor operations
-export([
    nt_zeros/1, nt_ones/1, nt_fill/2, nt_from_list/2,
    nt_to_list/1, nt_shape/1, nt_size/1,
    nt_add/2, nt_sub/2, nt_mul/2, nt_scale/2, nt_negate/1,
    nt_dot/2, nt_sum/1, nt_max/1, nt_min/1,
    nt_matmul/5, nt_transpose/1,
    nt_relu/1, nt_sigmoid/1, nt_exp/1, nt_log/1,
    %% In-place mutation (zero allocation)
    nt_add_mut/2, nt_scale_mut/2, nt_negate_mut/1, nt_relu_mut/1,
    %% Retro / fused kernels
    nt_saturn_blend/3, nt_fused_linear_relu/6,
    %% Resonance kernels (Log-Number System) - f64
    nt_resonance_mul/2, nt_resonance_power/2
]).

%% LNS (True Log-Number System) - f32 via IADD, 8x throughput
-export([
    lns_from_f64/1, lns_to_f64/1,
    lns_mul/2, lns_mul_corrected/2, lns_div/2,
    lns_sqrt/1, lns_rsqrt/1
]).

%% Horde - SoA Physics, 10K+ entities at 60fps
-export([
    horde_create/2,
    horde_set_positions/2, horde_set_velocities/2,
    horde_integrate/2, horde_dampen/2, horde_wrap/2,
    horde_get_positions/1, horde_get_velocities/1,
    horde_count/1, horde_kinetic_energy/1
]).

%% HDC - Hyperdimensional Computing, one-shot learning
-export([
    hdc_create/1, hdc_random/2,
    hdc_bind/2, hdc_similarity/2, hdc_permute/2, hdc_dim/1
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
%% SIMD Element-wise Multiply
%% ==========================================================================

simd_mul(A, B) ->
    case is_loaded() of
        true ->
            try nif_simd_mul(A, B)
            catch error:nif_not_loaded -> {ok, [X * Y || {X, Y} <- lists:zip(A, B)]}
            end;
        false -> {ok, [X * Y || {X, Y} <- lists:zip(A, B)]}
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

nif_simd_mul(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_matmul(_A, _B, _M, _N, _K) ->
    erlang:nif_error(nif_not_loaded).

nif_simd_available() ->
    erlang:nif_error(nif_not_loaded).

nif_backend_info() ->
    erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% NIF Resource API Stubs (replaced by C NIFs on load)
%% ==========================================================================

%% Constructors
nt_zeros(_Shape) -> erlang:nif_error(nif_not_loaded).
nt_ones(_Shape) -> erlang:nif_error(nif_not_loaded).
nt_fill(_Shape, _Value) -> erlang:nif_error(nif_not_loaded).
nt_from_list(_Data, _Shape) -> erlang:nif_error(nif_not_loaded).

%% Accessors
nt_to_list(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_shape(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_size(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Element-wise ops
nt_add(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_sub(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_scale(_Ref, _Scalar) -> erlang:nif_error(nif_not_loaded).
nt_negate(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Reductions
nt_dot(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_sum(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_max(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_min(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Matrix ops
nt_matmul(_A, _B, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).
nt_transpose(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Activations
nt_relu(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_sigmoid(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_exp(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_log(_Ref) -> erlang:nif_error(nif_not_loaded).

%% In-place mutation (zero allocation)
nt_add_mut(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_scale_mut(_Ref, _Scalar) -> erlang:nif_error(nif_not_loaded).
nt_negate_mut(_Ref) -> erlang:nif_error(nif_not_loaded).
nt_relu_mut(_Ref) -> erlang:nif_error(nif_not_loaded).

%% Retro / fused kernels
nt_saturn_blend(_Texture, _Shade, _Bias) -> erlang:nif_error(nif_not_loaded).
nt_fused_linear_relu(_A, _B, _Bias, _M, _N, _K) -> erlang:nif_error(nif_not_loaded).

%% Resonance kernels (Log-Number System) - f64
nt_resonance_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
nt_resonance_power(_Ref, _Exponent) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% LNS (True Log-Number System) - f32 via IADD, 8x throughput
%% ==========================================================================

lns_from_f64(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_to_f64(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_mul(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_mul_corrected(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_div(_A, _B) -> erlang:nif_error(nif_not_loaded).
lns_sqrt(_Ref) -> erlang:nif_error(nif_not_loaded).
lns_rsqrt(_Ref) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% Horde - SoA Physics, 10K+ entities at 60fps
%% ==========================================================================

horde_create(_EntityCount, _Dims) -> erlang:nif_error(nif_not_loaded).
horde_set_positions(_Horde, _Data) -> erlang:nif_error(nif_not_loaded).
horde_set_velocities(_Horde, _Data) -> erlang:nif_error(nif_not_loaded).
horde_integrate(_Horde, _Dt) -> erlang:nif_error(nif_not_loaded).
horde_dampen(_Horde, _Friction) -> erlang:nif_error(nif_not_loaded).
horde_wrap(_Horde, _MaxBound) -> erlang:nif_error(nif_not_loaded).
horde_get_positions(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_get_velocities(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_count(_Horde) -> erlang:nif_error(nif_not_loaded).
horde_kinetic_energy(_Horde) -> erlang:nif_error(nif_not_loaded).

%% ==========================================================================
%% HDC - Hyperdimensional Computing, one-shot learning
%% ==========================================================================

hdc_create(_Dim) -> erlang:nif_error(nif_not_loaded).
hdc_random(_Dim, _Seed) -> erlang:nif_error(nif_not_loaded).
hdc_bind(_A, _B) -> erlang:nif_error(nif_not_loaded).
hdc_similarity(_A, _B) -> erlang:nif_error(nif_not_loaded).
hdc_permute(_Vec, _Shift) -> erlang:nif_error(nif_not_loaded).
hdc_dim(_Vec) -> erlang:nif_error(nif_not_loaded).
