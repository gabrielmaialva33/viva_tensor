%%% Offline SmoothQuant-style calibration for FP8 quantization.
%%%
%%% SmoothQuant (Xiao et al., 2022, https://arxiv.org/abs/2211.10438)
%%% migrates activation outliers into the weights via a per-input-channel
%%% scaling factor:
%%%
%%%     s[c] = max_abs(activation[c]) ** alpha
%%%          / max_abs(weight[c, :])   ** (1 - alpha)
%%%
%%% Then mathematically:
%%%     W'[c, :] = W[c, :] * s[c]
%%%     x'[c]    = x[c]   / s[c]
%%%     (W' @ x') == (W @ x)
%%% but max_abs(x') is much smaller — better fit for FP8.
%%%
%%% Weights are consumed in the same row-major [InFeatures, OutFeatures]
%%% layout that viva_tensor_safetensors_ffi:transpose_fp32/3 produces
%%% (the "in-out" layout used by viva_tensor's prepack NIFs). That means
%%% row `c` of the binary is the OutFeatures-long fp32 slice W[c, :].
%%%
%%% Pure Erlang. No NIFs. Standalone utility — not yet wired into the
%%% existing forward path; future integration is a separate task.

-module(llama_calibration).
-export([smoothquant_scale/5, apply_smoothquant/4, test/0]).

%% ---------------------------------------------------------------------------
%% Public API
%% ---------------------------------------------------------------------------

%% Compute per-input-channel SmoothQuant scales.
%%
%% WeightFp32Bin :: <<F:32/float-little, ...>> shape [InFeatures, OutFeatures]
%% ActMaxAbs     :: [float()] length InFeatures
%% Alpha         :: float() (typical 0.5)
%% returns        :: [float()] length InFeatures
smoothquant_scale(WeightFp32Bin, ActMaxAbs, InFeatures, OutFeatures, Alpha)
        when is_binary(WeightFp32Bin),
             is_list(ActMaxAbs),
             is_integer(InFeatures),
             is_integer(OutFeatures),
             is_number(Alpha) ->
    Expected = InFeatures * OutFeatures * 4,
    case byte_size(WeightFp32Bin) of
        Expected -> ok;
        Got -> error({weight_size_mismatch, expected, Expected, got, Got})
    end,
    case length(ActMaxAbs) of
        InFeatures -> ok;
        N -> error({act_length_mismatch, expected, InFeatures, got, N})
    end,
    RowBytes = OutFeatures * 4,
    %% Walk InFeatures rows; for each, find max_abs along OutFeatures.
    compute_scales(WeightFp32Bin, ActMaxAbs, RowBytes, Alpha, []).

%% Apply per-input-channel scaling to the weight matrix.
%%
%% Returns {AdjustedWeightBin, ActScaleList}. The caller must divide
%% activations element-wise by ActScaleList before quantizing the new
%% weight matrix.
apply_smoothquant(WeightFp32Bin, ScaleList, InFeatures, OutFeatures)
        when is_binary(WeightFp32Bin),
             is_list(ScaleList),
             is_integer(InFeatures),
             is_integer(OutFeatures) ->
    Expected = InFeatures * OutFeatures * 4,
    case byte_size(WeightFp32Bin) of
        Expected -> ok;
        Got -> error({weight_size_mismatch, expected, Expected, got, Got})
    end,
    case length(ScaleList) of
        InFeatures -> ok;
        N -> error({scale_length_mismatch, expected, InFeatures, got, N})
    end,
    RowBytes = OutFeatures * 4,
    Adjusted = scale_rows(WeightFp32Bin, ScaleList, RowBytes, []),
    {Adjusted, ScaleList}.

%% ---------------------------------------------------------------------------
%% Internals
%% ---------------------------------------------------------------------------

compute_scales(<<>>, [], _RowBytes, _Alpha, Acc) ->
    lists:reverse(Acc);
compute_scales(Bin, [ActMax | RestAct], RowBytes, Alpha, Acc) ->
    <<Row:RowBytes/binary, Rest/binary>> = Bin,
    WMax = row_max_abs(Row, 0.0),
    Scale = smooth_scale(ActMax, WMax, Alpha),
    compute_scales(Rest, RestAct, RowBytes, Alpha, [Scale | Acc]).

row_max_abs(<<>>, M) -> M;
row_max_abs(<<F:32/float-little, Rest/binary>>, M) ->
    A = abs(F),
    case A > M of
        true  -> row_max_abs(Rest, A);
        false -> row_max_abs(Rest, M)
    end.

%% s = act**alpha / w**(1-alpha)
%% Numerical safety: if either side is zero, fall back to 1.0 — the row
%% is a no-op and we don't want to introduce inf/nan.
smooth_scale(_Act, WMax, _Alpha) when WMax =< 0.0 -> 1.0;
smooth_scale(Act, _WMax, _Alpha) when Act =< 0.0 -> 1.0;
smooth_scale(Act, WMax, Alpha) ->
    Num = math:pow(Act, Alpha),
    Den = math:pow(WMax, 1.0 - Alpha),
    case Den of
        +0.0 -> 1.0;
        _    -> Num / Den
    end.

scale_rows(<<>>, [], _RowBytes, Acc) ->
    iolist_to_binary(lists:reverse(Acc));
scale_rows(Bin, [Scale | RestScales], RowBytes, Acc) ->
    <<Row:RowBytes/binary, Rest/binary>> = Bin,
    Scaled = scale_row(Row, Scale, <<>>),
    scale_rows(Rest, RestScales, RowBytes, [Scaled | Acc]).

scale_row(<<>>, _Scale, Acc) -> Acc;
scale_row(<<F:32/float-little, Rest/binary>>, Scale, Acc) ->
    G = F * Scale,
    scale_row(Rest, Scale, <<Acc/binary, G:32/float-little>>).

%% ---------------------------------------------------------------------------
%% Helpers used only by the unit test
%% ---------------------------------------------------------------------------

%% Pack a list of floats as fp32 little-endian.
floats_to_fp32(Xs) ->
    iolist_to_binary([<<F:32/float-little>> || F <- Xs]).

%% Decode an fp32 binary back to a list of floats.
fp32_to_floats(<<>>) -> [];
fp32_to_floats(<<F:32/float-little, Rest/binary>>) ->
    [F | fp32_to_floats(Rest)].

%% Compute W^T @ x, where W is [InF, OutF] row-major (so W[c, :] is
%% the c-th row, contributing to every output through input channel c)
%% and x has length InF. Result length = OutF.
matvec_in_out(WeightBin, X, _InF, OutF) ->
    RowBytes = OutF * 4,
    Zeros = lists:duplicate(OutF, 0.0),
    matvec_loop(WeightBin, X, RowBytes, Zeros).

matvec_loop(<<>>, [], _RowBytes, Acc) -> Acc;
matvec_loop(Bin, [Xc | RestX], RowBytes, Acc) ->
    <<Row:RowBytes/binary, Rest/binary>> = Bin,
    RowFloats = fp32_to_floats(Row),
    NewAcc = add_scaled(Acc, RowFloats, Xc),
    matvec_loop(Rest, RestX, RowBytes, NewAcc).

add_scaled([], [], _S) -> [];
add_scaled([A | RA], [B | RB], S) -> [A + B * S | add_scaled(RA, RB, S)].

list_max_abs(Xs) ->
    lists:foldl(fun(X, M) ->
        A = abs(X),
        case A > M of true -> A; false -> M end
    end, 0.0, Xs).

%% Per-input-channel max_abs of weight rows (sanity for "balanced" check).
row_maxabs_per_input(WeightBin, InF, OutF) ->
    RowBytes = OutF * 4,
    [row_max_abs(R, 0.0)
     || R <- split_rows(WeightBin, RowBytes, InF)].

split_rows(_Bin, _RowBytes, 0) -> [];
split_rows(Bin, RowBytes, N) ->
    <<Row:RowBytes/binary, Rest/binary>> = Bin,
    [Row | split_rows(Rest, RowBytes, N - 1)].

approx_equal(A, B, Tol) ->
    abs(A - B) =< Tol.

list_approx_equal([], [], _Tol) -> true;
list_approx_equal([A | RA], [B | RB], Tol) ->
    case approx_equal(A, B, Tol) of
        true  -> list_approx_equal(RA, RB, Tol);
        false -> {mismatch, A, B}
    end.

%% ---------------------------------------------------------------------------
%% Unit test
%% ---------------------------------------------------------------------------

test() ->
    InF = 4,
    OutF = 2,
    Alpha = 0.5,

    %% Synthetic weight [InF=4, OutF=2] — input channel 0 has a 10x outlier.
    %% Row 0: [10.0, 0.0]   <-- big magnitude on input channel 0
    %% Row 1: [0.1,  1.0]
    %% Row 2: [0.2,  1.0]
    %% Row 3: [0.3,  1.0]
    Rows = [
        [10.0, 0.0],
        [0.1,  1.0],
        [0.2,  1.0],
        [0.3,  1.0]
    ],
    WeightBin = floats_to_fp32(lists:flatten(Rows)),
    ExpectedBytes = InF * OutF * 4,
    ExpectedBytes = byte_size(WeightBin),

    %% Activation max-abs per input channel — channel 0 is the outlier.
    ActMaxAbs = [100.0, 1.0, 1.0, 1.0],

    %% A specific activation vector compatible with ActMaxAbs.
    X = [100.0, 1.0, 1.0, 1.0],

    %% --- 1. Compute scales -------------------------------------------------
    Scales = smoothquant_scale(WeightBin, ActMaxAbs, InF, OutF, Alpha),
    io:format("scales        = ~p~n", [Scales]),

    %% Expected: scale[0] = sqrt(100/10) ~= 3.1623, others = sqrt(1/1) = 1.0
    [S0, S1, S2, S3] = Scales,
    true = approx_equal(S0, math:sqrt(100.0 / 10.0), 1.0e-5),
    true = approx_equal(S1, 1.0, 1.0e-5),
    true = approx_equal(S2, 1.0, 1.0e-5),
    true = approx_equal(S3, 1.0, 1.0e-5),

    %% --- 2. Stats before --------------------------------------------------
    MaxAbsBefore = row_maxabs_per_input(WeightBin, InF, OutF),
    io:format("w_max_abs pre = ~p~n", [MaxAbsBefore]),
    io:format("act_max  pre  = ~p~n", [ActMaxAbs]),

    %% --- 3. Apply scaling --------------------------------------------------
    {Adjusted, ActScales} = apply_smoothquant(WeightBin, Scales, InF, OutF),
    Scales = ActScales,  %% returned identical
    OrigBytes = byte_size(WeightBin),
    OrigBytes = byte_size(Adjusted),

    MaxAbsAfter = row_maxabs_per_input(Adjusted, InF, OutF),
    XAdj = [Xc / S || {Xc, S} <- lists:zip(X, Scales)],
    XAdjMax = list_max_abs(XAdj),
    io:format("w_max_abs post= ~p~n", [MaxAbsAfter]),
    io:format("act     post  = ~p (max_abs ~p)~n", [XAdj, XAdjMax]),

    %% --- 4. Balanced check -------------------------------------------------
    %% Channel 0 weight max_abs grew from 10 to ~31.6, activation shrank
    %% from 100 to ~31.6. They are now in the same ballpark (sqrt product).
    true = approx_equal(lists:nth(1, MaxAbsAfter),
                        10.0 * S0, 1.0e-4),
    %% Activation outlier killed: channel 0 went from 100 to ~sqrt(100*10).
    true = approx_equal(XAdjMax, 100.0 / S0, 1.0e-4),
    true = XAdjMax < 50.0,   %% well below the original 100

    %% --- 5. Mathematical invariant: W^T @ x preserved ---------------------
    YOrig = matvec_in_out(WeightBin, X,    InF, OutF),
    YAdj  = matvec_in_out(Adjusted,  XAdj, InF, OutF),
    io:format("y_orig        = ~p~n", [YOrig]),
    io:format("y_adjusted    = ~p~n", [YAdj]),
    true = list_approx_equal(YOrig, YAdj, 1.0e-4),

    io:format("OK llama_calibration:test/0 passed~n"),
    ok.
