%%% End-to-end smoke test: load TinyLlama-1.1B layer 0 weights and run
%%% a full transformer block forward through viva_tensor's FP8 linears.
%%%
%%% Loads from tmp/tinyllama/model.safetensors. Prints per-stage magnitudes
%%% (mean abs value of the hidden state) and asserts everything is finite.
%%%
%%% Run: erl -pa build/dev/erlang/viva_tensor/ebin -noshell -s llama_smoke run -s init stop

-module(llama_smoke).
-export([run/0, time_stage/2]).

-define(PATH, <<"tmp/tinyllama/model.safetensors">>).
-define(HIDDEN, 2048).
-define(KV_DIM, 256).         %% 4 KV heads × 64 head_dim
-define(NUM_HEADS, 32).
-define(NUM_KV_HEADS, 4).
-define(HEAD_DIM, 64).
-define(FFN, 5632).

run() ->
    io:format("~n=== TinyLlama-1.1B layer-0 forward smoke test ===~n~n"),
    Header = time_stage("Open header",
        fun() -> {ok, H} = viva_tensor_safetensors_ffi:open_header(?PATH), H end),

    %% Load + transpose all 7 linear weights (HF [out, in] -> viva [in, out]).
    QProj    = load_linear(Header, <<"model.layers.0.self_attn.q_proj.weight">>, ?HIDDEN, ?HIDDEN),
    KProj    = load_linear(Header, <<"model.layers.0.self_attn.k_proj.weight">>, ?KV_DIM, ?HIDDEN),
    VProj    = load_linear(Header, <<"model.layers.0.self_attn.v_proj.weight">>, ?KV_DIM, ?HIDDEN),
    OProj    = load_linear(Header, <<"model.layers.0.self_attn.o_proj.weight">>, ?HIDDEN, ?HIDDEN),
    GateProj = load_linear(Header, <<"model.layers.0.mlp.gate_proj.weight">>,    ?FFN,    ?HIDDEN),
    UpProj   = load_linear(Header, <<"model.layers.0.mlp.up_proj.weight">>,      ?FFN,    ?HIDDEN),
    DownProj = load_linear(Header, <<"model.layers.0.mlp.down_proj.weight">>,    ?HIDDEN, ?FFN),

    %% Load 1-D RMSNorm gammas.
    Norm1 = load_rmsnorm(Header, <<"model.layers.0.input_layernorm.weight">>),
    Norm2 = load_rmsnorm(Header, <<"model.layers.0.post_attention_layernorm.weight">>),

    %% Build a deterministic input hidden state [1, 2048] in fp32, then fp16.
    Seed  = 17,
    Xf32  = time_stage("Build input",
        fun() -> deterministic_floats(Seed, ?HIDDEN) end),
    Xf16  = floats_to_fp16(Xf32),

    io:format("~nInput  mean_abs=~p~n", [mean_abs(Xf32)]),

    %% Prepack all 7 linears in FP8.
    QPacked    = time_stage("Prepack Q",    fun() -> prepack(QProj,    ?HIDDEN, ?HIDDEN) end),
    KPacked    = time_stage("Prepack K",    fun() -> prepack(KProj,    ?HIDDEN, ?KV_DIM) end),
    VPacked    = time_stage("Prepack V",    fun() -> prepack(VProj,    ?HIDDEN, ?KV_DIM) end),
    OPacked    = time_stage("Prepack O",    fun() -> prepack(OProj,    ?HIDDEN, ?HIDDEN) end),
    GatePacked = time_stage("Prepack Gate", fun() -> prepack(GateProj, ?HIDDEN, ?FFN) end),
    UpPacked   = time_stage("Prepack Up",   fun() -> prepack(UpProj,   ?HIDDEN, ?FFN) end),
    DownPacked = time_stage("Prepack Down", fun() -> prepack(DownProj, ?FFN,    ?HIDDEN) end),

    %% RMSNorm 1 (input layernorm).
    XNorm1 = rmsnorm(Xf32, Norm1, 1.0e-5),
    XNorm1Fp16 = floats_to_fp16(XNorm1),
    io:format("~nAfter input_layernorm  mean_abs=~p~n", [mean_abs(XNorm1)]),

    %% Q/K/V projections via FP8 linears.
    Q = time_stage("Linear Q", fun() -> linear_fp8(XNorm1Fp16, QPacked, ?HIDDEN) end),
    K = time_stage("Linear K", fun() -> linear_fp8(XNorm1Fp16, KPacked, ?KV_DIM) end),
    V = time_stage("Linear V", fun() -> linear_fp8(XNorm1Fp16, VPacked, ?KV_DIM) end),
    io:format("Q mean_abs=~p  K mean_abs=~p  V mean_abs=~p~n",
              [mean_abs(Q), mean_abs(K), mean_abs(V)]),

    %% Attention: GQA with 32 Q heads, 4 KV heads. Single-token forward, no
    %% RoPE (would need theta-indexed rotation), no KV cache. With a single
    %% token of context the attention degenerates: softmax of one scalar = 1.
    %% So attn_out per head = v_head. Concatenate across heads.
    AttnOut = attention_single_token(Q, K, V),
    io:format("Attention out mean_abs=~p~n", [mean_abs(AttnOut)]),

    %% Output projection. Both CUTLASS and cuBLASLt FP8 paths now write
    %% FP32 device buffers, so no Inf saturation from FP16 cast.
    AttnOutFp16 = floats_to_fp16(AttnOut),
    O = time_stage("Linear O", fun() -> linear_fp8(AttnOutFp16, OPacked, ?HIDDEN) end),
    io:format("O proj mean_abs=~p~n", [mean_abs(O)]),

    %% Residual.
    Resid1 = list_add(Xf32, O),
    io:format("After residual 1  mean_abs=~p~n", [mean_abs(Resid1)]),

    %% RMSNorm 2 (post-attention).
    XNorm2 = rmsnorm(Resid1, Norm2, 1.0e-5),
    XNorm2Fp16 = floats_to_fp16(XNorm2),
    io:format("After post_attention_layernorm  mean_abs=~p~n", [mean_abs(XNorm2)]),

    %% SwiGLU FFN: intermediate = silu(gate(x)) * up(x), then down.
    %% SwiGLU: fused gate*silu*up via the dedicated NIF (single call), then
    %% the standard linear_fp8 for the down projection. Replaces the older
    %% manual 3-linear workaround now that the SwiGLU NIF's symbol link
    %% has been fixed.
    SwiGluInter = time_stage("SwiGLU gate*silu*up (fused NIF)",
        fun() -> linear_swiglu_intermediate(XNorm2, 1, ?HIDDEN, GatePacked, UpPacked, ?FFN) end),
    io:format("SwiGLU intermediate mean_abs=~p~n", [mean_abs(SwiGluInter)]),

    SwiGluInterFp16 = floats_to_fp16(SwiGluInter),
    FfnOut = time_stage("Linear Down",
        fun() -> linear_fp8(SwiGluInterFp16, DownPacked, ?HIDDEN) end),
    io:format("FFN out mean_abs=~p~n", [mean_abs(FfnOut)]),

    %% Residual 2.
    Resid2 = list_add(Resid1, FfnOut),
    io:format("Final hidden  mean_abs=~p  finite_count=~p / ~p~n",
              [mean_abs(Resid2), length([X || X <- Resid2, is_finite(X)]), length(Resid2)]),

    ok.

%% ---------------------------------------------------------------------------
%% Loaders
%% ---------------------------------------------------------------------------

load_linear(Header, Name, OutF, InF) ->
    time_stage(io_lib:format("Load+transpose ~s (~px~p)", [Name, OutF, InF]),
        fun() ->
            {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
            Fp32 = viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16),
            %% HF stores [out, in] row-major. viva_tensor prepack expects
            %% [in, out] row-major, so transpose.
            {ok, Trans} = viva_tensor_safetensors_ffi:transpose_fp32(Fp32, OutF, InF),
            Trans
        end).

load_rmsnorm(Header, Name) ->
    {ok, Bf16} = viva_tensor_safetensors_ffi:read_tensor_bf16(Header, Name),
    viva_tensor_safetensors_ffi:rmsnorm_weight_to_fp32_list(Bf16).

prepack(Bin, InF, OutF) when is_binary(Bin) ->
    case viva_tensor_zig:nt_prepack_fp8(Bin, [InF, OutF]) of
        {ok, {Resource, _, _, _}} -> Resource;
        {ok, Resource} when is_reference(Resource) -> Resource;
        Other -> error({prepack_failed, Other})
    end.

%% ---------------------------------------------------------------------------
%% Linear NIF wrappers
%% ---------------------------------------------------------------------------

%% linear_fp8(Input_FP16_binary, Packed, Bias_or_nil, Epilogue_int) -> {ok, OutFp16Binary}
linear_fp8(InputFp16, Packed, OutF) when is_binary(InputFp16) ->
    case viva_tensor_zig:nt_linear_fp8(InputFp16, Packed, nil, 0) of
        {ok, OutBin} ->
            Vals = viva_tensor_inference_ffi:fp16_binary_to_floats(OutBin),
            verify_size(Vals, OutF),
            Vals;
        Error ->
            error({linear_fp8_failed, Error})
    end.

%% linear_swiglu_fp8(InputDataList, InputShapeList, Gate, Up, BiasOrNil)
%% Returns the SwiGLU intermediate [B, ffn] — does NOT include the down
%% projection. Caller must apply linear_fp8 with down_proj afterwards.
linear_swiglu_intermediate(InputFp32List, B, InF, Gate, Up, FfnDim) ->
    case viva_tensor_zig:nt_linear_swiglu_fp8(InputFp32List, [B, InF], Gate, Up, nil) of
        {ok, OutBin} ->
            Vals = viva_tensor_inference_ffi:fp16_binary_to_floats(OutBin),
            verify_size(Vals, B * FfnDim),
            Vals;
        Error ->
            error({swiglu_failed, Error})
    end.

verify_size(L, Expected) ->
    case length(L) of
        Expected -> ok;
        N -> error({size_mismatch, got, N, expected, Expected})
    end.

%% ---------------------------------------------------------------------------
%% Plain Erlang math helpers
%% ---------------------------------------------------------------------------

deterministic_floats(Seed, N) ->
    deterministic_floats(Seed, N, []).
deterministic_floats(_, 0, Acc) -> lists:reverse(Acc);
deterministic_floats(S, N, Acc) ->
    Next = (S * 1664525 + 1013904223) rem 2147483647,
    X = (Next / 2147483647.0) * 2.0 - 1.0,
    deterministic_floats(Next, N - 1, [X | Acc]).

rmsnorm(X, Gamma, Eps) ->
    SumSq = lists:foldl(fun(V, A) -> A + V * V end, 0.0, X),
    N = length(X),
    InvRms = 1.0 / math:sqrt(SumSq / N + Eps),
    lists:zipwith(fun(V, G) -> V * InvRms * G end, X, Gamma).

list_add(A, B) ->
    lists:zipwith(fun(X, Y) -> X + Y end, A, B).

silu(X) when is_float(X) ->
    X / (1.0 + math:exp(-X)).

clamp_inf(L, Max) ->
    [clamp_one(X, Max) || X <- L].
clamp_one(X, Max) when X > Max  -> Max;
clamp_one(X, Max) when X < -Max -> -Max;
clamp_one(X, _) -> X.

%% Numerically stable mean_abs that survives Inf/extreme values: divides
%% as we go to avoid summing 2048+ huge numbers into a single overflowed
%% float. Also reports how many values were infinite.
mean_abs(L) ->
    {Sum, Count, InfCount, MaxAbs} = lists:foldl(
        fun(X, {S, N, Inf, M}) ->
            A = abs(X),
            case A > 1.0e30 of
                true  -> {S, N + 1, Inf + 1, max(M, A)};
                false -> {S + A / 1.0e6, N + 1, Inf, max(M, A)}
            end
        end, {0.0, 0, 0, 0.0}, L),
    case Count of
        0 -> 0.0;
        _ ->
            Mean = (Sum * 1.0e6) / Count,
            case InfCount of
                0 -> Mean;
                _ -> io:format("  (warning: ~p inf-ish values, max_abs=~p)~n", [InfCount, MaxAbs]), Mean
            end
    end.

is_finite(X) when is_float(X) ->
    %% Erlang only allows finite floats — Inf/NaN are not representable as
    %% the `float` type, so existence as `is_float(X)` is already the check.
    true;
is_finite(_) -> false.

floats_to_fp16(L) when is_list(L) ->
    viva_tensor_inference_ffi:floats_to_fp16_binary(L).

%% Single-token attention with GQA. Each Q head attends to its assigned KV
%% head. With a single token of context, attention collapses to v_head
%% (softmax of single element = 1.0). Output is concatenation of v_heads,
%% repeated 8× to match Q's head count via GQA broadcast.
attention_single_token(_Q, _K, V) ->
    %% V is [kv_dim=256] = 4 heads × 64 head_dim.
    %% Each Q head (32 total) reads its assigned KV head (kv_idx = q_idx / 8).
    %% Output [num_heads * head_dim = 2048].
    %% For single-token forward this reduces to repeat each KV head 8×.
    VHeads = chunks(V, ?HEAD_DIM),    %% list of 4 lists of 64 floats
    lists:flatten(
        [lists:nth(QHead div (?NUM_HEADS div ?NUM_KV_HEADS) + 1, VHeads)
         || QHead <- lists:seq(0, ?NUM_HEADS - 1)]
    ).

chunks([], _) -> [];
chunks(L, N) ->
    {H, T} = lists:split(N, L),
    [H | chunks(T, N)].

time_stage(Label, F) ->
    T0 = erlang:monotonic_time(millisecond),
    R = F(),
    T1 = erlang:monotonic_time(millisecond),
    io:format("[~5w ms] ~s~n", [T1 - T0, Label]),
    R.
