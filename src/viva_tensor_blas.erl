%% viva_tensor_blas.erl - Intelligent BLAS Backend Selection
%%
%% Auto-detects Intel MKL, OpenBLAS, or Zig SIMD and configures
%% for maximum performance on the current CPU.

-module(viva_tensor_blas).
-export([
    detect_backend/0,
    get_cpu_topology/0,
    configure_threads/1,
    set_affinity/1,
    auto_configure/0,
    matmul/5
]).

%% Detect the best available BLAS backend
detect_backend() ->
    case os:type() of
        {win32, _} ->
            %% Windows: Check for MKL DLLs
            case os:getenv("ONEAPI_ROOT") of
                false -> check_openblas_win();
                _ -> intel_mkl
            end;
        _ ->
            %% Unix: Check library availability
            case os:find_executable("ldd") of
                false -> zig_simd;
                _ -> detect_unix_blas()
            end
    end.

detect_unix_blas() ->
    %% Check if MKL is available
    MklPath = "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so",
    case filelib:is_file(MklPath) of
        true ->
            intel_mkl;
        false ->
            %% Check OpenBLAS
            case os:find_executable("libopenblas.so") of
                false ->
                    %% Check system OpenBLAS
                    case filelib:wildcard("/usr/lib/*/libopenblas*.so*") of
                        [] -> zig_simd;
                        _ -> open_blas
                    end;
                _ ->
                    open_blas
            end
    end.

check_openblas_win() ->
    case filelib:wildcard("C:/msys64/*/lib/libopenblas*.dll") of
        [] -> zig_simd;
        _ -> open_blas
    end.

%% Get CPU topology (calls into NIF or fallback)
get_cpu_topology() ->
    try viva_tensor_zig:cpu_topology() of
        Topo when is_map(Topo) -> Topo;
        _ -> fallback_topology()
    catch
        _:_ -> fallback_topology()
    end.

fallback_topology() ->
    Logical = erlang:system_info(logical_processors),
    #{
        physical_cores => Logical div 2,
        logical_cpus => Logical,
        l2_cache_kb => 256,
        l3_cache_kb => 8192,
        has_avx2 => true,
        has_avx512 => false
    }.

%% Configure threads for BLAS operations
configure_threads(NumThreads) ->
    %% Set environment variables for all BLAS implementations
    os:putenv("MKL_NUM_THREADS", integer_to_list(NumThreads)),
    os:putenv("OMP_NUM_THREADS", integer_to_list(NumThreads)),
    os:putenv("OPENBLAS_NUM_THREADS", integer_to_list(NumThreads)),
    os:putenv("GOTO_NUM_THREADS", integer_to_list(NumThreads)),
    os:putenv("MKL_DYNAMIC", "FALSE"),
    {ok, nil}.

%% Set thread affinity mode
set_affinity(Mode) ->
    ModeStr =
        case Mode of
            <<"scatter">> -> "scatter,granularity=fine";
            <<"compact">> -> "compact,granularity=fine";
            _ -> "scatter,granularity=fine"
        end,
    os:putenv("KMP_AFFINITY", ModeStr),
    os:putenv("GOMP_CPU_AFFINITY", "0-23"),
    %% MKL threading layer (INTEL is fastest for GEMM)
    os:putenv("MKL_THREADING_LAYER", "INTEL"),
    {ok, nil}.

%% Auto-configure for maximum performance
auto_configure() ->
    Backend = detect_backend(),
    Topo = get_cpu_topology(),
    Physical = maps:get(physical_cores, Topo, 8),
    Logical = maps:get(logical_cpus, Topo, 16),

    %% Use all physical cores (includes E-cores on hybrid)
    NumThreads = Physical,

    configure_threads(NumThreads),
    set_affinity(<<"scatter">>),

    io:format("~n=== VIVA TENSOR BLAS ===~n"),
    io:format("Backend: ~p~n", [Backend]),
    io:format("Threads: ~B (of ~B logical)~n", [NumThreads, Logical]),
    io:format("Affinity: scatter~n"),
    io:format("========================~n~n"),

    {ok, Backend}.

%% Smart matmul - uses best available backend
matmul(A, B, M, N, K) ->
    case detect_backend() of
        intel_mkl ->
            %% Use MKL via cblas_dgemm
            viva_tensor_zig:nt_matmul_blas(A, B, M, N, K);
        open_blas ->
            %% Use OpenBLAS via cblas_dgemm
            viva_tensor_zig:nt_matmul_blas(A, B, M, N, K);
        zig_simd ->
            %% Fallback to Zig GEMM
            viva_tensor_zig:nt_matmul(A, B, M, N, K)
    end.
