@echo off
REM =============================================================================
REM viva_tensor - Script de Build para Windows
REM =============================================================================
REM
REM Uso:
REM   make build       - Compila o projeto
REM   make test        - Roda os testes
REM   make bench       - Roda benchmarks Gleam
REM   make bench-fused - Roda benchmark Fused Quantized Matmul (MKL 800+ GFLOPS!)
REM   make demo        - Roda demonstracao
REM   make docs        - Gera documentacao
REM   make clean       - Limpa build
REM   make fmt         - Formata codigo
REM   make zig         - Build Zig NIF com MKL
REM   make help        - Mostra ajuda
REM
REM =============================================================================

setlocal enabledelayedexpansion

set "OUTPUT_DIR=output"
set "DATE=%date:~-4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "DATE=%DATE: =0%"

REM Intel oneAPI / MKL paths (winget install Intel.oneMKL)
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if exist "%ONEAPI_ROOT%\mkl\latest\bin" (
    set "PATH=%ONEAPI_ROOT%\mkl\latest\bin;%ONEAPI_ROOT%\compiler\latest\bin;%ONEAPI_ROOT%\tbb\latest\bin;%PATH%"
)

REM MKL tuning for maximum performance (i7-13700K = 24 threads)
set MKL_NUM_THREADS=24
set MKL_THREADING_LAYER=INTEL
set KMP_AFFINITY=scatter
set MKL_DYNAMIC=FALSE

if "%1"=="" goto help
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="bench" goto bench
if "%1"=="bench-fused" goto bench-fused
if "%1"=="demo" goto demo
if "%1"=="docs" goto docs
if "%1"=="clean" goto clean
if "%1"=="fmt" goto fmt
if "%1"=="check" goto check
if "%1"=="deps" goto deps
if "%1"=="zig" goto zig
if "%1"=="zig-clean" goto zig-clean
if "%1"=="build-all" goto build-all
if "%1"=="all" goto all
if "%1"=="help" goto help
goto help

:build
echo [BUILD] Compilando viva_tensor...
gleam build
echo [OK] Build concluido!
goto end

:test
echo [TEST] Executando testes...
gleam test
echo [OK] Testes concluidos!
goto end

:bench
call :ensure_output
echo [BENCH] Executando benchmarks...
gleam run -m viva_tensor/benchmark_full > "%OUTPUT_DIR%\benchmark_%DATE%.txt" 2>&1
echo [OK] Benchmark salvo em: %OUTPUT_DIR%\benchmark_%DATE%.txt
goto end

:bench-fused
call :ensure_output
echo [BENCH-FUSED] Fused Quantized Matmul Benchmark (MKL 800+ GFLOPS!)
echo.
echo MKL Config:
echo   MKL_NUM_THREADS=%MKL_NUM_THREADS%
echo   MKL_THREADING_LAYER=%MKL_THREADING_LAYER%
echo   KMP_AFFINITY=%KMP_AFFINITY%
echo.
escript bench\bench_fused_windows.erl
goto end

:demo
call :ensure_output
echo [DEMO] Executando demonstracao...
gleam run -m viva_tensor/demo > "%OUTPUT_DIR%\demo_%DATE%.txt" 2>&1
echo [OK] Demo salva em: %OUTPUT_DIR%\demo_%DATE%.txt
goto end

:docs
echo [DOCS] Gerando documentacao...
gleam docs build
echo [OK] Docs geradas em: build\docs\
goto end

:clean
echo [CLEAN] Limpando artefatos...
if exist build rmdir /S /Q build
echo [OK] Limpo!
goto end

:fmt
echo [FMT] Formatando codigo...
gleam format src test
echo [OK] Codigo formatado!
goto end

:check
echo [CHECK] Verificando tipos...
gleam check
echo [OK] Tipos OK!
goto end

:deps
echo [DEPS] Baixando dependencias...
gleam deps download
echo [OK] Dependencias instaladas!
goto end

:zig
echo [ZIG] Building Zig SIMD NIF...
if not exist priv mkdir priv
for /f "tokens=*" %%i in ('erl -noshell -eval "io:format([126,115,126,110],[filename:join([code:root_dir(),<<""erts-"">>,erlang:system_info(version),<<""include"">>])])" -s init stop 2^>nul') do set "ERL_INCLUDE=%%i"
echo [ZIG] Using Erlang include: %ERL_INCLUDE%
cd zig_src && zig build -Derl_include="%ERL_INCLUDE%" -Doptimize=ReleaseFast && cd ..
if exist zig_src\zig-out\bin\viva_tensor_zig.dll (
    copy zig_src\zig-out\bin\viva_tensor_zig.dll priv\viva_tensor_zig.dll >nul
    echo [OK] Zig NIF built: priv\viva_tensor_zig.dll
) else (
    echo [FAIL] Zig build failed
)
goto end

:zig-clean
echo [CLEAN] Cleaning Zig NIF...
if exist zig_src\zig-out rmdir /S /Q zig_src\zig-out
if exist zig_src\.zig-cache rmdir /S /Q zig_src\.zig-cache
if exist priv\viva_tensor_zig.dll del /Q priv\viva_tensor_zig.dll
echo [OK] Zig NIF limpo!
goto end

:build-all
call :build
call :zig
echo [OK] Full build (Gleam + Zig NIF) completo!
goto end

:all
call :build
call :test
call :bench
echo [OK] Build completo!
goto end

:ensure_output
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
goto :eof

:help
echo.
echo viva_tensor - Script de Build para Windows
echo ===========================================
echo.
echo Comandos:
echo   make build       - Compila o projeto
echo   make test        - Roda os testes
echo   make bench       - Roda benchmarks Gleam (salva em output/)
echo   make bench-fused - Roda Fused Quantized Matmul benchmark (MKL 800+ GFLOPS!)
echo   make demo        - Roda demonstracao
echo   make docs        - Gera documentacao
echo   make fmt         - Formata codigo
echo   make check       - Verifica tipos
echo   make clean       - Limpa build
echo   make deps        - Instala dependencias
echo   make zig         - Build Zig SIMD NIF com MKL (requer Zig 0.15+)
echo   make zig-clean   - Limpa artefatos Zig NIF
echo   make build-all   - Build Gleam + Zig NIF
echo   make all         - Build + test + bench
echo   make help        - Mostra esta ajuda
echo.
echo MKL Performance:
echo   MKL_NUM_THREADS=24  (use all i7-13700K cores)
echo   MKL_THREADING_LAYER=INTEL
echo   KMP_AFFINITY=scatter
echo   Expected: 800+ GFLOPS for 5000x5000 matmul
echo.
goto end

:end
endlocal
