@echo off
REM Intel MKL MAXIMUM PERFORMANCE tuning
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
set PATH=%ONEAPI_ROOT%\mkl\latest\bin;%ONEAPI_ROOT%\compiler\latest\bin;%ONEAPI_ROOT%\tbb\latest\bin;%PATH%

REM Use ALL cores (24 physical on i7-13700K = 16 P-cores + 8 E-cores)
set MKL_NUM_THREADS=24
set OMP_NUM_THREADS=24

REM Disable dynamic thread adjustment
set MKL_DYNAMIC=FALSE

REM Use Intel OpenMP (default, best for GEMM)
set MKL_THREADING_LAYER=INTEL

REM Scatter affinity across all cores
set KMP_AFFINITY=scatter,granularity=fine

cd /d "%~dp0..\.."
echo === MKL TUNED BENCHMARK (24 threads) ===
echo MKL_NUM_THREADS=%MKL_NUM_THREADS%
echo MKL_THREADING_LAYER=%MKL_THREADING_LAYER%
echo.
if not exist "build\dev\erlang\viva_tensor\ebin" gleam build
erlc -o bench\windows bench\erlang\bench_full.erl
erl -noshell -pa build\dev\erlang\viva_tensor\ebin -pa priv -pa bench\windows -s bench_full run -s init stop
