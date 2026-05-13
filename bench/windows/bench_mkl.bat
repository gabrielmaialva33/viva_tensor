@echo off
set PATH=C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin;%PATH%
set MKL_NUM_THREADS=8
cd /d "%~dp0..\.."
if not exist "build\dev\erlang\viva_tensor\ebin" gleam build
erlc -o bench\windows bench\erlang\bench_mkl.erl
erl -noshell -pa build\dev\erlang\viva_tensor\ebin -pa priv -pa bench\windows -s bench_mkl run -s init stop
