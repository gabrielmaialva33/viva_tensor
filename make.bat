@echo off
REM =============================================================================
REM viva_tensor - Script de Build para Windows
REM =============================================================================
REM
REM Uso:
REM   make build      - Compila o projeto
REM   make test       - Roda os testes
REM   make bench      - Roda benchmarks
REM   make demo       - Roda demonstracao
REM   make docs       - Gera documentacao
REM   make clean      - Limpa build
REM   make fmt        - Formata codigo
REM   make help       - Mostra ajuda
REM
REM =============================================================================

setlocal enabledelayedexpansion

set "OUTPUT_DIR=output"
set "DATE=%date:~-4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%"
set "DATE=%DATE: =0%"

if "%1"=="" goto help
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="bench" goto bench
if "%1"=="demo" goto demo
if "%1"=="docs" goto docs
if "%1"=="clean" goto clean
if "%1"=="fmt" goto fmt
if "%1"=="check" goto check
if "%1"=="deps" goto deps
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
echo   make build      - Compila o projeto
echo   make test       - Roda os testes
echo   make bench      - Roda benchmarks (salva em output/)
echo   make demo       - Roda demonstracao
echo   make docs       - Gera documentacao
echo   make fmt        - Formata codigo
echo   make check      - Verifica tipos
echo   make clean      - Limpa build
echo   make deps       - Instala dependencias
echo   make all        - Build + test + bench
echo   make help       - Mostra esta ajuda
echo.
goto end

:end
endlocal
