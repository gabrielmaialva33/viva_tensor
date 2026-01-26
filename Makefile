# =============================================================================
# viva_tensor - Makefile Cross-Platform (Unix/Windows)
# =============================================================================
#
# Uso:
#   make build      - Compila o projeto
#   make test       - Roda os testes
#   make bench      - Roda benchmarks e salva em output/
#   make demo       - Roda a demonstração
#   make docs       - Gera documentação
#   make clean      - Limpa artefatos de build
#   make fmt        - Formata o código
#   make check      - Verifica tipos
#   make all        - Build + test + bench
#
# =============================================================================

# Detecta o sistema operacional
ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
    RM := del /Q /F
    RMDIR := rmdir /S /Q
    MKDIR := mkdir
    SEP := \\
    EXT := .exe
    DATE := $(shell powershell -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'")
    COPY := copy
    NULL := NUL
else
    SHELL := /bin/bash
    RM := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    SEP := /
    EXT :=
    DATE := $(shell date +%Y-%m-%d_%H-%M-%S)
    COPY := cp
    NULL := /dev/null
endif

# Diretórios
SRC_DIR := src
TEST_DIR := test
OUTPUT_DIR := output
DOCS_DIR := docs
BUILD_DIR := build

# Arquivos de saída
BENCH_OUTPUT := $(OUTPUT_DIR)$(SEP)benchmark_$(DATE).txt
DEMO_OUTPUT := $(OUTPUT_DIR)$(SEP)demo_$(DATE).txt
METRICS_OUTPUT := $(OUTPUT_DIR)$(SEP)metrics_$(DATE).txt

# Cores para output (Unix only)
ifneq ($(OS),Windows_NT)
    GREEN := \033[0;32m
    RED := \033[0;31m
    YELLOW := \033[0;33m
    NC := \033[0m
else
    GREEN :=
    RED :=
    YELLOW :=
    NC :=
endif

# =============================================================================
# TARGETS PRINCIPAIS
# =============================================================================

.PHONY: all build test bench demo docs clean fmt check help

## Compila, testa e roda benchmarks
all: build test bench
	@echo "$(GREEN)[OK]$(NC) Build completo!"

## Compila o projeto
build:
	@echo "$(YELLOW)[BUILD]$(NC) Compilando viva_tensor..."
	gleam build
	@echo "$(GREEN)[OK]$(NC) Build concluido!"

## Roda os testes
test:
	@echo "$(YELLOW)[TEST]$(NC) Executando testes..."
	gleam test
	@echo "$(GREEN)[OK]$(NC) Testes concluidos!"

## Roda benchmarks e salva em output/
bench: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) Executando benchmarks..."
	@echo "=== viva_tensor Benchmark - $(DATE) ===" > $(BENCH_OUTPUT)
	@echo "" >> $(BENCH_OUTPUT)
	gleam run -m viva_tensor/benchmark_full >> $(BENCH_OUTPUT) 2>&1
	@echo "" >> $(BENCH_OUTPUT)
	@echo "=== Benchmark Concluido ===" >> $(BENCH_OUTPUT)
	@echo "$(GREEN)[OK]$(NC) Benchmark salvo em: $(BENCH_OUTPUT)"

## Roda métricas avançadas
metrics: build ensure-output
	@echo "$(YELLOW)[METRICS]$(NC) Executando metricas..."
	@echo "=== viva_tensor Metricas - $(DATE) ===" > $(METRICS_OUTPUT)
	gleam run -m viva_tensor/metrics >> $(METRICS_OUTPUT) 2>&1
	@echo "$(GREEN)[OK]$(NC) Metricas salvas em: $(METRICS_OUTPUT)"

## Roda a demonstração completa
demo: build ensure-output
	@echo "$(YELLOW)[DEMO]$(NC) Executando demonstracao..."
	@echo "=== viva_tensor Demo - $(DATE) ===" > $(DEMO_OUTPUT)
	gleam run -m viva_tensor/demo >> $(DEMO_OUTPUT) 2>&1
	@echo "$(GREEN)[OK]$(NC) Demo salva em: $(DEMO_OUTPUT)"

## Gera documentação
docs:
	@echo "$(YELLOW)[DOCS]$(NC) Gerando documentacao..."
	gleam docs build
	@echo "$(GREEN)[OK]$(NC) Docs geradas em: build/docs/"

## Formata o código
fmt:
	@echo "$(YELLOW)[FMT]$(NC) Formatando codigo..."
	gleam format $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)[OK]$(NC) Codigo formatado!"

## Verifica tipos sem compilar
check:
	@echo "$(YELLOW)[CHECK]$(NC) Verificando tipos..."
	gleam check
	@echo "$(GREEN)[OK]$(NC) Tipos OK!"

## Limpa artefatos de build
clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Limpando artefatos..."
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) $(RMDIR) $(BUILD_DIR)
else
	@$(RMDIR) $(BUILD_DIR) 2>$(NULL) || true
endif
	@echo "$(GREEN)[OK]$(NC) Limpo!"

## Cria diretório output se não existir
ensure-output:
ifeq ($(OS),Windows_NT)
	@if not exist $(OUTPUT_DIR) $(MKDIR) $(OUTPUT_DIR)
else
	@$(MKDIR) $(OUTPUT_DIR)
endif

# =============================================================================
# BENCHMARKS ESPECÍFICOS
# =============================================================================

## Benchmark INT8
bench-int8: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) INT8 Quantization..."
	gleam run -m viva_tensor/compression > $(OUTPUT_DIR)$(SEP)int8_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Salvo em output/"

## Benchmark NF4
bench-nf4: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) NF4 Quantization..."
	gleam run -m viva_tensor/nf4 > $(OUTPUT_DIR)$(SEP)nf4_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Salvo em output/"

## Benchmark AWQ
bench-awq: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) AWQ Quantization..."
	gleam run -m viva_tensor/awq > $(OUTPUT_DIR)$(SEP)awq_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Salvo em output/"

## Benchmark Flash Attention
bench-flash: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) Flash Attention..."
	gleam run -m viva_tensor/flash_attention > $(OUTPUT_DIR)$(SEP)flash_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Salvo em output/"

## Benchmark 2:4 Sparsity
bench-sparse: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) 2:4 Sparsity..."
	gleam run -m viva_tensor/sparsity > $(OUTPUT_DIR)$(SEP)sparsity_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Salvo em output/"

## Todos os benchmarks individuais
bench-all: bench-int8 bench-nf4 bench-awq bench-flash bench-sparse bench
	@echo "$(GREEN)[OK]$(NC) Todos os benchmarks concluidos!"

# =============================================================================
# DESENVOLVIMENTO
# =============================================================================

## Roda em modo watch (recompila ao salvar)
watch:
	@echo "$(YELLOW)[WATCH]$(NC) Modo watch ativado..."
	gleam run --watch

## Instala dependências
deps:
	@echo "$(YELLOW)[DEPS]$(NC) Baixando dependencias..."
	gleam deps download
	@echo "$(GREEN)[OK]$(NC) Dependencias instaladas!"

## Publica no Hex
publish:
	@echo "$(YELLOW)[PUBLISH]$(NC) Publicando no Hex..."
	gleam publish
	@echo "$(GREEN)[OK]$(NC) Publicado!"

# =============================================================================
# HELP
# =============================================================================

## Mostra ajuda
help:
	@echo ""
	@echo "viva_tensor - Makefile Cross-Platform"
	@echo "======================================"
	@echo ""
	@echo "Comandos principais:"
	@echo "  make build       - Compila o projeto"
	@echo "  make test        - Roda os testes"
	@echo "  make bench       - Roda benchmarks (salva em output/)"
	@echo "  make demo        - Roda demonstracao"
	@echo "  make docs        - Gera documentacao"
	@echo "  make fmt         - Formata codigo"
	@echo "  make check       - Verifica tipos"
	@echo "  make clean       - Limpa build"
	@echo "  make all         - Build + test + bench"
	@echo ""
	@echo "Benchmarks especificos:"
	@echo "  make bench-int8  - Benchmark INT8"
	@echo "  make bench-nf4   - Benchmark NF4"
	@echo "  make bench-awq   - Benchmark AWQ"
	@echo "  make bench-flash - Benchmark Flash Attention"
	@echo "  make bench-sparse- Benchmark 2:4 Sparsity"
	@echo "  make bench-all   - Todos os benchmarks"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make deps        - Instala dependencias"
	@echo "  make watch       - Modo watch"
	@echo "  make publish     - Publica no Hex"
	@echo ""
