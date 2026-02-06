# =============================================================================
# viva_tensor - Makefile Cross-Platform (Unix/Windows)
# =============================================================================
#
# Usage:
#   make build      - Build the project
#   make test       - Run tests
#   make bench      - Run benchmarks and save to output/
#   make demo       - Run the demonstration
#   make docs       - Generate documentation
#   make clean      - Clean build artifacts
#   make fmt        - Format code
#   make check      - Type check
#   make all        - Build + test + bench
#
# =============================================================================

# Detect operating system
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

# Directories
SRC_DIR := src
TEST_DIR := test
OUTPUT_DIR := output
DOCS_DIR := docs
BUILD_DIR := build

# Output files
BENCH_OUTPUT := $(OUTPUT_DIR)$(SEP)benchmark_$(DATE).txt
DEMO_OUTPUT := $(OUTPUT_DIR)$(SEP)demo_$(DATE).txt
METRICS_OUTPUT := $(OUTPUT_DIR)$(SEP)metrics_$(DATE).txt

# Colors for output (Unix only)
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
# MAIN TARGETS
# =============================================================================

.PHONY: all build test bench demo docs clean fmt check help

## Build, test, and run benchmarks
all: build test bench
	@echo "$(GREEN)[OK]$(NC) Build complete!"

## Build the project
build:
	@echo "$(YELLOW)[BUILD]$(NC) Building viva_tensor..."
	gleam build
	@echo "$(GREEN)[OK]$(NC) Build finished!"

## Run tests
test:
	@echo "$(YELLOW)[TEST]$(NC) Running tests..."
	gleam test
	@echo "$(GREEN)[OK]$(NC) Tests finished!"

## Run benchmarks and save to output/
bench: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) Running benchmarks..."
	@echo "=== viva_tensor Benchmark - $(DATE) ===" > $(BENCH_OUTPUT)
	@echo "" >> $(BENCH_OUTPUT)
	gleam run -m bench/full >> $(BENCH_OUTPUT) 2>&1
	@echo "" >> $(BENCH_OUTPUT)
	@echo "=== Benchmark Complete ===" >> $(BENCH_OUTPUT)
	@echo "$(GREEN)[OK]$(NC) Benchmark saved to: $(BENCH_OUTPUT)"

## Run advanced metrics
metrics: build ensure-output
	@echo "$(YELLOW)[METRICS]$(NC) Running metrics..."
	@echo "=== viva_tensor Metrics - $(DATE) ===" > $(METRICS_OUTPUT)
	gleam run -m viva_tensor/metrics >> $(METRICS_OUTPUT) 2>&1
	@echo "$(GREEN)[OK]$(NC) Metrics saved to: $(METRICS_OUTPUT)"

## Run the full demonstration
demo: build ensure-output
	@echo "$(YELLOW)[DEMO]$(NC) Running demonstration..."
	@echo "=== viva_tensor Demo - $(DATE) ===" > $(DEMO_OUTPUT)
	gleam run -m examples/demo >> $(DEMO_OUTPUT) 2>&1
	@echo "$(GREEN)[OK]$(NC) Demo saved to: $(DEMO_OUTPUT)"

## Generate documentation
docs:
	@echo "$(YELLOW)[DOCS]$(NC) Generating documentation..."
	gleam docs build
	@echo "$(GREEN)[OK]$(NC) Docs generated at: build/docs/"

## Format code
fmt:
	@echo "$(YELLOW)[FMT]$(NC) Formatting code..."
	gleam format $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)[OK]$(NC) Code formatted!"

## Type check without building
check:
	@echo "$(YELLOW)[CHECK]$(NC) Checking types..."
	gleam check
	@echo "$(GREEN)[OK]$(NC) Types OK!"

## Clean build artifacts
clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Cleaning artifacts..."
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) $(RMDIR) $(BUILD_DIR)
else
	@$(RMDIR) $(BUILD_DIR) 2>$(NULL) || true
endif
	@echo "$(GREEN)[OK]$(NC) Clean!"

## Create output directory if it doesn't exist
ensure-output:
ifeq ($(OS),Windows_NT)
	@if not exist $(OUTPUT_DIR) $(MKDIR) $(OUTPUT_DIR)
else
	@$(MKDIR) $(OUTPUT_DIR)
endif

# =============================================================================
# NIF BUILD (Apple Accelerate on macOS)
# =============================================================================

.PHONY: nif nif-clean nif-info zig zig-clean

# Erlang NIF headers (auto-detected)
ERL_ROOT := $(shell erl -noshell -eval 'io:format("~s", [code:root_dir()]).' -s init stop 2>$(NULL))
ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]).' -s init stop 2>$(NULL))

## Build Apple Accelerate NIF (macOS only)
nif:
ifeq ($(OS),Windows_NT)
	@echo "$(YELLOW)[SKIP]$(NC) NIF only supported on macOS"
else
ifeq ($(shell uname -s),Darwin)
	@echo "$(YELLOW)[NIF]$(NC) Building Apple Accelerate NIF..."
	@$(MAKE) -C c_src
	@echo "$(GREEN)[OK]$(NC) NIF built: priv/viva_tensor_nif.so"
else
	@echo "$(YELLOW)[SKIP]$(NC) NIF only supported on macOS"
endif
endif

## Build Zig SIMD NIF (cross-platform, requires Zig 0.15+)
zig:
	@echo "$(YELLOW)[ZIG]$(NC) Building Zig SIMD NIF..."
	@$(MKDIR) priv
	@cd zig_src && zig build -Derl_include=$(ERL_INCLUDE) -Doptimize=ReleaseFast
ifeq ($(OS),Windows_NT)
	@$(COPY) zig_src$(SEP)zig-out$(SEP)bin$(SEP)viva_tensor_zig.dll priv$(SEP)viva_tensor_zig.dll 2>$(NULL) || true
	@echo "$(GREEN)[OK]$(NC) Zig NIF built: priv/viva_tensor_zig.dll"
else
	@$(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.dylib priv$(SEP)viva_tensor_zig.so 2>$(NULL) || \
	 $(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.so priv$(SEP)viva_tensor_zig.so 2>$(NULL) || true
	@echo "$(GREEN)[OK]$(NC) Zig NIF built: priv/viva_tensor_zig.so"
endif

## Clean Zig NIF artifacts
zig-clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Cleaning Zig NIF..."
	@$(RMDIR) zig_src$(SEP)zig-out 2>$(NULL) || true
	@$(RMDIR) zig_src$(SEP).zig-cache 2>$(NULL) || true
	@$(RM) priv$(SEP)viva_tensor_zig.so 2>$(NULL) || true
	@echo "$(GREEN)[OK]$(NC) Zig NIF cleaned!"

## Clean C NIF artifacts
nif-clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Cleaning C NIF..."
	@$(MAKE) -C c_src clean 2>$(NULL) || true
	@$(RM) priv$(SEP)viva_tensor_nif.so 2>$(NULL) || true
	@echo "$(GREEN)[OK]$(NC) C NIF cleaned!"

## Show NIF build info
nif-info:
ifeq ($(shell uname -s),Darwin)
	@$(MAKE) -C c_src info
	@echo "Zig: $$(zig version 2>$(NULL) || echo 'not installed')"
	@echo "ERL_INCLUDE: $(ERL_INCLUDE)"
else
	@echo "C NIF only supported on macOS"
	@echo "Zig: $$(zig version 2>$(NULL) || echo 'not installed')"
	@echo "ERL_INCLUDE: $(ERL_INCLUDE)"
endif

## Full build including all NIFs
build-all: build nif zig
	@echo "$(GREEN)[OK]$(NC) Full build (Gleam + C NIF + Zig NIF) complete!"

# =============================================================================
# SPECIFIC BENCHMARKS
# =============================================================================

## Benchmark INT8
bench-int8: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) INT8 Quantization..."
	gleam run -m viva_tensor/compression > $(OUTPUT_DIR)$(SEP)int8_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Saved to output/"

## Benchmark NF4
bench-nf4: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) NF4 Quantization..."
	gleam run -m viva_tensor/nf4 > $(OUTPUT_DIR)$(SEP)nf4_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Saved to output/"

## Benchmark AWQ
bench-awq: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) AWQ Quantization..."
	gleam run -m viva_tensor/awq > $(OUTPUT_DIR)$(SEP)awq_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Saved to output/"

## Benchmark Flash Attention
bench-flash: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) Flash Attention..."
	gleam run -m viva_tensor/flash_attention > $(OUTPUT_DIR)$(SEP)flash_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Saved to output/"

## Benchmark 2:4 Sparsity
bench-sparse: build ensure-output
	@echo "$(YELLOW)[BENCH]$(NC) 2:4 Sparsity..."
	gleam run -m viva_tensor/sparsity > $(OUTPUT_DIR)$(SEP)sparsity_$(DATE).txt 2>&1
	@echo "$(GREEN)[OK]$(NC) Saved to output/"

## All individual benchmarks
bench-all: bench-int8 bench-nf4 bench-awq bench-flash bench-sparse bench
	@echo "$(GREEN)[OK]$(NC) All benchmarks complete!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

## Run in watch mode (recompiles on save)
watch:
	@echo "$(YELLOW)[WATCH]$(NC) Watch mode enabled..."
	gleam run --watch

## Install dependencies
deps:
	@echo "$(YELLOW)[DEPS]$(NC) Downloading dependencies..."
	gleam deps download
	@echo "$(GREEN)[OK]$(NC) Dependencies installed!"

## Publish to Hex
publish:
	@echo "$(YELLOW)[PUBLISH]$(NC) Publishing to Hex..."
	gleam publish
	@echo "$(GREEN)[OK]$(NC) Published!"

# =============================================================================
# HELP
# =============================================================================

## Show help
help:
	@echo ""
	@echo "viva_tensor - Makefile Cross-Platform"
	@echo "======================================"
	@echo ""
	@echo "Main commands:"
	@echo "  make build       - Build the project"
	@echo "  make test        - Run tests"
	@echo "  make bench       - Run benchmarks (saves to output/)"
	@echo "  make demo        - Run demonstration"
	@echo "  make docs        - Generate documentation"
	@echo "  make fmt         - Format code"
	@echo "  make check       - Type check"
	@echo "  make clean       - Clean build"
	@echo "  make all         - Build + test + bench"
	@echo ""
	@echo "NIFs:"
	@echo "  make nif         - Build Apple Accelerate NIF (macOS)"
	@echo "  make zig         - Build Zig SIMD NIF (cross-platform)"
	@echo "  make nif-clean   - Clean C NIF artifacts"
	@echo "  make zig-clean   - Clean Zig NIF artifacts"
	@echo "  make nif-info    - Show NIF build info"
	@echo "  make build-all   - Build Gleam + C NIF + Zig NIF"
	@echo ""
	@echo "Specific benchmarks:"
	@echo "  make bench-int8  - Benchmark INT8"
	@echo "  make bench-nf4   - Benchmark NF4"
	@echo "  make bench-awq   - Benchmark AWQ"
	@echo "  make bench-flash - Benchmark Flash Attention"
	@echo "  make bench-sparse- Benchmark 2:4 Sparsity"
	@echo "  make bench-all   - All benchmarks"
	@echo ""
	@echo "Development:"
	@echo "  make deps        - Install dependencies"
	@echo "  make watch       - Watch mode"
	@echo "  make publish     - Publish to Hex"
	@echo ""
