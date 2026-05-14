# =============================================================================
# viva_tensor - Makefile Cross-Platform (Unix/Windows)
# =============================================================================
#
# Usage:
#   make build      - Build the project
#   make test       - Run tests
#   make bench      - Run benchmarks and save to bench/reports/
#   make bench-regression - Run small stable API regression benchmark
#   make demo       - Run the demonstration
#   make docs       - Generate documentation
#   make clean      - Clean build artifacts
#   make clean-workspace - Remove generated local artifacts from the tree
#   make fmt        - Format code
#   make fmt-check  - Check formatting
#   make check      - Type check
#   make verify     - Run local quality gates
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
DEV_DIR := dev
OUTPUT_DIR := bench/reports
DOCS_DIR := docs
BUILD_DIR := build

# Output files
BENCH_OUTPUT := $(OUTPUT_DIR)$(SEP)benchmark_$(DATE).txt
DEMO_OUTPUT := $(OUTPUT_DIR)$(SEP)demo_$(DATE).txt
METRICS_OUTPUT := $(OUTPUT_DIR)$(SEP)metrics_$(DATE).txt

# Colors for output (Unix only)
ifneq ($(OS),Windows_NT)
    LOG := printf '%b\n'
    GREEN := \033[0;32m
    RED := \033[0;31m
    YELLOW := \033[0;33m
    NC := \033[0m
else
    LOG := echo
    GREEN :=
    RED :=
    YELLOW :=
    NC :=
endif

# =============================================================================
# MAIN TARGETS
# =============================================================================

.PHONY: all verify build test bench bench-regression bench-rtx metrics demo docs clean clean-workspace fmt fmt-check check help
.PHONY: zig zig-cpu zig-cuda cutlass-libs zig-clean zig-info build-all
.PHONY: bench-int8 bench-nf4 bench-awq bench-flash bench-sparse bench-all
.PHONY: watch deps publish

## Build, test, and run benchmarks
all: build test bench
	@$(LOG) "$(GREEN)[OK]$(NC) Build complete!"

## Run local quality gates used before commits and PRs
verify: fmt-check check test docs
	@$(LOG) "$(GREEN)[OK]$(NC) Verification complete!"

## Build the project
build:
	@$(LOG) "$(YELLOW)[BUILD]$(NC) Building viva_tensor..."
	gleam build
ifneq ($(OS),Windows_NT)
	@# Compile Erlang NIF wrapper modules (not compiled by Gleam)
	@erlc -o build/dev/erlang/viva_tensor/ebin src/viva_tensor_zig.erl 2>$(NULL) || true
	@erlc -o build/dev/erlang/viva_tensor/ebin src/viva_tensor_blas.erl 2>$(NULL) || true
endif
	@$(LOG) "$(GREEN)[OK]$(NC) Build finished!"

## Run tests
test:
	@$(LOG) "$(YELLOW)[TEST]$(NC) Running tests..."
	gleam test
	@$(LOG) "$(GREEN)[OK]$(NC) Tests finished!"

## Run tests with native NIF library temporarily hidden
test-no-nif:
	@$(LOG) "$(YELLOW)[TEST]$(NC) Running tests without native NIF..."
ifeq ($(OS),Windows_NT)
	@$(LOG) "$(YELLOW)[SKIP]$(NC) test-no-nif is only implemented for Unix shells."
else
	@set -e; \
	tmp="/tmp/viva_tensor_zig.so.$$$$"; \
	moved=0; \
	restore() { if [ "$$moved" = "1" ] && [ -f "$$tmp" ]; then mkdir -p priv; mv "$$tmp" priv/viva_tensor_zig.so; fi; }; \
	trap restore EXIT; \
	if [ -f priv/viva_tensor_zig.so ]; then mv priv/viva_tensor_zig.so "$$tmp"; moved=1; fi; \
	rm -f build/dev/erlang/viva_tensor/priv/viva_tensor_zig.so; \
	gleam test
endif
	@$(LOG) "$(GREEN)[OK]$(NC) NIF-free tests finished!"

## Run benchmarks and save to bench/reports/
bench: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) Running benchmarks..."
	@echo "=== viva_tensor Benchmark - $(DATE) ===" > $(BENCH_OUTPUT)
	@echo "" >> $(BENCH_OUTPUT)
	gleam run -m viva_tensor/bench/full >> $(BENCH_OUTPUT) 2>&1
	@echo "" >> $(BENCH_OUTPUT)
	@echo "=== Benchmark Complete ===" >> $(BENCH_OUTPUT)
	@$(LOG) "$(GREEN)[OK]$(NC) Benchmark saved to: $(BENCH_OUTPUT)"

## Run RTX-focused benchmark
bench-rtx: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) RTX 4090 benchmark..."
	@echo "=== viva_tensor RTX Benchmark - $(DATE) ===" > $(OUTPUT_DIR)$(SEP)rtx_$(DATE).txt
	gleam run -m viva_tensor/bench/rtx >> $(OUTPUT_DIR)$(SEP)rtx_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) RTX benchmark saved to: $(OUTPUT_DIR)$(SEP)rtx_$(DATE).txt"

## Run small stable API regression benchmark
bench-regression: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) Stable API regression benchmark..."
	@echo "=== viva_tensor Regression Benchmark - $(DATE) ===" > $(OUTPUT_DIR)$(SEP)regression_$(DATE).txt
	gleam run -m viva_tensor/bench/regression >> $(OUTPUT_DIR)$(SEP)regression_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Regression benchmark saved to: $(OUTPUT_DIR)$(SEP)regression_$(DATE).txt"

## Run advanced metrics
metrics: build ensure-output
	@$(LOG) "$(YELLOW)[METRICS]$(NC) Running metrics..."
	@echo "=== viva_tensor Metrics - $(DATE) ===" > $(METRICS_OUTPUT)
	gleam run -m viva_tensor/bench/full >> $(METRICS_OUTPUT) 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Metrics saved to: $(METRICS_OUTPUT)"

## Run the full demonstration
demo: build ensure-output
	@$(LOG) "$(YELLOW)[DEMO]$(NC) Running demonstration..."
	@echo "=== viva_tensor Demo - $(DATE) ===" > $(DEMO_OUTPUT)
	gleam run -m viva_tensor/examples/demo >> $(DEMO_OUTPUT) 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Demo saved to: $(DEMO_OUTPUT)"

## Generate documentation
docs:
	@$(LOG) "$(YELLOW)[DOCS]$(NC) Generating documentation..."
	gleam docs build
	@$(LOG) "$(GREEN)[OK]$(NC) Docs generated at: build/dev/docs/viva_tensor/index.html"

## Format code
fmt:
	@$(LOG) "$(YELLOW)[FMT]$(NC) Formatting code..."
	gleam format $(SRC_DIR) $(TEST_DIR) $(DEV_DIR)
	@$(LOG) "$(GREEN)[OK]$(NC) Code formatted!"

## Check code formatting
fmt-check:
	@$(LOG) "$(YELLOW)[FMT]$(NC) Checking code format..."
	gleam format --check $(SRC_DIR) $(TEST_DIR) $(DEV_DIR)
	@$(LOG) "$(GREEN)[OK]$(NC) Code format OK!"

## Type check without building
check:
	@$(LOG) "$(YELLOW)[CHECK]$(NC) Checking types..."
	gleam check
	@$(LOG) "$(GREEN)[OK]$(NC) Types OK!"

## Clean build artifacts
clean:
	@$(LOG) "$(YELLOW)[CLEAN]$(NC) Cleaning artifacts..."
ifeq ($(OS),Windows_NT)
	@if exist $(BUILD_DIR) $(RMDIR) $(BUILD_DIR)
else
	@$(RMDIR) $(BUILD_DIR) 2>$(NULL) || true
endif
	@$(LOG) "$(GREEN)[OK]$(NC) Clean!"

## Remove generated local artifacts that make the project tree noisy
clean-workspace: clean zig-clean
	@$(LOG) "$(YELLOW)[CLEAN]$(NC) Cleaning benchmark reports and native intermediates..."
ifeq ($(OS),Windows_NT)
	@if exist bench$(SEP)reports $(RMDIR) bench$(SEP)reports
	@if exist zig_src$(SEP)*.o $(RM) zig_src$(SEP)*.o
	@if exist zig_src$(SEP)*.a $(RM) zig_src$(SEP)*.a
else
	@$(RMDIR) bench$(SEP)reports 2>$(NULL) || true
	@$(RM) zig_src$(SEP)*.o 2>$(NULL) || true
	@$(RM) zig_src$(SEP)*.a 2>$(NULL) || true
endif
	@$(LOG) "$(GREEN)[OK]$(NC) Workspace artifacts removed. Research clones under tmp/ are left untouched."

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

# Erlang NIF headers (auto-detected)
ERL_ROOT := $(shell erl -noshell -eval 'io:format("~s", [code:root_dir()]).' -s init stop 2>$(NULL))
ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]).' -s init stop 2>$(NULL))

## CUDA / CUTLASS build settings (used by `cutlass-libs` and `zig-cuda`)
NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CUTLASS_INCLUDE ?= /usr/include
CUSPARSELT_INCLUDE ?= /opt/cusparselt/include
CUDA_INCLUDE ?= /usr/local/cuda/include
NVCC_FLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC \
              -I$(CUTLASS_INCLUDE) -I$(CUSPARSELT_INCLUDE) -I$(CUDA_INCLUDE)

## Build Zig NIF (cross-platform: Windows/Linux/macOS)
## Includes: SIMD kernels, Intel MKL, CUDA, Apple Accelerate
zig:
	@$(LOG) "$(YELLOW)[ZIG]$(NC) Building NIF (Zig + platform backends)..."
	@$(MKDIR) priv
	@cd zig_src && zig build -Derl_include=$(ERL_INCLUDE) -Doptimize=ReleaseFast
ifeq ($(OS),Windows_NT)
	@$(COPY) zig_src$(SEP)zig-out$(SEP)bin$(SEP)viva_tensor_zig.dll priv$(SEP)viva_tensor_zig.dll 2>$(NULL) || true
	@$(LOG) "$(GREEN)[OK]$(NC) NIF built: priv/viva_tensor_zig.dll"
else
	@$(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.dylib priv$(SEP)viva_tensor_zig.so 2>$(NULL) || \
	 $(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.so priv$(SEP)viva_tensor_zig.so 2>$(NULL) || true
	@$(LOG) "$(GREEN)[OK]$(NC) NIF built: priv/viva_tensor_zig.so"
endif

## Compile pre-baked CUTLASS / cuSPARSELt static libs needed by the full Zig NIF.
## Outputs (in zig_src/):
##   - libcutlass_fp8.a            (FP8 E4M3 GEMM, FP16 accum — 660 TOPS on Ada)
##   - libcusparselt_int8.a        (cuSPARSELt INT8 + CUTLASS sparse INT8 GEMM)
##   - libcutlass_int4_sparse.a    (CUTLASS sparse INT4 GEMM)
## Requires: nvcc (CUDA toolkit), cutlass headers, cusparseLt headers.
## Override with: make cutlass-libs CUDA_ARCH=sm_86 CUTLASS_INCLUDE=/path/cutlass/include
cutlass-libs:
	@$(LOG) "$(YELLOW)[NVCC]$(NC) Building CUTLASS static libs (arch=$(CUDA_ARCH))..."
	@command -v $(NVCC) >$(NULL) 2>&1 || { \
	  $(LOG) "$(RED)[ERR]$(NC) nvcc not found. Install CUDA toolkit or pass NVCC=/path/to/nvcc."; \
	  exit 1; \
	}
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_fp8_cutlass.cu          -o cuda_fp8_cutlass.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_sparse_int8_cutlass.cu  -o cuda_sparse_int8_cutlass.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_cusparselt_int8.cu      -o cuda_cusparselt_int8.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_int4_sparse_cutlass.cu  -o cuda_int4_sparse_cutlass.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_fp16_bench.cu          -o cuda_fp16_bench.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_graph_bench.cu         -o cuda_graph_bench.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_fp16_fused_bench.cu    -o cuda_fp16_fused_bench.o
	@cd zig_src && $(NVCC) $(NVCC_FLAGS) -c cuda_nvfp4_emu.cu           -o cuda_nvfp4_emu.o
	@cd zig_src && ar rcs libcutlass_fp8.a         cuda_fp8_cutlass.o cuda_fp16_bench.o cuda_graph_bench.o cuda_fp16_fused_bench.o cuda_nvfp4_emu.o
	@cd zig_src && ar rcs libcusparselt_int8.a     cuda_sparse_int8_cutlass.o cuda_cusparselt_int8.o
	@cd zig_src && ar rcs libcutlass_int4_sparse.a cuda_int4_sparse_cutlass.o
	@$(LOG) "$(GREEN)[OK]$(NC) Built: zig_src/libcutlass_fp8.a, libcusparselt_int8.a, libcutlass_int4_sparse.a"

## Build full Zig NIF with CUDA (RTX Tensor Cores enabled).
## Compiles CUTLASS libs first, then links them into the NIF.
zig-cuda: cutlass-libs zig

## Build Zig NIF without CUDA (CPU/MKL + SIMD only)
## Skips libcutlass_*.a, cuSPARSELt, and nif_cuda_* sources.
## Use in CI or on hosts without CUDA toolkit / cutlass artifacts.
zig-cpu:
	@$(LOG) "$(YELLOW)[ZIG]$(NC) Building NIF (CPU-only: MKL + SIMD, no CUDA)..."
	@$(MKDIR) priv
	@cd zig_src && zig build -Derl_include=$(ERL_INCLUDE) -Dcuda=false -Doptimize=ReleaseFast
ifeq ($(OS),Windows_NT)
	@$(COPY) zig_src$(SEP)zig-out$(SEP)bin$(SEP)viva_tensor_zig.dll priv$(SEP)viva_tensor_zig.dll 2>$(NULL) || true
	@$(LOG) "$(GREEN)[OK]$(NC) NIF built (CPU-only): priv/viva_tensor_zig.dll"
else
	@$(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.dylib priv$(SEP)viva_tensor_zig.so 2>$(NULL) || \
	 $(COPY) zig_src$(SEP)zig-out$(SEP)lib$(SEP)libviva_tensor_zig.so priv$(SEP)viva_tensor_zig.so 2>$(NULL) || true
	@$(LOG) "$(GREEN)[OK]$(NC) NIF built (CPU-only): priv/viva_tensor_zig.so"
endif

## Clean Zig NIF artifacts
zig-clean:
	@$(LOG) "$(YELLOW)[CLEAN]$(NC) Cleaning NIF..."
ifeq ($(OS),Windows_NT)
	@if exist zig_src$(SEP)zig-out $(RMDIR) zig_src$(SEP)zig-out
	@if exist zig_src$(SEP).zig-cache $(RMDIR) zig_src$(SEP).zig-cache
	@if exist priv$(SEP)viva_tensor_zig.so $(RM) priv$(SEP)viva_tensor_zig.so
	@if exist priv$(SEP)viva_tensor_zig.dll $(RM) priv$(SEP)viva_tensor_zig.dll
else
	@$(RMDIR) zig_src$(SEP)zig-out 2>$(NULL) || true
	@$(RMDIR) zig_src$(SEP).zig-cache 2>$(NULL) || true
	@$(RM) priv$(SEP)viva_tensor_zig.so 2>$(NULL) || true
	@$(RM) priv$(SEP)viva_tensor_zig.dll 2>$(NULL) || true
	@$(RM) zig_src$(SEP)*.o zig_src$(SEP)libcutlass_fp8.a zig_src$(SEP)libcusparselt_int8.a zig_src$(SEP)libcutlass_int4_sparse.a 2>$(NULL) || true
endif
	@$(LOG) "$(GREEN)[OK]$(NC) NIF cleaned!"

## Show NIF build info
zig-info:
	@echo "Zig: $$(zig version 2>$(NULL) || echo 'not installed')"
	@echo "ERL_INCLUDE: $(ERL_INCLUDE)"
ifeq ($(shell uname -s),Darwin)
	@echo "Backend: Apple Accelerate"
else ifeq ($(OS),Windows_NT)
	@echo "Backend: Intel MKL"
else
	@echo "Backend: Intel MKL + CUDA"
endif

## Full build including NIF
build-all: build zig
	@$(LOG) "$(GREEN)[OK]$(NC) Full build (Gleam + NIF) complete!"

# =============================================================================
# SPECIFIC BENCHMARKS
# =============================================================================

## Benchmark INT8
bench-int8: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) INT8 Quantization..."
	gleam run -m viva_tensor/quant/compression > $(OUTPUT_DIR)$(SEP)int8_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Saved to $(OUTPUT_DIR)$(SEP)int8_$(DATE).txt"

## Benchmark NF4
bench-nf4: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) NF4 Quantization..."
	gleam run -m viva_tensor/quant/nf4 > $(OUTPUT_DIR)$(SEP)nf4_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Saved to $(OUTPUT_DIR)$(SEP)nf4_$(DATE).txt"

## Benchmark AWQ
bench-awq: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) AWQ Quantization..."
	gleam run -m viva_tensor/quant/awq > $(OUTPUT_DIR)$(SEP)awq_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Saved to $(OUTPUT_DIR)$(SEP)awq_$(DATE).txt"

## Benchmark Flash Attention
bench-flash: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) Flash Attention..."
	gleam run -m viva_tensor/nn/flash_attention > $(OUTPUT_DIR)$(SEP)flash_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Saved to $(OUTPUT_DIR)$(SEP)flash_$(DATE).txt"

## Benchmark 2:4 Sparsity
bench-sparse: build ensure-output
	@$(LOG) "$(YELLOW)[BENCH]$(NC) 2:4 Sparsity..."
	gleam run -m viva_tensor/optim/sparsity > $(OUTPUT_DIR)$(SEP)sparsity_$(DATE).txt 2>&1
	@$(LOG) "$(GREEN)[OK]$(NC) Saved to $(OUTPUT_DIR)$(SEP)sparsity_$(DATE).txt"

## All individual benchmarks
bench-all: bench-int8 bench-nf4 bench-awq bench-flash bench-sparse bench
	@$(LOG) "$(GREEN)[OK]$(NC) All benchmarks complete!"

# =============================================================================
# DEVELOPMENT
# =============================================================================

## Run in watch mode (recompiles on save)
watch:
	@$(LOG) "$(YELLOW)[WATCH]$(NC) Watch mode enabled..."
	gleam run --watch

## Install dependencies
deps:
	@$(LOG) "$(YELLOW)[DEPS]$(NC) Downloading dependencies..."
	gleam deps download
	@$(LOG) "$(GREEN)[OK]$(NC) Dependencies installed!"

## Publish to Hex
publish:
	@$(LOG) "$(YELLOW)[PUBLISH]$(NC) Publishing to Hex..."
	gleam publish
	@$(LOG) "$(GREEN)[OK]$(NC) Published!"

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
	@echo "  make bench       - Run benchmarks (saves to bench/reports/)"
	@echo "  make demo        - Run demonstration"
	@echo "  make docs        - Generate documentation"
	@echo "  make fmt         - Format code"
	@echo "  make fmt-check   - Check formatting"
	@echo "  make check       - Type check"
	@echo "  make verify      - Format check + type check + tests + docs"
	@echo "  make clean       - Clean build"
	@echo "  make clean-workspace - Remove ignored local artifacts from the tree"
	@echo "  make all         - Build + test + bench"
	@echo ""
	@echo "NIF (Native):"
	@echo "  make zig         - Build NIF (cross-platform: MKL/CUDA/Accelerate)"
	@echo "  make zig-clean   - Clean NIF artifacts"
	@echo "  make zig-info    - Show NIF build info"
	@echo "  make build-all   - Build Gleam + NIF"
	@echo ""
	@echo "Specific benchmarks:"
	@echo "  make bench-rtx   - RTX 4090 benchmark"
	@echo "  make bench-int8  - Benchmark INT8"
	@echo "  make bench-nf4   - Benchmark NF4"
	@echo "  make bench-awq   - Benchmark AWQ"
	@echo "  make bench-flash - Benchmark Flash Attention"
	@echo "  make bench-sparse - Benchmark 2:4 Sparsity"
	@echo "  make bench-all   - All benchmarks"
	@echo ""
	@echo "Development:"
	@echo "  make deps        - Install dependencies"
	@echo "  make watch       - Watch mode"
	@echo "  make publish     - Publish to Hex"
	@echo ""
