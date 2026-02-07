#!/bin/bash
#
# viva_tensor Benchmark Runner
# ============================
#
# Usage:
#   ./bench/run_benchmarks.sh          # Run all benchmarks
#   ./bench/run_benchmarks.sh --quick  # Quick run (smaller sizes)
#   ./bench/run_benchmarks.sh --ci     # CI mode (save to artifacts)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================================"
echo "  viva_tensor Benchmark Suite"
echo "============================================================"
echo ""
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "ERROR: numpy not installed (pip install numpy)"
    exit 1
fi

# Check if viva_tensor NIF is available
if ! erl -pa build/dev/erlang/*/ebin ebin -noshell -eval \
    'case viva_tensor_zig:is_loaded() of true -> halt(0); false -> halt(1) end' 2>/dev/null; then
    echo "WARNING: viva_tensor NIF not loaded, building..."
    gleam build
fi

echo "Dependencies OK"
echo ""

# Set optimal environment
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}

echo "Environment:"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# Run Python benchmarks
echo "Running benchmarks..."
python3 bench/benchmark.py

# Run R analysis if available
if command -v Rscript &> /dev/null; then
    echo ""
    echo "Running statistical analysis..."
    Rscript bench/analysis.R 2>/dev/null || echo "R analysis skipped (missing packages)"
fi

echo ""
echo "============================================================"
echo "  Benchmark complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Data: bench/data/benchmark_latest.json"
echo "  - Report: bench/reports/benchmark_report.md"
echo "  - Plots: bench/reports/*.png"
echo ""
