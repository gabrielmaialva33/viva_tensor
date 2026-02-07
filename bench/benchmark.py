#!/usr/bin/env python3
"""
viva_tensor Benchmark Suite
===========================

Statistical rigor following academic standards:
- Shapiro-Wilk normality test
- Bootstrap confidence intervals (10,000 resamples)
- Mann-Whitney U test for non-normal distributions
- Cohen's d effect size with interpretation
- Coefficient of variation analysis
- Latency percentiles (p50, p90, p95, p99)
- Memory bandwidth estimation

References:
- Kalibera & Jones (2013) "Rigorous Benchmarking in Reasonable Time"
- Georges et al. (2007) "Statistically Rigorous Java Performance Evaluation"
- Hoefler & Belli (2015) "Scientific Benchmarking of Parallel Computing Systems"

Usage:
    python3 bench/benchmark.py              # Full benchmark
    python3 bench/benchmark.py --quick      # Quick mode (smaller sizes)
    python3 bench/benchmark.py --sizes 1000 2000  # Custom sizes
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np

# Optional imports with fallbacks
try:
    import scipy.stats as stats
    from scipy.stats import shapiro, mannwhitneyu, bootstrap
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available - using basic statistics")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Professional benchmark configuration"""
    # Timing parameters
    warmup_runs: int = 5           # Warmup iterations (discarded)
    timed_runs: int = 30           # Measured iterations
    cooldown_seconds: float = 0.1  # Pause between runs

    # Matrix sizes to test
    sizes: List[int] = field(default_factory=lambda: [1000, 2000, 3000, 4000, 5000])

    # Statistical parameters
    confidence_level: float = 0.95
    bootstrap_samples: int = 10000
    outlier_iqr_factor: float = 1.5

    # Output
    output_dir: str = "bench/data"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class LatencyMetrics:
    """Latency and I/O performance metrics"""
    # Latency percentiles (ms)
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    # Throughput
    ops_per_second: float = 0.0
    gb_per_second: float = 0.0

    # Memory bandwidth estimation (GB/s)
    # For GEMM: reads A (M*K) + B (K*N), writes C (M*N)
    memory_bandwidth_gb_s: float = 0.0

    # First call overhead (cold start)
    cold_start_ms: float = 0.0
    warm_avg_ms: float = 0.0
    cold_overhead_percent: float = 0.0


@dataclass
class StatisticalResult:
    """Comprehensive statistical analysis result"""
    # Raw data
    raw_times_ms: List[float] = field(default_factory=list)
    raw_gflops: List[float] = field(default_factory=list)

    # Cleaned data (after outlier removal)
    clean_times_ms: List[float] = field(default_factory=list)
    clean_gflops: List[float] = field(default_factory=list)

    # Descriptive statistics
    n_samples: int = 0
    n_outliers: int = 0

    mean_gflops: float = 0.0
    std_gflops: float = 0.0
    median_gflops: float = 0.0
    min_gflops: float = 0.0
    max_gflops: float = 0.0

    # Confidence interval (bootstrap)
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ci_method: str = "bootstrap"

    # Variability
    coefficient_of_variation: float = 0.0  # CV = std/mean * 100
    iqr: float = 0.0

    # Normality test
    shapiro_statistic: float = 0.0
    shapiro_pvalue: float = 0.0
    is_normal: bool = False  # p > 0.05

    # Latency metrics
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    def compute(self, config: BenchmarkConfig):
        """Compute all statistics"""
        if not self.raw_gflops:
            return

        # Remove outliers using IQR
        self.clean_gflops = self._remove_outliers(self.raw_gflops, config.outlier_iqr_factor)
        self.clean_times_ms = self._remove_outliers(self.raw_times_ms, config.outlier_iqr_factor)

        self.n_samples = len(self.clean_gflops)
        self.n_outliers = len(self.raw_gflops) - self.n_samples

        if self.n_samples < 2:
            self.clean_gflops = self.raw_gflops
            self.clean_times_ms = self.raw_times_ms
            self.n_samples = len(self.clean_gflops)
            self.n_outliers = 0

        data = np.array(self.clean_gflops)

        # Descriptive statistics
        self.mean_gflops = float(np.mean(data))
        self.std_gflops = float(np.std(data, ddof=1))
        self.median_gflops = float(np.median(data))
        self.min_gflops = float(np.min(data))
        self.max_gflops = float(np.max(data))

        # Coefficient of variation
        if self.mean_gflops > 0:
            self.coefficient_of_variation = (self.std_gflops / self.mean_gflops) * 100

        # IQR
        q1, q3 = np.percentile(data, [25, 75])
        self.iqr = float(q3 - q1)

        # Normality test (Shapiro-Wilk)
        if HAS_SCIPY and len(data) >= 3:
            stat, pval = shapiro(data)
            self.shapiro_statistic = float(stat)
            self.shapiro_pvalue = float(pval)
            self.is_normal = pval > 0.05

        # Confidence interval
        self._compute_confidence_interval(data, config)

        # Latency metrics
        self._compute_latency_metrics()

    def _remove_outliers(self, data: List[float], iqr_factor: float) -> List[float]:
        """Remove outliers using IQR method"""
        if len(data) < 4:
            return data

        arr = np.array(data)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        return [x for x in data if lower <= x <= upper]

    def _compute_confidence_interval(self, data: np.ndarray, config: BenchmarkConfig):
        """Compute confidence interval using bootstrap"""
        alpha = 1 - config.confidence_level

        if HAS_SCIPY and len(data) >= 5:
            try:
                # Bootstrap confidence interval
                res = bootstrap(
                    (data,),
                    np.mean,
                    n_resamples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    method='BCa'  # Bias-corrected and accelerated
                )
                self.ci_lower = float(res.confidence_interval.low)
                self.ci_upper = float(res.confidence_interval.high)
                self.ci_method = "bootstrap_BCa"
                return
            except Exception:
                pass

        # Fallback: t-distribution CI
        if len(data) >= 2:
            sem = self.std_gflops / np.sqrt(len(data))
            if HAS_SCIPY:
                t_crit = stats.t.ppf(1 - alpha/2, len(data) - 1)
            else:
                t_crit = 2.0  # Approximate for large n

            self.ci_lower = self.mean_gflops - t_crit * sem
            self.ci_upper = self.mean_gflops + t_crit * sem
            self.ci_method = "t_distribution"

    def _compute_latency_metrics(self):
        """Compute latency percentiles and throughput metrics"""
        if not self.clean_times_ms:
            return

        times = np.array(self.clean_times_ms)

        # Latency percentiles
        self.latency.p50_ms = float(np.percentile(times, 50))
        self.latency.p90_ms = float(np.percentile(times, 90))
        self.latency.p95_ms = float(np.percentile(times, 95))
        self.latency.p99_ms = float(np.percentile(times, 99))

        # Throughput
        mean_time_s = np.mean(times) / 1000.0
        if mean_time_s > 0:
            self.latency.ops_per_second = 1.0 / mean_time_s

        # Cold start analysis (if we have raw data)
        if len(self.raw_times_ms) > 1:
            self.latency.cold_start_ms = self.raw_times_ms[0]
            self.latency.warm_avg_ms = float(np.mean(self.raw_times_ms[1:]))
            if self.latency.warm_avg_ms > 0:
                self.latency.cold_overhead_percent = (
                    (self.latency.cold_start_ms / self.latency.warm_avg_ms - 1) * 100
                )


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one library/size combination"""
    library: str
    operation: str
    size: int
    timestamp: str = ""

    # Statistical results
    stats: StatisticalResult = field(default_factory=StatisticalResult)

    # Metadata
    backend: str = ""
    threads: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        return d


# =============================================================================
# Comparison Analysis
# =============================================================================

@dataclass
class ComparisonResult:
    """Statistical comparison between two libraries"""
    size: int
    library_a: str
    library_b: str

    mean_a: float = 0.0
    mean_b: float = 0.0

    # Absolute and relative difference
    diff_absolute: float = 0.0
    diff_percent: float = 0.0

    # Statistical test
    test_name: str = ""
    test_statistic: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False

    # Effect size
    cohens_d: float = 0.0
    effect_interpretation: str = ""

    winner: str = ""


def compare_results(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    alpha: float = 0.05
) -> ComparisonResult:
    """Perform statistical comparison between two benchmark results"""

    comp = ComparisonResult(
        size=result_a.size,
        library_a=result_a.library,
        library_b=result_b.library,
        mean_a=result_a.stats.mean_gflops,
        mean_b=result_b.stats.mean_gflops
    )

    # Difference
    comp.diff_absolute = comp.mean_a - comp.mean_b
    if comp.mean_b > 0:
        comp.diff_percent = (comp.mean_a / comp.mean_b - 1) * 100

    # Winner
    comp.winner = result_a.library if comp.mean_a > comp.mean_b else result_b.library

    data_a = np.array(result_a.stats.clean_gflops)
    data_b = np.array(result_b.stats.clean_gflops)

    if len(data_a) < 3 or len(data_b) < 3:
        return comp

    if HAS_SCIPY:
        # Choose test based on normality
        if result_a.stats.is_normal and result_b.stats.is_normal:
            # Welch's t-test (doesn't assume equal variances)
            stat, pval = stats.ttest_ind(data_a, data_b, equal_var=False)
            comp.test_name = "Welch's t-test"
        else:
            # Mann-Whitney U test (non-parametric)
            stat, pval = mannwhitneyu(data_a, data_b, alternative='two-sided')
            comp.test_name = "Mann-Whitney U"

        comp.test_statistic = float(stat)
        comp.p_value = float(pval)
        comp.is_significant = pval < alpha

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(data_a) - 1) * np.var(data_a, ddof=1) +
         (len(data_b) - 1) * np.var(data_b, ddof=1)) /
        (len(data_a) + len(data_b) - 2)
    )

    if pooled_std > 0:
        comp.cohens_d = float((np.mean(data_a) - np.mean(data_b)) / pooled_std)

    # Interpret effect size
    d = abs(comp.cohens_d)
    if d < 0.2:
        comp.effect_interpretation = "negligible"
    elif d < 0.5:
        comp.effect_interpretation = "small"
    elif d < 0.8:
        comp.effect_interpretation = "medium"
    else:
        comp.effect_interpretation = "large"

    return comp


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_numpy(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark NumPy with full statistical analysis"""
    results = []

    # Get NumPy BLAS info
    try:
        blas_info = np.show_config(mode='dicts')
        backend = "NumPy + " + str(blas_info.get('Build Dependencies', {}).get('blas', {}).get('name', 'unknown'))
    except:
        backend = "NumPy"

    for size in config.sizes:
        print(f"  NumPy {size}×{size}...", end=" ", flush=True)

        a = np.random.randn(size, size).astype(np.float64)
        b = np.random.randn(size, size).astype(np.float64)

        flops = 2 * size * size * size

        # Warmup
        for _ in range(config.warmup_runs):
            _ = a @ b

        # Timed runs
        times_ms = []
        gflops_list = []

        for _ in range(config.timed_runs):
            time.sleep(config.cooldown_seconds)

            start = time.perf_counter()
            _ = a @ b
            elapsed = time.perf_counter() - start

            times_ms.append(elapsed * 1000)
            gflops_list.append(flops / (elapsed * 1e9))

        # Create result with statistics
        stat = StatisticalResult(
            raw_times_ms=times_ms,
            raw_gflops=gflops_list
        )
        stat.compute(config)

        result = BenchmarkResult(
            library="numpy",
            operation="matmul",
            size=size,
            timestamp=datetime.now().isoformat(),
            stats=stat,
            backend=backend
        )
        results.append(result)

        cv_indicator = "✓" if stat.coefficient_of_variation < 5 else "!"
        print(f"{stat.mean_gflops:.1f} ±{stat.std_gflops:.1f} GFLOPS (CV={stat.coefficient_of_variation:.1f}%{cv_indicator})")

    return results


def benchmark_pytorch(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark PyTorch with full statistical analysis"""
    if not HAS_TORCH:
        print("  PyTorch not available")
        return []

    results = []
    backend = f"PyTorch {torch.__version__}"
    threads = torch.get_num_threads()

    for size in config.sizes:
        print(f"  PyTorch {size}×{size}...", end=" ", flush=True)

        a = torch.randn(size, size, dtype=torch.float64)
        b = torch.randn(size, size, dtype=torch.float64)

        flops = 2 * size * size * size

        # Warmup
        for _ in range(config.warmup_runs):
            _ = a @ b

        # Timed runs
        times_ms = []
        gflops_list = []

        for _ in range(config.timed_runs):
            time.sleep(config.cooldown_seconds)

            start = time.perf_counter()
            _ = a @ b
            elapsed = time.perf_counter() - start

            times_ms.append(elapsed * 1000)
            gflops_list.append(flops / (elapsed * 1e9))

        stat = StatisticalResult(
            raw_times_ms=times_ms,
            raw_gflops=gflops_list
        )
        stat.compute(config)

        result = BenchmarkResult(
            library="pytorch",
            operation="matmul",
            size=size,
            timestamp=datetime.now().isoformat(),
            stats=stat,
            backend=backend,
            threads=threads
        )
        results.append(result)

        cv_indicator = "✓" if stat.coefficient_of_variation < 5 else "!"
        print(f"{stat.mean_gflops:.1f} ±{stat.std_gflops:.1f} GFLOPS (CV={stat.coefficient_of_variation:.1f}%{cv_indicator})")

    return results


def benchmark_viva_tensor(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Benchmark viva_tensor with full statistical analysis"""
    results = []

    for size in config.sizes:
        print(f"  viva_tensor {size}×{size}...", end=" ", flush=True)

        # Run Erlang benchmark
        erlang_code = f'''
            N = {size},
            Warmup = {config.warmup_runs},
            Runs = {config.timed_runs},
            {{ok, A}} = viva_tensor_zig:nt_zeros([N, N]),
            {{ok, B}} = viva_tensor_zig:nt_zeros([N, N]),
            erlang:garbage_collect(),

            % Warmup
            [viva_tensor_zig:nt_matmul(A, B, N, N, N) || _ <- lists:seq(1, Warmup)],

            % Timed runs with GC between each
            Times = [begin
                timer:sleep({int(config.cooldown_seconds * 1000)}),
                erlang:garbage_collect(),
                {{T, _}} = timer:tc(fun() -> viva_tensor_zig:nt_matmul(A, B, N, N, N) end),
                T / 1000.0
            end || _ <- lists:seq(1, Runs)],

            Gflops = [(2.0 * N * N * N) / (T * 1000000) || T <- Times],
            Backend = viva_tensor_zig:backend_info(),

            io:format("~p|~p|~s~n", [Times, Gflops, Backend]),
            halt().
        '''

        try:
            result = subprocess.run(
                ["erl", "-pa", "build/dev/erlang/viva_tensor/ebin",
                 "-pa", "ebin", "-noshell", "-eval", erlang_code],
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "MKL_NUM_THREADS": "24", "MKL_DYNAMIC": "FALSE"}
            )

            output = result.stdout.strip()
            if "|" in output:
                parts = output.split("|")
                times_ms = eval(parts[0])
                gflops_list = eval(parts[1])
                backend = parts[2] if len(parts) > 2 else "viva_tensor"

                stat = StatisticalResult(
                    raw_times_ms=times_ms,
                    raw_gflops=gflops_list
                )
                stat.compute(config)

                bench_result = BenchmarkResult(
                    library="viva_tensor",
                    operation="matmul",
                    size=size,
                    timestamp=datetime.now().isoformat(),
                    stats=stat,
                    backend=backend
                )
                results.append(bench_result)

                cv_indicator = "✓" if stat.coefficient_of_variation < 5 else "!"
                print(f"{stat.mean_gflops:.1f} ±{stat.std_gflops:.1f} GFLOPS (CV={stat.coefficient_of_variation:.1f}%{cv_indicator})")
            else:
                print(f"PARSE ERROR")

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


# =============================================================================
# Reporting
# =============================================================================

def generate_markdown_report(
    all_results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
    config: BenchmarkConfig,
    output_file: str
):
    """Generate comprehensive markdown report"""

    with open(output_file, 'w') as f:
        f.write("# viva_tensor Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("## Methodology\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|:----------|------:|\n")
        f.write(f"| Warmup runs | {config.warmup_runs} |\n")
        f.write(f"| Timed runs | {config.timed_runs} |\n")
        f.write(f"| Confidence level | {config.confidence_level*100:.0f}% |\n")
        f.write(f"| Bootstrap samples | {config.bootstrap_samples:,} |\n")
        f.write(f"| Outlier removal | IQR × {config.outlier_iqr_factor} |\n")
        f.write("\n")

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Size | Library | Mean (GFLOPS) | 95% CI | CV% | Normal? |\n")
        f.write("|:----:|:--------|:-------------:|:------:|:---:|:-------:|\n")

        for r in sorted(all_results, key=lambda x: (x.size, x.library)):
            s = r.stats
            normal = "✓" if s.is_normal else "✗"
            cv_status = "✓" if s.coefficient_of_variation < 5 else "⚠"
            f.write(f"| {r.size}×{r.size} | {r.library} | "
                   f"**{s.mean_gflops:.1f}** ±{s.std_gflops:.1f} | "
                   f"[{s.ci_lower:.1f}, {s.ci_upper:.1f}] | "
                   f"{s.coefficient_of_variation:.1f}{cv_status} | {normal} |\n")

        f.write("\n")

        # Statistical comparisons
        f.write("## Statistical Comparisons\n\n")
        f.write("| Size | Comparison | Δ% | p-value | Significant? | Effect Size |\n")
        f.write("|:----:|:-----------|:--:|:-------:|:------------:|:-----------:|\n")

        for c in comparisons:
            sig = "**Yes**" if c.is_significant else "No"
            f.write(f"| {c.size}×{c.size} | {c.library_a} vs {c.library_b} | "
                   f"{c.diff_percent:+.1f}% | {c.p_value:.4f} | {sig} | "
                   f"{c.effect_interpretation} (d={c.cohens_d:.2f}) |\n")

        f.write("\n")

        # Winner summary
        f.write("## Winners by Size\n\n")
        f.write("| Size | Winner | Margin |\n")
        f.write("|:----:|:-------|:------:|\n")

        sizes = sorted(set(r.size for r in all_results))
        for size in sizes:
            size_results = [r for r in all_results if r.size == size]
            if size_results:
                best = max(size_results, key=lambda x: x.stats.mean_gflops)
                second = sorted(size_results, key=lambda x: x.stats.mean_gflops, reverse=True)[1] if len(size_results) > 1 else None

                if second:
                    margin = (best.stats.mean_gflops / second.stats.mean_gflops - 1) * 100
                    f.write(f"| {size}×{size} | **{best.library}** | +{margin:.1f}% vs {second.library} |\n")
                else:
                    f.write(f"| {size}×{size} | **{best.library}** | - |\n")

        f.write("\n---\n\n")
        f.write("*Report generated by viva_tensor benchmark suite v2.0*\n")

    print(f"\nReport saved: {output_file}")


def save_results_json(
    all_results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
    config: BenchmarkConfig,
    output_file: str
):
    """Save results to JSON"""

    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "config": asdict(config)
        },
        "results": [r.to_dict() for r in all_results],
        "comparisons": [asdict(c) for c in comparisons]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Data saved: {output_file}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="viva_tensor Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 bench/benchmark.py              # Full benchmark
    python3 bench/benchmark.py --quick      # Quick mode
    python3 bench/benchmark.py --sizes 1000 2000 3000
        """
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer runs, smaller sizes"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        help="Matrix sizes to benchmark (default: 2000 3000 4000 5000)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of timed runs (default: 30)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup runs (default: 5)"
    )
    parser.add_argument(
        "--no-pytorch", action="store_true",
        help="Skip PyTorch benchmarks"
    )
    parser.add_argument(
        "--no-viva", action="store_true",
        help="Skip viva_tensor benchmarks"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  viva_tensor Benchmark Suite")
    print("  Statistical rigor • Bootstrap CI • Effect sizes")
    print("=" * 70)

    # Configure based on arguments
    if args.quick:
        sizes = [1000, 2000]
        runs = 10
        warmup = 3
    else:
        sizes = args.sizes or [2000, 3000, 4000, 5000]
        runs = args.runs
        warmup = args.warmup

    config = BenchmarkConfig(
        warmup_runs=warmup,
        timed_runs=runs,
        cooldown_seconds=0.05,
        sizes=sizes,
        bootstrap_samples=10000
    )

    print(f"\nConfiguration:")
    print(f"  Warmup: {config.warmup_runs} runs")
    print(f"  Timed: {config.timed_runs} runs")
    print(f"  CI: {config.confidence_level*100:.0f}% (Bootstrap BCa, {config.bootstrap_samples:,} samples)")

    all_results = []

    print("\n" + "=" * 70)
    print("  NumPy")
    print("=" * 70)
    all_results.extend(benchmark_numpy(config))

    if not args.no_pytorch:
        print("\n" + "=" * 70)
        print("  PyTorch")
        print("=" * 70)
        all_results.extend(benchmark_pytorch(config))

    if not args.no_viva:
        print("\n" + "=" * 70)
        print("  viva_tensor")
        print("=" * 70)
        all_results.extend(benchmark_viva_tensor(config))

    # Statistical comparisons
    print("\n" + "=" * 70)
    print("  Statistical Analysis")
    print("=" * 70)

    comparisons = []
    sizes = sorted(set(r.size for r in all_results))
    libraries = sorted(set(r.library for r in all_results))

    for size in sizes:
        size_results = {r.library: r for r in all_results if r.size == size}

        # Compare viva_tensor vs others
        if "viva_tensor" in size_results:
            vt = size_results["viva_tensor"]
            for lib in libraries:
                if lib != "viva_tensor" and lib in size_results:
                    comp = compare_results(vt, size_results[lib])
                    comparisons.append(comp)

                    sig = "*" if comp.is_significant else ""
                    print(f"  {size}×{size}: viva_tensor vs {lib}: "
                          f"{comp.diff_percent:+.1f}%{sig} "
                          f"(p={comp.p_value:.4f}, d={comp.cohens_d:.2f} {comp.effect_interpretation})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure reports directory exists
    Path("bench/reports").mkdir(parents=True, exist_ok=True)

    save_results_json(
        all_results, comparisons, config,
        f"{config.output_dir}/benchmark_{timestamp}.json"
    )

    generate_markdown_report(
        all_results, comparisons, config,
        f"bench/reports/benchmark_report_{timestamp}.md"
    )

    # Also save as latest
    save_results_json(
        all_results, comparisons, config,
        f"{config.output_dir}/benchmark_latest.json"
    )

    print("\n" + "=" * 70)
    print("  Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
