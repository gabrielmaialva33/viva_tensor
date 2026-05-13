//// Observability for viva_tensor.
////
//// Structured logging, metrics collection, and statistical benchmarking
//// powered by viva_telemetry. Gives you full visibility into tensor
//// operations, backend selection, memory usage, and gradient computation.
////
//// ## Quick Start
////
//// ```gleam
//// import viva_tensor/telemetry
////
//// // Console logging at Info level + all metrics
//// telemetry.init()
////
//// // With JSON file output for production
//// telemetry.init_with_json("logs/viva_tensor.json")
////
//// // Full custom config
//// telemetry.init_with_config(telemetry.Config(
////   log_level: level.Debug,
////   json_path: option.Some("logs/tensor.json"),
//// ))
//// ```
////
//// ## Prometheus Metrics
////
//// ```gleam
//// let metrics_text = telemetry.prometheus()
//// // Returns all metrics in Prometheus exposition format
//// ```
////
//// ## Benchmarking
////
//// ```gleam
//// let result = telemetry.benchmark("matmul_100x100", fn() {
////   ops.matmul_auto(a, b)
//// })
//// telemetry.benchmark_print(result)
//// ```

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
import viva_telemetry/bench
import viva_telemetry/log
import viva_telemetry/log/level
import viva_telemetry/metrics

// --- Configuration -----------------------------------------------------------

/// Telemetry configuration.
///
/// log_level: Minimum severity for log output (default: Info)
/// json_path: Optional path for structured JSON log file
pub type Config {
  Config(log_level: level.Level, json_path: Option(String))
}

/// Default configuration: Info level, console only.
pub fn default_config() -> Config {
  Config(log_level: level.Info, json_path: None)
}

// --- Initialization ----------------------------------------------------------

/// Initialize telemetry with sensible defaults.
/// Console logging at Info level, all metrics enabled.
pub fn init() -> Nil {
  log.configure_console(level.Info)
  Nil
}

/// Initialize with console + JSON file logging.
/// JSON file gets all levels for post-mortem analysis.
pub fn init_with_json(path: String) -> Nil {
  log.configure_full(level.Info, path, level.Debug)
  Nil
}

/// Initialize with full custom configuration.
pub fn init_with_config(config: Config) -> Nil {
  case config.json_path {
    Some(path) -> log.configure_full(config.log_level, path, config.log_level)
    None -> log.configure_console(config.log_level)
  }
  Nil
}

// --- Metrics Definitions -----------------------------------------------------
// Pre-built metrics for tensor operations.
// ETS-backed, thread-safe, zero-config.

/// Counter for tensor operations by name.
pub fn op_counter(op: String) -> metrics.Counter {
  metrics.counter_with_labels("vt_ops_total", [#("op", op)])
}

/// Histogram for matmul latency in microseconds.
pub fn matmul_histogram() -> metrics.Histogram {
  metrics.histogram("vt_matmul_duration_us", [
    10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10_000.0, 50_000.0, 100_000.0,
  ])
}

/// Histogram for general operation latency in microseconds.
pub fn op_histogram() -> metrics.Histogram {
  metrics.histogram("vt_op_duration_us", [
    1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10_000.0,
  ])
}

/// Histogram for backward pass latency in microseconds.
pub fn backward_histogram() -> metrics.Histogram {
  metrics.histogram("vt_backward_duration_us", [
    100.0, 500.0, 1000.0, 5000.0, 10_000.0, 50_000.0, 100_000.0,
  ])
}

/// Counter for backend selections (accelerate, zig, erlang).
pub fn backend_counter(backend: String) -> metrics.Counter {
  metrics.counter_with_labels("vt_backend_total", [#("backend", backend)])
}

/// Gauge for BEAM process memory in bytes.
pub fn memory_gauge() -> metrics.Gauge {
  metrics.gauge("vt_memory_bytes")
}

/// Counter for quantization operations by format.
pub fn quant_counter(format: String) -> metrics.Counter {
  metrics.counter_with_labels("vt_quant_ops_total", [#("format", format)])
}

// --- Logging Helpers ---------------------------------------------------------

/// Log a tensor operation at debug level.
pub fn log_op(op: String, shape: List(Int)) -> Nil {
  log.debug("tensor op", [#("op", op), #("shape", format_shape(shape))])
}

/// Log matmul execution with performance details.
pub fn log_matmul(
  m: Int,
  k: Int,
  n: Int,
  backend: String,
  duration_us: Int,
) -> Nil {
  let flops = 2 * m * k * n
  let gflops = case duration_us > 0 {
    True ->
      format_float(int.to_float(flops) /. int.to_float(duration_us) /. 1000.0)
    False -> "inf"
  }
  log.info("matmul", [
    #(
      "dims",
      int.to_string(m)
        <> "x"
        <> int.to_string(k)
        <> " @ "
        <> int.to_string(k)
        <> "x"
        <> int.to_string(n),
    ),
    #("backend", backend),
    #("duration_us", int.to_string(duration_us)),
    #("gflops", gflops),
  ])
}

/// Log backend selection at debug level.
pub fn log_backend(backend: String, op: String) -> Nil {
  log.debug("backend selected", [#("backend", backend), #("op", op)])
}

/// Log backward pass completion.
pub fn log_backward(num_nodes: Int, duration_us: Int) -> Nil {
  log.info("backward", [
    #("nodes", int.to_string(num_nodes)),
    #("duration_us", int.to_string(duration_us)),
  ])
}

/// Log quantization operation.
pub fn log_quantize(format: String, ratio: Float) -> Nil {
  log.info("quantize", [
    #("format", format),
    #("ratio", format_float(ratio) <> "x"),
  ])
}

/// Log a tensor error.
pub fn log_error(op: String, reason: String) -> Nil {
  log.error("tensor error", [#("op", op), #("reason", reason)])
}

// --- Metrics Recording -------------------------------------------------------

/// Record a generic tensor operation with timing.
pub fn record_op(op: String, duration_us: Int) -> Nil {
  metrics.inc(op_counter(op))
  metrics.observe(op_histogram(), int.to_float(duration_us))
}

/// Record a matmul operation with full details.
pub fn record_matmul(
  m: Int,
  n: Int,
  k: Int,
  duration_us: Int,
  backend: String,
) -> Nil {
  metrics.inc(op_counter("matmul"))
  metrics.observe(matmul_histogram(), int.to_float(duration_us))
  metrics.inc(backend_counter(backend))
  log_matmul(m, k, n, backend, duration_us)
}

/// Record a dot product operation.
pub fn record_dot(size: Int, duration_us: Int, backend: String) -> Nil {
  metrics.inc(op_counter("dot"))
  metrics.observe(op_histogram(), int.to_float(duration_us))
  metrics.inc(backend_counter(backend))
  log.debug("dot", [
    #("size", int.to_string(size)),
    #("backend", backend),
    #("duration_us", int.to_string(duration_us)),
  ])
}

/// Record a backward pass.
pub fn record_backward(num_nodes: Int, duration_us: Int) -> Nil {
  metrics.observe(backward_histogram(), int.to_float(duration_us))
  log_backward(num_nodes, duration_us)
}

/// Record a quantization operation.
pub fn record_quantize(format: String, ratio: Float) -> Nil {
  metrics.inc(quant_counter(format))
  log_quantize(format, ratio)
}

// --- Memory ------------------------------------------------------------------

/// Report BEAM memory usage. Logs and records gauge.
pub fn report_memory() -> metrics.BeamMemory {
  let mem = metrics.beam_memory()
  metrics.set(memory_gauge(), int.to_float(mem.total))
  log.info("beam memory", [
    #("total_mb", int.to_string(mem.total / 1_048_576)),
    #("processes_mb", int.to_string(mem.processes / 1_048_576)),
    #("binary_mb", int.to_string(mem.binary / 1_048_576)),
    #("ets_mb", int.to_string(mem.ets / 1_048_576)),
  ])
  mem
}

// --- Prometheus ---------------------------------------------------------------

/// Export all collected metrics in Prometheus text format.
pub fn prometheus() -> String {
  metrics.to_prometheus()
}

// --- Benchmarking ------------------------------------------------------------
// Wraps viva_telemetry/bench for convenient benchmarking.

/// Run a benchmark with default config (100 warmup, 1000 iterations).
pub fn benchmark(name: String, f: fn() -> a) -> bench.BenchResult {
  bench.run(name, f)
}

/// Run a benchmark with custom iterations.
pub fn benchmark_with_config(
  name: String,
  f: fn() -> a,
  warmup: Int,
  iterations: Int,
) -> bench.BenchResult {
  bench.run_with_config(name, f, bench.config(warmup, iterations))
}

/// Run multiple benchmarks and return all results.
pub fn benchmark_all(
  benchmarks: List(#(String, fn() -> a)),
) -> List(bench.BenchResult) {
  bench.run_all(benchmarks, bench.default_config())
}

/// Compare two benchmark results. Returns speedup and significance.
pub fn benchmark_compare(
  baseline: bench.BenchResult,
  target: bench.BenchResult,
) -> bench.Comparison {
  bench.compare(baseline, target)
}

/// Print benchmark result to stdout.
pub fn benchmark_print(result: bench.BenchResult) -> Nil {
  bench.print(result)
}

/// Print comparison to stdout.
pub fn benchmark_print_comparison(comparison: bench.Comparison) -> Nil {
  bench.print_comparison(comparison)
}

/// Convert benchmark result to markdown.
pub fn benchmark_to_markdown(result: bench.BenchResult) -> String {
  bench.to_markdown(result)
}

/// Convert multiple results to a markdown comparison table.
pub fn benchmark_table(results: List(bench.BenchResult)) -> String {
  bench.to_markdown_table(results)
}

// --- Internal ----------------------------------------------------------------

fn format_shape(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
}

fn format_float(f: Float) -> String {
  let rounded = float.round(f *. 100.0)
  float.to_string(int.to_float(rounded) /. 100.0)
}
