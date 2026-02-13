//// TFLOPS - Tera Floating Point Operations Per Second
////
//// Multi-platform computational throughput measurement and auto-dispatch.
//// From Pure Erlang (~0.001 TFLOPS) to CUDA Sparse 2:4 (~660 TFLOPS).
////
//// The `Auto` backend automatically selects the fastest available compute:
//// GPU Sparse > GPU FP16 > GPU INT8 > GPU FP32 > CPU MKL > CPU SIMD > Erlang
////
//// ```gleam
//// import viva_tensor/tflops
////
//// // Auto-select fastest backend
//// let result = tflops.measure_matmul(tflops.Auto, 2048, 2048, 2048)
//// io.println(tflops.format_result(result))
////
//// // Benchmark all available backends
//// let backends = tflops.detect_backends()
//// let results = list.map(backends, fn(b) { tflops.measure_matmul(b, 1024, 1024, 1024) })
//// io.println(tflops.format_table(results))
//// ```

import gleam/float
import gleam/int
import gleam/list
import gleam/string
import viva_tensor/core/ffi

// =============================================================================
// TYPES
// =============================================================================

/// Compute backend — ordered from slowest to fastest
pub type Backend {
  /// Pure Erlang lists — ~0.001 TFLOPS (baseline, always available)
  PureErlang
  /// Zig SIMD NIF — ~1.5 TFLOPS (AVX2/SSE, portable)
  ZigSIMD
  /// Intel MKL BLAS — ~2.0 TFLOPS (multi-threaded SGEMM)
  MklBLAS
  /// CUDA FP32 cuBLAS — ~59 TFLOPS (RTX 4090 measured)
  CudaFP32
  /// CUDA FP16 Tensor Cores — ~172 TFLOPS (HMMA, RTX 4090 measured)
  CudaFP16
  /// CUDA INT8 IMMA Tensor Cores — ~330 TOPS
  CudaINT8
  /// CUDA 2:4 Sparse FP16 — ~660 TFLOPS (cuSPARSELt)
  CudaSparse
  /// Auto-select fastest available backend
  Auto
}

/// TFLOPS measurement result
pub type TflopsResult {
  TflopsResult(
    backend: Backend,
    matrix_size: Int,
    flops: Int,
    time_us: Int,
    tflops: Float,
    gflops: Float,
    efficiency: Float,
  )
}

// =============================================================================
// API
// =============================================================================

/// Detect the fastest available backend
pub fn best_backend() -> Backend {
  case ffi.sparse_available() {
    True -> CudaSparse
    False ->
      case ffi.ct16_available() {
        True -> CudaFP16
        False ->
          case ffi.ct_int8_available() {
            True -> CudaINT8
            False ->
              case cuda_fp32_available() {
                True -> CudaFP32
                False ->
                  case ffi.zig_is_loaded() {
                    True -> MklBLAS
                    False -> PureErlang
                  }
              }
          }
      }
  }
}

/// Measure single matmul TFLOPS for a backend
pub fn measure_matmul(backend: Backend, m: Int, n: Int, k: Int) -> TflopsResult {
  let actual = resolve_backend(backend)
  let flops = 2 * m * n * k
  let data_a = random_floats(m * k)
  let data_b = random_floats(k * n)

  let time_us = run_matmul(actual, data_a, data_b, m, n, k)
  let tflops = compute_tflops(flops, time_us)
  let peak = theoretical_peak(actual)
  let eff = case peak >. 0.0 {
    True -> tflops /. peak *. 100.0
    False -> 0.0
  }

  TflopsResult(
    backend: actual,
    matrix_size: m,
    flops: flops,
    time_us: time_us,
    tflops: tflops,
    gflops: tflops *. 1000.0,
    efficiency: eff,
  )
}

/// Measure averaged TFLOPS (warmup + iterations)
pub fn measure_matmul_averaged(
  backend: Backend,
  m: Int,
  n: Int,
  k: Int,
  iterations: Int,
) -> TflopsResult {
  let actual = resolve_backend(backend)
  let flops = 2 * m * n * k
  let data_a = random_floats(m * k)
  let data_b = random_floats(k * n)

  // Warmup: 2 iterations discarded
  let _ = run_matmul(actual, data_a, data_b, m, n, k)
  let _ = run_matmul(actual, data_a, data_b, m, n, k)

  // Measured iterations
  let times = measure_n(actual, data_a, data_b, m, n, k, iterations, [])
  let total_us = list.fold(times, 0, fn(acc, t) { acc + t })
  let avg_us = total_us / iterations

  let tflops = compute_tflops(flops, avg_us)
  let peak = theoretical_peak(actual)
  let eff = case peak >. 0.0 {
    True -> tflops /. peak *. 100.0
    False -> 0.0
  }

  TflopsResult(
    backend: actual,
    matrix_size: m,
    flops: flops,
    time_us: avg_us,
    tflops: tflops,
    gflops: tflops *. 1000.0,
    efficiency: eff,
  )
}

/// Detect all available backends (ordered slowest to fastest)
pub fn detect_backends() -> List(Backend) {
  let base = [PureErlang]

  let with_zig = case ffi.zig_is_loaded() {
    True -> list.append(base, [ZigSIMD, MklBLAS])
    False -> base
  }

  let with_cuda = case cuda_fp32_available() {
    True -> list.append(with_zig, [CudaFP32])
    False -> with_zig
  }

  let with_fp16 = case ffi.ct16_available() {
    True -> list.append(with_cuda, [CudaFP16])
    False -> with_cuda
  }

  let with_int8 = case ffi.ct_int8_available() {
    True -> list.append(with_fp16, [CudaINT8])
    False -> with_fp16
  }

  case ffi.sparse_available() {
    True -> list.append(with_int8, [CudaSparse])
    False -> with_int8
  }
}

/// Theoretical peak TFLOPS for a backend (RTX 4090 / i9-13900K)
pub fn theoretical_peak(backend: Backend) -> Float {
  case backend {
    PureErlang -> 0.001
    ZigSIMD -> 1.5
    MklBLAS -> 2.0
    CudaFP32 -> 82.6
    CudaFP16 -> 330.3
    CudaINT8 -> 660.0
    CudaSparse -> 660.6
    Auto -> theoretical_peak(best_backend())
  }
}

/// Backend name as string
pub fn backend_name(backend: Backend) -> String {
  case backend {
    PureErlang -> "Pure Erlang"
    ZigSIMD -> "Zig SIMD"
    MklBLAS -> "MKL BLAS"
    CudaFP32 -> "CUDA FP32"
    CudaFP16 -> "CUDA FP16"
    CudaINT8 -> "CUDA INT8"
    CudaSparse -> "Sparse 2:4"
    Auto -> "Auto"
  }
}

/// Format single result as a one-line string
pub fn format_result(result: TflopsResult) -> String {
  backend_name(result.backend)
  <> " "
  <> int.to_string(result.matrix_size)
  <> "x"
  <> int.to_string(result.matrix_size)
  <> ": "
  <> format_tflops(result.tflops)
  <> " TFLOPS ("
  <> format_float2(result.gflops)
  <> " GFLOPS, "
  <> format_float1(result.efficiency)
  <> "% eff, "
  <> format_time_ms(result.time_us)
  <> " ms)"
}

/// Format list of results as a table
pub fn format_table(results: List(TflopsResult)) -> String {
  let header =
    "  ┌──────────────────┬────────────┬──────────┬──────────┐\n"
    <> "  │ Backend          │ Time (ms)  │ TFLOPS   │ Eff %    │\n"
    <> "  ├──────────────────┼────────────┼──────────┼──────────┤"

  let rows =
    list.map(results, fn(r) {
      "  │ "
      <> pad_right(backend_name(r.backend), 16)
      <> " │ "
      <> pad_right(format_time_ms(r.time_us), 10)
      <> " │ "
      <> pad_right(format_tflops(r.tflops), 8)
      <> " │ "
      <> pad_right(format_float1(r.efficiency) <> "%", 8)
      <> " │"
    })

  let footer = "  └──────────────────┴────────────┴──────────┴──────────┘"

  string.join([header, ..list.append(rows, [footer])], "\n")
}

// =============================================================================
// INTERNAL - Backend resolution
// =============================================================================

fn resolve_backend(backend: Backend) -> Backend {
  case backend {
    Auto -> best_backend()
    other -> other
  }
}

// =============================================================================
// INTERNAL - Matmul dispatch
// =============================================================================

fn run_matmul(
  backend: Backend,
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  case backend {
    PureErlang -> run_pure_erlang(data_a, data_b, m, n, k)
    ZigSIMD -> run_zig_simd(data_a, data_b, m, n, k)
    MklBLAS -> run_mkl_blas(data_a, data_b, m, n, k)
    CudaFP32 -> run_cuda_fp32(data_a, data_b, m, n, k)
    CudaFP16 -> run_cuda_fp16(data_a, data_b, m, n, k)
    CudaINT8 -> run_cuda_int8(data_a, data_b, m, n, k)
    CudaSparse -> run_cuda_sparse(data_a, data_b, m, n, k)
    Auto -> run_matmul(best_backend(), data_a, data_b, m, n, k)
  }
}

fn run_pure_erlang(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  let a = ffi.list_to_array(data_a)
  let b = ffi.list_to_array(data_b)
  let start = ffi.now_microseconds()
  let _ = ffi.array_matmul(a, b, m, n, k)
  let end = ffi.now_microseconds()
  end - start
}

fn run_zig_simd(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  let start = ffi.now_microseconds()
  let _ = ffi.zig_matmul(data_a, data_b, m, n, k)
  let end = ffi.now_microseconds()
  end - start
}

fn run_mkl_blas(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  // NativeTensorRef path uses MKL cblas_dgemm under the hood
  case ffi.nt_from_list(data_a, [m, k]) {
    Ok(a_nt) ->
      case ffi.nt_from_list(data_b, [k, n]) {
        Ok(b_nt) -> {
          let start = ffi.now_microseconds()
          let _ = ffi.nt_matmul(a_nt, b_nt, m, n, k)
          let end = ffi.now_microseconds()
          end - start
        }
        Error(_) -> 0
      }
    Error(_) -> 0
  }
}

fn run_cuda_fp32(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  case ffi.ct_from_list(data_a, [m, k]) {
    Ok(a_ct) ->
      case ffi.ct_from_list(data_b, [k, n]) {
        Ok(b_ct) -> {
          let start = ffi.now_microseconds()
          let _ = ffi.ct_matmul(a_ct, b_ct, m, n, k)
          let end = ffi.now_microseconds()
          end - start
        }
        Error(_) -> 0
      }
    Error(_) -> 0
  }
}

fn run_cuda_fp16(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  case ffi.ct16_from_list(data_a, [m, k]) {
    Ok(a_ct16) ->
      case ffi.ct16_from_list(data_b, [k, n]) {
        Ok(b_ct16) -> {
          let start = ffi.now_microseconds()
          let _ = ffi.ct16_matmul(a_ct16, b_ct16, m, n, k)
          let end = ffi.now_microseconds()
          end - start
        }
        Error(_) -> 0
      }
    Error(_) -> 0
  }
}

fn run_cuda_int8(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  case ffi.ct_int8_from_list(data_a, [m, k]) {
    Ok(a_int8) ->
      case ffi.ct_int8_from_list(data_b, [k, n]) {
        Ok(b_int8) -> {
          let start = ffi.now_microseconds()
          let _ = ffi.ct_int8_matmul(a_int8, b_int8, m, n, k)
          let end = ffi.now_microseconds()
          end - start
        }
        Error(_) -> 0
      }
    Error(_) -> 0
  }
}

fn run_cuda_sparse(
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Int {
  case ffi.ct16_from_list(data_a, [m, k]) {
    Ok(a_ct16) ->
      case ffi.sparse_from_ct16(a_ct16) {
        Ok(a_sparse) ->
          case ffi.ct16_from_list(data_b, [k, n]) {
            Ok(b_ct16) -> {
              let start = ffi.now_microseconds()
              let _ = ffi.sparse_matmul(a_sparse, b_ct16, m, n, k)
              let end = ffi.now_microseconds()
              end - start
            }
            Error(_) -> 0
          }
        Error(_) -> 0
      }
    Error(_) -> 0
  }
}

fn measure_n(
  backend: Backend,
  data_a: List(Float),
  data_b: List(Float),
  m: Int,
  n: Int,
  k: Int,
  remaining: Int,
  acc: List(Int),
) -> List(Int) {
  case remaining <= 0 {
    True -> list.reverse(acc)
    False -> {
      let time_us = run_matmul(backend, data_a, data_b, m, n, k)
      measure_n(backend, data_a, data_b, m, n, k, remaining - 1, [
        time_us,
        ..acc
      ])
    }
  }
}

fn cuda_fp32_available() -> Bool {
  case ffi.ct_from_list([1.0, 0.0, 0.0, 1.0], [2, 2]) {
    Ok(_) -> True
    Error(_) -> False
  }
}

// =============================================================================
// INTERNAL - Math
// =============================================================================

fn compute_tflops(flops: Int, time_us: Int) -> Float {
  case time_us > 0 {
    True -> int.to_float(flops) /. { int.to_float(time_us) *. 1_000_000.0 }
    False -> 0.0
  }
}

fn random_floats(n: Int) -> List(Float) {
  random_floats_acc(n, [])
}

fn random_floats_acc(remaining: Int, acc: List(Float)) -> List(Float) {
  case remaining <= 0 {
    True -> acc
    False -> random_floats_acc(remaining - 1, [ffi.random_uniform(), ..acc])
  }
}

// =============================================================================
// INTERNAL - Formatting
// =============================================================================

fn format_tflops(t: Float) -> String {
  case t <. 0.001 {
    True -> "<0.001"
    False -> format_float3(t)
  }
}

fn format_time_ms(time_us: Int) -> String {
  let ms = int.to_float(time_us) /. 1000.0
  format_float2(ms)
}

fn format_float1(f: Float) -> String {
  let rounded = float.round(f *. 10.0)
  let whole = rounded / 10
  let frac = abs_int(rounded - whole * 10)
  int.to_string(whole) <> "." <> int.to_string(frac)
}

fn format_float2(f: Float) -> String {
  let rounded = float.round(f *. 100.0)
  let whole = rounded / 100
  let frac = abs_int(rounded - whole * 100)
  int.to_string(whole) <> "." <> pad_left_zero(int.to_string(frac), 2)
}

fn format_float3(f: Float) -> String {
  let rounded = float.round(f *. 1000.0)
  let whole = rounded / 1000
  let frac = abs_int(rounded - whole * 1000)
  int.to_string(whole) <> "." <> pad_left_zero(int.to_string(frac), 3)
}

fn abs_int(n: Int) -> Int {
  case n < 0 {
    True -> 0 - n
    False -> n
  }
}

fn pad_left_zero(s: String, width: Int) -> String {
  let len = string_length(s)
  case len < width {
    True -> repeat_string("0", width - len) <> s
    False -> s
  }
}

fn pad_right(s: String, width: Int) -> String {
  let len = string_length(s)
  case len < width {
    True -> s <> repeat_string(" ", width - len)
    False -> s
  }
}

fn repeat_string(s: String, n: Int) -> String {
  case n <= 0 {
    True -> ""
    False -> s <> repeat_string(s, n - 1)
  }
}

@external(erlang, "string", "length")
fn string_length(s: String) -> Int
