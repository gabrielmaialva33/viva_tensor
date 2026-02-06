//// Tensor operations - where the actual computation happens.
////
//// Design philosophy: correctness first, then optimize the hot paths.
//// The naive O(n³) matmul is fine for small matrices. For large ones,
//// we delegate to BLAS via NIF (Apple Accelerate on macOS, OpenBLAS elsewhere).
////
//// Broadcasting follows NumPy semantics exactly because (1) it's well-documented,
//// (2) everyone expects it, and (3) I tried inventing my own rules once. Never again.
////
//// Historical note: Hadamard product (element-wise mul) is named after
//// Jacques Hadamard, who used it in his 1893 theorem on determinants.
//// Most people just call it "element-wise multiplication" now.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ffi
import viva_tensor/core/tensor.{type Tensor}
import viva_tensor/telemetry

// --- Element-wise Ops -------------------------------------------------------

/// Map a function over all elements. The workhorse of tensor ops.
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  let data = tensor.to_list(t)
  let new_data = list.map(data, f)
  case tensor.new(new_data, tensor.shape(t)) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Like map but you also get the index. Useful for positional encoding.
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  let data = tensor.to_list(t)
  let new_data = list.index_map(data, fn(x, i) { f(x, i) })
  case tensor.new(new_data, tensor.shape(t)) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Element-wise add. Shapes must match (use add_broadcast for different shapes).
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x +. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// a - b, element-wise.
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x -. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Hadamard product (element-wise multiply). Not to be confused with matmul!
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x *. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// a / b element-wise. Watch out for division by zero (you get Infinity).
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x /. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Scalar multiplication. The s stands for scalar, not "slow".
pub fn scale(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x *. s })
}

/// Add constant to all elements. Useful for bias terms.
pub fn add_scalar(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x +. s })
}

/// Negate all elements (multiply by -1).
pub fn negate(t: Tensor) -> Tensor {
  scale(t, -1.0)
}

/// Element-wise absolute value
pub fn abs(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.abs(x) })
}

/// Element-wise square
pub fn square(t: Tensor) -> Tensor {
  map(t, fn(x) { x *. x })
}

/// Element-wise square root
pub fn sqrt(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.sqrt(x) })
}

/// Element-wise exponential
pub fn exp(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.exp(x) })
}

/// Element-wise natural log
pub fn log(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.log(x) })
}

/// Element-wise power
pub fn pow(t: Tensor, exponent: Float) -> Tensor {
  map(t, fn(x) { ffi.pow(x, exponent) })
}

// --- Reductions -------------------------------------------------------------
// Collapse tensor to scalar. The "aggregation" in "aggregate functions".
// Numerically: we use Kahan summation for sum() to reduce float error.
// Just kidding, we use naive summation. Sue me. The error is O(n*ε) where
// ε ≈ 2.2e-16 for f64. For typical ML workloads, this is noise.
//
// TODO: add axis parameter for sum_axis, mean_axis, etc.
// TODO: consider pairwise summation for better numerical stability

/// Sum all elements. O(n) time, O(1) space.
pub fn sum(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  list.fold(data, 0.0, fn(acc, x) { acc +. x })
}

/// Product of all elements
pub fn product(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  list.fold(data, 1.0, fn(acc, x) { acc *. x })
}

/// Mean of all elements
pub fn mean(t: Tensor) -> Float {
  let s = sum(t)
  let n = int.to_float(tensor.size(t))
  case n >. 0.0 {
    True -> s /. n
    False -> 0.0
  }
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
  }
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.min(acc, x) })
  }
}

/// Index of max element. Returns 0 for empty tensors (debatable choice).
pub fn argmax(t: Tensor) -> Int {
  let data = tensor.to_list(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      // Track best index, best value, and current position
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x >. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

/// Index of minimum element
pub fn argmin(t: Tensor) -> Int {
  let data = tensor.to_list(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x <. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

/// Variance of all elements. Single pass: computes sum and sum_sq simultaneously.
pub fn variance(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  let n = int.to_float(tensor.size(t))
  case n >. 0.0 {
    True -> {
      let #(total, total_sq) =
        list.fold(data, #(0.0, 0.0), fn(acc, x) {
          let #(s, sq) = acc
          #(s +. x, sq +. x *. x)
        })
      let m = total /. n
      // Var = E[X²] - (E[X])²
      total_sq /. n -. m *. m
    }
    False -> 0.0
  }
}

/// Standard deviation
pub fn std(t: Tensor) -> Float {
  ffi.sqrt(variance(t))
}

/// L2 norm
pub fn norm(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  ffi.sqrt(sum_sq)
}

/// Normalize to unit length
pub fn normalize(t: Tensor) -> Tensor {
  let n = norm(t)
  case n >. 0.0001 {
    True -> scale(t, 1.0 /. n)
    False -> t
  }
}

// --- Matrix Operations ------------------------------------------------------
// The crown jewels. matmul is O(n³) for naive implementation.
// Strassen (1969) gets O(n^2.807), but the constant factor is huge.
// Current record is O(n^2.373) by Alman-Williams (2020). Impractical.
// For real speedups, we use BLAS which is cache-optimized and SIMD'd.

/// Dot product: Σ(a_i * b_i). The foundation of all neural networks, really.
/// Uses array-based access for large vectors, list-based for small ones.
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case
    tensor.rank(a) == 1
    && tensor.rank(b) == 1
    && tensor.size(a) == tensor.size(b)
  {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      // list.map2 + fold is fine here - dot is O(n) either way
      let products = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
/// Uses array-based O(1) access for O(m*n) total instead of O(m*n^2).
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(mat), tensor.shape(vec) {
    [m, n], [vec_n] if n == vec_n -> {
      let mat_arr = ffi.list_to_array(tensor.to_list(mat))
      let vec_arr = ffi.list_to_array(tensor.to_list(vec))
      let result_data =
        list.range(0, m - 1)
        |> list.map(fn(row_idx) {
          let start = row_idx * n
          list.range(0, n - 1)
          |> list.fold(0.0, fn(acc, k) {
            acc +. ffi.array_get(mat_arr, start + k) *. ffi.array_get(vec_arr, k)
          })
        })
      tensor.new(result_data, [m])
    }
    [_m, n], [vec_n] -> Error(error.ShapeMismatch([n], [vec_n]))
    _, _ -> Error(error.DimensionError("Expected matrix and vector"))
  }
}

/// Matrix multiplication. The operation that launched a thousand GPUs.
/// C[i,j] = Σ_k A[i,k] * B[k,j]
///
/// Uses array-based O(1) access for O(mnp) total.
/// For serious work, use matmul_auto() which delegates to BLAS.
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, n], [n2, p] if n == n2 -> {
      let a_arr = ffi.list_to_array(tensor.to_list(a))
      let b_arr = ffi.list_to_array(tensor.to_list(b))
      let result_data =
        list.range(0, m - 1)
        |> list.flat_map(fn(i) {
          let row_start = i * n
          list.range(0, p - 1)
          |> list.map(fn(j) {
            list.range(0, n - 1)
            |> list.fold(0.0, fn(acc, k) {
              acc +. ffi.array_get(a_arr, row_start + k) *. ffi.array_get(b_arr, k * p + j)
            })
          })
        })
      tensor.new(result_data, [m, p])
    }
    [_m, n], [n2, _p] -> Error(error.ShapeMismatch([n, -1], [n2, -1]))
    _, _ -> Error(error.DimensionError("Expected two matrices"))
  }
}

// --- Optimized Ops (Erlang :array) ------------------------------------------
// These use O(1) array access instead of list traversal.
// Makes a real difference for vectors > 1000 elements.

/// Faster dot using Erlang arrays. ~2-3x speedup for large vectors.
pub fn dot_fast(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case
    tensor.rank(a) == 1
    && tensor.rank(b) == 1
    && tensor.size(a) == tensor.size(b)
  {
    True -> {
      let a_arr = ffi.list_to_array(tensor.to_list(a))
      let b_arr = ffi.list_to_array(tensor.to_list(b))
      Ok(ffi.array_dot(a_arr, b_arr))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Faster matmul using arrays. Use this for matrices > 50x50.
pub fn matmul_fast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, k], [k2, n] if k == k2 -> {
      let a_arr = ffi.list_to_array(tensor.to_list(a))
      let b_arr = ffi.list_to_array(tensor.to_list(b))
      let result_arr = ffi.array_matmul(a_arr, b_arr, m, n, k)
      let result_list = ffi.array_to_list(result_arr)
      tensor.new(result_list, [m, n])
    }
    [_m, k], [k2, _n] -> Error(error.ShapeMismatch([k, -1], [k2, -1]))
    _, _ -> Error(error.DimensionError("Expected two matrices"))
  }
}

// --- Auto-selecting Ops (best available backend) ----------------------------
// Backend priority: Apple Accelerate > Zig SIMD > Pure Erlang
//
// Why this order?
// - Accelerate: Apple's BLAS, uses AMX on M-series chips. Insanely fast.
// - Zig SIMD: Cross-platform, we wrote it, ~10-50x speedup over naive.
// - Pure Erlang: Always works, no native deps, reasonable for small tensors.
//
// Benchmarks (1000x1000 matmul on M1 Mac):
// - Pure Gleam: ~45 seconds (yes, seconds)
// - Erlang arrays: ~15 seconds
// - Zig SIMD: ~800ms
// - Accelerate: ~30ms
//
// The 1500x difference is why we bother with NIFs.

/// Smart dot - delegates to fastest available backend at runtime.
/// Instrumented: records latency and backend selection metrics.
pub fn dot_auto(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case
    tensor.rank(a) == 1
    && tensor.rank(b) == 1
    && tensor.size(a) == tensor.size(b)
  {
    True -> {
      let t0 = ffi.now_microseconds()
      // Try Native path first (zero list conversion)
      let #(result, backend) = case
        tensor.native_ref(a),
        tensor.native_ref(b)
      {
        Ok(ref_a), Ok(ref_b) ->
          case ffi.nt_dot(ref_a, ref_b) {
            Ok(r) -> #(Ok(r), "native")
            Error(_) -> dot_list_fallback(a, b)
          }
        _, _ -> dot_list_fallback(a, b)
      }
      telemetry.record_dot(tensor.size(a), ffi.now_microseconds() - t0, backend)
      result
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

fn dot_list_fallback(
  a: Tensor,
  b: Tensor,
) -> #(Result(Float, TensorError), String) {
  let a_list = tensor.to_list(a)
  let b_list = tensor.to_list(b)
  case ffi.is_nif_loaded() {
    True ->
      case ffi.nif_dot(a_list, b_list) {
        Ok(r) -> #(Ok(r), "accelerate")
        Error(_) -> dot_with_zig_fallback(a_list, b_list, a, b)
      }
    False -> dot_with_zig_fallback(a_list, b_list, a, b)
  }
}

fn dot_with_zig_fallback(
  a_list: List(Float),
  b_list: List(Float),
  a: Tensor,
  b: Tensor,
) -> #(Result(Float, TensorError), String) {
  // Try Zig SIMD (cross-platform fast)
  case ffi.zig_is_loaded() {
    True -> {
      case ffi.zig_dot(a_list, b_list) {
        Ok(r) -> #(Ok(r), "zig")
        Error(_) -> #(dot_fast(a, b), "erlang")
      }
    }
    False -> #(dot_fast(a, b), "erlang")
  }
}

/// Smart matmul. Can be 1400x faster than pure Gleam for 500x500 matrices.
/// Instrumented: logs backend selection and records latency metrics.
pub fn matmul_auto(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, k], [k2, n] if k == k2 -> {
      let t0 = ffi.now_microseconds()
      // Try Native path first (zero list conversion)
      let #(result, backend) = case
        tensor.native_ref(a),
        tensor.native_ref(b)
      {
        Ok(ref_a), Ok(ref_b) ->
          case ffi.nt_matmul(ref_a, ref_b, m, n, k) {
            Ok(ref_c) -> #(
              Ok(tensor.from_native_ref(ref_c, [m, n])),
              "native",
            )
            Error(_) -> matmul_list_fallback(a, b, m, n, k)
          }
        _, _ -> matmul_list_fallback(a, b, m, n, k)
      }
      telemetry.record_matmul(m, n, k, ffi.now_microseconds() - t0, backend)
      result
    }
    [_m, k], [k2, _n] -> Error(error.ShapeMismatch([k, -1], [k2, -1]))
    _, _ -> Error(error.DimensionError("Expected two matrices"))
  }
}

fn matmul_list_fallback(
  a: Tensor,
  b: Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> #(Result(Tensor, TensorError), String) {
  let a_list = tensor.to_list(a)
  let b_list = tensor.to_list(b)
  case ffi.is_nif_loaded() {
    True ->
      case ffi.nif_matmul(a_list, b_list, m, n, k) {
        Ok(result_list) -> #(tensor.new(result_list, [m, n]), "accelerate")
        Error(_) -> matmul_with_zig_fallback(a_list, b_list, m, n, k, a, b)
      }
    False -> matmul_with_zig_fallback(a_list, b_list, m, n, k, a, b)
  }
}

fn matmul_with_zig_fallback(
  a_list: List(Float),
  b_list: List(Float),
  m: Int,
  n: Int,
  k: Int,
  a: Tensor,
  b: Tensor,
) -> #(Result(Tensor, TensorError), String) {
  // Try Zig SIMD (cross-platform, often faster for small-medium matrices)
  case ffi.zig_is_loaded() {
    True -> {
      case ffi.zig_matmul(a_list, b_list, m, n, k) {
        Ok(result_list) -> #(tensor.new(result_list, [m, n]), "zig")
        Error(_) -> #(matmul_fast(a, b), "erlang")
      }
    }
    False -> #(matmul_fast(a, b), "erlang")
  }
}

/// Get current backend info string
/// Shows best available backend for tensor operations
pub fn backend_info() -> String {
  case ffi.zig_is_loaded() {
    True -> ffi.zig_backend_info() <> " + NIF Resources"
    False ->
      case ffi.is_nif_loaded() {
        True -> ffi.nif_backend_info()
        False -> "Pure Erlang (O(1) array access)"
      }
  }
}

/// Get detailed status of all backends
pub fn all_backends_info() -> String {
  let zig_status = case ffi.zig_is_loaded() {
    True -> "✓ " <> ffi.zig_backend_info()
    False -> "✗ Not available"
  }
  let accel_status = case ffi.is_nif_loaded() {
    True -> "✓ " <> ffi.nif_backend_info()
    False -> "✗ Not available"
  }
  "Zig SIMD: "
  <> zig_status
  <> "\nApple Accelerate: "
  <> accel_status
  <> "\nPure Erlang: ✓ Always available"
}

/// Auto-selecting sum reduction.
/// Priority: Zig SIMD > Apple Accelerate > Pure Erlang.
/// Instrumented: records operation latency.
pub fn sum_auto(t: Tensor) -> Float {
  let t0 = ffi.now_microseconds()
  // Try Native path first
  let result = case tensor.native_ref(t) {
    Ok(ref) ->
      case ffi.nt_sum(ref) {
        Ok(r) -> r
        Error(_) -> sum_list_fallback(t)
      }
    Error(_) -> sum_list_fallback(t)
  }
  telemetry.record_op("sum", ffi.now_microseconds() - t0)
  result
}

fn sum_list_fallback(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  case ffi.zig_is_loaded() {
    True ->
      case ffi.zig_sum(data) {
        Ok(r) -> r
        Error(_) -> sum_with_accel_fallback(data)
      }
    False -> sum_with_accel_fallback(data)
  }
}

fn sum_with_accel_fallback(data: List(Float)) -> Float {
  case ffi.is_nif_loaded() {
    True -> {
      case ffi.nif_sum(data) {
        Ok(result) -> result
        Error(_) -> list.fold(data, 0.0, fn(acc, x) { acc +. x })
      }
    }
    False -> list.fold(data, 0.0, fn(acc, x) { acc +. x })
  }
}

/// Auto-selecting element-wise add. Delegates to Zig SIMD or Accelerate.
pub fn add_auto(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True ->
      case tensor.native_ref(a), tensor.native_ref(b) {
        Ok(ref_a), Ok(ref_b) ->
          case ffi.nt_add(ref_a, ref_b) {
            Ok(ref_c) -> Ok(tensor.from_native_ref(ref_c, tensor.shape(a)))
            Error(_) -> add_list_fallback(a, b)
          }
        _, _ -> add_list_fallback(a, b)
      }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

fn add_list_fallback(
  a: Tensor,
  b: Tensor,
) -> Result(Tensor, TensorError) {
  let a_data = tensor.to_list(a)
  let b_data = tensor.to_list(b)
  let result_data = case ffi.zig_is_loaded() {
    True ->
      case ffi.zig_add(a_data, b_data) {
        Ok(r) -> r
        Error(_) -> list.map2(a_data, b_data, fn(x, y) { x +. y })
      }
    False -> list.map2(a_data, b_data, fn(x, y) { x +. y })
  }
  tensor.new(result_data, tensor.shape(a))
}

/// Auto-selecting element-wise subtract.
pub fn sub_auto(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True ->
      case tensor.native_ref(a), tensor.native_ref(b) {
        Ok(ref_a), Ok(ref_b) ->
          case ffi.nt_sub(ref_a, ref_b) {
            Ok(ref_c) -> Ok(tensor.from_native_ref(ref_c, tensor.shape(a)))
            Error(_) -> sub_list_fallback(a, b)
          }
        _, _ -> sub_list_fallback(a, b)
      }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

fn sub_list_fallback(
  a: Tensor,
  b: Tensor,
) -> Result(Tensor, TensorError) {
  let a_data = tensor.to_list(a)
  let b_data = tensor.to_list(b)
  let result_data = case ffi.zig_is_loaded() {
    True ->
      case ffi.zig_scale(b_data, -1.0) {
        Ok(neg_b) ->
          case ffi.zig_add(a_data, neg_b) {
            Ok(r) -> r
            Error(_) -> list.map2(a_data, b_data, fn(x, y) { x -. y })
          }
        Error(_) -> list.map2(a_data, b_data, fn(x, y) { x -. y })
      }
    False -> list.map2(a_data, b_data, fn(x, y) { x -. y })
  }
  tensor.new(result_data, tensor.shape(a))
}

/// Auto-selecting element-wise multiply. Delegates to Zig SIMD when available.
pub fn mul_auto(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True ->
      case tensor.native_ref(a), tensor.native_ref(b) {
        Ok(ref_a), Ok(ref_b) ->
          case ffi.nt_mul(ref_a, ref_b) {
            Ok(ref_c) -> Ok(tensor.from_native_ref(ref_c, tensor.shape(a)))
            Error(_) -> mul_list_fallback(a, b)
          }
        _, _ -> mul_list_fallback(a, b)
      }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

fn mul_list_fallback(
  a: Tensor,
  b: Tensor,
) -> Result(Tensor, TensorError) {
  let a_data = tensor.to_list(a)
  let b_data = tensor.to_list(b)
  let result_data = case ffi.zig_is_loaded() {
    True ->
      case ffi.zig_mul(a_data, b_data) {
        Ok(r) -> r
        Error(_) -> list.map2(a_data, b_data, fn(x, y) { x *. y })
      }
    False -> list.map2(a_data, b_data, fn(x, y) { x *. y })
  }
  tensor.new(result_data, tensor.shape(a))
}

/// Auto-selecting scalar multiplication. Delegates to Zig SIMD or Accelerate.
pub fn scale_auto(t: Tensor, s: Float) -> Tensor {
  // Try Native path first
  case tensor.native_ref(t) {
    Ok(ref) ->
      case ffi.nt_scale(ref, s) {
        Ok(ref_c) -> tensor.from_native_ref(ref_c, tensor.shape(t))
        Error(_) -> scale_list_fallback(t, s)
      }
    Error(_) -> scale_list_fallback(t, s)
  }
}

fn scale_list_fallback(t: Tensor, s: Float) -> Tensor {
  let data = tensor.to_list(t)
  let result_data = case ffi.zig_is_loaded() {
    True ->
      case ffi.zig_scale(data, s) {
        Ok(r) -> r
        Error(_) -> scale_with_accel_fallback(data, s)
      }
    False -> scale_with_accel_fallback(data, s)
  }
  case tensor.new(result_data, tensor.shape(t)) {
    Ok(result) -> result
    Error(_) -> t
  }
}

fn scale_with_accel_fallback(data: List(Float), s: Float) -> List(Float) {
  case ffi.is_nif_loaded() {
    True -> {
      case ffi.nif_scale(data, s) {
        Ok(result) -> result
        Error(_) -> list.map(data, fn(x) { x *. s })
      }
    }
    False -> list.map(data, fn(x) { x *. s })
  }
}

/// Matrix transpose. Uses array-based O(1) access for O(m*n) total.
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(t) {
    [m, n] -> {
      let arr = ffi.list_to_array(tensor.to_list(t))
      let result_data =
        list.range(0, n - 1)
        |> list.flat_map(fn(j) {
          list.range(0, m - 1)
          |> list.map(fn(i) { ffi.array_get(arr, i * n + j) })
        })
      tensor.new(result_data, [n, m])
    }
    _ -> Error(error.DimensionError("Transpose requires 2D tensor"))
  }
}

/// Outer product: [m] @ [n] -> [m, n]
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.rank(a) == 1 && tensor.rank(b) == 1 {
    True -> {
      let m = tensor.size(a)
      let n = tensor.size(b)
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let result_data =
        list.flat_map(a_data, fn(ai) { list.map(b_data, fn(bj) { ai *. bj }) })
      tensor.new(result_data, [m, n])
    }
    False -> Error(error.DimensionError("Outer product requires two vectors"))
  }
}

// --- Activations & Utils ----------------------------------------------------

/// Clamp to [min, max]. Useful for gradient clipping.
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}

/// ReLU activation
pub fn relu(t: Tensor) -> Tensor {
  map(t, fn(x) { float.max(x, 0.0) })
}

/// Sigmoid activation
pub fn sigmoid(t: Tensor) -> Tensor {
  map(t, fn(x) { 1.0 /. { 1.0 +. ffi.exp(0.0 -. x) } })
}

/// Tanh activation
pub fn tanh(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.tanh(x) })
}

/// Softmax: exp(x_i) / Σexp(x_j). Converts logits to probabilities.
///
/// The "subtract max" trick prevents overflow. Without it:
///   softmax([1000, 1001, 1002]) = [exp(1000)/..., ...] = [Inf/Inf, Inf/Inf, Inf/Inf] = NaN
/// With it:
///   softmax([1000, 1001, 1002]) = softmax([0, 1, 2]) = [0.09, 0.24, 0.67] ✓
///
/// Mathematically equivalent because softmax(x) = softmax(x - c) for any c.
///
/// TODO: add axis parameter, this only works on 1D vectors right now
pub fn softmax(t: Tensor) -> Tensor {
  let data = tensor.to_list(t)
  let max_val = max(t)
  let shifted = list.map(data, fn(x) { ffi.exp(x -. max_val) })
  let sum_exp = list.fold(shifted, 0.0, fn(acc, x) { acc +. x })
  let result = list.map(shifted, fn(x) { x /. sum_exp })
  case tensor.new(result, tensor.shape(t)) {
    Ok(r) -> r
    Error(_) -> t
  }
}

// --- Broadcasting -----------------------------------------------------------
// The rules (from NumPy, we copy them verbatim):
// 1. Align shapes from the right: [3,4] and [4] become [3,4] and [1,4]
// 2. At each position, dims must be equal OR one of them must be 1
// 3. The "1" dimension gets stretched to match the other
//
// Example: [2,3,4] + [3,1] works!
//   [2,3,4]     [2,3,4]
//     [3,1]  →  [1,3,1]  →  [2,3,4] (broadcasted)
//
// This is genuinely clever. Whoever designed it at APL in the 1960s was a genius.

/// Check if shapes can broadcast together.
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  let #(longer, shorter) = case list.length(a) >= list.length(b) {
    True -> #(a, b)
    False -> #(b, a)
  }

  let diff = list.length(longer) - list.length(shorter)
  let padded = list.append(list.repeat(1, diff), shorter)

  list.zip(longer, padded)
  |> list.all(fn(pair) {
    let #(dim_a, dim_b) = pair
    dim_a == dim_b || dim_a == 1 || dim_b == 1
  })
}

/// Compute broadcast shape
pub fn broadcast_shape(
  a: List(Int),
  b: List(Int),
) -> Result(List(Int), TensorError) {
  case can_broadcast(a, b) {
    False -> Error(error.BroadcastError(a, b))
    True -> {
      let max_rank = int.max(list.length(a), list.length(b))
      let diff_a = max_rank - list.length(a)
      let diff_b = max_rank - list.length(b)
      let padded_a = list.append(list.repeat(1, diff_a), a)
      let padded_b = list.append(list.repeat(1, diff_b), b)

      let result_shape =
        list.zip(padded_a, padded_b)
        |> list.map(fn(pair) {
          let #(dim_a, dim_b) = pair
          int.max(dim_a, dim_b)
        })

      Ok(result_shape)
    }
  }
}

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(
    tensor.shape(a),
    tensor.shape(b),
  ))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  add(a_bc, b_bc)
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(
    tensor.shape(a),
    tensor.shape(b),
  ))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  mul(a_bc, b_bc)
}

/// Broadcast tensor to target shape
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  let src_shape = tensor.shape(t)
  case can_broadcast(src_shape, target_shape) {
    False -> Error(error.BroadcastError(src_shape, target_shape))
    True -> {
      case src_shape == target_shape {
        True -> Ok(t)
        False -> {
          let data = broadcast_data(t, target_shape)
          tensor.new(data, target_shape)
        }
      }
    }
  }
}

fn broadcast_data(t: Tensor, target_shape: List(Int)) -> List(Float) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_shape = tensor.shape(t)
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  let data = tensor.to_list(t)

  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  list.range(0, target_size - 1)
  |> list.map(fn(flat_idx) {
    let target_indices = flat_to_multi(flat_idx, target_shape)

    let src_indices =
      list.zip(target_indices, padded_shape)
      |> list.map(fn(pair) {
        let #(idx, dim) = pair
        case dim == 1 {
          True -> 0
          False -> idx
        }
      })
      |> list.drop(diff)

    let src_flat = multi_to_flat(src_indices, src_shape)
    case list_at_float(data, src_flat) {
      Ok(v) -> v
      Error(_) -> 0.0
    }
  })
}

fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(indices, _) =
    list.fold(reversed, #([], flat), fn(acc, dim) {
      let #(idxs, remaining) = acc
      let idx = remaining % dim
      let next = remaining / dim
      #([idx, ..idxs], next)
    })
  indices
}

fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)
  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
}

fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}
