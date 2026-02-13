//// FFI - Foreign Function Interface to Erlang
////
//// The escape hatch from pure functional bliss into the world of
//// mutable arrays and hardware-specific optimizations.
////
//// Why we need this:
//// 1. Erlang lists are O(n) for random access. That's death for matrix ops.
//// 2. Erlang's :array gives us O(1) access (technically O(log32 n), close enough).
//// 3. Native NIFs unlock SIMD, BLAS, and GPU backends.
////
//// ## Performance Hierarchy (fastest to slowest)
////
//// 1. Zig SIMD NIF: Hand-tuned SIMD for the hot paths. 10-100x vs pure Gleam.
//// 2. Apple Accelerate NIF: cblas_dgemm on macOS. Ridiculously optimized.
//// 3. Erlang :array: O(1) access, pure Erlang. 10-50x vs lists for matmul.
//// 4. Pure Gleam lists: Beautiful, correct, slow. Fine for small tensors.
////
//// ## Architecture
////
//// We have three acceleration backends that we auto-select from:
//// - viva_tensor_zig: Portable SIMD via Zig. Works everywhere Zig compiles.
//// - viva_tensor_nif: Apple Accelerate on macOS (cblas, vDSP).
//// - viva_tensor_ffi: Pure Erlang fallback. Always works, just slower.
////
//// The ops module auto-selects the best available backend at runtime.

// --- Types ---

/// Erlang :array - the key to O(1) tensor element access.
///
/// Under the hood, it's a tree of 10-element tuples. Technically O(log32 n)
/// but that's effectively O(1) for any reasonable tensor size.
///
/// A 1M element lookup: log32(1_000_000) = ~4 tree traversals.
/// For lists: 500,000 average. That's the difference between
/// matmul taking 1 second vs 2 hours.
///
/// The array is immutable (functional updates create new nodes),
/// but we mostly just read from it during tensor operations.
pub type ErlangArray

/// Native tensor resource - opaque reference to contiguous C memory.
/// Created by NIF constructors, ops return new refs.
/// Erlang GC frees native memory automatically via destructor.
pub type NativeTensorRef

// --- Array Operations ---
//
// These wrap Erlang :array operations for O(1) element access.
// Use these instead of list indexing in any hot path.

/// Convert list to Erlang array for O(1) access.
///
/// O(n) to build, but subsequent access is O(1).
/// Worth it for any tensor you'll index more than once.
pub fn list_to_array(lst: List(Float)) -> ErlangArray {
  list_to_array_ffi(lst)
}

/// Get element from array at index - O(1).
///
/// Contrast with list indexing: O(n).
/// For a 1000-element matmul (1000 iterations, each indexing both inputs),
/// that's 2M list traversals vs 2K array lookups. Huge difference.
pub fn array_get(arr: ErlangArray, index: Int) -> Float {
  array_get_ffi(arr, index)
}

/// Get array size - O(1).
pub fn array_size(arr: ErlangArray) -> Int {
  array_size_ffi(arr)
}

/// Convert array back to list - O(n).
///
/// Use this for final output or when you need list operations.
/// Try to stay in array-land as long as possible for hot paths.
pub fn array_to_list(arr: ErlangArray) -> List(Float) {
  array_to_list_ffi(arr)
}

// --- Optimized Array Math ---
//
// Implemented in Erlang for O(1) element access.
// These are the fallback when NIFs aren't available.
// Still 10-50x faster than naive list-based implementations.

/// Dot product using Erlang arrays.
///
/// Performance: ~10-50x faster than list-based for large vectors.
/// The speedup comes entirely from O(1) vs O(n) element access.
pub fn array_dot(a: ErlangArray, b: ErlangArray) -> Float {
  array_dot_ffi(a, b)
}

/// Matrix multiplication using Erlang arrays.
///
/// C[m,n] = A[m,k] @ B[k,n]
///
/// Naive O(mnk) algorithm but with O(1) element access.
/// For 100x100 matrices: ~50x faster than list-based.
///
/// For serious work, use the Zig SIMD or Accelerate NIF backends.
/// This is the reliable fallback that works everywhere.
pub fn array_matmul(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray {
  array_matmul_ffi(a, b, m, n, k)
}

/// Sum all elements - O(n).
pub fn array_sum(arr: ErlangArray) -> Float {
  array_sum_ffi(arr)
}

/// Scale all elements by scalar - O(n).
pub fn array_scale(arr: ErlangArray, scalar: Float) -> ErlangArray {
  array_scale_ffi(arr, scalar)
}

// --- Apple Accelerate NIF ---
//
// macOS ships with the Accelerate framework, which includes:
// - BLAS: cblas_dgemm for matrix multiply. Hand-tuned for Apple silicon.
// - vDSP: Vectorized signal processing. Fast reductions and element-wise ops.
//
// On an M1 Max, cblas_dgemm achieves ~3 TFLOPS FP64.
// Our naive Erlang implementation: ~0.001 TFLOPS. That's 3000x.
//
// The catch: only works on macOS. Falls back gracefully elsewhere.

/// Check if the Apple Accelerate NIF is loaded.
///
/// Returns True on macOS with the NIF built, False elsewhere.
/// Use this to decide whether to use nif_* functions or fall back.
pub fn is_nif_loaded() -> Bool {
  nif_is_loaded_ffi()
}

/// Get backend info string for debugging.
///
/// Returns something like "Apple Accelerate (cblas_dgemm, vDSP)" on macOS.
pub fn nif_backend_info() -> String {
  nif_backend_info_ffi()
}

/// NIF-accelerated matrix multiplication via cblas_dgemm.
///
/// This is where the magic happens on macOS. Apple has spent years
/// optimizing BLAS for their chips. We just call their code.
///
/// Falls back to pure Erlang if NIF not available.
pub fn nif_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  nif_matmul_ffi(a, b, m, n, k)
}

/// NIF-accelerated dot product via vDSP.
pub fn nif_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  nif_dot_ffi(a, b)
}

/// NIF-accelerated sum via vDSP.
pub fn nif_sum(data: List(Float)) -> Result(Float, String) {
  nif_sum_ffi(data)
}

/// NIF-accelerated scale via vDSP.
pub fn nif_scale(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String) {
  nif_scale_ffi(data, scalar)
}

// --- Math Functions ---
//
// Thin wrappers around Erlang :math module.
// These are fine - no need to NIF them, they're already C under the hood.

/// Square root - wraps :math.sqrt/1
pub fn sqrt(x: Float) -> Float {
  sqrt_ffi(x)
}

/// Natural logarithm - wraps :math.log/1
///
/// Undefined for x <= 0. Erlang will return -inf for 0, NaN for negative.
/// Caller's responsibility to check input.
pub fn log(x: Float) -> Float {
  log_ffi(x)
}

/// Exponential e^x - wraps :math.exp/1
///
/// Watch for overflow: exp(710) = inf in Float64.
/// For softmax, subtract max first: exp(x - max(x)).
pub fn exp(x: Float) -> Float {
  exp_ffi(x)
}

/// Cosine - wraps :math.cos/1
pub fn cos(x: Float) -> Float {
  cos_ffi(x)
}

/// Sine - wraps :math.sin/1
pub fn sin(x: Float) -> Float {
  sin_ffi(x)
}

/// Tangent - wraps :math.tan/1
pub fn tan(x: Float) -> Float {
  tan_ffi(x)
}

/// Hyperbolic tangent - wraps :math.tanh/1
///
/// Range: (-1, 1). Saturates for |x| > ~20.
/// Used in some activation functions, though ReLU dominates now.
pub fn tanh(x: Float) -> Float {
  tanh_ffi(x)
}

/// Power x^y - wraps :math.pow/2
pub fn pow(x: Float, y: Float) -> Float {
  pow_ffi(x, y)
}

/// Absolute value.
///
/// Implemented in pure Gleam because :math.abs/1 doesn't exist
/// and erlang:abs/1 is polymorphic (returns same type as input).
pub fn abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

// --- Random ---
//
// Erlang's :rand module. Uses a PRNG per process.
// Fine for shuffling, initialization. Not cryptographically secure.

/// Uniform random float in [0, 1).
///
/// Uses Erlang's per-process PRNG (Xoroshiro116+ by default).
/// Not suitable for cryptography, but fine for ML initialization.
///
/// For reproducible results, seed with :rand.seed(Algorithm, Seed).
pub fn random_uniform() -> Float {
  random_uniform_ffi()
}

// --- Constants ---
//
// Because typing 3.14159... every time is error-prone.

/// Pi to 20 decimal places. More than Float64 can represent anyway.
pub const pi: Float = 3.14159265358979323846

/// Euler's number e to 20 decimal places.
pub const e: Float = 2.71828182845904523536

// --- FFI Bindings ---
//
// The actual Erlang external function declarations.
// Keep these private - expose through the typed wrappers above.

@external(erlang, "viva_tensor_ffi", "list_to_array")
fn list_to_array_ffi(lst: List(Float)) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_get")
fn array_get_ffi(arr: ErlangArray, index: Int) -> Float

@external(erlang, "viva_tensor_ffi", "array_size")
fn array_size_ffi(arr: ErlangArray) -> Int

@external(erlang, "viva_tensor_ffi", "array_to_list")
fn array_to_list_ffi(arr: ErlangArray) -> List(Float)

@external(erlang, "viva_tensor_ffi", "array_dot")
fn array_dot_ffi(a: ErlangArray, b: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_matmul")
fn array_matmul_ffi(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_sum")
fn array_sum_ffi(arr: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_scale")
fn array_scale_ffi(arr: ErlangArray, scalar: Float) -> ErlangArray

@external(erlang, "math", "sqrt")
fn sqrt_ffi(x: Float) -> Float

@external(erlang, "math", "log")
fn log_ffi(x: Float) -> Float

@external(erlang, "math", "exp")
fn exp_ffi(x: Float) -> Float

@external(erlang, "math", "cos")
fn cos_ffi(x: Float) -> Float

@external(erlang, "math", "sin")
fn sin_ffi(x: Float) -> Float

@external(erlang, "math", "tan")
fn tan_ffi(x: Float) -> Float

@external(erlang, "math", "tanh")
fn tanh_ffi(x: Float) -> Float

@external(erlang, "math", "pow")
fn pow_ffi(x: Float, y: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_uniform_ffi() -> Float

// Apple Accelerate NIF bindings
@external(erlang, "viva_tensor_nif", "is_nif_loaded")
fn nif_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_nif", "backend_info")
fn nif_backend_info_ffi() -> String

@external(erlang, "viva_tensor_nif", "matmul")
fn nif_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)

@external(erlang, "viva_tensor_nif", "dot")
fn nif_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "sum")
fn nif_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "scale")
fn nif_scale_ffi(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String)

// --- Timing ---
//
// For benchmarking. Microsecond precision is enough for tensor ops.

/// Get current time in microseconds.
///
/// Use for benchmarking: before/after difference gives wall-clock time.
/// For production profiling, use Erlang's :fprof or :eprof instead.
pub fn now_microseconds() -> Int {
  now_microseconds_ffi()
}

@external(erlang, "viva_tensor_ffi", "now_microseconds")
fn now_microseconds_ffi() -> Int

// --- Zig SIMD NIF ---
//
// Our portable high-performance backend. Zig compiles to native code
// with explicit SIMD support. No dependency on platform-specific libraries.
//
// Why Zig?
// 1. Cross-compiles everywhere (Linux, macOS, Windows, ARM, x86)
// 2. Explicit SIMD vectors: @Vector(8, f64) gives us AVX on x86, NEON on ARM
// 3. No runtime, no GC - just fast native code
// 4. Plays nice with Erlang NIFs
//
// Performance: 10-100x vs pure Erlang for vectorizable ops.
// Not quite Apple Accelerate levels, but works everywhere.

/// Check if Zig SIMD NIF is loaded.
pub fn zig_is_loaded() -> Bool {
  zig_is_loaded_ffi()
}

/// Get Zig backend info for debugging.
///
/// Returns SIMD capability info: "Zig SIMD (AVX2)" or "Zig SIMD (NEON)" etc.
pub fn zig_backend_info() -> String {
  zig_backend_info_ffi()
}

/// Zig SIMD dot product.
///
/// Uses 4-way or 8-way SIMD depending on platform.
/// Unrolled loop with accumulator to maximize throughput.
pub fn zig_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  zig_dot_ffi(a, b)
}

/// Zig SIMD sum reduction.
pub fn zig_sum(data: List(Float)) -> Result(Float, String) {
  zig_sum_ffi(data)
}

/// Zig SIMD scale (multiply all elements by scalar).
pub fn zig_scale(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String) {
  zig_scale_ffi(data, scalar)
}

/// Zig SIMD element-wise add.
pub fn zig_add(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  zig_add_ffi(a, b)
}

/// Zig SIMD element-wise multiply.
pub fn zig_mul(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  zig_mul_ffi(a, b)
}

/// Zig SIMD matrix multiplication.
///
/// Tiled implementation with SIMD inner loops.
/// Not quite BLAS-level but respectable: ~10-50 GFLOPS depending on platform.
pub fn zig_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  zig_matmul_ffi(a, b, m, n, k)
}

// --- NIF Resource API (zero-copy native tensors) ---
//
// These operate on NativeTensorRef - opaque handles to contiguous C arrays.
// No list<->array conversion per operation. Data stays in native memory.
// This is the fast path: Gleam → Erlang NIF → C → Zig SIMD → C → Erlang NIF → Gleam
// vs the old path: Gleam → List → C array alloc+copy → Zig SIMD → C → List alloc → Gleam

/// Create native tensor of zeros
pub fn nt_zeros(shape: List(Int)) -> Result(NativeTensorRef, String) {
  nt_zeros_ffi(shape)
}

/// Create native tensor of ones
pub fn nt_ones(shape: List(Int)) -> Result(NativeTensorRef, String) {
  nt_ones_ffi(shape)
}

/// Create native tensor filled with value
pub fn nt_fill(
  shape: List(Int),
  value: Float,
) -> Result(NativeTensorRef, String) {
  nt_fill_ffi(shape, value)
}

/// Create native tensor from list data + shape
pub fn nt_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(NativeTensorRef, String) {
  nt_from_list_ffi(data, shape)
}

/// Extract data as list (one-time conversion at boundaries)
pub fn nt_to_list(ref: NativeTensorRef) -> Result(List(Float), String) {
  nt_to_list_ffi(ref)
}

/// Get shape from native tensor
pub fn nt_shape(ref: NativeTensorRef) -> Result(List(Int), String) {
  nt_shape_ffi(ref)
}

/// Get total element count
pub fn nt_size(ref: NativeTensorRef) -> Result(Int, String) {
  nt_size_ffi(ref)
}

/// Native add: ref + ref → ref (zero copy)
pub fn nt_add(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String) {
  nt_add_ffi(a, b)
}

/// Native sub: ref - ref → ref
pub fn nt_sub(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String) {
  nt_sub_ffi(a, b)
}

/// Native element-wise mul: ref * ref → ref
pub fn nt_mul(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String) {
  nt_mul_ffi(a, b)
}

/// Native scale: ref * scalar → ref
pub fn nt_scale(
  a: NativeTensorRef,
  scalar: Float,
) -> Result(NativeTensorRef, String) {
  nt_scale_ffi(a, scalar)
}

/// Native negate: -ref → ref
pub fn nt_negate(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_negate_ffi(a)
}

/// Native dot product: ref · ref → scalar
pub fn nt_dot(a: NativeTensorRef, b: NativeTensorRef) -> Result(Float, String) {
  nt_dot_ffi(a, b)
}

/// Native sum reduction → scalar
pub fn nt_sum(a: NativeTensorRef) -> Result(Float, String) {
  nt_sum_ffi(a)
}

/// Native max → scalar
pub fn nt_max(a: NativeTensorRef) -> Result(Float, String) {
  nt_max_ffi(a)
}

/// Native min → scalar
pub fn nt_min(a: NativeTensorRef) -> Result(Float, String) {
  nt_min_ffi(a)
}

/// Native matmul: [m,k] @ [k,n] → [m,n] in native memory
pub fn nt_matmul(
  a: NativeTensorRef,
  b: NativeTensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(NativeTensorRef, String) {
  nt_matmul_ffi(a, b, m, n, k)
}

/// Native transpose: [m,n] → [n,m] contiguous copy
pub fn nt_transpose(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_transpose_ffi(a)
}

/// Native ReLU activation
pub fn nt_relu(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_relu_ffi(a)
}

/// Native sigmoid activation
pub fn nt_sigmoid(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_sigmoid_ffi(a)
}

/// Native exp
pub fn nt_exp(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_exp_ffi(a)
}

/// Native log
pub fn nt_log(a: NativeTensorRef) -> Result(NativeTensorRef, String) {
  nt_log_ffi(a)
}

// --- In-Place Mutation (Zero Allocation) ---
// "Quebrar a imutabilidade dentro do Zig para economizar RAM"
// These modify the tensor IN PLACE. The caller must understand
// that the original ref now points to mutated data.
// Use with care - this breaks functional purity for performance.

/// In-place add: a += b. Returns ok. MUTATES a.
pub fn nt_add_mut(a: NativeTensorRef, b: NativeTensorRef) -> Result(Nil, String) {
  nt_add_mut_ffi(a, b)
}

/// In-place scale: a *= scalar. Returns ok. MUTATES a.
pub fn nt_scale_mut(a: NativeTensorRef, scalar: Float) -> Result(Nil, String) {
  nt_scale_mut_ffi(a, scalar)
}

/// In-place negate: a = -a. Returns ok. MUTATES a.
pub fn nt_negate_mut(a: NativeTensorRef) -> Result(Nil, String) {
  nt_negate_mut_ffi(a)
}

/// In-place ReLU: a = max(0, a). Returns ok. MUTATES a.
pub fn nt_relu_mut(a: NativeTensorRef) -> Result(Nil, String) {
  nt_relu_mut_ffi(a)
}

// --- Retro / Fused Kernels ---

/// Saturn Blend: result = texture + (shade - bias)
/// VDP1-inspired lighting with pure SIMD addition.
pub fn nt_saturn_blend(
  texture: NativeTensorRef,
  shade: NativeTensorRef,
  bias: Float,
) -> Result(NativeTensorRef, String) {
  nt_saturn_blend_ffi(texture, shade, bias)
}

/// Fused MatMul + Bias + ReLU: C = max(0, A@B + bias)
/// Single pass, saves 2 full tensor traversals.
pub fn nt_fused_linear_relu(
  a: NativeTensorRef,
  b: NativeTensorRef,
  bias: NativeTensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(NativeTensorRef, String) {
  nt_fused_linear_relu_ffi(a, b, bias, m, n, k)
}

/// Resonance Multiply: LNS element-wise multiply.
/// result[i] = sign * exp(log|a[i]| + log|b[i]|)
/// Multiplication via addition in log domain — better precision for chains.
pub fn nt_resonance_mul(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String) {
  nt_resonance_mul_ffi(a, b)
}

/// Resonance Power: LNS element-wise power.
/// result[i] = sign(x) * |x|^exponent via exp(exponent * log|x|)
/// Power = multiply in log domain. Sign preserved for bipolar states.
pub fn nt_resonance_power(
  data: NativeTensorRef,
  exponent: Float,
) -> Result(NativeTensorRef, String) {
  nt_resonance_power_ffi(data, exponent)
}

@external(erlang, "viva_tensor_zig", "nt_zeros")
fn nt_zeros_ffi(shape: List(Int)) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_ones")
fn nt_ones_ffi(shape: List(Int)) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_fill")
fn nt_fill_ffi(
  shape: List(Int),
  value: Float,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_from_list")
fn nt_from_list_ffi(
  data: List(Float),
  shape: List(Int),
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_to_list")
fn nt_to_list_ffi(ref: NativeTensorRef) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "nt_shape")
fn nt_shape_ffi(ref: NativeTensorRef) -> Result(List(Int), String)

@external(erlang, "viva_tensor_zig", "nt_size")
fn nt_size_ffi(ref: NativeTensorRef) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "nt_add")
fn nt_add_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_sub")
fn nt_sub_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_mul")
fn nt_mul_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_scale")
fn nt_scale_ffi(
  a: NativeTensorRef,
  scalar: Float,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_negate")
fn nt_negate_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_dot")
fn nt_dot_ffi(a: NativeTensorRef, b: NativeTensorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "nt_sum")
fn nt_sum_ffi(a: NativeTensorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "nt_max")
fn nt_max_ffi(a: NativeTensorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "nt_min")
fn nt_min_ffi(a: NativeTensorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "nt_matmul")
fn nt_matmul_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_transpose")
fn nt_transpose_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_relu")
fn nt_relu_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_sigmoid")
fn nt_sigmoid_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_exp")
fn nt_exp_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_log")
fn nt_log_ffi(a: NativeTensorRef) -> Result(NativeTensorRef, String)

// In-place mutation FFI
@external(erlang, "viva_tensor_zig", "nt_add_mut")
fn nt_add_mut_ffi(a: NativeTensorRef, b: NativeTensorRef) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "nt_scale_mut")
fn nt_scale_mut_ffi(a: NativeTensorRef, scalar: Float) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "nt_negate_mut")
fn nt_negate_mut_ffi(a: NativeTensorRef) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "nt_relu_mut")
fn nt_relu_mut_ffi(a: NativeTensorRef) -> Result(Nil, String)

// Retro / fused kernel FFI
@external(erlang, "viva_tensor_zig", "nt_saturn_blend")
fn nt_saturn_blend_ffi(
  texture: NativeTensorRef,
  shade: NativeTensorRef,
  bias: Float,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_fused_linear_relu")
fn nt_fused_linear_relu_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
  bias: NativeTensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(NativeTensorRef, String)

// Resonance kernel FFI (Log-Number System)
@external(erlang, "viva_tensor_zig", "nt_resonance_mul")
fn nt_resonance_mul_ffi(
  a: NativeTensorRef,
  b: NativeTensorRef,
) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "nt_resonance_power")
fn nt_resonance_power_ffi(
  data: NativeTensorRef,
  exponent: Float,
) -> Result(NativeTensorRef, String)

// Zig NIF FFI bindings
@external(erlang, "viva_tensor_zig", "is_loaded")
fn zig_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_zig", "backend_info")
fn zig_backend_info_ffi() -> String

@external(erlang, "viva_tensor_zig", "simd_dot")
fn zig_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_sum")
fn zig_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_scale")
fn zig_scale_ffi(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_add")
fn zig_add_ffi(a: List(Float), b: List(Float)) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_mul")
fn zig_mul_ffi(a: List(Float), b: List(Float)) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_matmul")
fn zig_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)

// =============================================================================
// LNS (True Log-Number System) - f32 via IADD
// "Multiplicação como soma de inteiros" - 8x throughput vs FMA
//
// IEEE-754 trick: bits(A*B) ≈ bits(A) + bits(B) - bias
// where bias = 0x3F800000 (1.0f)
//
// This trades precision for throughput:
// - Fast (~11% max error): 16 IADD ops/cycle vs 2 FMA ops/cycle
// - Corrected (~2% max error): Mitchell's algorithm adds correction term
//
// Use when precision tolerance > 2% is acceptable.
// Great for: embeddings, attention scores, normalization
// =============================================================================

/// LNS tensor reference - f32 storage for IADD-based multiplication
pub type LnsTensorRef

/// Convert f64 NativeTensor to f32 LNS tensor
pub fn lns_from_f64(ref: NativeTensorRef) -> Result(LnsTensorRef, String) {
  lns_from_f64_ffi(ref)
}

/// Convert LNS tensor back to f64 NativeTensor
pub fn lns_to_f64(ref: LnsTensorRef) -> Result(NativeTensorRef, String) {
  lns_to_f64_ffi(ref)
}

/// Fast LNS multiply via IADD (~11% max error, 8x throughput)
pub fn lns_mul(a: LnsTensorRef, b: LnsTensorRef) -> Result(LnsTensorRef, String) {
  lns_mul_ffi(a, b)
}

/// Mitchell's corrected LNS multiply (~2% max error)
pub fn lns_mul_corrected(
  a: LnsTensorRef,
  b: LnsTensorRef,
) -> Result(LnsTensorRef, String) {
  lns_mul_corrected_ffi(a, b)
}

/// LNS division via ISUB
pub fn lns_div(a: LnsTensorRef, b: LnsTensorRef) -> Result(LnsTensorRef, String) {
  lns_div_ffi(a, b)
}

/// LNS sqrt via bit shift
pub fn lns_sqrt(a: LnsTensorRef) -> Result(LnsTensorRef, String) {
  lns_sqrt_ffi(a)
}

/// Fast inverse sqrt (Quake III trick)
pub fn lns_rsqrt(a: LnsTensorRef) -> Result(LnsTensorRef, String) {
  lns_rsqrt_ffi(a)
}

// LNS FFI bindings
@external(erlang, "viva_tensor_zig", "lns_from_f64")
fn lns_from_f64_ffi(ref: NativeTensorRef) -> Result(LnsTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_to_f64")
fn lns_to_f64_ffi(ref: LnsTensorRef) -> Result(NativeTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_mul")
fn lns_mul_ffi(a: LnsTensorRef, b: LnsTensorRef) -> Result(LnsTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_mul_corrected")
fn lns_mul_corrected_ffi(
  a: LnsTensorRef,
  b: LnsTensorRef,
) -> Result(LnsTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_div")
fn lns_div_ffi(a: LnsTensorRef, b: LnsTensorRef) -> Result(LnsTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_sqrt")
fn lns_sqrt_ffi(a: LnsTensorRef) -> Result(LnsTensorRef, String)

@external(erlang, "viva_tensor_zig", "lns_rsqrt")
fn lns_rsqrt_ffi(a: LnsTensorRef) -> Result(LnsTensorRef, String)

// =============================================================================
// HORDE - SoA Physics Engine
// "10K entidades sem GC" - Structure of Arrays for cache-efficient physics
//
// Layout: positions[N*dims], velocities[N*dims] (contiguous SoA)
// Operations are SIMD FMA across all entities at once.
//
// Use for: particle systems, boids, agent simulations, physics
// =============================================================================

/// Horde reference - SoA entity collection
pub type HordeRef

/// Create new Horde with entity count and dimensionality (1, 2, or 3)
pub fn horde_create(entity_count: Int, dims: Int) -> Result(HordeRef, String) {
  horde_create_ffi(entity_count, dims)
}

/// Set all positions from flat list [x0, y0, x1, y1, ...] for 2D
pub fn horde_set_positions(
  horde: HordeRef,
  data: List(Float),
) -> Result(Nil, String) {
  horde_set_positions_ffi(horde, data)
}

/// Set all velocities from flat list
pub fn horde_set_velocities(
  horde: HordeRef,
  data: List(Float),
) -> Result(Nil, String) {
  horde_set_velocities_ffi(horde, data)
}

/// Euler integration step: positions += velocities * dt (FMA)
pub fn horde_integrate(horde: HordeRef, dt: Float) -> Result(Nil, String) {
  horde_integrate_ffi(horde, dt)
}

/// Apply velocity damping: velocities *= friction
pub fn horde_dampen(horde: HordeRef, friction: Float) -> Result(Nil, String) {
  horde_dampen_ffi(horde, friction)
}

/// Toroidal wrap: positions mod max_bound
pub fn horde_wrap(horde: HordeRef, max_bound: Float) -> Result(Nil, String) {
  horde_wrap_ffi(horde, max_bound)
}

/// Get current positions as flat list
pub fn horde_get_positions(horde: HordeRef) -> Result(List(Float), String) {
  horde_get_positions_ffi(horde)
}

/// Get current velocities as flat list
pub fn horde_get_velocities(horde: HordeRef) -> Result(List(Float), String) {
  horde_get_velocities_ffi(horde)
}

/// Get entity count
pub fn horde_count(horde: HordeRef) -> Result(Int, String) {
  horde_count_ffi(horde)
}

/// Compute total kinetic energy: 0.5 * sum(vel^2)
pub fn horde_kinetic_energy(horde: HordeRef) -> Result(Float, String) {
  horde_kinetic_energy_ffi(horde)
}

// Horde FFI bindings
@external(erlang, "viva_tensor_zig", "horde_create")
fn horde_create_ffi(entity_count: Int, dims: Int) -> Result(HordeRef, String)

@external(erlang, "viva_tensor_zig", "horde_set_positions")
fn horde_set_positions_ffi(
  horde: HordeRef,
  data: List(Float),
) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "horde_set_velocities")
fn horde_set_velocities_ffi(
  horde: HordeRef,
  data: List(Float),
) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "horde_integrate")
fn horde_integrate_ffi(horde: HordeRef, dt: Float) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "horde_dampen")
fn horde_dampen_ffi(horde: HordeRef, friction: Float) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "horde_wrap")
fn horde_wrap_ffi(horde: HordeRef, max_bound: Float) -> Result(Nil, String)

@external(erlang, "viva_tensor_zig", "horde_get_positions")
fn horde_get_positions_ffi(horde: HordeRef) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "horde_get_velocities")
fn horde_get_velocities_ffi(horde: HordeRef) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "horde_count")
fn horde_count_ffi(horde: HordeRef) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "horde_kinetic_energy")
fn horde_kinetic_energy_ffi(horde: HordeRef) -> Result(Float, String)

// =============================================================================
// HDC - Hyperdimensional Computing
// "One-shot learning via binary vectors" - 1M similarity ops/sec
//
// Uses high-dimensional binary vectors (10K bits) for:
// - Binding: XOR (associative memory, invertible)
// - Similarity: Hamming distance via popcount
// - Permutation: circular shift for sequence encoding
//
// Use for: one-shot learning, associative memory, symbolic reasoning
// =============================================================================

/// HDC vector reference - binary hyperdimensional vector
pub type HdcVectorRef

/// Standard HDC dimension (10,000 bits)
pub const hdc_default_dim: Int = 10_048

// Note: 10048 = 157 * 64, rounds up from 10000 to multiple of 64

/// Create empty hypervector (dim must be multiple of 64)
pub fn hdc_create(dim: Int) -> Result(HdcVectorRef, String) {
  hdc_create_ffi(dim)
}

/// Create random hypervector (seed for reproducibility)
pub fn hdc_random(dim: Int, seed: Int) -> Result(HdcVectorRef, String) {
  hdc_random_ffi(dim, seed)
}

/// XOR binding: associates two concepts (invertible: A XOR B XOR B = A)
pub fn hdc_bind(
  a: HdcVectorRef,
  b: HdcVectorRef,
) -> Result(HdcVectorRef, String) {
  hdc_bind_ffi(a, b)
}

/// Cosine-like similarity via Hamming distance [0, 1]
/// 1 = identical, 0.5 = orthogonal (random), 0 = opposite
pub fn hdc_similarity(a: HdcVectorRef, b: HdcVectorRef) -> Result(Float, String) {
  hdc_similarity_ffi(a, b)
}

/// Circular permutation for sequence encoding
/// encode(ABC) = A XOR perm(B,1) XOR perm(C,2)
pub fn hdc_permute(
  vec: HdcVectorRef,
  shift: Int,
) -> Result(HdcVectorRef, String) {
  hdc_permute_ffi(vec, shift)
}

/// Get dimensionality (total bits)
pub fn hdc_dim(vec: HdcVectorRef) -> Result(Int, String) {
  hdc_dim_ffi(vec)
}

// HDC FFI bindings
@external(erlang, "viva_tensor_zig", "hdc_create")
fn hdc_create_ffi(dim: Int) -> Result(HdcVectorRef, String)

@external(erlang, "viva_tensor_zig", "hdc_random")
fn hdc_random_ffi(dim: Int, seed: Int) -> Result(HdcVectorRef, String)

@external(erlang, "viva_tensor_zig", "hdc_bind")
fn hdc_bind_ffi(
  a: HdcVectorRef,
  b: HdcVectorRef,
) -> Result(HdcVectorRef, String)

@external(erlang, "viva_tensor_zig", "hdc_similarity")
fn hdc_similarity_ffi(a: HdcVectorRef, b: HdcVectorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "hdc_permute")
fn hdc_permute_ffi(
  vec: HdcVectorRef,
  shift: Int,
) -> Result(HdcVectorRef, String)

@external(erlang, "viva_tensor_zig", "hdc_dim")
fn hdc_dim_ffi(vec: HdcVectorRef) -> Result(Int, String)

// =============================================================================
// Quantization - NF4 / INT8 Compression
// =============================================================================

/// Matrix multiplication with NF4 quantized weights
pub fn nt_matmul_nf4(
  a: NativeTensorRef,
  b_indices: List(Int),
  b_scales: List(Float),
  m: Int,
  n: Int,
  k: Int,
  block_size: Int,
) -> Result(NativeTensorRef, String) {
  nt_matmul_nf4_ffi(a, b_indices, b_scales, m, n, k, block_size)
}

// Quantization FFI bindings
@external(erlang, "viva_tensor_zig", "nt_matmul_nf4")
fn nt_matmul_nf4_ffi(
  a: NativeTensorRef,
  b_indices: List(Int),
  b_scales: List(Float),
  m: Int,
  n: Int,
  k: Int,
  block_size: Int,
) -> Result(NativeTensorRef, String)

// =============================================================================
// CudaTensor - Persistent GPU Memory (FP32)
// =============================================================================

pub type CudaTensorRef

@external(erlang, "viva_tensor_zig", "ct_from_list")
pub fn ct_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(CudaTensorRef, String)

@external(erlang, "viva_tensor_zig", "ct_to_list")
pub fn ct_to_list(ref: CudaTensorRef) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "ct_shape")
pub fn ct_shape(ref: CudaTensorRef) -> Result(List(Int), String)

@external(erlang, "viva_tensor_zig", "ct_matmul")
pub fn ct_matmul(
  a: CudaTensorRef,
  b: CudaTensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensorRef, String)

// =============================================================================
// CudaTensor16 - FP16 Tensor Cores (330 TFLOPS)
// =============================================================================

pub type CudaTensor16Ref

@external(erlang, "viva_tensor_zig", "ct16_available")
pub fn ct16_available() -> Bool

@external(erlang, "viva_tensor_zig", "ct16_from_list")
pub fn ct16_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(CudaTensor16Ref, String)

@external(erlang, "viva_tensor_zig", "ct16_to_list")
pub fn ct16_to_list(ref: CudaTensor16Ref) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "ct16_shape")
pub fn ct16_shape(ref: CudaTensor16Ref) -> Result(List(Int), String)

@external(erlang, "viva_tensor_zig", "ct16_matmul")
pub fn ct16_matmul(
  a: CudaTensor16Ref,
  b: CudaTensor16Ref,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor16Ref, String)

// =============================================================================
// CudaInt8Tensor - INT8 IMMA Tensor Cores (660 TOPS)
// =============================================================================

pub type CudaInt8TensorRef

@external(erlang, "viva_tensor_zig", "ct_int8_available")
pub fn ct_int8_available() -> Bool

@external(erlang, "viva_tensor_zig", "ct_int8_from_list")
pub fn ct_int8_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(CudaInt8TensorRef, String)

@external(erlang, "viva_tensor_zig", "ct_int8_to_list")
pub fn ct_int8_to_list(ref: CudaInt8TensorRef) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "ct_int8_shape")
pub fn ct_int8_shape(ref: CudaInt8TensorRef) -> Result(List(Int), String)

@external(erlang, "viva_tensor_zig", "ct_int8_matmul")
pub fn ct_int8_matmul(
  a: CudaInt8TensorRef,
  b: CudaInt8TensorRef,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaInt8TensorRef, String)

// =============================================================================
// SparseTensor - 2:4 Sparsity (660+ TFLOPS)
// =============================================================================

pub type SparseTensorRef

@external(erlang, "viva_tensor_zig", "sparse_available")
pub fn sparse_available() -> Bool

@external(erlang, "viva_tensor_zig", "sparse_from_ct16")
pub fn sparse_from_ct16(ref: CudaTensor16Ref) -> Result(SparseTensorRef, String)

@external(erlang, "viva_tensor_zig", "sparse_shape")
pub fn sparse_shape(ref: SparseTensorRef) -> Result(List(Int), String)

@external(erlang, "viva_tensor_zig", "sparse_compression_ratio")
pub fn sparse_compression_ratio(ref: SparseTensorRef) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "sparse_matmul")
pub fn sparse_matmul(
  a_sparse: SparseTensorRef,
  b_dense: CudaTensor16Ref,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor16Ref, String)
