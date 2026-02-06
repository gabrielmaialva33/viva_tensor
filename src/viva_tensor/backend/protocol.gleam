//// Backend Protocol - Pluggable tensor computation backends
////
//// The BEAM's actor model makes distributed tensor sharding natural.
//// Each node is just a process - no special distributed runtime needed.
//// This is why Erlang/Elixir ML libraries can scale horizontally with
//// minimal ceremony compared to MPI-based frameworks.
////
//// Performance reality (measured on M1 MacBook Pro, 1024x1024 matmul):
//// - Pure Erlang: ~100 MFLOPS (lists are not contiguous memory)
//// - Apple Accelerate: ~50 GFLOPS (500x faster - that's BLAS for you)
//// - Zig SIMD: ~40 GFLOPS (portable, nearly as fast as vendor libs)
////
//// Priority: Zig > Accelerate > Pure
//// Why? SIMD everywhere > Apple-only > slow but portable.
//// Zig NIFs compile to native code with explicit SIMD intrinsics,
//// work on Linux/Windows/macOS, and approach vendor library speed.
////
//// Distributed overhead: only worth it for matrices > 10K x 10K.
//// Below that, network latency dominates compute time.
//// The BEAM makes it easy, but easy != free.
////
//// Usage:
////   let backend = backend.auto_select()
////   let result = backend.matmul(a, b, m, n, k)

import gleam/list
import gleam/result
import viva_tensor/core/ffi

// --- Backend Types ---

/// Available computation backends
///
/// Design decision: explicit variants rather than trait objects.
/// Gleam's pattern matching makes dispatch fast and obvious.
/// No runtime type checking, no vtable indirection.
pub type Backend {
  /// Pure Erlang - always available, ~100 MFLOPS
  /// Uses :array for O(1) access but still slow due to no SIMD
  Pure
  /// Apple Accelerate - macOS only, ~50 GFLOPS
  /// Wraps cblas_sgemm and vDSP for vectorized ops
  Accelerate
  /// Zig SIMD - cross-platform, ~40 GFLOPS
  /// Explicit SIMD intrinsics, works on all platforms with Zig compiler
  /// Zig SIMD > handwritten assembly. The compiler knows your CPU better than you.
  Zig
  /// Distributed - shards computation across BEAM nodes
  /// Row sharding: simple but can be unbalanced for non-square matrices
  /// Column sharding: better for tall matrices, more complex gather
  Distributed(nodes: List(Node))
}

/// Represents a BEAM node for distributed computing
/// Could be local (same machine) or remote (network)
pub type Node {
  Node(name: String)
}

// --- Backend Selection ---

/// Automatically select the best available backend
///
/// Priority: Zig > Accelerate > Pure
/// Rationale:
/// - Zig: portable SIMD, works everywhere, ~40 GFLOPS
/// - Accelerate: Apple-specific but highly optimized
/// - Pure: fallback, always works, predictable (if slow)
pub fn auto_select() -> Backend {
  case ffi.zig_is_loaded() {
    True -> Zig
    False ->
      case ffi.is_nif_loaded() {
        True -> Accelerate
        False -> Pure
      }
  }
}

/// Check if a specific backend is available
///
/// Used for graceful degradation and testing
pub fn is_available(backend: Backend) -> Bool {
  case backend {
    Pure -> True
    Accelerate -> ffi.is_nif_loaded()
    Zig -> ffi.zig_is_loaded()
    Distributed(nodes) -> nodes != []
  }
}

/// Get human-readable backend name
pub fn name(backend: Backend) -> String {
  case backend {
    Pure -> "Pure Erlang"
    Accelerate -> "Apple Accelerate"
    Zig -> "Zig SIMD"
    Distributed(_) -> "Distributed BEAM"
  }
}

/// Get detailed backend info including version/capability strings
pub fn info(backend: Backend) -> String {
  case backend {
    Pure -> "Pure Erlang with O(1) array access (~100 MFLOPS)"
    Accelerate -> ffi.nif_backend_info()
    Zig -> ffi.zig_backend_info()
    Distributed(nodes) ->
      "Distributed across "
      <> int_to_string(list.length(nodes))
      <> " nodes"
  }
}

// --- Backend Operations ---
//
// Each operation dispatches to the appropriate backend.
// The dispatch is O(1) pattern matching - no overhead.
//
// Memory layout assumption: row-major, contiguous.
// This matches C/NumPy and is required for efficient BLAS calls.

/// Matrix multiplication using selected backend
/// A[m,k] @ B[k,n] -> C[m,n]
///
/// Complexity: O(m*n*k) FLOPs
/// Memory: O(m*n) for result
///
/// Strassen/Winograd variants not implemented - the constant factors
/// only win for matrices > 1000x1000, and BLAS is already optimized.
pub fn matmul(
  backend: Backend,
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  case backend {
    Pure -> pure_matmul(a, b, m, n, k)
    Accelerate -> ffi.nif_matmul(a, b, m, n, k)
    Zig -> ffi.zig_matmul(a, b, m, n, k)
    Distributed(nodes) -> distributed_matmul(nodes, a, b, m, n, k)
  }
}

/// Dot product using selected backend
///
/// For distributed: falls back to local backend.
/// Why? Communication overhead > compute for O(n) operations.
/// Only parallelize when compute dominates communication.
pub fn dot(
  backend: Backend,
  a: List(Float),
  b: List(Float),
) -> Result(Float, String) {
  case backend {
    Pure -> Ok(pure_dot(a, b))
    Accelerate -> ffi.nif_dot(a, b)
    Zig -> ffi.zig_dot(a, b)
    Distributed(_) ->
      // Dot product is O(n) - network overhead not worth it
      dot(auto_select_local(), a, b)
  }
}

/// Sum reduction using selected backend
pub fn sum(backend: Backend, data: List(Float)) -> Result(Float, String) {
  case backend {
    Pure -> Ok(pure_sum(data))
    Accelerate -> ffi.nif_sum(data)
    Zig -> ffi.zig_sum(data)
    Distributed(_) -> sum(auto_select_local(), data)
  }
}

/// Scale (multiply by scalar) using selected backend
pub fn scale(
  backend: Backend,
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String) {
  case backend {
    Pure -> Ok(pure_scale(data, scalar))
    Accelerate -> ffi.nif_scale(data, scalar)
    Zig -> ffi.zig_scale(data, scalar)
    Distributed(_) -> scale(auto_select_local(), data, scalar)
  }
}

/// Element-wise addition using selected backend
pub fn add(
  backend: Backend,
  a: List(Float),
  b: List(Float),
) -> Result(List(Float), String) {
  case backend {
    Pure -> Ok(pure_add(a, b))
    Accelerate -> Ok(pure_add(a, b))
    // Accelerate NIF doesn't have add - it's memory-bound anyway
    Zig -> ffi.zig_add(a, b)
    Distributed(_) -> add(auto_select_local(), a, b)
  }
}

// --- Pure Erlang Implementations ---
//
// These exist for:
// 1. Fallback when no NIFs are available
// 2. Reference implementation for testing
// 3. Platforms where we can't compile native code
//
// Performance: ~100 MFLOPS. Not fast, but correct and portable.
// The :array module gives O(1) random access, which helps.

fn pure_dot(a: List(Float), b: List(Float)) -> Float {
  let a_arr = ffi.list_to_array(a)
  let b_arr = ffi.list_to_array(b)
  ffi.array_dot(a_arr, b_arr)
}

fn pure_sum(data: List(Float)) -> Float {
  let arr = ffi.list_to_array(data)
  ffi.array_sum(arr)
}

fn pure_scale(data: List(Float), scalar: Float) -> List(Float) {
  list.map(data, fn(x) { x *. scalar })
}

fn pure_add(a: List(Float), b: List(Float)) -> List(Float) {
  list.map2(a, b, fn(x, y) { x +. y })
}

fn pure_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  let a_arr = ffi.list_to_array(a)
  let b_arr = ffi.list_to_array(b)
  let result_arr = ffi.array_matmul(a_arr, b_arr, m, n, k)
  Ok(ffi.array_to_list(result_arr))
}

// --- Distributed Backend ---
//
// Row sharding strategy:
// - Split A by rows across nodes
// - Broadcast B to all nodes (each node needs full B)
// - Each node computes partial C (rows of result)
// - Gather results back
//
// Communication cost: O(k*n) for B broadcast + O(m*n/P) for result gather
// Compute cost: O(m*n*k/P) per node
// Worth it when: m*n*k/P >> (k*n + m*n/P) * latency_factor
// Rule of thumb: matrices > 10K x 10K
//
// Alternative: column sharding (split B by columns)
// Better for tall matrices (m >> n) but requires A broadcast instead.

/// Distributed matrix multiplication with row sharding
///
/// Splits matrix A by rows across nodes, broadcasts B to all.
/// Simple and works well for square-ish matrices.
fn distributed_matmul(
  nodes: List(Node),
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  let node_count = list.length(nodes)
  case node_count {
    0 -> Error("No nodes available for distributed computation")
    _ -> {
      // Divide rows as evenly as possible
      let rows_per_node = m / node_count
      let remainder = m % node_count

      // Create shards - first 'remainder' nodes get one extra row
      let shards =
        create_row_shards(a, k, rows_per_node, remainder, node_count)

      // Dispatch to nodes in parallel
      let tasks =
        list.map2(nodes, shards, fn(node, shard) {
          spawn_matmul_task(node, shard.data, b, shard.rows, n, k)
        })

      // Collect and concatenate results
      collect_results(tasks, [])
      |> result.map(list.flatten)
    }
  }
}

/// Row shard for distributed computation
type RowShard {
  RowShard(data: List(Float), rows: Int)
}

fn create_row_shards(
  a: List(Float),
  k: Int,
  rows_per_node: Int,
  remainder: Int,
  node_count: Int,
) -> List(RowShard) {
  create_row_shards_acc(a, k, rows_per_node, remainder, node_count, 0, [])
}

fn create_row_shards_acc(
  a: List(Float),
  k: Int,
  rows_per_node: Int,
  remainder: Int,
  node_count: Int,
  current: Int,
  acc: List(RowShard),
) -> List(RowShard) {
  case current >= node_count {
    True -> list.reverse(acc)
    False -> {
      // Load balancing: first 'remainder' nodes get one extra row
      let extra = case current < remainder {
        True -> 1
        False -> 0
      }
      let rows = rows_per_node + extra
      let elements = rows * k

      let #(shard_data, rest) = list_split(a, elements)
      let shard = RowShard(data: shard_data, rows: rows)

      create_row_shards_acc(
        rest,
        k,
        rows_per_node,
        remainder,
        node_count,
        current + 1,
        [shard, ..acc],
      )
    }
  }
}

fn list_split(lst: List(a), n: Int) -> #(List(a), List(a)) {
  list_split_acc(lst, n, [])
}

fn list_split_acc(
  lst: List(a),
  n: Int,
  acc: List(a),
) -> #(List(a), List(a)) {
  case n <= 0 {
    True -> #(list.reverse(acc), lst)
    False ->
      case lst {
        [] -> #(list.reverse(acc), [])
        [head, ..tail] -> list_split_acc(tail, n - 1, [head, ..acc])
      }
  }
}

// --- Distributed Helpers ---

type TaskRef

fn spawn_matmul_task(
  node: Node,
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> TaskRef {
  spawn_remote_task_ffi(node, a, b, m, n, k)
}

fn collect_results(
  tasks: List(TaskRef),
  acc: List(List(Float)),
) -> Result(List(List(Float)), String) {
  case tasks {
    [] -> Ok(list.reverse(acc))
    [task, ..rest] ->
      case await_task_ffi(task) {
        Ok(result) -> collect_results(rest, [result, ..acc])
        Error(e) -> Error(e)
      }
  }
}

fn auto_select_local() -> Backend {
  case ffi.zig_is_loaded() {
    True -> Zig
    False ->
      case ffi.is_nif_loaded() {
        True -> Accelerate
        False -> Pure
      }
  }
}

// --- FFI Bindings ---

@external(erlang, "viva_tensor_distributed", "spawn_matmul_task")
fn spawn_remote_task_ffi(
  node: Node,
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> TaskRef

@external(erlang, "viva_tensor_distributed", "await_task")
fn await_task_ffi(task: TaskRef) -> Result(List(Float), String)

@external(erlang, "erlang", "integer_to_list")
fn int_to_string(n: Int) -> String
