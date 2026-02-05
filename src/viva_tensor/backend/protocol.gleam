//// Backend Protocol - Pluggable tensor computation backends
////
//// Provides a unified interface for different execution backends:
//// - Pure: Pure Erlang implementation (always available)
//// - Accelerate: Apple Accelerate framework (macOS only)
//// - Zig: Cross-platform SIMD via Zig NIFs
//// - Distributed: Tensor sharding across BEAM nodes
////
//// Usage:
////   let backend = backend.auto_select()
////   let result = backend.matmul(a, b, m, n, k)

import gleam/list
import gleam/result
import viva_tensor/core/ffi

// =============================================================================
// BACKEND TYPE
// =============================================================================

/// Available computation backends
pub type Backend {
  /// Pure Erlang - always available, portable
  Pure
  /// Apple Accelerate - macOS only, uses cblas/vDSP
  Accelerate
  /// Zig SIMD - cross-platform, portable SIMD
  Zig
  /// Distributed - shards computation across BEAM nodes
  Distributed(nodes: List(Node))
}

/// Represents a BEAM node for distributed computing
pub type Node {
  Node(name: String)
}

// =============================================================================
// BACKEND SELECTION
// =============================================================================

/// Automatically select the best available backend
/// Priority: Zig > Accelerate > Pure
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

/// Get detailed backend info
pub fn info(backend: Backend) -> String {
  case backend {
    Pure -> "Pure Erlang with O(1) array access"
    Accelerate -> ffi.nif_backend_info()
    Zig -> ffi.zig_backend_info()
    Distributed(nodes) ->
      "Distributed across "
      <> int_to_string(list.length(nodes))
      <> " nodes"
  }
}

// =============================================================================
// BACKEND OPERATIONS
// =============================================================================

/// Matrix multiplication using selected backend
/// A[m,k] @ B[k,n] -> C[m,n]
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
      // For dot product, distributed overhead not worth it
      // Fall back to best local backend
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
    // Accelerate doesn't have add NIF
    Zig -> ffi.zig_add(a, b)
    Distributed(_) -> add(auto_select_local(), a, b)
  }
}

// =============================================================================
// PURE ERLANG IMPLEMENTATIONS
// =============================================================================

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

// =============================================================================
// DISTRIBUTED BACKEND
// =============================================================================

/// Distributed matrix multiplication with row sharding
/// Splits matrix A by rows across nodes, broadcasts B
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
      // Calculate rows per node
      let rows_per_node = m / node_count
      let remainder = m % node_count

      // Shard A by rows and dispatch to nodes
      let shards =
        create_row_shards(a, k, rows_per_node, remainder, node_count)

      // Spawn computation on each node
      let tasks =
        list.map2(nodes, shards, fn(node, shard) {
          spawn_matmul_task(node, shard.data, b, shard.rows, n, k)
        })

      // Collect results
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
      // Give extra row to first 'remainder' nodes
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

// =============================================================================
// DISTRIBUTED HELPERS (FFI to Erlang)
// =============================================================================

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

// =============================================================================
// FFI BINDINGS
// =============================================================================

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
