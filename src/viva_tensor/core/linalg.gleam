//// Pure-Gleam linear algebra primitives.
////
//// Operates on 2-D matrices stored in row-major order inside the public
//// `viva_tensor/tensor` `Tensor(data, shape)` value.
////
//// These implementations prioritize correctness over speed. They are not
//// intended to compete with LAPACK; NIF-backed counterparts can replace them
//// later behind the same API.
////
//// All operations are pure Gleam (no NIF) and work on dense list-backed
//// tensors. Native tensors are materialized to a dense list first.
////
//// Algorithms used:
//// - `solve` / `inv`: Gaussian elimination with partial pivoting.
//// - `det` / `lu`: LU decomposition with partial pivoting (Doolittle).
//// - `cholesky`: classical right-looking Cholesky for symmetric PD matrices.
//// - `qr`: classical Gram-Schmidt (numerically inferior to Householder for
////   near-singular columns; documented in the function comment).
//// - `svd` / `eig`: stubbed — return `DimensionError` pointing to a future
////   NIF-backed implementation.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{
  type TensorError, DimensionError, InvalidShape, ShapeMismatch,
}
import viva_tensor/tensor.{type Tensor, Tensor}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Solve `A x = b` for a square coefficient matrix `A` using Gaussian
/// elimination with partial pivoting.
///
/// `b` may be a 1-D vector or a 2-D matrix of right-hand sides; the result
/// keeps the rank of `b`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [3.0, 2.0, 1.0, 2.0])
/// let b = tensor.from_list([7.0, 5.0])
/// let assert Ok(x) = linalg.solve(a, b)
/// tensor.to_list(x)
/// // -> [1.5, 1.75]
/// ```
pub fn solve(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use #(n, a_rows) <- result.try(square_matrix_rows(a))
  use #(b_rows, nrhs, b_is_vector) <- result.try(rhs_rows(b, n))
  case b_rows == n {
    False -> Error(ShapeMismatch(expected: [n, nrhs], got: [b_rows, nrhs]))
    True -> {
      use rows <- result.try(rows_with_rhs(a_rows, b, n, nrhs))
      use solved <- result.try(gauss_eliminate(rows, n, nrhs))
      let flat = list.flatten(solved)
      case b_is_vector {
        True -> Ok(Tensor(data: flat, shape: [n]))
        False -> Ok(Tensor(data: flat, shape: [n, nrhs]))
      }
    }
  }
}

/// Matrix inverse via `solve(a, I)`.
///
/// Returns `InvalidShape("matrix is singular")` when `A` is singular.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [4.0, 7.0, 2.0, 6.0])
/// let assert Ok(ai) = linalg.inv(a)
/// tensor.to_list(ai)
/// // -> [0.6, -0.7, -0.2, 0.4]
/// ```
pub fn inv(a: Tensor) -> Result(Tensor, TensorError) {
  use #(n, _) <- result.try(square_matrix_rows(a))
  let identity = eye(n)
  case solve(a, identity) {
    Ok(x) -> Ok(x)
    Error(InvalidShape("matrix is singular")) ->
      Error(InvalidShape("matrix is singular"))
    Error(other) -> Error(other)
  }
}

/// Determinant via LU decomposition.
///
/// Returns `0.0` for singular matrices instead of erroring — that is the
/// mathematically correct result.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
/// let assert Ok(d) = linalg.det(a)
/// d
/// // -> -2.0
/// ```
pub fn det(a: Tensor) -> Result(Float, TensorError) {
  use #(n, rows) <- result.try(square_matrix_rows(a))
  case lu_decompose(rows, n) {
    Error(LuSingular) -> Ok(0.0)
    Error(LuError(e)) -> Error(e)
    Ok(decomp) -> {
      let diag_product =
        range_int(0, n - 1)
        |> list.fold(1.0, fn(acc, i) {
          let row = unsafe_at(decomp.u, i)
          acc *. unsafe_at(row, i)
        })
      let sign = case int.is_even(decomp.swaps) {
        True -> 1.0
        False -> -1.0
      }
      Ok(sign *. diag_product)
    }
  }
}

/// LU decomposition with partial pivoting.
///
/// Returns `#(L, U, perm)` where:
/// - `L` is unit lower-triangular,
/// - `U` is upper-triangular,
/// - `perm` is the row permutation such that `A[perm[i]] = (L @ U)[i]`
///   (equivalently `P @ A = L @ U`, where `P` is the permutation matrix
///   built from `perm`).
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [4.0, 3.0, 6.0, 3.0])
/// let assert Ok(#(_l, _u, _perm)) = linalg.lu(a)
/// Nil
/// ```
pub fn lu(a: Tensor) -> Result(#(Tensor, Tensor, List(Int)), TensorError) {
  use #(n, rows) <- result.try(square_matrix_rows(a))
  case lu_decompose(rows, n) {
    Error(LuSingular) -> Error(InvalidShape("matrix is singular"))
    Error(LuError(e)) -> Error(e)
    Ok(decomp) -> {
      let l_flat = list.flatten(decomp.l)
      let u_flat = list.flatten(decomp.u)
      Ok(#(
        Tensor(data: l_flat, shape: [n, n]),
        Tensor(data: u_flat, shape: [n, n]),
        decomp.perm,
      ))
    }
  }
}

/// Cholesky decomposition `A = L @ L^T` for a symmetric positive-definite
/// matrix.
///
/// Returns the lower-triangular factor `L`. Errors with
/// `InvalidShape("matrix is not positive definite")` when a pivot is
/// non-positive (e.g. the matrix is indefinite or only positive
/// semi-definite).
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [4.0, 12.0, 12.0, 37.0])
/// let assert Ok(l) = linalg.cholesky(a)
/// tensor.to_list(l)
/// // -> [2.0, 0.0, 6.0, 1.0]
/// ```
pub fn cholesky(a: Tensor) -> Result(Tensor, TensorError) {
  use #(n, rows) <- result.try(square_matrix_rows(a))
  use l_rows <- result.try(cholesky_rows(rows, n))
  Ok(Tensor(data: list.flatten(l_rows), shape: [n, n]))
}

/// QR decomposition via classical Gram-Schmidt.
///
/// Returns `#(Q, R)` where `Q` has orthonormal columns and `R` is upper
/// triangular such that `A ≈ Q @ R`.
///
/// Numerical caveat: classical Gram-Schmidt loses orthogonality on
/// near-linearly-dependent columns. Prefer modified Gram-Schmidt or
/// Householder reflections for ill-conditioned inputs. This v1 trades
/// stability for clarity.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
/// let assert Ok(#(_q, _r)) = linalg.qr(a)
/// Nil
/// ```
pub fn qr(a: Tensor) -> Result(#(Tensor, Tensor), TensorError) {
  case a.shape {
    [m, n] -> {
      use rows <- result.try(to_rows(a, m, n))
      let cols = transpose_rows(rows, m, n)
      use #(q_cols, r_rows) <- result.try(gram_schmidt(cols, m, n))
      let q_rows = transpose_rows(q_cols, n, m)
      Ok(#(
        Tensor(data: list.flatten(q_rows), shape: [m, n]),
        Tensor(data: list.flatten(r_rows), shape: [n, n]),
      ))
    }
    _ -> Error(DimensionError("qr requires a 2D matrix"))
  }
}

/// Identity matrix of size `n` by `n`.
///
/// Convenience wrapper used by other linalg routines. Equivalent to
/// `tensor.eye(n)`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
///
/// linalg.eye(2)
/// // -> Tensor(data: [1.0, 0.0, 0.0, 1.0], shape: [2, 2])
/// ```
pub fn eye(n: Int) -> Tensor {
  let data =
    range_int(0, n - 1)
    |> list.flat_map(fn(i) {
      range_int(0, n - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  Tensor(data: data, shape: [n, n])
}

/// SVD is not implemented in v1. Use a NIF-backed implementation in v2.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [1.0, 0.0, 0.0, 1.0])
/// let result = linalg.svd(a)
/// // -> Error(DimensionError(...))
/// result
/// ```
pub fn svd(_a: Tensor) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  Error(DimensionError("svd: not implemented in v1; use a NIF in v2"))
}

/// Eigendecomposition is not implemented in v1. Use a NIF-backed
/// implementation in v2.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/core/linalg
/// import viva_tensor/tensor
///
/// let assert Ok(a) = tensor.matrix(2, 2, [1.0, 0.0, 0.0, 1.0])
/// let result = linalg.eig(a)
/// // -> Error(DimensionError(...))
/// result
/// ```
pub fn eig(_a: Tensor) -> Result(#(Tensor, Tensor), TensorError) {
  Error(DimensionError("eig: not implemented in v1; use a NIF in v2"))
}

// =============================================================================
// INTERNAL TYPES
// =============================================================================

type LuFailure {
  LuSingular
  LuError(TensorError)
}

type LuDecomp {
  LuDecomp(
    l: List(List(Float)),
    u: List(List(Float)),
    perm: List(Int),
    swaps: Int,
  )
}

// =============================================================================
// SHAPE VALIDATION
// =============================================================================

fn square_matrix_rows(
  a: Tensor,
) -> Result(#(Int, List(List(Float))), TensorError) {
  case a.shape {
    [n, m] if n == m -> {
      use rows <- result.try(to_rows(a, n, n))
      Ok(#(n, rows))
    }
    [n, m] -> Error(ShapeMismatch(expected: [n, n], got: [n, m]))
    _ -> Error(DimensionError("expected a 2D square matrix"))
  }
}

fn rhs_rows(b: Tensor, n: Int) -> Result(#(Int, Int, Bool), TensorError) {
  case b.shape {
    [rows] -> Ok(#(rows, 1, True))
    [rows, cols] -> Ok(#(rows, cols, False))
    _ -> {
      let _ = n
      Error(DimensionError("right-hand side must be 1D or 2D"))
    }
  }
}

fn to_rows(
  a: Tensor,
  m: Int,
  n: Int,
) -> Result(List(List(Float)), TensorError) {
  use data <- result.try(tensor.try_to_list(a))
  case list.length(data) == m * n {
    True -> Ok(chunk(data, n))
    False ->
      Error(InvalidShape(
        "tensor data length "
        <> int.to_string(list.length(data))
        <> " does not match shape "
        <> int.to_string(m)
        <> "x"
        <> int.to_string(n),
      ))
  }
}

fn rows_with_rhs(
  a_rows: List(List(Float)),
  b: Tensor,
  n: Int,
  nrhs: Int,
) -> Result(List(List(Float)), TensorError) {
  use b_data <- result.try(tensor.try_to_list(b))
  case list.length(b_data) == n * nrhs {
    False ->
      Error(InvalidShape(
        "right-hand side data length "
        <> int.to_string(list.length(b_data))
        <> " does not match expected "
        <> int.to_string(n * nrhs),
      ))
    True -> {
      let b_rows = chunk(b_data, nrhs)
      Ok(
        list.zip(a_rows, b_rows)
        |> list.map(fn(pair) {
          let #(arow, brow) = pair
          list.append(arow, brow)
        }),
      )
    }
  }
}

// =============================================================================
// GAUSSIAN ELIMINATION (used by solve)
// =============================================================================
// Augmented rows are length n + nrhs. After elimination we read back the last
// nrhs columns of each row as the solution.

fn gauss_eliminate(
  rows: List(List(Float)),
  n: Int,
  nrhs: Int,
) -> Result(List(List(Float)), TensorError) {
  case eliminate_loop(rows, 0, n) {
    Error(e) -> Error(e)
    Ok(triangular) -> Ok(back_substitute(triangular, n, nrhs))
  }
}

fn eliminate_loop(
  rows: List(List(Float)),
  k: Int,
  n: Int,
) -> Result(List(List(Float)), TensorError) {
  case k == n {
    True -> Ok(rows)
    False -> {
      let pivot_idx = find_pivot(rows, k, n)
      case pivot_idx {
        Error(_) -> Error(InvalidShape("matrix is singular"))
        Ok(idx) -> {
          let swapped = swap_rows(rows, k, idx)
          let pivot_row = unsafe_at(swapped, k)
          let pivot_value = unsafe_at(pivot_row, k)
          case float.absolute_value(pivot_value) <. 1.0e-12 {
            True -> Error(InvalidShape("matrix is singular"))
            False -> {
              let updated =
                swapped
                |> list.index_map(fn(row, i) {
                  case i == k {
                    True -> row
                    False -> {
                      let factor = unsafe_at(row, k) /. pivot_value
                      eliminate_row(row, pivot_row, factor)
                    }
                  }
                })
              eliminate_loop(updated, k + 1, n)
            }
          }
        }
      }
    }
  }
}

fn find_pivot(rows: List(List(Float)), k: Int, n: Int) -> Result(Int, Nil) {
  let candidates =
    range_int(k, n - 1)
    |> list.map(fn(i) { #(i, float.absolute_value(value_at(rows, i, k))) })
  case candidates {
    [] -> Error(Nil)
    [first, ..rest] -> {
      let #(best_idx, best_val) =
        list.fold(rest, first, fn(acc, candidate) {
          let #(_, acc_val) = acc
          let #(_, cand_val) = candidate
          case cand_val >. acc_val {
            True -> candidate
            False -> acc
          }
        })
      case best_val <. 1.0e-12 {
        True -> Error(Nil)
        False -> Ok(best_idx)
      }
    }
  }
}

fn eliminate_row(
  row: List(Float),
  pivot_row: List(Float),
  factor: Float,
) -> List(Float) {
  list.zip(row, pivot_row)
  |> list.map(fn(pair) {
    let #(r, p) = pair
    r -. factor *. p
  })
}

fn back_substitute(
  rows: List(List(Float)),
  n: Int,
  nrhs: Int,
) -> List(List(Float)) {
  // Convert to vectors keyed by row index. Then walk from bottom to top.
  let row_array = rows
  let solution_init = list.repeat(list.repeat(0.0, nrhs), n)
  back_substitute_loop(row_array, solution_init, n - 1, n, nrhs)
}

fn back_substitute_loop(
  rows: List(List(Float)),
  solution: List(List(Float)),
  i: Int,
  n: Int,
  nrhs: Int,
) -> List(List(Float)) {
  case i < 0 {
    True -> solution
    False -> {
      let row = unsafe_at(rows, i)
      let a_ii = unsafe_at(row, i)
      let b_part =
        row
        |> list.drop(n)
        |> list.take(nrhs)
      let summed =
        int_range(i + 1, n - 1)
        |> list.fold(b_part, fn(acc, j) {
          let a_ij = unsafe_at(row, j)
          let x_j = unsafe_at(solution, j)
          list.zip(acc, x_j)
          |> list.map(fn(pair) {
            let #(a_val, x_val) = pair
            a_val -. a_ij *. x_val
          })
        })
      let x_i = list.map(summed, fn(v) { v /. a_ii })
      let new_solution = set_at(solution, i, x_i)
      back_substitute_loop(rows, new_solution, i - 1, n, nrhs)
    }
  }
}

// =============================================================================
// LU DECOMPOSITION (used by det and lu)
// =============================================================================
// Doolittle variant with partial pivoting: L has unit diagonal, U is upper.
// We build L and U row-by-row from a permuted working matrix.

fn lu_decompose(
  rows: List(List(Float)),
  n: Int,
) -> Result(LuDecomp, LuFailure) {
  let perm = range_int(0, n - 1)
  let l_init =
    range_int(0, n - 1)
    |> list.map(fn(i) {
      range_int(0, n - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  let u_init = list.repeat(list.repeat(0.0, n), n)
  lu_loop(rows, l_init, u_init, perm, 0, 0, n)
}

fn lu_loop(
  working: List(List(Float)),
  l: List(List(Float)),
  u: List(List(Float)),
  perm: List(Int),
  k: Int,
  swaps: Int,
  n: Int,
) -> Result(LuDecomp, LuFailure) {
  case k == n {
    True -> Ok(LuDecomp(l: l, u: u, perm: perm, swaps: swaps))
    False -> {
      let pivot_idx = find_pivot_lu(working, k, n)
      case pivot_idx {
        Error(_) -> Error(LuSingular)
        Ok(idx) -> {
          let #(swapped, swap_inc) = case idx == k {
            True -> #(working, 0)
            False -> #(swap_rows(working, k, idx), 1)
          }
          let perm_swapped = case idx == k {
            True -> perm
            False -> swap_int_list(perm, k, idx)
          }
          let l_swapped = case idx == k {
            True -> l
            False -> swap_l_below_k(l, k, idx)
          }
          let pivot_row = unsafe_at(swapped, k)
          let pivot_val = unsafe_at(pivot_row, k)
          case float.absolute_value(pivot_val) <. 1.0e-12 {
            True -> Error(LuSingular)
            False -> {
              let u_updated = set_at(u, k, pivot_row)
              let #(working_next, l_next) =
                lu_eliminate_below(swapped, l_swapped, pivot_row, k, n)
              lu_loop(
                working_next,
                l_next,
                u_updated,
                perm_swapped,
                k + 1,
                swaps + swap_inc,
                n,
              )
            }
          }
        }
      }
    }
  }
}

fn find_pivot_lu(rows: List(List(Float)), k: Int, n: Int) -> Result(Int, Nil) {
  find_pivot(rows, k, n)
}

fn lu_eliminate_below(
  working: List(List(Float)),
  l: List(List(Float)),
  pivot_row: List(Float),
  k: Int,
  n: Int,
) -> #(List(List(Float)), List(List(Float))) {
  let pivot_val = unsafe_at(pivot_row, k)
  let updated =
    working
    |> list.index_map(fn(row, i) {
      case i <= k {
        True -> #(row, 0.0)
        False -> {
          let factor = unsafe_at(row, k) /. pivot_val
          let new_row = eliminate_row(row, pivot_row, factor)
          #(new_row, factor)
        }
      }
    })
  let working_next =
    updated
    |> list.map(fn(pair) { pair.0 })
  let _ = n
  let l_next =
    l
    |> list.index_map(fn(row, i) {
      case i <= k {
        True -> row
        False -> {
          let factor = case list.drop(updated, i) {
            [#(_, f), ..] -> f
            _ -> 0.0
          }
          set_in_row(row, k, factor)
        }
      }
    })
  #(working_next, l_next)
}

// =============================================================================
// CHOLESKY
// =============================================================================

fn cholesky_rows(
  rows: List(List(Float)),
  n: Int,
) -> Result(List(List(Float)), TensorError) {
  let l_init = list.repeat(list.repeat(0.0, n), n)
  cholesky_loop(rows, l_init, 0, n)
}

fn cholesky_loop(
  a: List(List(Float)),
  l: List(List(Float)),
  i: Int,
  n: Int,
) -> Result(List(List(Float)), TensorError) {
  case i == n {
    True -> Ok(l)
    False -> {
      case cholesky_row(a, l, i, n) {
        Error(e) -> Error(e)
        Ok(new_l) -> cholesky_loop(a, new_l, i + 1, n)
      }
    }
  }
}

fn cholesky_row(
  a: List(List(Float)),
  l: List(List(Float)),
  i: Int,
  _n: Int,
) -> Result(List(List(Float)), TensorError) {
  let a_i = unsafe_at(a, i)
  cholesky_row_loop(a_i, l, i, 0)
}

fn cholesky_row_loop(
  a_i: List(Float),
  l: List(List(Float)),
  i: Int,
  j: Int,
) -> Result(List(List(Float)), TensorError) {
  case j > i {
    True -> Ok(l)
    False -> {
      let a_ij = unsafe_at(a_i, j)
      let l_i = unsafe_at(l, i)
      let l_j = unsafe_at(l, j)
      let dot_sum =
        int_range(0, j - 1)
        |> list.fold(0.0, fn(acc, k) {
          acc +. unsafe_at(l_i, k) *. unsafe_at(l_j, k)
        })
      let raw = a_ij -. dot_sum
      case i == j {
        True ->
          case raw <=. 0.0 {
            True -> Error(InvalidShape("matrix is not positive definite"))
            False -> {
              let value = sqrt_float(raw)
              let new_l_i = set_in_row(l_i, j, value)
              Ok(set_at(l, i, new_l_i))
            }
          }
        False -> {
          let l_jj = unsafe_at(l_j, j)
          case float.absolute_value(l_jj) <. 1.0e-15 {
            True -> Error(InvalidShape("matrix is not positive definite"))
            False -> {
              let value = raw /. l_jj
              let new_l_i = set_in_row(l_i, j, value)
              let new_l = set_at(l, i, new_l_i)
              cholesky_row_loop(a_i, new_l, i, j + 1)
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// QR (classical Gram-Schmidt)
// =============================================================================

fn gram_schmidt(
  cols: List(List(Float)),
  m: Int,
  n: Int,
) -> Result(#(List(List(Float)), List(List(Float))), TensorError) {
  let r_init = list.repeat(list.repeat(0.0, n), n)
  let q_init = list.repeat(list.repeat(0.0, m), n)
  gram_schmidt_loop(cols, q_init, r_init, 0, n)
}

fn gram_schmidt_loop(
  cols: List(List(Float)),
  q: List(List(Float)),
  r: List(List(Float)),
  k: Int,
  n: Int,
) -> Result(#(List(List(Float)), List(List(Float))), TensorError) {
  case k == n {
    True -> Ok(#(q, r))
    False -> {
      let a_k = unsafe_at(cols, k)
      let #(v, r_updated) =
        int_range(0, k - 1)
        |> list.fold(#(a_k, r), fn(acc, j) {
          let #(current_v, current_r) = acc
          let q_j = unsafe_at(q, j)
          let r_jk = dot(q_j, a_k)
          let new_v =
            list.zip(current_v, q_j)
            |> list.map(fn(pair) {
              let #(vv, qq) = pair
              vv -. r_jk *. qq
            })
          let r_row_j = unsafe_at(current_r, j)
          let new_r_row_j = set_in_row(r_row_j, k, r_jk)
          let new_r = set_at(current_r, j, new_r_row_j)
          #(new_v, new_r)
        })
      let norm_v = sqrt_float(dot(v, v))
      case norm_v <. 1.0e-15 {
        True -> Error(InvalidShape("qr: columns are linearly dependent"))
        False -> {
          let q_k = list.map(v, fn(x) { x /. norm_v })
          let q_next = set_at(q, k, q_k)
          let r_row_k = unsafe_at(r_updated, k)
          let new_r_row_k = set_in_row(r_row_k, k, norm_v)
          let r_next = set_at(r_updated, k, new_r_row_k)
          gram_schmidt_loop(cols, q_next, r_next, k + 1, n)
        }
      }
    }
  }
}

// =============================================================================
// LOW-LEVEL HELPERS
// =============================================================================

/// Inclusive integer range that returns an empty list when `stop < start`.
/// Differs from the old stdlib range helper which always emitted at least one element.
fn int_range(start: Int, stop: Int) -> List(Int) {
  case stop < start {
    True -> []
    False -> range_int(start, stop)
  }
}

fn chunk(data: List(Float), n: Int) -> List(List(Float)) {
  case data {
    [] -> []
    _ -> [list.take(data, n), ..chunk(list.drop(data, n), n)]
  }
}

fn unsafe_at(lst: List(a), idx: Int) -> a {
  let assert Ok(value) = list_at(lst, idx)
  value
}

fn list_at(lst: List(a), idx: Int) -> Result(a, Nil) {
  case lst, idx {
    [], _ -> Error(Nil)
    [head, ..], 0 -> Ok(head)
    [_, ..rest], _ -> list_at(rest, idx - 1)
  }
}

fn value_at(rows: List(List(Float)), i: Int, j: Int) -> Float {
  unsafe_at(unsafe_at(rows, i), j)
}

fn swap_rows(rows: List(List(Float)), i: Int, j: Int) -> List(List(Float)) {
  case i == j {
    True -> rows
    False -> {
      let row_i = unsafe_at(rows, i)
      let row_j = unsafe_at(rows, j)
      rows
      |> list.index_map(fn(row, idx) {
        case idx == i, idx == j {
          True, _ -> row_j
          _, True -> row_i
          _, _ -> row
        }
      })
    }
  }
}

fn swap_int_list(lst: List(Int), i: Int, j: Int) -> List(Int) {
  case i == j {
    True -> lst
    False -> {
      let v_i = unsafe_at_int(lst, i)
      let v_j = unsafe_at_int(lst, j)
      lst
      |> list.index_map(fn(v, idx) {
        case idx == i, idx == j {
          True, _ -> v_j
          _, True -> v_i
          _, _ -> v
        }
      })
    }
  }
}

fn unsafe_at_int(lst: List(Int), idx: Int) -> Int {
  let assert Ok(v) = list_at(lst, idx)
  v
}

fn swap_l_below_k(l: List(List(Float)), i: Int, j: Int) -> List(List(Float)) {
  // Swap the parts of rows i and j that lie strictly to the left of column k.
  // Required because partial-pivoting updates rows of L that were already
  // partially filled in earlier elimination steps.
  let row_i = unsafe_at(l, i)
  let row_j = unsafe_at(l, j)
  let new_i = take_below_k_from(row_j, row_i, i)
  let new_j = take_below_k_from(row_i, row_j, j)
  l
  |> list.index_map(fn(row, idx) {
    case idx == i, idx == j {
      True, _ -> new_i
      _, True -> new_j
      _, _ -> row
    }
  })
}

fn take_below_k_from(
  source: List(Float),
  target: List(Float),
  k: Int,
) -> List(Float) {
  // For column index < k take from `source`, otherwise from `target`.
  list.zip(source, target)
  |> list.index_map(fn(pair, idx) {
    let #(s, t) = pair
    case idx < k {
      True -> s
      False -> t
    }
  })
}

fn set_at(lst: List(a), idx: Int, value: a) -> List(a) {
  lst
  |> list.index_map(fn(item, i) {
    case i == idx {
      True -> value
      False -> item
    }
  })
}

fn set_in_row(row: List(Float), idx: Int, value: Float) -> List(Float) {
  set_at(row, idx, value)
}

fn dot(a: List(Float), b: List(Float)) -> Float {
  list.zip(a, b)
  |> list.fold(0.0, fn(acc, pair) {
    let #(x, y) = pair
    acc +. x *. y
  })
}

fn transpose_rows(
  rows: List(List(Float)),
  _m: Int,
  n: Int,
) -> List(List(Float)) {
  range_int(0, n - 1)
  |> list.map(fn(j) {
    rows
    |> list.map(fn(row) { unsafe_at(row, j) })
  })
}

@external(erlang, "math", "sqrt")
fn sqrt_float(x: Float) -> Float

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}
