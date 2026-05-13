//// Einstein summation (`einsum`) kernel for viva_tensor.
////
//// Implements the explicit-mode subset of NumPy's `einsum` for up to two
//// operands. The public wrapper lives in `viva_tensor/tensor`; this module
//// works on materialized row-major `(data, shape)` pairs and returns the
//// result `(data, shape)` so it has no dependency on the higher-level
//// `Tensor` type and avoids cyclic imports.
////
//// Out-of-scope features (return a `DimensionError` with a clear message):
////   * Ellipsis `...`
////   * More than 2 input operands
////   * Implicit output mode (no `->`)
////   * Optimization paths

import gleam/int
import gleam/list
import gleam/result
import gleam/string
import viva_tensor/core/error.{
  type TensorError, DimensionError, InvalidShape, ShapeMismatch,
}

// --- Public types -----------------------------------------------------------

/// Materialized tensor view: row-major data plus shape.
pub type Operand {
  Operand(data: List(Float), shape: List(Int))
}

// --- Public API -------------------------------------------------------------

/// Run einsum on an equation against a list of `Operand`s. Returns the
/// computed `(data, shape)` pair or a `TensorError`.
///
/// Recognized fast-path patterns are dispatched to caller-provided callbacks
/// in `viva_tensor/tensor`; this module only implements the generic
/// nested-loop kernel and validation. The wrapper module wires up the fast
/// paths.
pub fn run(
  equation: String,
  operands: List(Operand),
) -> Result(Operand, TensorError) {
  use parsed <- result.try(parse_equation(equation))
  let Parsed(lhs_labels, rhs_labels) = parsed
  use _ <- result.try(check_operand_arity(lhs_labels, operands))
  use _ <- result.try(check_rank_match(lhs_labels, operands))
  use dim_map <- result.try(infer_dims(lhs_labels, operands))
  use _ <- result.try(check_rhs_labels(rhs_labels, lhs_labels))

  case list.length(operands) > 2 {
    True ->
      Error(DimensionError(
        "einsum: feature not supported in v1: more than 2 input operands",
      ))
    False -> general_kernel(operands, lhs_labels, rhs_labels, dim_map)
  }
}

/// Public access to the parsed labels for the wrapper's dispatch logic.
pub fn parse(
  equation: String,
) -> Result(#(List(List(String)), List(String)), TensorError) {
  use parsed <- result.try(parse_equation(equation))
  let Parsed(lhs, rhs) = parsed
  Ok(#(lhs, rhs))
}

/// Validate parsed labels against given operands (rank + dim consistency
/// + RHS label coverage / uniqueness).
pub fn validate(
  lhs: List(List(String)),
  rhs: List(String),
  operands: List(Operand),
) -> Result(List(#(String, Int)), TensorError) {
  use _ <- result.try(check_operand_arity(lhs, operands))
  use _ <- result.try(check_rank_match(lhs, operands))
  use dim_map <- result.try(infer_dims(lhs, operands))
  use _ <- result.try(check_rhs_labels(rhs, lhs))
  Ok(dim_map)
}

/// Run the generic nested-loop kernel directly (used by the wrapper after
/// fast-path dispatch declines to handle the equation).
pub fn general(
  operands: List(Operand),
  lhs: List(List(String)),
  rhs: List(String),
  dim_map: List(#(String, Int)),
) -> Result(Operand, TensorError) {
  general_kernel(operands, lhs, rhs, dim_map)
}

// --- Parser ----------------------------------------------------------------

type Parsed {
  Parsed(lhs: List(List(String)), rhs: List(String))
}

fn parse_equation(equation: String) -> Result(Parsed, TensorError) {
  let normalized = remove_whitespace(equation)
  case string.contains(normalized, "...") {
    True ->
      Error(DimensionError("einsum: feature not supported in v1: ellipsis"))
    False ->
      case string.split(normalized, on: "->") {
        [lhs, rhs] -> {
          let lhs_parts = string.split(lhs, on: ",")
          let rhs_labels = string.to_graphemes(rhs)
          use _ <- result.try(check_labels(rhs_labels, "right-hand side"))
          let lhs_labels = list.map(lhs_parts, string.to_graphemes)
          use _ <- result.try(
            list.try_each(lhs_labels, fn(labels) {
              check_labels(labels, "left-hand side")
            }),
          )
          Ok(Parsed(lhs: lhs_labels, rhs: rhs_labels))
        }
        [_] ->
          Error(DimensionError(
            "einsum: feature not supported in v1: implicit output mode (missing '->')",
          ))
        _ ->
          Error(InvalidShape(
            "einsum: malformed equation (multiple '->' separators)",
          ))
      }
  }
}

fn remove_whitespace(s: String) -> String {
  s
  |> string.to_graphemes
  |> list.filter(fn(g) { g != " " && g != "\t" && g != "\n" })
  |> string.concat
}

fn check_labels(
  labels: List(String),
  context: String,
) -> Result(Nil, TensorError) {
  case list.all(labels, is_label_char) {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape(
        "einsum: invalid label in " <> context <> " (expected ASCII letters)",
      ))
  }
}

fn is_label_char(g: String) -> Bool {
  case g {
    "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" -> True
    "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" -> True
    "u" | "v" | "w" | "x" | "y" | "z" -> True
    "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" -> True
    "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" -> True
    "U" | "V" | "W" | "X" | "Y" | "Z" -> True
    _ -> False
  }
}

// --- Validation -------------------------------------------------------------

fn check_operand_arity(
  lhs: List(List(String)),
  operands: List(Operand),
) -> Result(Nil, TensorError) {
  let expected = list.length(lhs)
  let got = list.length(operands)
  case expected == got {
    True -> Ok(Nil)
    False ->
      Error(DimensionError(
        "einsum: equation expects "
        <> int.to_string(expected)
        <> " operand(s) but got "
        <> int.to_string(got),
      ))
  }
}

fn check_rank_match(
  lhs: List(List(String)),
  operands: List(Operand),
) -> Result(Nil, TensorError) {
  list.zip(lhs, operands)
  |> list.try_each(fn(pair) {
    let #(labels, op) = pair
    let r = list.length(op.shape)
    case list.length(labels) == r {
      True -> Ok(Nil)
      False ->
        Error(DimensionError(
          "einsum: operand rank "
          <> int.to_string(r)
          <> " does not match label count "
          <> int.to_string(list.length(labels)),
        ))
    }
  })
}

fn infer_dims(
  lhs: List(List(String)),
  operands: List(Operand),
) -> Result(List(#(String, Int)), TensorError) {
  list.zip(lhs, operands)
  |> list.try_fold([], fn(acc, pair) {
    let #(labels, op) = pair
    merge_dims(acc, list.zip(labels, op.shape))
  })
}

fn merge_dims(
  acc: List(#(String, Int)),
  pairs: List(#(String, Int)),
) -> Result(List(#(String, Int)), TensorError) {
  list.try_fold(pairs, acc, fn(acc, pair) {
    let #(label, dim) = pair
    case lookup(acc, label) {
      Ok(existing) ->
        case existing == dim {
          True -> Ok(acc)
          False -> Error(ShapeMismatch(expected: [existing], got: [dim]))
        }
      Error(_) -> Ok([#(label, dim), ..acc])
    }
  })
}

fn lookup(pairs: List(#(String, Int)), key: String) -> Result(Int, Nil) {
  case pairs {
    [] -> Error(Nil)
    [#(k, v), ..rest] ->
      case k == key {
        True -> Ok(v)
        False -> lookup(rest, key)
      }
  }
}

fn check_rhs_labels(
  rhs: List(String),
  lhs: List(List(String)),
) -> Result(Nil, TensorError) {
  let all_lhs = list.flatten(lhs)
  use _ <- result.try(
    list.try_each(rhs, fn(label) {
      case list.contains(all_lhs, label) {
        True -> Ok(Nil)
        False ->
          Error(InvalidShape(
            "einsum: output label '"
            <> label
            <> "' does not appear in any input",
          ))
      }
    }),
  )
  case has_unique(rhs, []) {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape(
        "einsum: output labels must be unique (repeated label on RHS)",
      ))
  }
}

fn has_unique(items: List(String), seen: List(String)) -> Bool {
  case items {
    [] -> True
    [x, ..rest] ->
      case list.contains(seen, x) {
        True -> False
        False -> has_unique(rest, [x, ..seen])
      }
  }
}

// --- General nested-loop kernel --------------------------------------------

/// Generic einsum: walk the cartesian product of output indices; for each
/// output cell, sum the product of operand values across the cartesian
/// product of contracted (summed-out) labels.
fn general_kernel(
  operands: List(Operand),
  lhs: List(List(String)),
  rhs: List(String),
  dim_map: List(#(String, Int)),
) -> Result(Operand, TensorError) {
  let all_labels = dedupe(list.flatten(lhs), [])
  let summed = list.filter(all_labels, fn(label) { !list.contains(rhs, label) })

  let out_shape =
    list.map(rhs, fn(label) {
      case lookup(dim_map, label) {
        Ok(d) -> d
        Error(_) -> 0
      }
    })

  let out_combos =
    cartesian(list.map(out_shape, fn(d) { list.range(0, d - 1) }))

  let sum_dims =
    list.map(summed, fn(label) {
      case lookup(dim_map, label) {
        Ok(d) -> d
        Error(_) -> 0
      }
    })
  let sum_combos = cartesian(list.map(sum_dims, fn(d) { list.range(0, d - 1) }))

  let lhs_data = list.zip(lhs, operands)

  let data =
    list.map(out_combos, fn(out_idx) {
      let out_bindings = list.zip(rhs, out_idx)
      list.fold(sum_combos, 0.0, fn(acc, sum_idx) {
        let sum_bindings = list.zip(summed, sum_idx)
        let bindings = list.append(out_bindings, sum_bindings)
        let product =
          list.fold(lhs_data, 1.0, fn(p, pair) {
            let #(labels, op) = pair
            let idx = label_indices(labels, bindings)
            let flat = flat_index(idx, op.shape)
            case nth(op.data, flat) {
              Ok(v) -> p *. v
              Error(_) -> p
            }
          })
        acc +. product
      })
    })

  case out_shape {
    [] ->
      case data {
        [v] -> Ok(Operand(data: [v], shape: []))
        _ -> Ok(Operand(data: [0.0], shape: []))
      }
    _ -> Ok(Operand(data: data, shape: out_shape))
  }
}

fn dedupe(items: List(String), seen: List(String)) -> List(String) {
  case items {
    [] -> list.reverse(seen)
    [x, ..rest] ->
      case list.contains(seen, x) {
        True -> dedupe(rest, seen)
        False -> dedupe(rest, [x, ..seen])
      }
  }
}

fn cartesian(ranges: List(List(Int))) -> List(List(Int)) {
  case ranges {
    [] -> [[]]
    [first, ..rest] -> {
      let tails = cartesian(rest)
      list.flat_map(first, fn(v) { list.map(tails, fn(t) { [v, ..t] }) })
    }
  }
}

fn label_indices(
  labels: List(String),
  bindings: List(#(String, Int)),
) -> List(Int) {
  list.map(labels, fn(label) {
    case lookup_binding(bindings, label) {
      Ok(v) -> v
      Error(_) -> 0
    }
  })
}

fn lookup_binding(
  bindings: List(#(String, Int)),
  key: String,
) -> Result(Int, Nil) {
  case bindings {
    [] -> Error(Nil)
    [#(k, v), ..rest] ->
      case k == key {
        True -> Ok(v)
        False -> lookup_binding(rest, key)
      }
  }
}

fn flat_index(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)
  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) { acc + pair.0 * pair.1 })
}

/// Row-major strides for `shape`. For `[2, 3]` produces `[3, 1]`.
fn compute_strides(shape: List(Int)) -> List(Int) {
  let rev = list.reverse(shape)
  let #(strides, _) =
    list.fold(rev, #([], 1), fn(state, dim) {
      let #(acc, product) = state
      #([product, ..acc], product * dim)
    })
  strides
}

fn nth(items: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False -> nth_loop(items, index)
  }
}

fn nth_loop(items: List(Float), index: Int) -> Result(Float, Nil) {
  case items, index {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> nth_loop(rest, n - 1)
  }
}
