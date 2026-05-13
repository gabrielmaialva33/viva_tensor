//// ONNX (Open Neural Network Exchange) graph import — JSON intermediate.
////
//// Protobuf parsing in pure Gleam is impractical, so this module accepts an
//// ONNX `GraphProto` that has already been converted to JSON. A typical
//// conversion script in Python:
////
//// ```python
//// import onnx
//// import json
//// from google.protobuf.json_format import MessageToDict
////
//// m = onnx.load("model.onnx")
//// d = MessageToDict(m.graph, preserving_proto_field_name=True)
//// # then re-shape into the schema this module expects (see parse_graph docs)
//// json.dump(d, open("graph.json", "w"))
//// ```
////
//// The expected JSON shape is:
////
//// ```json
//// {
////   "nodes": [
////     {
////       "op_type": "Gemm",
////       "name": "linear1",
////       "inputs": ["x", "W", "b"],
////       "outputs": ["h"],
////       "attributes": {
////         "alpha": {"type": "float", "value": 1.0},
////         "beta":  {"type": "float", "value": 1.0},
////         "transB": {"type": "int", "value": 1}
////       }
////     }
////   ],
////   "inputs": ["x"],
////   "outputs": ["h"],
////   "initializers": {
////     "W": {"shape": [4, 8], "data": [...]},
////     "b": {"shape": [8],   "data": [...]}
////   }
//// }
//// ```
////
//// Attribute `type` is one of: `"int"`, `"float"`, `"string"`, `"ints"`,
//// `"floats"`, or `"tensor"`. Tensor attributes use the same
//// `{"shape": [...], "data": [...]}` shape as initializers.
////
//// ## v1 supported ops
////
//// `Add`, `Sub`, `Mul`, `MatMul`, `Gemm`, `Relu`, `Sigmoid`, `Tanh`, `Gelu`,
//// `Softmax`, `Transpose`, `Reshape`, `Constant`, `LayerNormalization`.
////
//// ## Out of scope (v2+)
////
//// `Conv`, `MaxPool`, `AveragePool`, `Dropout`, `LSTM`, `GRU`,
//// `BatchNormalization`, `Cast`, `Slice`, `Concat`, `Split`, `Gather`,
//// `Squeeze`, `Unsqueeze`, `ReduceMean`, `ReduceSum`. Encountering any of
//// these in a graph returns `UnsupportedOp`.

import gleam/dict.{type Dict}
import gleam/dynamic/decode
import gleam/float
import gleam/int
import gleam/json
import gleam/list
import gleam/result
import gleam/string
import viva_tensor/nn/activations
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Public types ----------------------------------------------------------

/// A single ONNX node (operator instance) in the graph.
pub type OnnxNode {
  OnnxNode(
    op_type: String,
    name: String,
    inputs: List(String),
    outputs: List(String),
    attributes: Dict(String, OnnxAttribute),
  )
}

/// An ONNX attribute value. ONNX has more attribute kinds (graph, sparse
/// tensor, type protos, ...) but v1 only models the scalar/list/tensor kinds
/// needed by the supported op set.
pub type OnnxAttribute {
  IntAttr(value: Int)
  FloatAttr(value: Float)
  StringAttr(value: String)
  IntsAttr(value: List(Int))
  FloatsAttr(value: List(Float))
  TensorAttr(value: Tensor)
}

/// A parsed ONNX graph ready to run.
///
/// `nodes` must already be in topological order — `parse_graph` does not
/// re-sort them, mirroring how ONNX serializes graphs in execution order.
pub type OnnxGraph {
  OnnxGraph(
    nodes: List(OnnxNode),
    inputs: List(String),
    outputs: List(String),
    initializers: Dict(String, Tensor),
  )
}

/// Failure modes for ONNX import and execution.
pub type OnnxError {
  /// The JSON document could not be parsed or did not match the expected
  /// schema. `reason` is a human-readable description.
  ParseError(reason: String)
  /// A node references an op_type that this version does not implement.
  UnsupportedOp(op_type: String)
  /// A node references an input name that is not in the initializers, feeds,
  /// or previously-produced outputs.
  MissingInput(name: String)
  /// A tensor's shape or rank is incompatible with the operator's contract.
  ShapeError(reason: String)
}

// --- Public API ------------------------------------------------------------

/// Parse a JSON-encoded ONNX graph.
///
/// Returns `ParseError` if the JSON is malformed or the schema is wrong.
/// Supported op set (v1): `Add`, `Sub`, `Mul`, `MatMul`, `Gemm`, `Relu`,
/// `Sigmoid`, `Tanh`, `Gelu`, `Softmax`, `Transpose`, `Reshape`, `Constant`,
/// `LayerNormalization`. Any other op_type compiles fine here and only fails
/// at `run_graph` time with `UnsupportedOp`.
pub fn parse_graph(json_str: String) -> Result(OnnxGraph, OnnxError) {
  case json.parse(from: json_str, using: graph_decoder()) {
    Ok(graph) -> Ok(graph)
    Error(err) -> Error(ParseError(json_error_to_string(err)))
  }
}

/// Execute the graph against a dict of named input tensors.
///
/// The execution table starts with `initializers ∪ feeds`. Each node looks
/// up its inputs in that table, runs its op, and writes its outputs back so
/// later nodes can use them.
///
/// Returns the full output table (initializers, feeds, and all produced
/// intermediates) — callers can pick the named graph outputs from it.
///
/// Supported op set (v1): `Add`, `Sub`, `Mul`, `MatMul`, `Gemm`, `Relu`,
/// `Sigmoid`, `Tanh`, `Gelu`, `Softmax`, `Transpose`, `Reshape`, `Constant`,
/// `LayerNormalization`. Anything else returns `UnsupportedOp`.
pub fn run_graph(
  graph: OnnxGraph,
  feeds: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  let initial = merge_dicts(graph.initializers, feeds)
  run_nodes(graph.nodes, initial)
}

/// Return the list of ONNX op_types this module handles in v1.
///
/// Use this for capability discovery / pre-flight validation before calling
/// `run_graph`.
pub fn supported_ops() -> List(String) {
  [
    "Add", "Sub", "Mul", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh", "Gelu",
    "Softmax", "Transpose", "Reshape", "Constant", "LayerNormalization",
  ]
}

// --- JSON decoding ---------------------------------------------------------

fn graph_decoder() -> decode.Decoder(OnnxGraph) {
  use nodes <- decode.field("nodes", decode.list(node_decoder()))
  use inputs <- decode.optional_field("inputs", [], decode.list(decode.string))
  use outputs <- decode.optional_field(
    "outputs",
    [],
    decode.list(decode.string),
  )
  use initializers <- decode.optional_field(
    "initializers",
    dict.new(),
    decode.dict(decode.string, tensor_decoder()),
  )
  decode.success(OnnxGraph(nodes, inputs, outputs, initializers))
}

fn node_decoder() -> decode.Decoder(OnnxNode) {
  use op_type <- decode.field("op_type", decode.string)
  use name <- decode.optional_field("name", "", decode.string)
  use inputs <- decode.optional_field("inputs", [], decode.list(decode.string))
  use outputs <- decode.optional_field(
    "outputs",
    [],
    decode.list(decode.string),
  )
  use attributes <- decode.optional_field(
    "attributes",
    dict.new(),
    decode.dict(decode.string, attribute_decoder()),
  )
  decode.success(OnnxNode(op_type, name, inputs, outputs, attributes))
}

fn attribute_decoder() -> decode.Decoder(OnnxAttribute) {
  use type_tag <- decode.field("type", decode.string)
  case type_tag {
    "int" -> {
      use v <- decode.field("value", decode.int)
      decode.success(IntAttr(v))
    }
    "float" -> {
      // Accept both `1.0` and `1` for floats — JSON often loses the dot.
      use v <- decode.field("value", float_or_int_as_float())
      decode.success(FloatAttr(v))
    }
    "string" -> {
      use v <- decode.field("value", decode.string)
      decode.success(StringAttr(v))
    }
    "ints" -> {
      use v <- decode.field("value", decode.list(decode.int))
      decode.success(IntsAttr(v))
    }
    "floats" -> {
      use v <- decode.field("value", decode.list(float_or_int_as_float()))
      decode.success(FloatsAttr(v))
    }
    "tensor" -> {
      use v <- decode.field("value", tensor_decoder())
      decode.success(TensorAttr(v))
    }
    other ->
      decode.failure(
        IntAttr(0),
        "ONNX attribute type must be one of int|float|string|ints|floats|tensor, got "
          <> other,
      )
  }
}

fn float_or_int_as_float() -> decode.Decoder(Float) {
  decode.one_of(decode.float, [decode.map(decode.int, int.to_float)])
}

fn tensor_decoder() -> decode.Decoder(Tensor) {
  use shape <- decode.field("shape", decode.list(decode.int))
  use data <- decode.field("data", decode.list(float_or_int_as_float()))
  decode.success(Tensor(data: data, shape: shape))
}

fn json_error_to_string(err: json.DecodeError) -> String {
  case err {
    json.UnexpectedEndOfInput -> "unexpected end of JSON input"
    json.UnexpectedByte(byte) -> "unexpected byte " <> byte
    json.UnexpectedSequence(seq) -> "unexpected sequence " <> seq
    json.UnableToDecode(errors) -> "schema mismatch: " <> describe_errors(errors)
  }
}

fn describe_errors(errors: List(decode.DecodeError)) -> String {
  errors
  |> list.map(fn(e) {
    let decode.DecodeError(expected, found, path) = e
    "expected="
    <> expected
    <> " found="
    <> found
    <> " at="
    <> string.join(path, ".")
  })
  |> string.join("; ")
}

// --- Execution -------------------------------------------------------------

fn run_nodes(
  nodes: List(OnnxNode),
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case nodes {
    [] -> Ok(table)
    [node, ..rest] -> {
      use new_table <- result.try(run_node(node, table))
      run_nodes(rest, new_table)
    }
  }
}

fn run_node(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.op_type {
    "Add" -> binary_op(node, table, tensor.add_broadcast)
    "Sub" -> binary_op(node, table, tensor.sub_broadcast)
    "Mul" -> binary_op(node, table, tensor.mul_broadcast)
    "MatMul" -> binary_op(node, table, tensor.matmul)
    "Gemm" -> run_gemm(node, table)
    "Relu" -> unary_op(node, table, fn(t) { Ok(activations.relu(t)) })
    "Sigmoid" -> unary_op(node, table, fn(t) { Ok(activations.sigmoid(t)) })
    "Tanh" -> unary_op(node, table, fn(t) { Ok(activations.tanh(t)) })
    "Gelu" -> unary_op(node, table, fn(t) { Ok(activations.gelu(t)) })
    "Softmax" -> run_softmax(node, table)
    "Transpose" -> run_transpose(node, table)
    "Reshape" -> run_reshape(node, table)
    "Constant" -> run_constant(node, table)
    "LayerNormalization" -> run_layer_norm(node, table)
    other -> Error(UnsupportedOp(other))
  }
}

// --- Op helpers ------------------------------------------------------------

fn lookup(
  table: Dict(String, Tensor),
  name: String,
) -> Result(Tensor, OnnxError) {
  case dict.get(table, name) {
    Ok(t) -> Ok(t)
    Error(_) -> Error(MissingInput(name))
  }
}

fn first_output(node: OnnxNode) -> Result(String, OnnxError) {
  case node.outputs {
    [name, ..] -> Ok(name)
    [] ->
      Error(ShapeError(
        "node " <> node.op_type <> "/" <> node.name <> " has no outputs",
      ))
  }
}

fn write_output(
  table: Dict(String, Tensor),
  node: OnnxNode,
  value: Tensor,
) -> Result(Dict(String, Tensor), OnnxError) {
  use name <- result.try(first_output(node))
  Ok(dict.insert(table, name, value))
}

fn unary_op(
  node: OnnxNode,
  table: Dict(String, Tensor),
  op: fn(Tensor) -> Result(Tensor, anything),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [x_name, ..] -> {
      use x <- result.try(lookup(table, x_name))
      case op(x) {
        Ok(v) -> write_output(table, node, v)
        Error(_) ->
          Error(ShapeError(node.op_type <> ": op failed on input " <> x_name))
      }
    }
    [] -> Error(ShapeError(node.op_type <> ": expected 1 input, got 0"))
  }
}

fn binary_op(
  node: OnnxNode,
  table: Dict(String, Tensor),
  op: fn(Tensor, Tensor) -> Result(Tensor, anything),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [a_name, b_name, ..] -> {
      use a <- result.try(lookup(table, a_name))
      use b <- result.try(lookup(table, b_name))
      case op(a, b) {
        Ok(v) -> write_output(table, node, v)
        Error(_) ->
          Error(ShapeError(
            node.op_type <> ": op failed on inputs " <> a_name <> ", " <> b_name,
          ))
      }
    }
    _ ->
      Error(ShapeError(
        node.op_type
          <> ": expected 2 inputs, got "
          <> int.to_string(list.length(node.inputs)),
      ))
  }
}

// --- Gemm: y = alpha * (op(A) @ op(B)) + beta * C --------------------------

fn run_gemm(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  let alpha = float_attr(node.attributes, "alpha", 1.0)
  let beta = float_attr(node.attributes, "beta", 1.0)
  let trans_a = int_attr(node.attributes, "transA", 0) != 0
  let trans_b = int_attr(node.attributes, "transB", 0) != 0
  case node.inputs {
    [a_name, b_name, ..rest] -> {
      use a_raw <- result.try(lookup(table, a_name))
      use b_raw <- result.try(lookup(table, b_name))
      use a <- result.try(maybe_transpose(a_raw, trans_a, "Gemm A"))
      use b <- result.try(maybe_transpose(b_raw, trans_b, "Gemm B"))
      use ab <- result.try(
        tensor.matmul(a, b)
        |> result.map_error(fn(_) { ShapeError("Gemm: matmul failed") }),
      )
      let ab_scaled = tensor.scale(ab, alpha)
      let final = case rest {
        [c_name, ..] -> {
          use c <- result.try(lookup(table, c_name))
          let c_scaled = tensor.scale(c, beta)
          tensor.add_broadcast(ab_scaled, c_scaled)
          |> result.map_error(fn(_) { ShapeError("Gemm: bias add failed") })
        }
        [] -> Ok(ab_scaled)
      }
      use out <- result.try(final)
      write_output(table, node, out)
    }
    _ -> Error(ShapeError("Gemm: expected at least 2 inputs (A, B)"))
  }
}

fn maybe_transpose(
  t: Tensor,
  do_transpose: Bool,
  label: String,
) -> Result(Tensor, OnnxError) {
  case do_transpose {
    False -> Ok(t)
    True ->
      tensor.transpose(t)
      |> result.map_error(fn(_) {
        ShapeError(label <> ": transpose requires 2D tensor")
      })
  }
}

// --- Softmax ---------------------------------------------------------------

fn run_softmax(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [x_name, ..] -> {
      use x <- result.try(lookup(table, x_name))
      // ONNX softmax default axis is -1 (last) since opset 13.
      let raw_axis = int_attr(node.attributes, "axis", -1)
      let rank = list.length(tensor.shape(x))
      let axis = case raw_axis < 0 {
        True -> rank + raw_axis
        False -> raw_axis
      }
      case axis >= 0 && axis < rank {
        False ->
          Error(ShapeError(
            "Softmax: axis "
              <> int.to_string(raw_axis)
              <> " out of bounds for rank "
              <> int.to_string(rank),
          ))
        True ->
          case activations.softmax(x, axis) {
            Ok(v) -> write_output(table, node, v)
            Error(_) -> Error(ShapeError("Softmax: forward failed"))
          }
      }
    }
    [] -> Error(ShapeError("Softmax: expected 1 input"))
  }
}

// --- Transpose -------------------------------------------------------------

fn run_transpose(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [x_name, ..] -> {
      use x <- result.try(lookup(table, x_name))
      let shape = tensor.shape(x)
      let rank = list.length(shape)
      let perm = case dict.get(node.attributes, "perm") {
        Ok(IntsAttr(p)) -> p
        _ -> list.reverse(list.range(0, rank - 1))
      }
      transpose_with_perm(x, shape, perm)
      |> result.try(fn(v) { write_output(table, node, v) })
    }
    [] -> Error(ShapeError("Transpose: expected 1 input"))
  }
}

fn transpose_with_perm(
  x: Tensor,
  shape: List(Int),
  perm: List(Int),
) -> Result(Tensor, OnnxError) {
  let rank = list.length(shape)
  case rank, perm, shape {
    // 2D explicit swap — delegate to tensor.transpose.
    2, [1, 0], _ ->
      tensor.transpose(x)
      |> result.map_error(fn(_) { ShapeError("Transpose: tensor.transpose failed") })
    // 1D or identity permutation — return as-is.
    _, _, _ -> {
      let identity = list.range(0, rank - 1)
      case perm == identity {
        True -> Ok(x)
        False ->
          Error(UnsupportedOp(
            "Transpose: only 2D swaps and identity permutations are supported in v1",
          ))
      }
    }
  }
}

// --- Reshape ---------------------------------------------------------------

fn run_reshape(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [data_name, shape_name, ..] -> {
      use data <- result.try(lookup(table, data_name))
      use shape_tensor <- result.try(lookup(table, shape_name))
      // ONNX encodes shape as an int64 tensor; in our JSON it arrives as a
      // Float tensor — round to int.
      let shape_floats = tensor.to_list(shape_tensor)
      let new_shape =
        list.map(shape_floats, fn(f) { float.round(f) })
      use resolved <- result.try(resolve_reshape_dims(
        new_shape,
        tensor.shape(data),
      ))
      case tensor.reshape(data, resolved) {
        Ok(v) -> write_output(table, node, v)
        Error(_) ->
          Error(ShapeError("Reshape: size mismatch with target shape"))
      }
    }
    _ -> Error(ShapeError("Reshape: expected 2 inputs (data, shape)"))
  }
}

fn resolve_reshape_dims(
  target: List(Int),
  source: List(Int),
) -> Result(List(Int), OnnxError) {
  let source_size = list.fold(source, 1, fn(acc, d) { acc * d })
  let negative_count = list.count(target, fn(d) { d == -1 })
  case negative_count {
    0 -> Ok(target)
    1 -> {
      let known =
        list.fold(target, 1, fn(acc, d) {
          case d == -1 {
            True -> acc
            False -> acc * d
          }
        })
      case known == 0 {
        True ->
          Error(ShapeError("Reshape: cannot infer dim with zero in target"))
        False -> {
          case source_size % known == 0 {
            True -> {
              let inferred = source_size / known
              Ok(list.map(target, fn(d) {
                case d == -1 {
                  True -> inferred
                  False -> d
                }
              }))
            }
            False ->
              Error(ShapeError(
                "Reshape: source size not divisible by known dims",
              ))
          }
        }
      }
    }
    _ -> Error(ShapeError("Reshape: only one -1 placeholder allowed"))
  }
}

// --- Constant --------------------------------------------------------------

fn run_constant(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case dict.get(node.attributes, "value") {
    Ok(TensorAttr(t)) -> write_output(table, node, t)
    _ ->
      Error(ShapeError(
        "Constant: missing or wrong-typed `value` attribute (expected tensor)",
      ))
  }
}

// --- LayerNormalization ----------------------------------------------------
//
// ONNX semantics (opset 17+):
//   X normalized along `axis` and all trailing axes; Y = (X - mean)/sqrt(var+eps) * scale + bias
//
// v1 here only supports the common "last-axis" case (axis = -1 or axis = rank-1)
// because the underlying viva_tensor LayerNorm layer normalizes along the last
// dimension. Other axes return UnsupportedOp.

fn run_layer_norm(
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  case node.inputs {
    [x_name, scale_name, ..rest] -> {
      use x <- result.try(lookup(table, x_name))
      use scale <- result.try(lookup(table, scale_name))
      let bias = case rest {
        [bias_name, ..] ->
          case lookup(table, bias_name) {
            Ok(b) -> Ok(b)
            Error(e) -> Error(e)
          }
        [] -> Ok(tensor.zeros_like(scale))
      }
      use bias_tensor <- result.try(bias)
      let raw_axis = int_attr(node.attributes, "axis", -1)
      let eps = float_attr(node.attributes, "epsilon", 1.0e-5)
      let rank = list.length(tensor.shape(x))
      let axis = case raw_axis < 0 {
        True -> rank + raw_axis
        False -> raw_axis
      }
      case axis == rank - 1 {
        False ->
          Error(UnsupportedOp(
            "LayerNormalization: only axis = -1 (last dim) is supported in v1",
          ))
        True -> layer_norm_last_axis(x, scale, bias_tensor, eps, node, table)
      }
    }
    _ ->
      Error(ShapeError(
        "LayerNormalization: expected at least 2 inputs (X, scale)",
      ))
  }
}

fn layer_norm_last_axis(
  x: Tensor,
  scale: Tensor,
  bias: Tensor,
  eps: Float,
  node: OnnxNode,
  table: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  let x_shape = tensor.shape(x)
  let scale_shape = tensor.shape(scale)
  let bias_shape = tensor.shape(bias)
  use last <- result.try(case list.last(x_shape) {
    Ok(d) -> Ok(d)
    Error(_) -> Error(ShapeError("LayerNormalization: input has no dimensions"))
  })
  case scale_shape == [last] && bias_shape == [last] {
    False ->
      Error(ShapeError(
        "LayerNormalization: scale/bias must have shape ["
          <> int.to_string(last)
          <> "]",
      ))
    True -> {
      let x_data = tensor.to_list(x)
      let scale_data = tensor.to_list(scale)
      let bias_data = tensor.to_list(bias)
      let normalized =
        chunk_by(x_data, last)
        |> list.map(fn(chunk) {
          let n = int.to_float(list.length(chunk))
          let sum = list.fold(chunk, 0.0, fn(acc, v) { acc +. v })
          let mean = sum /. n
          let var =
            list.fold(chunk, 0.0, fn(acc, v) {
              let d = v -. mean
              acc +. d *. d
            })
            /. n
          let denom = safe_sqrt(var +. eps)
          list.map(
            list.zip(chunk, list.zip(scale_data, bias_data)),
            fn(triple) {
              let #(v, sb) = triple
              let #(s, b) = sb
              { v -. mean } /. denom *. s +. b
            },
          )
        })
        |> list.flatten
      let out = Tensor(data: normalized, shape: x_shape)
      write_output(table, node, out)
    }
  }
}

fn safe_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn chunk_by(data: List(Float), n: Int) -> List(List(Float)) {
  case n <= 0, data {
    True, _ -> []
    _, [] -> []
    _, _ -> {
      let head = list.take(data, n)
      let tail = list.drop(data, n)
      [head, ..chunk_by(tail, n)]
    }
  }
}

// --- Attribute helpers -----------------------------------------------------

fn int_attr(attrs: Dict(String, OnnxAttribute), name: String, default: Int) -> Int {
  case dict.get(attrs, name) {
    Ok(IntAttr(v)) -> v
    _ -> default
  }
}

fn float_attr(
  attrs: Dict(String, OnnxAttribute),
  name: String,
  default: Float,
) -> Float {
  case dict.get(attrs, name) {
    Ok(FloatAttr(v)) -> v
    Ok(IntAttr(i)) -> int.to_float(i)
    _ -> default
  }
}

// --- Dict utilities --------------------------------------------------------

fn merge_dicts(
  a: Dict(String, Tensor),
  b: Dict(String, Tensor),
) -> Dict(String, Tensor) {
  dict.fold(b, a, fn(acc, k, v) { dict.insert(acc, k, v) })
}
