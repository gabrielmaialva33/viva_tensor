import gleam/dict
import gleam/list
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/io/onnx
import viva_tensor/tensor.{Tensor}

// --- Parsing ---------------------------------------------------------------

pub fn parse_simple_graph_test() {
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Add\",\"name\":\"add1\","
    <> "\"inputs\":[\"a\",\"b\"],\"outputs\":[\"c\"],\"attributes\":{}}],"
    <> "\"inputs\":[\"a\",\"b\"],\"outputs\":[\"c\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  list.length(graph.nodes) |> should.equal(1)
  graph.inputs |> should.equal(["a", "b"])
  graph.outputs |> should.equal(["c"])
  let assert [node] = graph.nodes
  node.op_type |> should.equal("Add")
  node.name |> should.equal("add1")
  node.inputs |> should.equal(["a", "b"])
  node.outputs |> should.equal(["c"])
}

pub fn parse_with_initializer_test() {
  let json_str =
    "{\"nodes\":[],\"inputs\":[],\"outputs\":[\"W\"],"
    <> "\"initializers\":{\"W\":{\"shape\":[2,2],\"data\":[1.0,2.0,3.0,4.0]}}}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let assert Ok(w) = dict.get(graph.initializers, "W")
  t.shape(w) |> should.equal([2, 2])
  t.to_list(w) |> should.equal([1.0, 2.0, 3.0, 4.0])
}

pub fn parse_invalid_json_test() {
  let assert Error(err) = onnx.parse_graph("{not valid json")
  case err {
    onnx.ParseError(_) -> Nil
    _ -> should.equal("expected ParseError", "")
  }
}

// --- Execution -------------------------------------------------------------

pub fn run_add_test() {
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Add\",\"name\":\"\","
    <> "\"inputs\":[\"x\",\"y\"],\"outputs\":[\"z\"]}],"
    <> "\"inputs\":[\"x\",\"y\"],\"outputs\":[\"z\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("x", Tensor(data: [1.0, 2.0, 3.0], shape: [3])),
      #("y", Tensor(data: [10.0, 20.0, 30.0], shape: [3])),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(z) = dict.get(outputs, "z")
  t.shape(z) |> should.equal([3])
  t.to_list(z) |> should.equal([11.0, 22.0, 33.0])
}

pub fn run_gemm_test() {
  // y = alpha * A @ B + beta * C
  // A = [[1,2],[3,4]]  shape [2,2]
  // B = [[5,6],[7,8]]  shape [2,2]
  // A @ B = [[19,22],[43,50]]
  // alpha=2, beta=0.5, C = [1,1] (broadcast)
  // y = 2 * [[19,22],[43,50]] + 0.5 * [1,1]
  //   = [[38.5, 44.5], [86.5, 100.5]]
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Gemm\",\"name\":\"g\","
    <> "\"inputs\":[\"A\",\"B\",\"C\"],\"outputs\":[\"Y\"],"
    <> "\"attributes\":{"
    <> "\"alpha\":{\"type\":\"float\",\"value\":2.0},"
    <> "\"beta\":{\"type\":\"float\",\"value\":0.5}"
    <> "}}],"
    <> "\"inputs\":[\"A\",\"B\",\"C\"],\"outputs\":[\"Y\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("A", Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])),
      #("B", Tensor(data: [5.0, 6.0, 7.0, 8.0], shape: [2, 2])),
      #("C", Tensor(data: [1.0, 1.0], shape: [2])),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(y) = dict.get(outputs, "Y")
  t.shape(y) |> should.equal([2, 2])
  numerics.lists_close(
    t.to_list(y),
    [38.5, 44.5, 86.5, 100.5],
    1.0e-6,
    1.0e-9,
  )
  |> should.be_true
}

pub fn run_relu_test() {
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Relu\",\"name\":\"r\","
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}],"
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("x", Tensor(data: [-1.0, 0.0, 2.0, -3.0], shape: [4])),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(y) = dict.get(outputs, "y")
  t.to_list(y) |> should.equal([0.0, 0.0, 2.0, 0.0])
}

pub fn run_softmax_axis_test() {
  // softmax along axis 1 of a [2,3] tensor; each row sums to 1.
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Softmax\",\"name\":\"s\","
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"],"
    <> "\"attributes\":{\"axis\":{\"type\":\"int\",\"value\":1}}}],"
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #(
        "x",
        Tensor(data: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], shape: [2, 3]),
      ),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(y) = dict.get(outputs, "y")
  t.shape(y) |> should.equal([2, 3])

  let data = t.to_list(y)
  // row 0 and row 1 are identical -> ~[0.0900, 0.2447, 0.6652]
  let expected = [
    0.09003057317038046, 0.24472847105479764, 0.6652409557748219,
    0.09003057317038046, 0.24472847105479764, 0.6652409557748219,
  ]
  numerics.lists_close(data, expected, 1.0e-5, 1.0e-8) |> should.be_true
}

pub fn run_chained_test() {
  // h = MatMul(x, W); y = Relu(h)
  // x = [[1, -1]]    shape [1,2]
  // W = [[1, 2], [3, 4]]    shape [2,2]
  // h = [[-2, -2]]
  // y = [[0, 0]]
  let json_str =
    "{\"nodes\":["
    <> "{\"op_type\":\"MatMul\",\"name\":\"mm\","
    <> "\"inputs\":[\"x\",\"W\"],\"outputs\":[\"h\"]},"
    <> "{\"op_type\":\"Relu\",\"name\":\"r\","
    <> "\"inputs\":[\"h\"],\"outputs\":[\"y\"]}"
    <> "],"
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"],"
    <> "\"initializers\":{\"W\":{\"shape\":[2,2],\"data\":[1.0,2.0,3.0,4.0]}}}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("x", Tensor(data: [1.0, -1.0], shape: [1, 2])),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(y) = dict.get(outputs, "y")
  t.shape(y) |> should.equal([1, 2])
  t.to_list(y) |> should.equal([0.0, 0.0])
}

pub fn run_layer_norm_test() {
  // axis=-1, scale=[1,1,1,1], bias=[0,0,0,0]
  // For a single row [1,2,3,4]:
  //   mean = 2.5, var = ((1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 4) = 1.25
  //   denom = sqrt(1.25 + eps) ~ 1.118034
  //   y = (x - 2.5) / 1.118034
  //     ~ [-1.34164, -0.44721, 0.44721, 1.34164]
  let json_str =
    "{\"nodes\":[{\"op_type\":\"LayerNormalization\",\"name\":\"ln\","
    <> "\"inputs\":[\"x\",\"scale\",\"bias\"],\"outputs\":[\"y\"],"
    <> "\"attributes\":{"
    <> "\"axis\":{\"type\":\"int\",\"value\":-1},"
    <> "\"epsilon\":{\"type\":\"float\",\"value\":1.0e-5}"
    <> "}}],"
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"],"
    <> "\"initializers\":{"
    <> "\"scale\":{\"shape\":[4],\"data\":[1.0,1.0,1.0,1.0]},"
    <> "\"bias\":{\"shape\":[4],\"data\":[0.0,0.0,0.0,0.0]}"
    <> "}}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("x", Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [1, 4])),
    ])
  let assert Ok(outputs) = onnx.run_graph(graph, feeds)
  let assert Ok(y) = dict.get(outputs, "y")
  t.shape(y) |> should.equal([1, 4])
  let data = t.to_list(y)
  let expected = [
    -1.3416407413193976, -0.4472135804397992, 0.4472135804397992,
    1.3416407413193976,
  ]
  numerics.lists_close(data, expected, 1.0e-3, 1.0e-4) |> should.be_true
}

pub fn run_unsupported_op_test() {
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Conv\",\"name\":\"c\","
    <> "\"inputs\":[\"x\",\"w\"],\"outputs\":[\"y\"]}],"
    <> "\"inputs\":[\"x\"],\"outputs\":[\"y\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let feeds =
    dict.from_list([
      #("x", Tensor(data: [1.0], shape: [1])),
      #("w", Tensor(data: [1.0], shape: [1])),
    ])
  let assert Error(err) = onnx.run_graph(graph, feeds)
  case err {
    onnx.UnsupportedOp("Conv") -> Nil
    _ -> should.equal("expected UnsupportedOp(Conv)", "")
  }
}

pub fn run_missing_input_test() {
  let json_str =
    "{\"nodes\":[{\"op_type\":\"Add\",\"name\":\"a\","
    <> "\"inputs\":[\"missing_a\",\"missing_b\"],\"outputs\":[\"y\"]}],"
    <> "\"inputs\":[],\"outputs\":[\"y\"]}"

  let assert Ok(graph) = onnx.parse_graph(json_str)
  let assert Error(err) = onnx.run_graph(graph, dict.new())
  case err {
    onnx.MissingInput("missing_a") -> Nil
    _ -> should.equal("expected MissingInput(missing_a)", "")
  }
}

pub fn supported_ops_test() {
  let ops = onnx.supported_ops()
  list.contains(ops, "Add") |> should.be_true
  list.contains(ops, "Gemm") |> should.be_true
  list.contains(ops, "LayerNormalization") |> should.be_true
  list.contains(ops, "Conv") |> should.be_false
}
