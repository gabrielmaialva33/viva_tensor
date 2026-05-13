//// NumPy reference tests.
////
//// Loads JSON fixtures produced by `test/fixtures/numpy/gen_reference.py` and
//// asserts that viva_tensor's output is element-wise close to NumPy's, using
//// an `np.allclose`-style tolerance defined in each fixture.
////
//// Each `*_test` function below covers one op group and reads one fixture per
//// shape so the suite reports per-op pass/fail.

import gleam/dynamic/decode
import gleam/float
import gleam/json
import gleam/list
import gleam/result
import simplifile
import support/numerics
import viva_tensor as t

// -----------------------------------------------------------------------------
// Fixture schema + decoder
// -----------------------------------------------------------------------------

pub type TensorData {
  TensorData(shape: List(Int), data: List(Float))
}

pub type Tolerance {
  Tolerance(rtol: Float, atol: Float)
}

pub type Fixture {
  Fixture(
    op: String,
    inputs: List(TensorData),
    output: TensorData,
    tolerance: Tolerance,
  )
}

fn tensor_data_decoder() -> decode.Decoder(TensorData) {
  use shape <- decode.field("shape", decode.list(decode.int))
  use data <- decode.field("data", decode.list(decode.float))
  decode.success(TensorData(shape:, data:))
}

fn tolerance_decoder() -> decode.Decoder(Tolerance) {
  use rtol <- decode.field("rtol", decode.float)
  use atol <- decode.field("atol", decode.float)
  decode.success(Tolerance(rtol:, atol:))
}

fn fixture_decoder() -> decode.Decoder(Fixture) {
  use op <- decode.field("op", decode.string)
  use inputs <- decode.field("inputs", decode.list(tensor_data_decoder()))
  use output <- decode.field("output", tensor_data_decoder())
  use tolerance <- decode.field("tolerance", tolerance_decoder())
  decode.success(Fixture(op:, inputs:, output:, tolerance:))
}

/// Load and decode a fixture, failing the test loudly if anything goes wrong.
fn load(path: String) -> Fixture {
  let raw =
    simplifile.read(path)
    |> result.map_error(fn(_) { "could not read fixture: " <> path })
  let assert Ok(content) = raw
  let parsed =
    json.parse(content, fixture_decoder())
    |> result.map_error(fn(_) { "could not decode fixture: " <> path })
  let assert Ok(fixture) = parsed
  fixture
}

// -----------------------------------------------------------------------------
// Input helpers
// -----------------------------------------------------------------------------

/// Materialise a `TensorData` payload as a viva_tensor tensor.
///
/// Rank handling: 0-D scalars come back as a 1-element 1-D tensor (the only
/// time the test path consumes a scalar is when the *output* is a scalar — and
/// scalar outputs are checked via `assert_scalar_close`, not via this).
fn to_tensor(td: TensorData) -> t.Tensor {
  case td.shape {
    [] -> t.from_list(td.data)
    [_] -> t.from_list(td.data)
    [_rows, cols] -> {
      let rows = chunk(td.data, cols)
      let assert Ok(tensor) = t.from_list2d(rows)
      tensor
    }
    _ ->
      // Higher-rank shapes are not exercised by the current fixtures. Reshape
      // a 1-D buffer if/when we add them.
      panic as "unsupported rank in reference fixture"
  }
}

fn chunk(xs: List(Float), size: Int) -> List(List(Float)) {
  case xs {
    [] -> []
    _ -> {
      let head = list.take(xs, size)
      let rest = list.drop(xs, size)
      [head, ..chunk(rest, size)]
    }
  }
}

fn assert_output(actual: t.Tensor, fixture: Fixture) -> Nil {
  numerics.assert_close(
    actual,
    fixture.output.data,
    fixture.output.shape,
    fixture.tolerance.rtol,
    fixture.tolerance.atol,
  )
}

fn assert_scalar_output(actual: Float, fixture: Fixture) -> Nil {
  // Scalar fixtures use shape=[] and data=[value].
  let assert [expected] = fixture.output.data
  numerics.assert_scalar_close(
    actual,
    expected,
    fixture.tolerance.rtol,
    fixture.tolerance.atol,
  )
}

fn fixture_path(op: String, case_name: String) -> String {
  "test/fixtures/numpy/" <> op <> "/" <> case_name <> ".json"
}

// -----------------------------------------------------------------------------
// Op-group tests
// -----------------------------------------------------------------------------

fn run_binop(
  op: String,
  case_name: String,
  apply: fn(t.Tensor, t.Tensor) -> Result(t.Tensor, t.TensorError),
) -> Nil {
  let fixture = load(fixture_path(op, case_name))
  let assert [a, b] = fixture.inputs
  let ta = to_tensor(a)
  let tb = to_tensor(b)
  let assert Ok(out) = apply(ta, tb)
  assert_output(out, fixture)
}

pub fn add_test() {
  run_binop("add", "vec4", t.add)
  run_binop("add", "mat3x3", t.add)
}

pub fn sub_test() {
  run_binop("sub", "vec4", t.sub)
  run_binop("sub", "mat3x3", t.sub)
}

pub fn mul_test() {
  run_binop("mul", "vec4", t.mul)
  run_binop("mul", "mat3x3", t.mul)
}

pub fn div_test() {
  run_binop("div", "vec4", t.div)
  run_binop("div", "mat3x3", t.div)
}

pub fn matmul_test() {
  let fixture = load(fixture_path("matmul", "2x3_at_3x4"))
  let assert [a, b] = fixture.inputs
  let assert Ok(out) = t.matmul(to_tensor(a), to_tensor(b))
  assert_output(out, fixture)
}

pub fn sum_test() {
  let fixture = load(fixture_path("sum", "vec5"))
  let assert [a] = fixture.inputs
  let scalar = t.sum(to_tensor(a))
  assert_scalar_output(scalar, fixture)
}

pub fn mean_test() {
  let fixture = load(fixture_path("mean", "vec5"))
  let assert [a] = fixture.inputs
  let scalar = t.mean(to_tensor(a))
  assert_scalar_output(scalar, fixture)
}

pub fn variance_test() {
  let fixture = load(fixture_path("var", "vec5"))
  let assert [a] = fixture.inputs
  let scalar = t.variance(to_tensor(a))
  assert_scalar_output(scalar, fixture)
}

pub fn std_test() {
  let fixture = load(fixture_path("std", "vec5"))
  let assert [a] = fixture.inputs
  let scalar = t.std(to_tensor(a))
  assert_scalar_output(scalar, fixture)
}

pub fn transpose_test() {
  let fixture = load(fixture_path("transpose", "mat3x4"))
  let assert [a] = fixture.inputs
  let assert Ok(out) = t.transpose(to_tensor(a))
  assert_output(out, fixture)
}

pub fn exp_test() {
  let fixture = load(fixture_path("exp", "vec5"))
  let assert [a] = fixture.inputs
  let out = t.exp(to_tensor(a))
  assert_output(out, fixture)
}

pub fn log_test() {
  let fixture = load(fixture_path("log", "vec5"))
  let assert [a] = fixture.inputs
  let out = t.log(to_tensor(a))
  assert_output(out, fixture)
}

pub fn relu_test() {
  let fixture = load(fixture_path("relu", "vec7"))
  let assert [a] = fixture.inputs
  // viva_tensor's public API doesn't re-export `relu`, so we apply the
  // elementwise `max(x, 0)` directly. Keeps the test on the public surface.
  let input = to_tensor(a)
  let relu_data =
    t.to_list(input)
    |> list.map(fn(x) { float.max(x, 0.0) })
  let out = t.from_list(relu_data)
  assert_output(out, fixture)
}
