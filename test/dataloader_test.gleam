import gleam/list
import gleam/order
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/data/dataloader.{
  type Sample, Batch, Sample, data_loader_batches, data_loader_len,
  data_loader_new, dataset_from_lists, dataset_from_samples, dataset_get,
  dataset_len,
}
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// HELPERS
// =============================================================================

fn make_sample(input: List(Float), target: List(Float)) -> Sample {
  Sample(input: tensor.from_list(input), target: tensor.from_list(target))
}

fn three_samples() -> List(Sample) {
  [
    make_sample([1.0, 2.0], [10.0]),
    make_sample([3.0, 4.0], [20.0]),
    make_sample([5.0, 6.0], [30.0]),
  ]
}

// =============================================================================
// DATASET
// =============================================================================

pub fn dataset_from_samples_test() {
  let ds = dataset_from_samples(three_samples())
  dataset_len(ds) |> should.equal(3)
}

pub fn dataset_from_lists_test() {
  let xs = [
    tensor.from_list([1.0, 2.0]),
    tensor.from_list([3.0, 4.0]),
  ]
  let ys = [tensor.from_list([10.0]), tensor.from_list([20.0])]
  let assert Ok(ds) = dataset_from_lists(xs, ys)
  dataset_len(ds) |> should.equal(2)

  let assert Ok(sample) = dataset_get(ds, 0)
  numerics.lists_close(tensor.to_list(sample.input), [1.0, 2.0], 1.0e-9, 1.0e-9)
  |> should.be_true
  numerics.lists_close(tensor.to_list(sample.target), [10.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn dataset_from_lists_size_mismatch_test() {
  let xs = [tensor.from_list([1.0]), tensor.from_list([2.0])]
  let ys = [tensor.from_list([10.0])]
  dataset_from_lists(xs, ys)
  |> should.be_error
}

pub fn dataset_get_test() {
  let ds = dataset_from_samples(three_samples())
  let assert Ok(sample) = dataset_get(ds, 1)
  numerics.lists_close(tensor.to_list(sample.input), [3.0, 4.0], 1.0e-9, 1.0e-9)
  |> should.be_true
  numerics.floats_close(
    list_first(tensor.to_list(sample.target)),
    20.0,
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn dataset_get_negative_index_test() {
  let ds = dataset_from_samples(three_samples())
  let assert Ok(sample) = dataset_get(ds, -1)
  numerics.lists_close(tensor.to_list(sample.input), [5.0, 6.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn dataset_get_oob_test() {
  let ds = dataset_from_samples(three_samples())
  dataset_get(ds, 3) |> should.be_error
  dataset_get(ds, -4) |> should.be_error
}

// =============================================================================
// DATALOADER
// =============================================================================

pub fn data_loader_one_batch_test() {
  let samples = [
    make_sample([1.0, 2.0], [1.0]),
    make_sample([3.0, 4.0], [2.0]),
    make_sample([5.0, 6.0], [3.0]),
    make_sample([7.0, 8.0], [4.0]),
  ]
  let ds = dataset_from_samples(samples)
  let loader = data_loader_new(ds, 4, False, False)
  let assert Ok(batches) = data_loader_batches(loader)
  list.length(batches) |> should.equal(1)
  let assert [Batch(inputs, targets)] = batches
  tensor.shape(inputs) |> should.equal([4, 2])
  tensor.shape(targets) |> should.equal([4, 1])
  numerics.lists_close(
    tensor.to_list(inputs),
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
  numerics.lists_close(
    tensor.to_list(targets),
    [1.0, 2.0, 3.0, 4.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn data_loader_partial_batch_test() {
  let samples = [
    make_sample([1.0], [1.0]),
    make_sample([2.0], [2.0]),
    make_sample([3.0], [3.0]),
    make_sample([4.0], [4.0]),
    make_sample([5.0], [5.0]),
  ]
  let ds = dataset_from_samples(samples)
  let loader = data_loader_new(ds, 2, False, False)
  let assert Ok(batches) = data_loader_batches(loader)
  list.length(batches) |> should.equal(3)
  let sizes =
    list.map(batches, fn(b) {
      let Batch(inputs, _) = b
      case tensor.shape(inputs) {
        [n, ..] -> n
        [] -> 0
      }
    })
  sizes |> should.equal([2, 2, 1])
}

pub fn data_loader_drop_last_test() {
  let samples = [
    make_sample([1.0], [1.0]),
    make_sample([2.0], [2.0]),
    make_sample([3.0], [3.0]),
    make_sample([4.0], [4.0]),
    make_sample([5.0], [5.0]),
  ]
  let ds = dataset_from_samples(samples)
  let loader = data_loader_new(ds, 2, False, True)
  let assert Ok(batches) = data_loader_batches(loader)
  list.length(batches) |> should.equal(2)
}

pub fn data_loader_shuffle_test() {
  let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let samples = list.map(inputs, fn(v) { make_sample([v], [v]) })
  let ds = dataset_from_samples(samples)
  let loader = data_loader_new(ds, 8, True, False)
  let assert Ok(batches) = data_loader_batches(loader)
  let assert [Batch(batch_inputs, _)] = batches
  let observed = tensor.to_list(batch_inputs)
  // Contents preserved (same multiset).
  let sorted_obs = list.sort(observed, by: float_compare)
  let sorted_exp = list.sort(inputs, by: float_compare)
  numerics.lists_close(sorted_obs, sorted_exp, 1.0e-9, 1.0e-9) |> should.be_true
  // Length preserved.
  list.length(observed) |> should.equal(list.length(inputs))
}

pub fn data_loader_len_test() {
  let samples = [
    make_sample([1.0], [1.0]),
    make_sample([2.0], [2.0]),
    make_sample([3.0], [3.0]),
    make_sample([4.0], [4.0]),
    make_sample([5.0], [5.0]),
  ]
  let ds = dataset_from_samples(samples)
  data_loader_len(data_loader_new(ds, 2, False, False)) |> should.equal(3)
  data_loader_len(data_loader_new(ds, 2, False, True)) |> should.equal(2)
  data_loader_len(data_loader_new(ds, 5, False, False)) |> should.equal(1)
  data_loader_len(data_loader_new(ds, 10, False, False)) |> should.equal(1)
  data_loader_len(data_loader_new(ds, 10, False, True)) |> should.equal(0)
}

// --- local helpers ----------------------------------------------------------

fn list_first(xs: List(Float)) -> Float {
  case xs {
    [head, ..] -> head
    [] -> 0.0
  }
}

fn float_compare(a: Float, b: Float) -> order.Order {
  case a <. b {
    True -> order.Lt
    False ->
      case a >. b {
        True -> order.Gt
        False -> order.Eq
      }
  }
}
