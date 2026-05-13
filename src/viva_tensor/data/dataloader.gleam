//// Dataset and DataLoader abstractions for batched tensor iteration.
////
//// This module provides a minimal training-loop substrate: a `Dataset` holds
//// labeled `Sample`s, and a `DataLoader` yields `Batch`es with stacked input
//// and target tensors along axis 0. Pure Gleam, no NIF, no autograd.
////
//// Shuffle determinism: shuffling uses `int.random` which is the BEAM's
//// non-seedable PRNG. Iteration order is therefore non-deterministic across
//// runs. If you need reproducible shuffles, pre-shuffle samples upstream.

import gleam/int
import gleam/list
import gleam/order
import gleam/result
import viva_tensor/core/error.{
  type TensorError, IndexOutOfBounds, ShapeMismatch,
}
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Types ------------------------------------------------------------------

/// A labeled training example: an input tensor paired with a target tensor.
///
/// ## Example
///
/// ```gleam
/// let x = viva_tensor/tensor.from_list([1.0, 2.0])
/// let y = viva_tensor/tensor.from_list([1.0])
/// let _sample = Sample(input: x, target: y)
/// ```
pub type Sample {
  Sample(input: Tensor, target: Tensor)
}

/// In-memory dataset wrapping a list of labeled samples.
///
/// Constructed via `dataset_from_samples` or `dataset_from_lists`.
pub opaque type Dataset {
  Dataset(samples: List(Sample))
}

/// A batch carries stacked input and target tensors. Both tensors have an
/// extra leading axis (`batch_size`).
///
/// ## Example
///
/// ```gleam
/// // Inputs of shape [2] stacked into a batch of 3 → inputs.shape == [3, 2].
/// ```
pub type Batch {
  Batch(inputs: Tensor, targets: Tensor)
}

/// Iterator-style data loader over a `Dataset`.
///
/// Fields:
/// - `dataset`     – samples to iterate.
/// - `batch_size`  – samples per batch (must be > 0).
/// - `shuffle`     – when `True`, sample order is randomized on each call to
///   `data_loader_batches`.
/// - `drop_last`   – when `True`, a trailing partial batch is discarded.
pub type DataLoader {
  DataLoader(
    dataset: Dataset,
    batch_size: Int,
    shuffle: Bool,
    drop_last: Bool,
  )
}

// --- Dataset constructors ---------------------------------------------------

/// Build an in-memory dataset from a list of labeled samples.
///
/// ## Example
///
/// ```gleam
/// let x = viva_tensor/tensor.from_list([1.0])
/// let y = viva_tensor/tensor.from_list([0.0])
/// let _ds = dataset_from_samples([Sample(input: x, target: y)])
/// ```
pub fn dataset_from_samples(samples: List(Sample)) -> Dataset {
  Dataset(samples: samples)
}

/// Build a dataset from parallel input and target tensors. Each input becomes
/// one sample paired with the target at the same index.
///
/// Returns `Error(ShapeMismatch)` when the two lists have different lengths,
/// or when the inputs (resp. targets) have inconsistent tensor shapes.
///
/// ## Example
///
/// ```gleam
/// let xs = [viva_tensor/tensor.from_list([1.0])]
/// let ys = [viva_tensor/tensor.from_list([0.0])]
/// let assert Ok(_ds) = dataset_from_lists(xs, ys)
/// ```
pub fn dataset_from_lists(
  inputs: List(Tensor),
  targets: List(Tensor),
) -> Result(Dataset, TensorError) {
  let n_inputs = list.length(inputs)
  let n_targets = list.length(targets)
  case n_inputs == n_targets {
    False ->
      Error(ShapeMismatch(expected: [n_inputs], got: [n_targets]))
    True -> {
      use _ <- result.try(validate_uniform_shapes(inputs, "input"))
      use _ <- result.try(validate_uniform_shapes(targets, "target"))
      let samples =
        list.map2(inputs, targets, fn(x, y) {
          Sample(input: x, target: y)
        })
      Ok(Dataset(samples: samples))
    }
  }
}

/// Number of samples in the dataset.
///
/// ## Example
///
/// ```gleam
/// let ds = dataset_from_samples([])
/// let _zero = dataset_len(ds)
/// ```
pub fn dataset_len(d: Dataset) -> Int {
  list.length(d.samples)
}

/// Fetch the i-th sample (zero-indexed). Negative indices wrap from the end
/// (`-1` is the last sample).
///
/// Returns `Error(IndexOutOfBounds)` when `|index|` exceeds the dataset
/// length.
///
/// ## Example
///
/// ```gleam
/// let x = viva_tensor/tensor.from_list([1.0])
/// let y = viva_tensor/tensor.from_list([0.0])
/// let ds = dataset_from_samples([Sample(input: x, target: y)])
/// let assert Ok(_first) = dataset_get(ds, 0)
/// ```
pub fn dataset_get(d: Dataset, index: Int) -> Result(Sample, TensorError) {
  let n = list.length(d.samples)
  case n {
    0 -> Error(IndexOutOfBounds(index: index, size: 0))
    _ -> {
      let resolved = case index < 0 {
        True -> index + n
        False -> index
      }
      case resolved >= 0 && resolved < n {
        False -> Error(IndexOutOfBounds(index: index, size: n))
        True ->
          case list_at(d.samples, resolved) {
            Ok(sample) -> Ok(sample)
            Error(_) -> Error(IndexOutOfBounds(index: index, size: n))
          }
      }
    }
  }
}

// --- DataLoader -------------------------------------------------------------

/// Create a new data loader. `batch_size` must be > 0.
///
/// ## Example
///
/// ```gleam
/// let ds = dataset_from_samples([])
/// let _loader = data_loader_new(ds, 32, True, False)
/// ```
pub fn data_loader_new(
  dataset: Dataset,
  batch_size: Int,
  shuffle: Bool,
  drop_last: Bool,
) -> DataLoader {
  DataLoader(
    dataset: dataset,
    batch_size: batch_size,
    shuffle: shuffle,
    drop_last: drop_last,
  )
}

/// Iterate the loader once, returning all batches.
///
/// When `loader.shuffle` is `True`, the underlying sample order is randomized
/// for this call. When `loader.drop_last` is `True` and the trailing batch
/// would be smaller than `batch_size`, it is dropped. Otherwise the last batch
/// may contain fewer samples than `batch_size`.
///
/// Returns `Error(...)` if `batch_size <= 0` or if stacking fails because of
/// shape inconsistencies.
///
/// ## Example
///
/// ```gleam
/// let x = viva_tensor/tensor.from_list([1.0])
/// let y = viva_tensor/tensor.from_list([0.0])
/// let ds = dataset_from_samples([Sample(input: x, target: y)])
/// let loader = data_loader_new(ds, 1, False, False)
/// let assert Ok(_batches) = data_loader_batches(loader)
/// ```
pub fn data_loader_batches(
  loader: DataLoader,
) -> Result(List(Batch), TensorError) {
  case loader.batch_size <= 0 {
    True ->
      Error(error.InvalidShape("batch_size must be > 0"))
    False -> {
      let samples = loader.dataset.samples
      let ordered = case loader.shuffle {
        True -> shuffle_samples(samples)
        False -> samples
      }
      let groups = chunk(ordered, loader.batch_size)
      let kept = case loader.drop_last {
        True ->
          list.filter(groups, fn(g) {
            list.length(g) == loader.batch_size
          })
        False -> groups
      }
      stack_groups(kept, [])
    }
  }
}

/// Total number of batches a single iteration will yield.
///
/// ## Example
///
/// ```gleam
/// let ds = dataset_from_samples([])
/// let loader = data_loader_new(ds, 4, False, False)
/// let _zero = data_loader_len(loader)
/// ```
pub fn data_loader_len(loader: DataLoader) -> Int {
  case loader.batch_size <= 0 {
    True -> 0
    False -> {
      let n = list.length(loader.dataset.samples)
      let full = n / loader.batch_size
      let remainder = n % loader.batch_size
      case remainder == 0 || loader.drop_last {
        True -> full
        False -> full + 1
      }
    }
  }
}

// --- Internal helpers -------------------------------------------------------

fn list_at(xs: List(a), index: Int) -> Result(a, Nil) {
  case xs, index {
    [], _ -> Error(Nil)
    [head, ..], 0 -> Ok(head)
    [_, ..rest], _ -> list_at(rest, index - 1)
  }
}

fn validate_uniform_shapes(
  tensors: List(Tensor),
  _label: String,
) -> Result(Nil, TensorError) {
  case tensors {
    [] -> Ok(Nil)
    [first, ..rest] -> {
      let expected = tensor.shape(first)
      case
        list.find(rest, fn(t) { tensor.shape(t) != expected })
      {
        Ok(bad) -> Error(ShapeMismatch(expected: expected, got: tensor.shape(bad)))
        Error(_) -> Ok(Nil)
      }
    }
  }
}

fn chunk(xs: List(a), size: Int) -> List(List(a)) {
  case xs {
    [] -> []
    _ -> {
      let head = list.take(xs, size)
      let tail = list.drop(xs, size)
      [head, ..chunk(tail, size)]
    }
  }
}

fn stack_groups(
  groups: List(List(Sample)),
  acc: List(Batch),
) -> Result(List(Batch), TensorError) {
  case groups {
    [] -> Ok(list.reverse(acc))
    [group, ..rest] -> {
      use batch <- result.try(stack_samples(group))
      stack_groups(rest, [batch, ..acc])
    }
  }
}

fn stack_samples(samples: List(Sample)) -> Result(Batch, TensorError) {
  case samples {
    [] ->
      Error(error.InvalidShape("cannot stack an empty batch"))
    [first, ..] -> {
      let input_shape = tensor.shape(first.input)
      let target_shape = tensor.shape(first.target)
      use inputs <- result.try(
        stack_tensors(
          list.map(samples, fn(s) { s.input }),
          input_shape,
        ),
      )
      use targets <- result.try(
        stack_tensors(
          list.map(samples, fn(s) { s.target }),
          target_shape,
        ),
      )
      Ok(Batch(inputs: inputs, targets: targets))
    }
  }
}

fn stack_tensors(
  tensors: List(Tensor),
  expected_shape: List(Int),
) -> Result(Tensor, TensorError) {
  case list.find(tensors, fn(t) { tensor.shape(t) != expected_shape }) {
    Ok(bad) ->
      Error(ShapeMismatch(expected: expected_shape, got: tensor.shape(bad)))
    Error(_) -> {
      let batch = list.length(tensors)
      let flat =
        tensors
        |> list.map(tensor.to_list)
        |> list.flatten
      let shape = [batch, ..expected_shape]
      Ok(Tensor(data: flat, shape: shape))
    }
  }
}

// --- Shuffle ----------------------------------------------------------------
//
// Non-deterministic shuffle via `int.random`. The BEAM does not expose a
// portable seedable PRNG without an extra dependency, so two calls on the
// same input may return different orderings.

fn shuffle_samples(samples: List(Sample)) -> List(Sample) {
  samples
  |> list.map(fn(s) { #(int.random(1_000_000_000), s) })
  |> list.sort(fn(a, b) {
    let #(ka, _) = a
    let #(kb, _) = b
    case ka < kb {
      True -> order.Lt
      False ->
        case ka > kb {
          True -> order.Gt
          False -> order.Eq
        }
    }
  })
  |> list.map(fn(pair) {
    let #(_, s) = pair
    s
  })
}
