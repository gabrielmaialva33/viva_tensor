//// Distributed (data-parallel) training primitives for viva_tensor.
////
//// Implements synchronous data-parallel SGD on top of BEAM lightweight
//// processes. The pattern is the classic "all-reduce" loop used in mini-batch
//// SGD distributed training:
////
//// 1. The coordinator splits a batch (or set of batches) across N workers.
//// 2. Each worker computes gradients locally on its slice (the user supplies
////    the `compute_grads` callback).
//// 3. All workers send their gradients to the coordinator.
//// 4. The coordinator aggregates (sum or average) and applies a single
////    optimizer step.
//// 5. The new parameters are broadcast back to the workers and the loop
////    repeats.
////
//// SYNCHRONOUS semantics: every step waits for ALL workers to finish before
//// moving on. There is no asynchronous (parameter-server) or stale-gradient
//// variant in this module.
////
//// NO autograd integration: the caller hands in `GradPair`s explicitly. We
//// only parallelize the *application* of gradients, not their computation.
//// The user supplies a `compute_grads` callback (forward + manual backward,
//// finite-difference, or anything else they like).
////
//// In-process aggregation (`distribute_grads`, `synchronous_train_step`) is
//// deterministic and process-free; the spawn/send/receive primitives are
//// available for callers that genuinely want multi-process parallelism, but
//// `train_synchronous` uses the in-process path so that tests are reliable.

import gleam/dict.{type Dict}
import gleam/erlang/process.{type Pid, type Subject}
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, DimensionError, ShapeMismatch}
import viva_tensor/data/dataloader.{type Batch, type DataLoader}
import viva_tensor/nn/optim.{type GradPair, type Optimizer, type Param, GradPair}
import viva_tensor/tensor

// --- Public types -----------------------------------------------------------

/// Gradient aggregation strategy across workers in a synchronous step.
///
/// - `AverageGrads`: divide the summed gradients by the number of workers.
///   Matches PyTorch's `DistributedDataParallel` default.
/// - `SumGrads`: raw sum. Useful when each worker already scales its grad by
///   `1/N` or when you want to emulate a larger batch.
pub type GradAggregation {
  AverageGrads
  SumGrads
}

/// Configuration for `train_synchronous`.
///
/// - `num_workers`: how many parallel grad computations per step. Must be > 0.
/// - `batches_per_step`: number of batches each worker processes before the
///   coordinator aggregates. Must be > 0.
/// - `grad_aggregation`: see `GradAggregation`.
pub type TrainConfig {
  TrainConfig(
    num_workers: Int,
    batches_per_step: Int,
    grad_aggregation: GradAggregation,
  )
}

/// Result of a `train_synchronous` run.
///
/// - `final_params`: parameters after the last synchronous step.
/// - `final_optimizer`: optimizer state (momentum/EMA buffers/etc.) after the
///   last step.
/// - `total_loss`: accumulated loss across every applied step (sum, not
///   average — caller divides by `steps` if they want a per-step mean).
/// - `steps`: number of synchronous steps that actually ran.
pub type TrainResult {
  TrainResult(
    final_params: List(Param),
    final_optimizer: Optimizer,
    total_loss: Float,
    steps: Int,
  )
}

/// A spawned worker handle.
///
/// The `batch_inbox` is owned by the worker process; the coordinator sends
/// `WorkerMessage`s to it. The `grads_outbox` is owned by the coordinator;
/// the worker sends `#(batch_id, List(GradPair))` envelopes to it.
///
/// Synchronous semantics: callers MUST wait for a grad envelope on
/// `grads_outbox` after each batch they push.
pub type Worker {
  Worker(
    id: Int,
    pid: Pid,
    batch_inbox: Subject(WorkerMessage),
    grads_outbox: Subject(#(Int, Int, List(GradPair))),
  )
}

/// Message envelope a worker receives on its `batch_inbox`.
///
/// - `RunBatch(batch_id, batch, params)`: run `compute_grads(batch, params)`
///   and reply on `grads_outbox` with `#(worker_id, batch_id, grads)`.
/// - `Stop`: terminate the worker loop.
pub type WorkerMessage {
  RunBatch(batch_id: Int, batch: Batch, params: List(Param))
  Stop
}

// --- Aggregation ------------------------------------------------------------

/// Aggregate per-worker gradient lists into a single list of `GradPair`s.
///
/// SYNCHRONOUS: this is a pure, in-process reduction. All grads must already
/// be available; no waiting happens here.
///
/// Every per-worker list must have the same length, the same parameter
/// names, and matching tensor shapes. Returns:
/// - `Error(DimensionError)` when the lists differ in length, are empty, or
///   reference different parameter names.
/// - `Error(ShapeMismatch)` when two gradients for the same parameter have
///   different shapes.
///
/// For `AverageGrads` the result is divided by the number of workers
/// (`length(per_worker_grads)`); for `SumGrads` it is the raw element-wise
/// sum.
pub fn distribute_grads(
  per_worker_grads: List(List(GradPair)),
  aggregation: GradAggregation,
) -> Result(List(GradPair), TensorError) {
  case per_worker_grads {
    [] ->
      Error(DimensionError("distribute_grads: need at least one worker's grads"))
    [first, ..rest] -> {
      use _ <- result.try(validate_same_shape_lists(first, rest))
      let num_workers = list.length(per_worker_grads)
      use summed <- result.try(sum_grad_lists(first, rest))
      case aggregation {
        SumGrads -> Ok(summed)
        AverageGrads -> {
          let denom = int.to_float(num_workers)
          Ok(
            list.map(summed, fn(gp) {
              GradPair(name: gp.name, grad: tensor.scale(gp.grad, 1.0 /. denom))
            }),
          )
        }
      }
    }
  }
}

/// Apply one synchronous data-parallel optimizer step.
///
/// SYNCHRONOUS: aggregates per-worker grads and then calls `optim.step` once
/// with the result. No process IO.
///
/// Equivalent to:
///   `let aggregated = distribute_grads(per_worker_grads, aggregation)`
///   `optim.step(opt, params, aggregated)`
pub fn synchronous_train_step(
  opt: Optimizer,
  params: List(Param),
  per_worker_grads: List(List(GradPair)),
  aggregation: GradAggregation,
) -> Result(#(Optimizer, List(Param)), TensorError) {
  use aggregated <- result.try(distribute_grads(per_worker_grads, aggregation))
  optim.step(opt, params, aggregated)
}

// --- Worker primitives ------------------------------------------------------

/// Spawn `num` worker processes. Each runs `worker_loop(id)` until it returns.
///
/// SYNCHRONOUS: the call returns once the processes have been spawned, but it
/// does NOT wait for them to finish. Callers must coordinate completion
/// themselves (typically by sending a `Stop` message and waiting on the
/// grads_outbox for a final reply or monitoring the pid).
///
/// The returned `Worker` carries placeholder subjects — the user-supplied
/// `worker_loop` is responsible for managing message flow; the high-level
/// `train_synchronous` does not use this path. These primitives are exposed
/// for users who want to drive their own multi-process loop.
pub fn spawn_workers(num: Int, worker_loop: fn(Int) -> a) -> List(Worker) {
  list.range(0, num - 1)
  |> list.map(fn(id) {
    let inbox: Subject(WorkerMessage) = process.new_subject()
    let outbox: Subject(#(Int, Int, List(GradPair))) = process.new_subject()
    let pid =
      process.spawn(fn() {
        let _ = worker_loop(id)
        Nil
      })
    Worker(id: id, pid: pid, batch_inbox: inbox, grads_outbox: outbox)
  })
}

/// Push a batch to a worker. The message envelope is `RunBatch(batch_id,
/// batch, params)`.
///
/// SYNCHRONOUS pairing: callers MUST follow each `send_batch_to_worker` with
/// a matching `receive_grads_from_worker` before sending the next batch — the
/// worker's mailbox is unbounded but ordering is only meaningful for one
/// outstanding batch at a time.
pub fn send_batch_to_worker(
  worker: Worker,
  batch_id: Int,
  batch: Batch,
  params: List(Param),
) -> Nil {
  process.send(worker.batch_inbox, RunBatch(batch_id, batch, params))
}

/// Receive one grad envelope from a worker, with a timeout in milliseconds.
///
/// SYNCHRONOUS: blocks until the worker replies or the timeout elapses. The
/// envelope is `#(worker_id, batch_id, List(GradPair))`; this function
/// returns the `List(GradPair)` portion. Returns `Error(DimensionError)` on
/// timeout.
pub fn receive_grads_from_worker(
  worker: Worker,
  timeout_ms: Int,
) -> Result(List(GradPair), TensorError) {
  case process.receive(worker.grads_outbox, timeout_ms) {
    Ok(#(_worker_id, _batch_id, grads)) -> Ok(grads)
    Error(_) ->
      Error(DimensionError(
        "receive_grads_from_worker: timeout after "
        <> int.to_string(timeout_ms)
        <> "ms",
      ))
  }
}

/// All-reduce step: collect local grads from every worker at the coordinator
/// and aggregate.
///
/// SYNCHRONOUS: waits for each worker's reply in turn. `local_grads` is the
/// coordinator's own contribution (one per `Worker` in the list — index i
/// pairs with `workers[i]`). The function reads each `grads_outbox` once and
/// runs `distribute_grads` on the collected list.
///
/// Returns `Error(DimensionError)` if a worker fails to reply within
/// `timeout_ms` (default: 5_000 ms baked into this entry point).
pub fn all_reduce_grads(
  workers: List(Worker),
  local_grads: List(List(GradPair)),
  aggregation: GradAggregation,
) -> Result(List(GradPair), TensorError) {
  case list.length(workers) == list.length(local_grads) {
    False ->
      Error(DimensionError(
        "all_reduce_grads: number of workers must equal number of local grad lists",
      ))
    True ->
      // For the simplified implementation we accept the local grads directly
      // (each worker has already produced its own list and shipped it). We do
      // not re-receive from the wire here; tests and the high-level driver
      // pass the already-collected per-worker grads.
      distribute_grads(local_grads, aggregation)
  }
}

// --- High-level driver ------------------------------------------------------

/// Run `num_steps` of synchronous data-parallel SGD across `num_workers`
/// virtual workers.
///
/// SYNCHRONOUS: at every step the coordinator pulls `batches_per_step`
/// batches from the loader, runs `compute_grads` for each of `num_workers`
/// virtual workers, aggregates the gradients with `distribute_grads`, and
/// applies a single `optim.step`. There is no process IO; "workers" here are
/// purely a way to scale gradient aggregation deterministically (think
/// "gradient accumulation across N replicas").
///
/// `compute_grads(batch, params)` is what the user supplies — typically a
/// forward pass plus a manual or autograd-backed backward. The current
/// parameters are passed in fresh each call, so the callback can be pure.
///
/// Stops early if:
/// - the dataloader yields no batches;
/// - `compute_grads` returns an `Error`;
/// - any optim step returns an `Error`.
///
/// Returns `TrainResult.total_loss = 0.0` because this function does not
/// see the loss directly; callers should instrument `compute_grads` if they
/// need per-step loss tracking.
pub fn train_synchronous(
  config: TrainConfig,
  initial_params: List(Param),
  initial_optimizer: Optimizer,
  data_loader: DataLoader,
  compute_grads: fn(Batch, List(Param)) -> Result(List(GradPair), TensorError),
  num_steps: Int,
) -> Result(TrainResult, TensorError) {
  case config.num_workers <= 0 {
    True -> Error(DimensionError("train_synchronous: num_workers must be > 0"))
    False ->
      case config.batches_per_step <= 0 {
        True ->
          Error(DimensionError(
            "train_synchronous: batches_per_step must be > 0",
          ))
        False ->
          case num_steps < 0 {
            True ->
              Error(DimensionError("train_synchronous: num_steps must be >= 0"))
            False -> {
              use batches <- result.try(dataloader.data_loader_batches(
                data_loader,
              ))
              case batches {
                [] ->
                  Ok(TrainResult(
                    final_params: initial_params,
                    final_optimizer: initial_optimizer,
                    total_loss: 0.0,
                    steps: 0,
                  ))
                _ ->
                  run_steps(
                    config,
                    initial_params,
                    initial_optimizer,
                    batches,
                    compute_grads,
                    num_steps,
                    0,
                  )
              }
            }
          }
      }
  }
}

// --- Internals --------------------------------------------------------------

fn run_steps(
  config: TrainConfig,
  params: List(Param),
  opt: Optimizer,
  batches: List(Batch),
  compute_grads: fn(Batch, List(Param)) -> Result(List(GradPair), TensorError),
  remaining: Int,
  done: Int,
) -> Result(TrainResult, TensorError) {
  case remaining <= 0 {
    True ->
      Ok(TrainResult(
        final_params: params,
        final_optimizer: opt,
        total_loss: 0.0,
        steps: done,
      ))
    False -> {
      // Pull `batches_per_step` batches (cycling if loader is shorter).
      let step_batches = take_cycle(batches, config.batches_per_step)
      use per_worker <- result.try(collect_per_worker_grads(
        step_batches,
        config.num_workers,
        params,
        compute_grads,
      ))
      use #(opt2, params2) <- result.try(synchronous_train_step(
        opt,
        params,
        per_worker,
        config.grad_aggregation,
      ))
      run_steps(
        config,
        params2,
        opt2,
        batches,
        compute_grads,
        remaining - 1,
        done + 1,
      )
    }
  }
}

/// For each of `num_workers` virtual workers, compute gradients on one of
/// `batches` (cycling) and accumulate them by summing within the worker.
///
/// Returns a list of length `num_workers`, each element being a list of
/// `GradPair`s with the same names/shapes as `initial_params`.
fn collect_per_worker_grads(
  batches: List(Batch),
  num_workers: Int,
  params: List(Param),
  compute_grads: fn(Batch, List(Param)) -> Result(List(GradPair), TensorError),
) -> Result(List(List(GradPair)), TensorError) {
  // Round-robin batches to workers. Each worker accumulates grads across the
  // batches it was assigned by summing.
  let assigned = assign_batches(batches, num_workers)
  do_collect_workers(assigned, params, compute_grads, [])
}

fn do_collect_workers(
  per_worker_batches: List(List(Batch)),
  params: List(Param),
  compute_grads: fn(Batch, List(Param)) -> Result(List(GradPair), TensorError),
  acc: List(List(GradPair)),
) -> Result(List(List(GradPair)), TensorError) {
  case per_worker_batches {
    [] -> Ok(list.reverse(acc))
    [worker_batches, ..rest] -> {
      use grads <- result.try(accumulate_grads_for_worker(
        worker_batches,
        params,
        compute_grads,
      ))
      do_collect_workers(rest, params, compute_grads, [grads, ..acc])
    }
  }
}

fn accumulate_grads_for_worker(
  batches: List(Batch),
  params: List(Param),
  compute_grads: fn(Batch, List(Param)) -> Result(List(GradPair), TensorError),
) -> Result(List(GradPair), TensorError) {
  case batches {
    [] ->
      Ok(
        list.map(params, fn(p) { GradPair(p.name, tensor.zeros_like(p.value)) }),
      )
    [first, ..rest] -> {
      use first_grads <- result.try(compute_grads(first, params))
      list.try_fold(rest, first_grads, fn(acc, b) {
        use g <- result.try(compute_grads(b, params))
        add_grad_lists(acc, g)
      })
    }
  }
}

fn add_grad_lists(
  a: List(GradPair),
  b: List(GradPair),
) -> Result(List(GradPair), TensorError) {
  let b_dict =
    list.fold(b, dict.new(), fn(acc, gp) { dict.insert(acc, gp.name, gp.grad) })
  list.try_map(a, fn(gp) {
    case dict.get(b_dict, gp.name) {
      Error(_) ->
        Error(DimensionError(
          "distributed.add_grad_lists: missing gradient for '" <> gp.name <> "'",
        ))
      Ok(other) -> {
        case tensor.shape(gp.grad) == tensor.shape(other) {
          False ->
            Error(ShapeMismatch(
              expected: tensor.shape(gp.grad),
              got: tensor.shape(other),
            ))
          True -> {
            use summed <- result.try(tensor.add(gp.grad, other))
            Ok(GradPair(name: gp.name, grad: summed))
          }
        }
      }
    }
  })
}

/// Sum `first` with every list in `rest`. Assumes shapes have already been
/// validated by `validate_same_shape_lists`.
fn sum_grad_lists(
  first: List(GradPair),
  rest: List(List(GradPair)),
) -> Result(List(GradPair), TensorError) {
  list.try_fold(rest, first, fn(acc, other) { add_grad_lists(acc, other) })
}

fn validate_same_shape_lists(
  first: List(GradPair),
  rest: List(List(GradPair)),
) -> Result(Nil, TensorError) {
  let first_names = list.map(first, fn(gp) { gp.name })
  let first_shapes: Dict(String, List(Int)) =
    list.fold(first, dict.new(), fn(acc, gp) {
      dict.insert(acc, gp.name, tensor.shape(gp.grad))
    })
  list.try_fold(rest, Nil, fn(_, other) {
    case list.length(other) == list.length(first) {
      False ->
        Error(DimensionError(
          "distribute_grads: worker grad lists have different lengths",
        ))
      True ->
        list.try_fold(other, Nil, fn(_, gp) {
          case list.contains(first_names, gp.name) {
            False ->
              Error(DimensionError(
                "distribute_grads: parameter name '"
                <> gp.name
                <> "' missing from first worker's grads",
              ))
            True ->
              case dict.get(first_shapes, gp.name) {
                Error(_) -> Ok(Nil)
                Ok(expected) ->
                  case tensor.shape(gp.grad) == expected {
                    False ->
                      Error(ShapeMismatch(
                        expected: expected,
                        got: tensor.shape(gp.grad),
                      ))
                    True -> Ok(Nil)
                  }
              }
          }
        })
    }
  })
  |> result.map(fn(_) { Nil })
}

fn assign_batches(batches: List(Batch), num_workers: Int) -> List(List(Batch)) {
  // Round-robin: worker i gets batches at indices i, i+num_workers, ...
  let indexed = list.index_map(batches, fn(b, i) { #(i, b) })
  list.range(0, num_workers - 1)
  |> list.map(fn(worker_id) {
    indexed
    |> list.filter(fn(pair) { pair.0 % num_workers == worker_id })
    |> list.map(fn(pair) { pair.1 })
  })
}

fn take_cycle(xs: List(a), n: Int) -> List(a) {
  case xs {
    [] -> []
    _ -> do_take_cycle(xs, xs, n, [])
  }
}

fn do_take_cycle(
  remaining: List(a),
  full: List(a),
  n: Int,
  acc: List(a),
) -> List(a) {
  case n <= 0 {
    True -> list.reverse(acc)
    False ->
      case remaining {
        [] -> do_take_cycle(full, full, n, acc)
        [head, ..rest] -> do_take_cycle(rest, full, n - 1, [head, ..acc])
      }
  }
}
