//// Tests for `viva_tensor/distributed/trainer`.
////
//// These tests stay in-process (no `process.spawn`) so they are
//// deterministic in CI. The convergence test fakes "two workers" by
//// calling `compute_grads` twice — once per virtual replica.

import gleam/float
import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/core/error.{type TensorError, ShapeMismatch}
import viva_tensor/data/dataloader.{
  type Batch, Sample, data_loader_new, dataset_from_samples,
}
import viva_tensor/distributed/trainer.{AverageGrads, SumGrads, TrainConfig}
import viva_tensor/nn/optim.{type GradPair, type Param, GradPair, Param}
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

// --- Helpers ---------------------------------------------------------------

fn close(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <. tol
}

fn list_close(a: List(Float), b: List(Float), tol: Float) -> Bool {
  list.length(a) == list.length(b)
  && list.zip(a, b)
  |> list.all(fn(pair) { close(pair.0, pair.1, tol) })
}

fn grad(name: String, values: List(Float)) -> GradPair {
  GradPair(name: name, grad: tensor.from_list(values))
}

fn param(name: String, values: List(Float)) -> Param {
  Param(name: name, value: tensor.from_list(values))
}

// --- distribute_grads ------------------------------------------------------

pub fn distribute_grads_average_test() {
  // 2 workers, both with [Grad("w", [2.0])]
  // Sum = [4.0], avg = [2.0]
  let worker_grads = [[grad("w", [2.0])], [grad("w", [2.0])]]
  let assert Ok(aggregated) =
    trainer.distribute_grads(worker_grads, AverageGrads)
  let assert [first] = aggregated
  first.name |> should.equal("w")
  list_close(tensor.to_list(first.grad), [2.0], 1.0e-9)
  |> should.be_true()
}

pub fn distribute_grads_sum_test() {
  // Same input as the average test, but with SumGrads: raw 2.0 + 2.0 = 4.0.
  let worker_grads = [[grad("w", [2.0])], [grad("w", [2.0])]]
  let assert Ok(aggregated) = trainer.distribute_grads(worker_grads, SumGrads)
  let assert [first] = aggregated
  list_close(tensor.to_list(first.grad), [4.0], 1.0e-9)
  |> should.be_true()
}

pub fn distribute_grads_multi_param_average_test() {
  // Two parameters across two workers, AverageGrads.
  let w0 = [grad("w", [1.0, 1.0]), grad("b", [2.0])]
  let w1 = [grad("w", [3.0, 3.0]), grad("b", [4.0])]
  let assert Ok(aggregated) = trainer.distribute_grads([w0, w1], AverageGrads)
  // Names preserved in input order.
  let assert [g_w, g_b] = aggregated
  g_w.name |> should.equal("w")
  g_b.name |> should.equal("b")
  list_close(tensor.to_list(g_w.grad), [2.0, 2.0], 1.0e-9)
  |> should.be_true()
  list_close(tensor.to_list(g_b.grad), [3.0], 1.0e-9)
  |> should.be_true()
}

pub fn distribute_grads_shape_mismatch_test() {
  let w0 = [grad("w", [1.0, 1.0])]
  let w1 = [grad("w", [1.0])]
  case trainer.distribute_grads([w0, w1], AverageGrads) {
    Error(ShapeMismatch(expected: [2], got: [1])) -> Nil
    other -> {
      other |> should.equal(other)
      panic as "expected ShapeMismatch([2], [1])"
    }
  }
}

pub fn distribute_grads_empty_workers_test() {
  // No workers at all is an error — we need at least one grad list.
  case trainer.distribute_grads([], AverageGrads) {
    Error(_) -> Nil
    Ok(_) -> panic as "expected Error for empty worker list"
  }
}

// --- synchronous_train_step ------------------------------------------------

pub fn synchronous_train_step_test() {
  // 2 workers' grads on a single-param SGD model.
  // Each worker grad is [1.0]; AverageGrads → aggregated grad = [1.0].
  // With lr=0.1, expected param update: 1.0 - 0.1 * 1.0 = 0.9.
  let opt = optim.sgd(0.1)
  let p = param("w", [1.0])
  let w0_grads = [grad("w", [1.0])]
  let w1_grads = [grad("w", [1.0])]

  let assert Ok(#(_opt2, params2)) =
    trainer.synchronous_train_step(opt, [p], [w0_grads, w1_grads], AverageGrads)
  let assert [updated] = params2
  list_close(tensor.to_list(updated.value), [0.9], 1.0e-9)
  |> should.be_true()

  // Sanity: equivalent to applying optim.step with the averaged grad.
  let avg_grad = [grad("w", [1.0])]
  let assert Ok(#(_opt_ref, params_ref)) = optim.step(opt, [p], avg_grad)
  let assert [ref_param] = params_ref
  list_close(
    tensor.to_list(updated.value),
    tensor.to_list(ref_param.value),
    1.0e-9,
  )
  |> should.be_true()
}

pub fn synchronous_train_step_sum_test() {
  // SumGrads: aggregated grad = [1.0] + [1.0] = [2.0].
  // lr=0.1 → 1.0 - 0.1 * 2.0 = 0.8.
  let opt = optim.sgd(0.1)
  let p = param("w", [1.0])
  let assert Ok(#(_opt2, params2)) =
    trainer.synchronous_train_step(
      opt,
      [p],
      [[grad("w", [1.0])], [grad("w", [1.0])]],
      SumGrads,
    )
  let assert [updated] = params2
  list_close(tensor.to_list(updated.value), [0.8], 1.0e-9)
  |> should.be_true()
}

// --- train_synchronous convergence ----------------------------------------

/// y = 2x + 1.
/// Param "w" learned from grad = 2 * (w*x + b - y) * x (MSE wrt one sample).
/// Param "b" learned from grad = 2 * (w*x + b - y).
/// We simulate two workers by calling compute_grads twice on the same batch.
fn linreg_grads(
  batch: Batch,
  params: List(Param),
) -> Result(List(GradPair), TensorError) {
  let assert [Param(name: "w", value: wv), Param(name: "b", value: bv)] = params
  let assert [w] = tensor.to_list(wv)
  let assert [b] = tensor.to_list(bv)
  let xs = tensor.to_list(batch.inputs)
  let ys = tensor.to_list(batch.targets)
  let pairs = list.zip(xs, ys)
  let n = case list.length(pairs) {
    0 -> 1
    k -> k
  }
  let inv_n = 1.0 /. int.to_float(n)
  let #(gw_sum, gb_sum) =
    list.fold(pairs, #(0.0, 0.0), fn(acc, p) {
      let #(gw, gb) = acc
      let #(x, y) = p
      let err = w *. x +. b -. y
      #(gw +. 2.0 *. err *. x, gb +. 2.0 *. err)
    })
  Ok([
    GradPair(name: "w", grad: tensor.from_list([gw_sum *. inv_n])),
    GradPair(name: "b", grad: tensor.from_list([gb_sum *. inv_n])),
  ])
}

fn linreg_loss(batch: Batch, params: List(Param)) -> Float {
  let assert [Param(name: "w", value: wv), Param(name: "b", value: bv)] = params
  let assert [w] = tensor.to_list(wv)
  let assert [b] = tensor.to_list(bv)
  let xs = tensor.to_list(batch.inputs)
  let ys = tensor.to_list(batch.targets)
  list.zip(xs, ys)
  |> list.map(fn(p) {
    let #(x, y) = p
    let err = w *. x +. b -. y
    err *. err
  })
  |> float.sum
}

pub fn train_synchronous_convergence_test() {
  // 4 single-sample batches drawn from y = 2x + 1 over x in [1, 2, 3, 4].
  let xs = [1.0, 2.0, 3.0, 4.0]
  let ys = list.map(xs, fn(x) { 2.0 *. x +. 1.0 })
  let samples =
    list.zip(xs, ys)
    |> list.map(fn(pair) {
      let #(x, y) = pair
      Sample(input: tensor.from_list([x]), target: tensor.from_list([y]))
    })
  let ds = dataset_from_samples(samples)
  let loader = data_loader_new(ds, 1, False, False)

  let initial_params = [param("w", [0.0]), param("b", [0.0])]
  let opt = optim.sgd(0.05)

  // Loss at t=0 (sum over the 4 batches, w=0/b=0).
  let assert Ok(batches0) = dataloader.data_loader_batches(loader)
  let initial_loss =
    batches0
    |> list.map(fn(b) { linreg_loss(b, initial_params) })
    |> float.sum

  let config =
    TrainConfig(
      num_workers: 2,
      batches_per_step: 4,
      grad_aggregation: AverageGrads,
    )

  let assert Ok(result) =
    trainer.train_synchronous(
      config,
      initial_params,
      opt,
      loader,
      linreg_grads,
      10,
    )

  // After training: loss should have dropped.
  let final_loss =
    batches0
    |> list.map(fn(b) { linreg_loss(b, result.final_params) })
    |> float.sum

  // Strictly less than the starting loss. Linreg with these hyperparameters
  // converges very fast, so even after 10 steps the drop is dramatic.
  { final_loss <. initial_loss } |> should.be_true()
  // And the optimizer ran exactly 10 steps.
  result.steps |> should.equal(10)
  // And we got two parameters back, same names as input.
  let names = list.map(result.final_params, fn(p) { p.name })
  names |> should.equal(["w", "b"])
}

// --- Misc edge cases -------------------------------------------------------

pub fn train_synchronous_zero_steps_test() {
  // num_steps=0 should return the initial state untouched.
  let samples = [
    Sample(input: tensor.from_list([1.0]), target: tensor.from_list([1.0])),
  ]
  let loader = data_loader_new(dataset_from_samples(samples), 1, False, False)
  let initial_params = [param("w", [0.0])]
  let opt = optim.sgd(0.1)
  let config =
    TrainConfig(
      num_workers: 1,
      batches_per_step: 1,
      grad_aggregation: AverageGrads,
    )
  let dummy_grads = fn(_b: Batch, _p: List(Param)) { Ok([grad("w", [1.0])]) }
  let assert Ok(result) =
    trainer.train_synchronous(
      config,
      initial_params,
      opt,
      loader,
      dummy_grads,
      0,
    )
  result.steps |> should.equal(0)
  let assert [back] = result.final_params
  list_close(tensor.to_list(back.value), [0.0], 1.0e-12)
  |> should.be_true()
}
