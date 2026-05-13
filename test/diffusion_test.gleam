//// Tests for `viva_tensor/diffusion/samplers`. Six tests covering schedule
//// construction (linear + cosine), DDPM boundary at t=0, DDIM determinism
//// at eta=0, and the full sampling loop with a trivial constant model.

import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/diffusion/samplers
import viva_tensor/tensor.{type Tensor, Tensor}

pub fn main() -> Nil {
  gleeunit.main()
}

// --- Schedule construction --------------------------------------------------

pub fn build_linear_schedule_test() {
  let state = t.build_schedule(samplers.LinearSchedule(0.001, 0.02, 5))
  state.num_steps |> should.equal(5)
  // betas linear from 0.001 → 0.02 in 5 steps. Step = (0.02 - 0.001) / 4 =
  // 0.00475.
  let expected = [0.001, 0.005_75, 0.010_5, 0.015_25, 0.02]
  numerics.lists_close(state.betas, expected, 0.000_001, 0.000_001)
  |> should.be_true()

  // alphas[t] = 1 - betas[t]
  let expected_alphas = list.map(expected, fn(b) { 1.0 -. b })
  numerics.lists_close(state.alphas, expected_alphas, 0.000_001, 0.000_001)
  |> should.be_true()
}

pub fn build_cosine_schedule_test() {
  // For the cosine schedule, ᾱ_t = f(t)/f(0) where
  // f(t) = cos²(((t/T + s)/(1+s)) · π/2). It is monotonically decreasing
  // and ᾱ_T ≈ 0.
  let num_steps = 10
  let state = t.build_schedule(samplers.CosineSchedule(num_steps))
  state.num_steps |> should.equal(num_steps)
  list.length(state.alpha_bars) |> should.equal(num_steps)

  // Strict monotonicity: each successive ᾱ_t <= ᾱ_{t-1} (within rounding).
  let pairs =
    state.alpha_bars
    |> list.window_by_2
  let monotone =
    list.all(pairs, fn(pair) {
      let #(prev, next) = pair
      next <=. prev +. 0.000_001
    })
  monotone |> should.be_true()

  // First entry must be close to ᾱ_1 = f(1)/f(0); last entry near 0.
  let first = case state.alpha_bars {
    [head, ..] -> head
    [] -> -1.0
  }
  { first <. 1.0 && first >. 0.9 } |> should.be_true()

  let last = case list.last(state.alpha_bars) {
    Ok(value) -> value
    Error(_) -> -1.0
  }
  { last >. 0.0 && last <. 0.05 } |> should.be_true()
}

// --- DDPM / DDIM steps ------------------------------------------------------

pub fn ddpm_step_at_t0_test() {
  // At t=0, the DDPM update drops the stochastic noise term: x_0 is fully
  // determined by x_1 and the noise prediction. Compare two consecutive
  // calls — equality up to numerics means there was no random kick.
  let state = t.build_schedule(samplers.LinearSchedule(0.001, 0.02, 5))
  let x_t = Tensor(data: [0.3, -0.5, 0.7, 1.1], shape: [4])
  let pred = Tensor(data: [0.1, 0.0, -0.2, 0.05], shape: [4])
  case t.ddpm_step(state, x_t, pred, 0), t.ddpm_step(state, x_t, pred, 0) {
    Ok(a), Ok(b) -> {
      t.shape(a) |> should.equal([4])
      numerics.lists_close(t.to_list(a), t.to_list(b), 0.000_000_1, 0.000_000_1)
      |> should.be_true()
    }
    _, _ -> should.fail()
  }
}

pub fn ddim_step_deterministic_test() {
  // eta = 0 makes the DDIM update fully deterministic — no draws from the
  // PRNG. Two calls with identical inputs must coincide exactly.
  let state = t.build_schedule(samplers.LinearSchedule(0.0001, 0.02, 20))
  let x_t = Tensor(data: [0.5, -0.3, 1.2, -0.7], shape: [4])
  let pred = Tensor(data: [0.1, 0.2, -0.1, 0.0], shape: [4])
  case
    t.ddim_step(state, x_t, pred, 10, 0.0),
    t.ddim_step(state, x_t, pred, 10, 0.0)
  {
    Ok(a), Ok(b) -> {
      numerics.lists_close(t.to_list(a), t.to_list(b), 0.0, 0.0)
      |> should.be_true()
    }
    _, _ -> should.fail()
  }
}

pub fn sample_loop_constant_model_test() {
  // Model that always predicts zero noise: DDIM reduces to scaling
  // x_t → √(ᾱ_{t-1}/ᾱ_t) · x_t at every step. The end result is finite,
  // has the right shape, and is not all NaN/Inf.
  let num_steps = 5
  let state = t.build_schedule(samplers.LinearSchedule(0.001, 0.02, num_steps))
  let config =
    samplers.SamplerConfig(
      schedule: samplers.LinearSchedule(0.001, 0.02, num_steps),
      eta: 0.0,
    )
  let model_fn = fn(x_t: Tensor, _t: Int) -> Result(Tensor, t.TensorError) {
    Ok(t.zeros_like(x_t))
  }
  case t.sample(config, state, [3], model_fn) {
    Ok(result) -> {
      t.shape(result) |> should.equal([3])
      let values = t.to_list(result)
      list.length(values) |> should.equal(3)
      // All finite.
      list.all(values, fn(v) { float.absolute_value(v) <. 1.0e9 })
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn sample_shape_test() {
  let num_steps = 4
  let state = t.build_schedule(samplers.LinearSchedule(0.001, 0.02, num_steps))
  let config =
    samplers.SamplerConfig(
      schedule: samplers.LinearSchedule(0.001, 0.02, num_steps),
      eta: 0.0,
    )
  let model_fn = fn(x_t: Tensor, _t: Int) -> Result(Tensor, t.TensorError) {
    Ok(t.zeros_like(x_t))
  }
  case t.sample(config, state, [2, 3], model_fn) {
    Ok(result) -> {
      t.shape(result) |> should.equal([2, 3])
      list.length(t.to_list(result)) |> should.equal(6)
    }
    Error(_) -> should.fail()
  }
}
