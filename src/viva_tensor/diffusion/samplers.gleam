//// Forward-only diffusion samplers (DDPM/DDIM).
////
//// The module is **inference only**: there is no training loop and no
//// autograd. The user supplies a noise-prediction model as a callback
//// `fn(x_t: Tensor, t: Int) -> Result(Tensor, TensorError)` and the
//// sampler drives the reverse diffusion process.
////
//// Conventions match Ho et al. 2020 ("Denoising Diffusion Probabilistic
//// Models", https://arxiv.org/abs/2006.11239) for DDPM and Song et al. 2021
//// ("Denoising Diffusion Implicit Models", https://arxiv.org/abs/2010.02502)
//// for DDIM. Tensors live in the standard Gleam dense form (`Tensor(data,
//// shape)`).

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, DimensionError, InvalidShape}
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor, Tensor}

// =============================================================================
// SCHEDULES
// =============================================================================

/// Noise schedule specification. Two flavors:
///
/// - `LinearSchedule(beta_start, beta_end, num_steps)` follows Ho et al. 2020
///   with `β_t` linearly spaced over the interval.
/// - `CosineSchedule(num_steps)` follows Nichol & Dhariwal 2021
///   ("Improved DDPMs", https://arxiv.org/abs/2102.09672):
///   ```
///   ᾱ_t = cos²((t/T + s) / (1 + s) · π/2)   with  s = 0.008
///   β_t = clamp(1 - ᾱ_t / ᾱ_{t-1}, 0, 0.999)
///   ```
pub type NoiseSchedule {
  LinearSchedule(beta_start: Float, beta_end: Float, num_steps: Int)
  CosineSchedule(num_steps: Int)
}

/// Sampler configuration: schedule + stochasticity weight.
///
/// `eta` is the DDIM stochasticity coefficient:
///   - `0.0` → fully deterministic DDIM update.
///   - `1.0` → coincides with DDPM's posterior variance.
pub type SamplerConfig {
  SamplerConfig(schedule: NoiseSchedule, eta: Float)
}

/// Precomputed schedule lookup tables.
///
/// All lists are indexed by `t ∈ [0, num_steps)`:
///
/// - `betas[t]   = β_t`
/// - `alphas[t]  = 1 - β_t`
/// - `alpha_bars[t] = Π_{s ≤ t} α_s`
pub type SchedulerState {
  SchedulerState(
    betas: List(Float),
    alphas: List(Float),
    alpha_bars: List(Float),
    num_steps: Int,
  )
}

/// Materialize the cumulative product tables for a schedule.
///
/// For the linear schedule the betas are `linspace(beta_start, beta_end, T)`.
/// For the cosine schedule we use the Nichol & Dhariwal formulation, with the
/// betas clamped to `[0, 0.999]` to avoid `1 - β < 0` blowing up later.
pub fn build_schedule(schedule: NoiseSchedule) -> SchedulerState {
  case schedule {
    LinearSchedule(beta_start, beta_end, num_steps) -> {
      let betas = linspace_floats(beta_start, beta_end, num_steps)
      schedule_from_betas(betas, num_steps)
    }
    CosineSchedule(num_steps) -> {
      let s_offset = 0.008
      let denom = 1.0 +. s_offset
      let f = fn(t: Int) -> Float {
        let x =
          { int.to_float(t) /. int.to_float(num_steps) +. s_offset }
          /. denom
          *. ffi.pi
          /. 2.0
        let c = ffi.cos(x)
        c *. c
      }
      let f0 = f(0)
      // Normalize so ᾱ_0 (effectively f(0)/f(0)) is well-defined.
      let alpha_bars =
        list.range(1, num_steps)
        |> list.map(fn(t) {
          case f0 <=. 0.0 {
            True -> 1.0
            False -> f(t) /. f0
          }
        })
      let betas = betas_from_alpha_bars(alpha_bars, 1.0)
      let alphas = list.map(betas, fn(b) { 1.0 -. b })
      SchedulerState(
        betas: betas,
        alphas: alphas,
        alpha_bars: alpha_bars,
        num_steps: num_steps,
      )
    }
  }
}

fn schedule_from_betas(
  betas: List(Float),
  num_steps: Int,
) -> SchedulerState {
  let alphas = list.map(betas, fn(b) { 1.0 -. b })
  let alpha_bars = cumulative_product(alphas)
  SchedulerState(
    betas: betas,
    alphas: alphas,
    alpha_bars: alpha_bars,
    num_steps: num_steps,
  )
}

fn cumulative_product(xs: List(Float)) -> List(Float) {
  list.fold(xs, #([], 1.0), fn(acc, value) {
    let #(accum, running) = acc
    let next = running *. value
    #([next, ..accum], next)
  }).0
  |> list.reverse
}

fn linspace_floats(start: Float, stop: Float, steps: Int) -> List(Float) {
  case steps <= 1 {
    True -> [start]
    False -> {
      let delta = { stop -. start } /. int.to_float(steps - 1)
      list.range(0, steps - 1)
      |> list.map(fn(i) { start +. delta *. int.to_float(i) })
    }
  }
}

fn betas_from_alpha_bars(
  alpha_bars: List(Float),
  prev_seed: Float,
) -> List(Float) {
  // β_t = 1 - ᾱ_t / ᾱ_{t-1}, clamped to [0, 0.999].
  list.fold(alpha_bars, #([], prev_seed), fn(acc, ab) {
    let #(out, prev) = acc
    let beta = case prev <=. 0.0 {
      True -> 0.999
      False -> float.clamp(1.0 -. ab /. prev, 0.0, 0.999)
    }
    #([beta, ..out], ab)
  }).0
  |> list.reverse
}

// =============================================================================
// STEPS
// =============================================================================

/// One DDPM reverse step at index `t`.
///
/// Given `x_t`, the noise prediction `ε_θ(x_t, t)`, and the schedule:
/// ```
/// mean = (1 / √α_t) · ( x_t - ((1 - α_t) / √(1 - ᾱ_t)) · ε_θ )
/// var  = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
/// x_{t-1} = mean + √var · z       (z ~ N(0, I); no noise when t = 0)
/// ```
/// We clamp negative variance produced by floating-point round-off in the
/// cosine schedule to `0.0`, and skip the stochastic term when `t = 0`.
pub fn ddpm_step(
  state: SchedulerState,
  x_t: Tensor,
  model_pred: Tensor,
  t: Int,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(validate_step(state, t))
  let alpha_t = at(state.alphas, t, 1.0)
  let alpha_bar_t = at(state.alpha_bars, t, 1.0)
  let beta_t = at(state.betas, t, 0.0)
  let alpha_bar_prev = case t == 0 {
    True -> 1.0
    False -> at(state.alpha_bars, t - 1, 1.0)
  }

  let sqrt_alpha_t = ffi.sqrt(float.max(alpha_t, 0.0))
  let sqrt_one_minus_bar = ffi.sqrt(float.max(1.0 -. alpha_bar_t, 0.0))

  let xs = tensor.to_list(x_t)
  let preds = tensor.to_list(model_pred)
  use _ <- result.try(ensure_same_length("ddpm_step", xs, preds))

  let coef = case sqrt_one_minus_bar <=. 0.0 {
    True -> 0.0
    False -> { 1.0 -. alpha_t } /. sqrt_one_minus_bar
  }
  let inv_sqrt_alpha = case sqrt_alpha_t <=. 0.0 {
    True -> 0.0
    False -> 1.0 /. sqrt_alpha_t
  }

  // Posterior variance, clamped to ≥ 0.
  let raw_var = case 1.0 -. alpha_bar_t <=. 0.0 {
    True -> 0.0
    False -> beta_t *. { 1.0 -. alpha_bar_prev } /. { 1.0 -. alpha_bar_t }
  }
  let variance = float.max(raw_var, 0.0)
  let sigma = case t == 0 {
    True -> 0.0
    False -> ffi.sqrt(variance)
  }

  let next =
    list.map(list.zip(xs, preds), fn(pair) {
      let #(x, eps) = pair
      let mean = inv_sqrt_alpha *. { x -. coef *. eps }
      let z = case sigma <=. 0.0 {
        True -> 0.0
        False -> standard_normal()
      }
      mean +. sigma *. z
    })
  Ok(Tensor(data: next, shape: tensor.shape(x_t)))
}

/// One DDIM reverse step at index `t` with stochasticity `eta`.
///
/// Algorithm (Song et al. 2021, Eq. 12):
/// ```
/// pred_x0 = (x_t - √(1 - ᾱ_t) · ε_θ) / √ᾱ_t
/// σ_t² = eta² · (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · (1 - ᾱ_t / ᾱ_{t-1})
/// dir_xt = √(max(1 - ᾱ_{t-1} - σ_t², 0)) · ε_θ
/// x_{t-1} = √ᾱ_{t-1} · pred_x0 + dir_xt + σ_t · z       (z ~ N(0, I))
/// ```
/// Edge cases we clamp:
/// - `1 - ᾱ_{t-1} - σ²` can dip negative under floating-point round-off in
///   the cosine schedule; we `max(·, 0)` before taking the square root.
/// - `1 - ᾱ_t / ᾱ_{t-1}` can be slightly negative if the schedule is
///   non-monotone; same treatment.
/// - At `t = 0` we treat `ᾱ_{t-1} = 1` (no previous step) and skip the
///   stochastic kick regardless of `eta`.
pub fn ddim_step(
  state: SchedulerState,
  x_t: Tensor,
  model_pred: Tensor,
  t: Int,
  eta: Float,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(validate_step(state, t))
  let alpha_bar_t = at(state.alpha_bars, t, 1.0)
  let alpha_bar_prev = case t == 0 {
    True -> 1.0
    False -> at(state.alpha_bars, t - 1, 1.0)
  }

  let one_minus_bar = float.max(1.0 -. alpha_bar_t, 0.0)
  let one_minus_prev = float.max(1.0 -. alpha_bar_prev, 0.0)
  let ratio = case alpha_bar_prev <=. 0.0 {
    True -> 0.0
    False -> float.max(1.0 -. alpha_bar_t /. alpha_bar_prev, 0.0)
  }
  let sigma_sq = case one_minus_bar <=. 0.0 {
    True -> 0.0
    False -> eta *. eta *. one_minus_prev /. one_minus_bar *. ratio
  }
  let sigma_sq = float.max(sigma_sq, 0.0)
  let sigma = case t == 0 {
    True -> 0.0
    False -> ffi.sqrt(sigma_sq)
  }
  let dir_coef = ffi.sqrt(float.max(one_minus_prev -. sigma_sq, 0.0))
  let sqrt_alpha_bar = ffi.sqrt(float.max(alpha_bar_t, 0.0))
  let sqrt_alpha_bar_prev = ffi.sqrt(float.max(alpha_bar_prev, 0.0))
  let sqrt_one_minus_bar = ffi.sqrt(one_minus_bar)

  let xs = tensor.to_list(x_t)
  let preds = tensor.to_list(model_pred)
  use _ <- result.try(ensure_same_length("ddim_step", xs, preds))

  let next =
    list.map(list.zip(xs, preds), fn(pair) {
      let #(x, eps) = pair
      let pred_x0 = case sqrt_alpha_bar <=. 0.0 {
        True -> 0.0
        False -> { x -. sqrt_one_minus_bar *. eps } /. sqrt_alpha_bar
      }
      let dir = dir_coef *. eps
      let z = case sigma <=. 0.0 {
        True -> 0.0
        False -> standard_normal()
      }
      sqrt_alpha_bar_prev *. pred_x0 +. dir +. sigma *. z
    })
  Ok(Tensor(data: next, shape: tensor.shape(x_t)))
}

// =============================================================================
// SAMPLING LOOP
// =============================================================================

/// Full reverse sampling loop. Starts from `x_T ~ N(0, I)` with the requested
/// shape and walks `t = T-1 → 0` calling `model_fn(x_t, t)` at each step.
///
/// When `config.eta == 0.0` and the schedule is reasonable, this is the
/// deterministic DDIM loop; otherwise it falls back to the DDPM update (more
/// noise per step). Pure: no progress bar, no telemetry — drop those in
/// caller code if needed.
pub fn sample(
  config: SamplerConfig,
  state: SchedulerState,
  shape: List(Int),
  model_fn: fn(Tensor, Int) -> Result(Tensor, TensorError),
) -> Result(Tensor, TensorError) {
  case state.num_steps <= 0 {
    True ->
      Error(InvalidShape(
        "sample: scheduler has " <> int.to_string(state.num_steps) <> " steps",
      ))
    False -> {
      let total = element_count(shape)
      case total <= 0 {
        True -> Error(InvalidShape("sample: empty target shape"))
        False -> {
          let x_t =
            Tensor(
              data: list.range(1, total) |> list.map(fn(_) { standard_normal() }),
              shape: shape,
            )
          sampling_loop(config, state, x_t, state.num_steps - 1, model_fn)
        }
      }
    }
  }
}

fn sampling_loop(
  config: SamplerConfig,
  state: SchedulerState,
  x_t: Tensor,
  t: Int,
  model_fn: fn(Tensor, Int) -> Result(Tensor, TensorError),
) -> Result(Tensor, TensorError) {
  case t < 0 {
    True -> Ok(x_t)
    False -> {
      use pred <- result.try(model_fn(x_t, t))
      use next <- result.try(case config.eta <=. 0.0 {
        True -> ddim_step(state, x_t, pred, t, 0.0)
        False ->
          case config.eta >=. 1.0 {
            True -> ddpm_step(state, x_t, pred, t)
            False -> ddim_step(state, x_t, pred, t, config.eta)
          }
      })
      sampling_loop(config, state, next, t - 1, model_fn)
    }
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn validate_step(state: SchedulerState, t: Int) -> Result(Nil, TensorError) {
  case t < 0 || t >= state.num_steps {
    True ->
      Error(DimensionError(
        "diffusion step: t="
        <> int.to_string(t)
        <> " out of range for "
        <> int.to_string(state.num_steps)
        <> "-step schedule",
      ))
    False -> Ok(Nil)
  }
}

fn ensure_same_length(
  op: String,
  a: List(Float),
  b: List(Float),
) -> Result(Nil, TensorError) {
  case list.length(a) == list.length(b) {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape(
        op
        <> ": x_t and model_pred have different element counts ("
        <> int.to_string(list.length(a))
        <> " vs "
        <> int.to_string(list.length(b))
        <> ")",
      ))
  }
}

fn at(xs: List(Float), i: Int, default: Float) -> Float {
  case list.drop(xs, i) {
    [v, ..] -> v
    [] -> default
  }
}

fn element_count(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, dim) { acc * dim })
}

fn standard_normal() -> Float {
  let u1 = float.max(ffi.random_uniform(), 1.0e-12)
  let u2 = ffi.random_uniform()
  ffi.sqrt(-2.0 *. ffi.log(u1)) *. ffi.cos(2.0 *. ffi.pi *. u2)
}
