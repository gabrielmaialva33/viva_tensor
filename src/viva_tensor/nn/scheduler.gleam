//// Learning-rate schedulers for `viva_tensor/nn/optim`.
////
//// Schedulers are pure data: a `Scheduler` record carries its tag, the base
//// learning rate, the current step count, and the hyperparameters needed by
//// any of the supported schedules. `scheduler_step` advances the step count
//// and returns the new learning rate; `apply_to_optimizer` is sugar that
//// pipes that learning rate into an `Optimizer`.
////
//// Curves at a glance:
////
//// - **StepLR**: piecewise-constant staircase. `lr = base_lr * gamma^(step / step_size)`
////   with integer division, so the rate drops by a factor of `gamma` every
////   `step_size` steps.
//// - **CosineAnnealingLR**: smooth half-cosine from `base_lr` down to
////   `eta_min` over `t_max` steps.
//// - **LinearWarmup**: linear ramp from 0 to `base_lr` over `warmup_steps`,
////   then constant `base_lr`.
//// - **OneCycleLR**: linear warmup `base_lr -> max_lr` for the first
////   `pct_start * total_steps` steps, then cosine anneal `max_lr -> base_lr`
////   for the remaining steps.
//// - **ExponentialLR**: continuous exponential decay. `lr = base_lr * gamma^step`.

import gleam/float
import gleam/int
import viva_math/constants as vm_constants
import viva_math/scalar as vm_scalar
import viva_tensor/nn/optim

/// Identifies which schedule a `Scheduler` is implementing.
pub type SchedulerKind {
  StepLr
  CosineAnnealingLr
  LinearWarmup
  OneCycleLr
  ExponentialLr
}

/// Scheduler state. The record holds the fields for every supported
/// schedule. Constructors fill in the relevant ones and leave the rest at
/// neutral defaults — fine because each `kind` only reads its own fields.
pub type Scheduler {
  Scheduler(
    kind: SchedulerKind,
    base_lr: Float,
    step_count: Int,
    // StepLR fields:
    step_size: Int,
    gamma: Float,
    // CosineAnnealingLR fields:
    t_max: Int,
    eta_min: Float,
    // LinearWarmup fields:
    warmup_steps: Int,
    // OneCycleLR fields:
    max_lr: Float,
    total_steps: Int,
    pct_start: Float,
  )
}

// --- Constructors -----------------------------------------------------------

/// Build a StepLR scheduler.
///
/// Formula: `lr = base_lr * gamma^floor(step / step_size)`.
///
/// The learning rate is multiplied by `gamma` every `step_size` steps, so
/// the curve is a descending staircase.
pub fn step_lr(base_lr: Float, step_size: Int, gamma: Float) -> Scheduler {
  Scheduler(
    kind: StepLr,
    base_lr: base_lr,
    step_count: 0,
    step_size: step_size,
    gamma: gamma,
    t_max: 0,
    eta_min: 0.0,
    warmup_steps: 0,
    max_lr: 0.0,
    total_steps: 0,
    pct_start: 0.0,
  )
}

/// Build a cosine-annealing scheduler.
///
/// Formula: `lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step / t_max))`.
///
/// At `step = 0` the rate is `base_lr`; at `step = t_max` it reaches
/// `eta_min`; halfway through it is the midpoint of the two.
pub fn cosine_annealing_lr(
  base_lr: Float,
  t_max: Int,
  eta_min: Float,
) -> Scheduler {
  Scheduler(
    kind: CosineAnnealingLr,
    base_lr: base_lr,
    step_count: 0,
    step_size: 0,
    gamma: 0.0,
    t_max: t_max,
    eta_min: eta_min,
    warmup_steps: 0,
    max_lr: 0.0,
    total_steps: 0,
    pct_start: 0.0,
  )
}

/// Build a linear-warmup scheduler.
///
/// Formula: `lr = base_lr * min(step / warmup_steps, 1.0)`.
///
/// The rate ramps linearly from 0 at `step = 0` to `base_lr` at
/// `step = warmup_steps`, then stays at `base_lr` forever.
pub fn linear_warmup(base_lr: Float, warmup_steps: Int) -> Scheduler {
  Scheduler(
    kind: LinearWarmup,
    base_lr: base_lr,
    step_count: 0,
    step_size: 0,
    gamma: 0.0,
    t_max: 0,
    eta_min: 0.0,
    warmup_steps: warmup_steps,
    max_lr: 0.0,
    total_steps: 0,
    pct_start: 0.0,
  )
}

/// Build a OneCycleLR scheduler.
///
/// Phase 1 (warmup), `step <= pct_start * total_steps`:
/// `lr = base_lr + (max_lr - base_lr) * (step / warmup_end)`.
///
/// Phase 2 (anneal), beyond that:
/// `lr = base_lr + 0.5 * (max_lr - base_lr) * (1 + cos(pi * progress))`
/// where `progress = (step - warmup_end) / (total_steps - warmup_end)`.
///
/// So the curve linearly rises from `base_lr` to `max_lr` and then cosine
/// anneals back to `base_lr` at `step = total_steps`.
pub fn one_cycle_lr(
  base_lr: Float,
  max_lr: Float,
  total_steps: Int,
  pct_start: Float,
) -> Scheduler {
  Scheduler(
    kind: OneCycleLr,
    base_lr: base_lr,
    step_count: 0,
    step_size: 0,
    gamma: 0.0,
    t_max: 0,
    eta_min: 0.0,
    warmup_steps: 0,
    max_lr: max_lr,
    total_steps: total_steps,
    pct_start: pct_start,
  )
}

/// Build an exponential-decay scheduler.
///
/// Formula: `lr = base_lr * gamma^step`.
///
/// Every step the rate is multiplied by `gamma`, giving a smooth
/// exponential decay (or growth, if `gamma > 1`).
pub fn exponential_lr(base_lr: Float, gamma: Float) -> Scheduler {
  Scheduler(
    kind: ExponentialLr,
    base_lr: base_lr,
    step_count: 0,
    step_size: 0,
    gamma: gamma,
    t_max: 0,
    eta_min: 0.0,
    warmup_steps: 0,
    max_lr: 0.0,
    total_steps: 0,
    pct_start: 0.0,
  )
}

// --- Step / query -----------------------------------------------------------

/// Compute the learning rate at the scheduler's current step without
/// advancing the step count.
pub fn scheduler_lr(s: Scheduler) -> Float {
  case s.kind {
    StepLr -> step_lr_value(s)
    CosineAnnealingLr -> cosine_value(s)
    LinearWarmup -> warmup_value(s)
    OneCycleLr -> one_cycle_value(s)
    ExponentialLr -> exponential_value(s)
  }
}

/// Advance the scheduler by one step and return the new learning rate.
pub fn scheduler_step(s: Scheduler) -> #(Scheduler, Float) {
  let next = Scheduler(..s, step_count: s.step_count + 1)
  #(next, scheduler_lr(next))
}

/// Apply the next learning rate to an optimizer. Advances the scheduler by
/// one step and returns the updated `(scheduler, optimizer)` pair.
pub fn apply_to_optimizer(
  s: Scheduler,
  opt: optim.Optimizer,
) -> #(Scheduler, optim.Optimizer) {
  let #(next, lr) = scheduler_step(s)
  #(next, optim.Optimizer(..opt, lr: lr))
}

// --- Internal helpers -------------------------------------------------------

fn step_lr_value(s: Scheduler) -> Float {
  let drops = case s.step_size {
    0 -> 0
    size -> s.step_count / size
  }
  s.base_lr *. pow(s.gamma, int.to_float(drops))
}

fn cosine_value(s: Scheduler) -> Float {
  let progress = case s.t_max {
    0 -> 0.0
    t -> int.to_float(s.step_count) /. int.to_float(t)
  }
  let cos_term = vm_scalar.cos(vm_constants.pi *. progress)
  s.eta_min +. 0.5 *. { s.base_lr -. s.eta_min } *. { 1.0 +. cos_term }
}

fn warmup_value(s: Scheduler) -> Float {
  case s.warmup_steps {
    0 -> s.base_lr
    w -> {
      let frac = int.to_float(s.step_count) /. int.to_float(w)
      let clamped = float.min(frac, 1.0)
      s.base_lr *. clamped
    }
  }
}

fn one_cycle_value(s: Scheduler) -> Float {
  let total_f = int.to_float(s.total_steps)
  let warmup_end_f = s.pct_start *. total_f
  let step_f = int.to_float(s.step_count)
  case step_f <=. warmup_end_f {
    True -> {
      let ramp = case warmup_end_f >. 0.0 {
        True -> step_f /. warmup_end_f
        False -> 1.0
      }
      s.base_lr +. { s.max_lr -. s.base_lr } *. ramp
    }
    False -> {
      let denom = total_f -. warmup_end_f
      let progress = case denom >. 0.0 {
        True -> { step_f -. warmup_end_f } /. denom
        False -> 1.0
      }
      let cos_term = vm_scalar.cos(vm_constants.pi *. progress)
      s.base_lr +. 0.5 *. { s.max_lr -. s.base_lr } *. { 1.0 +. cos_term }
    }
  }
}

fn exponential_value(s: Scheduler) -> Float {
  s.base_lr *. pow(s.gamma, int.to_float(s.step_count))
}

fn pow(base: Float, exp: Float) -> Float {
  case float.power(base, exp) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}
