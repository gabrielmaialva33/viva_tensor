import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/nn/optim
import viva_tensor/nn/scheduler

pub fn main() {
  gleeunit.main()
}

// StepLR: step_size=2, gamma=0.5, base_lr=0.1
// Expected lr at steps 0, 2, 4: 0.1, 0.05, 0.025.
pub fn step_lr_basic_test() {
  let s0 = scheduler.step_lr(0.1, 2, 0.5)
  let #(s1, lr1) = scheduler.scheduler_step(s0)
  let #(s2, lr2) = scheduler.scheduler_step(s1)
  let #(s3, lr3) = scheduler.scheduler_step(s2)
  let #(_s4, lr4) = scheduler.scheduler_step(s3)

  numerics.lists_close(
    [scheduler.scheduler_lr(s0), lr2, lr4],
    [0.1, 0.05, 0.025],
    1.0e-9,
    1.0e-12,
  )
  |> should.be_true()

  // lr halves only on the boundary, not before
  numerics.floats_close(lr1, 0.1, 1.0e-9, 1.0e-12) |> should.be_true()
  numerics.floats_close(lr3, 0.05, 1.0e-9, 1.0e-12) |> should.be_true()
}

pub fn cosine_annealing_at_zero_test() {
  let s = scheduler.cosine_annealing_lr(0.1, 10, 0.001)
  numerics.floats_close(scheduler.scheduler_lr(s), 0.1, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

pub fn cosine_annealing_at_t_max_test() {
  let s = scheduler.cosine_annealing_lr(0.1, 10, 0.001)
  let advanced = advance(s, 10)
  numerics.floats_close(
    scheduler.scheduler_lr(advanced),
    0.001,
    1.0e-9,
    1.0e-12,
  )
  |> should.be_true()
}

pub fn cosine_annealing_at_half_test() {
  let base_lr = 0.1
  let eta_min = 0.001
  let s = scheduler.cosine_annealing_lr(base_lr, 10, eta_min)
  let advanced = advance(s, 5)
  numerics.floats_close(
    scheduler.scheduler_lr(advanced),
    { base_lr +. eta_min } /. 2.0,
    1.0e-9,
    1.0e-12,
  )
  |> should.be_true()
}

pub fn linear_warmup_zero_test() {
  let s = scheduler.linear_warmup(0.1, 10)
  numerics.floats_close(scheduler.scheduler_lr(s), 0.0, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

pub fn linear_warmup_at_warmup_test() {
  let s = scheduler.linear_warmup(0.1, 10)
  let advanced = advance(s, 10)
  numerics.floats_close(scheduler.scheduler_lr(advanced), 0.1, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

pub fn linear_warmup_past_warmup_test() {
  let s = scheduler.linear_warmup(0.1, 10)
  let advanced = advance(s, 20)
  numerics.floats_close(scheduler.scheduler_lr(advanced), 0.1, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

pub fn one_cycle_ramp_test() {
  // total_steps=100, pct_start=0.3 -> warmup_end_step=30
  let s = scheduler.one_cycle_lr(0.01, 0.1, 100, 0.3)

  // step 0 -> base_lr
  numerics.floats_close(scheduler.scheduler_lr(s), 0.01, 1.0e-9, 1.0e-12)
  |> should.be_true()

  // step = pct_start * total_steps = 30 -> max_lr
  let at_peak = advance(s, 30)
  numerics.floats_close(scheduler.scheduler_lr(at_peak), 0.1, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

pub fn one_cycle_end_test() {
  let s = scheduler.one_cycle_lr(0.01, 0.1, 100, 0.3)
  let at_end = advance(s, 100)
  numerics.floats_close(scheduler.scheduler_lr(at_end), 0.01, 1.0e-9, 1.0e-12)
  |> should.be_true()
}

// ExponentialLR: base_lr=0.1, gamma=0.9
// Expected lr at steps 0, 1, 2: 0.1, 0.09, 0.081.
pub fn exponential_lr_test() {
  let s0 = scheduler.exponential_lr(0.1, 0.9)
  let #(s1, lr1) = scheduler.scheduler_step(s0)
  let #(_s2, lr2) = scheduler.scheduler_step(s1)

  numerics.lists_close(
    [scheduler.scheduler_lr(s0), lr1, lr2],
    [0.1, 0.09, 0.081],
    1.0e-9,
    1.0e-12,
  )
  |> should.be_true()
}

pub fn apply_to_optimizer_test() {
  // Cosine schedule guarantees a real lr change after one step.
  let s = scheduler.cosine_annealing_lr(0.1, 10, 0.001)
  let opt = optim.sgd(0.1)

  let #(next_s, next_opt) = scheduler.apply_to_optimizer(s, opt)

  // Scheduler step count advanced by 1.
  next_s.step_count |> should.equal(1)

  // Optimizer lr matches scheduler's lr at step 1.
  let expected_lr = scheduler.scheduler_lr(next_s)
  numerics.floats_close(next_opt.lr, expected_lr, 1.0e-12, 1.0e-12)
  |> should.be_true()

  // lr actually changed.
  case next_opt.lr == 0.1 {
    True -> should.fail()
    False -> Nil
  }
}

fn advance(s: scheduler.Scheduler, n: Int) -> scheduler.Scheduler {
  case n {
    0 -> s
    _ -> {
      let #(next, _) = scheduler.scheduler_step(s)
      advance(next, n - 1)
    }
  }
}
