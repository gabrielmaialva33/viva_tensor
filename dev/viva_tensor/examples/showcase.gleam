//// End-to-end showcase exercising the new features from Sprints 1-4.
////
//// Run with:
////   gleam run -m viva_tensor/examples/showcase

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{None}
import viva_tensor as t
import viva_tensor/data/dataloader
import viva_tensor/nn/conv
import viva_tensor/nn/optim

pub fn main() {
  io.println("== viva_tensor showcase ==\n")

  pretty_repr_demo()
  io.println("")
  activations_demo()
  io.println("")
  einsum_demo()
  io.println("")
  linalg_demo()
  io.println("")
  normalization_demo()
  io.println("")
  attention_demo()
  io.println("")
  rnn_demo()
  io.println("")
  conv_demo()
  io.println("")
  training_loop_demo()

  io.println("\n== showcase complete ==")
}

// =============================================================================
// 1. Pretty repr (Sprint 1)
// =============================================================================

fn pretty_repr_demo() {
  io.println("-- 1. Pretty Repr --")
  let v = t.from_list([1.0, 22.0, 333.0, 4.0, 55.0])
  io.println(t.to_string(v))

  let assert Ok(m) =
    t.matrix(3, 4, [
      1.0, 2.5, 3.0, 4.25, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0, 11.5, 12.0,
    ])
  io.println(t.to_string(m))

  // Force scientific notation
  let mixed = t.from_list([1.0e-5, 1.0, 2.0, 3.0, 4.0])
  io.println(t.to_string(mixed))
}

// =============================================================================
// 2. Activations (Sprint 3)
// =============================================================================

fn activations_demo() {
  io.println("-- 2. Activations --")
  let x = t.from_list([-2.0, -1.0, 0.0, 1.0, 2.0])
  io.println("input:   " <> t.to_string(x))
  io.println("sigmoid: " <> t.to_string(t.sigmoid(x)))
  io.println("tanh:    " <> t.to_string(t.tanh(x)))
  io.println("relu:    " <> t.to_string(t.relu(x)))
  io.println("gelu:    " <> t.to_string(t.gelu(x)))
  io.println("swish:   " <> t.to_string(t.swish(x)))

  let logits = t.from_list([1.0, 2.0, 3.0, 4.0])
  let assert Ok(probs) = t.softmax(logits, 0)
  io.println("softmax([1,2,3,4]): " <> t.to_string(probs))
}

// =============================================================================
// 3. Einsum (Sprint 2)
// =============================================================================

fn einsum_demo() {
  io.println("-- 3. einsum --")
  let assert Ok(a) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(b) = t.matrix(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])

  // ij,jk->ik (matmul)
  let assert Ok(prod) = t.einsum("ij,jk->ik", [a, b])
  io.println("matmul (ij,jk->ik):")
  io.println(t.to_string(prod))

  // ij->i (row sums)
  let assert Ok(row_sums) = t.einsum("ij->i", [a])
  io.println("row sums (ij->i): " <> t.to_string(row_sums))

  // Outer product
  let v1 = t.from_list([1.0, 2.0, 3.0])
  let v2 = t.from_list([10.0, 20.0])
  let assert Ok(outer) = t.einsum("i,j->ij", [v1, v2])
  io.println("outer (i,j->ij):")
  io.println(t.to_string(outer))
}

// =============================================================================
// 4. Linalg (Sprint 3)
// =============================================================================

fn linalg_demo() {
  io.println("-- 4. Linear Algebra --")
  // Solve Ax = b for known answer x = [1, 1, 1]
  let assert Ok(a_mat) =
    t.matrix(3, 3, [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
  let b_vec = t.from_list([3.0, 5.0, 3.0])
  let assert Ok(x) = t.solve(a_mat, b_vec)
  io.println("solve Ax=b for [3,5,3]: " <> t.to_string(x))

  let assert Ok(det_val) = t.det(a_mat)
  io.println("det(A) = " <> float.to_string(det_val))

  let assert Ok(inv_a) = t.inv(a_mat)
  io.println("inv(A):")
  io.println(t.to_string(inv_a))

  let assert Ok(#(q, r)) = t.qr(a_mat)
  io.println("QR decomposition Q:")
  io.println(t.to_string(q))
  io.println("R:")
  io.println(t.to_string(r))
}

// =============================================================================
// 5. Normalization (Sprint 3)
// =============================================================================

fn normalization_demo() {
  io.println("-- 5. Normalization --")
  let layer = t.layer_norm_init(4)
  let assert Ok(input) =
    t.matrix(2, 4, [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
  let assert Ok(normalized) = t.layer_norm_forward(layer, input)
  io.println("LayerNorm output (mean≈0, var≈1 per row):")
  io.println(t.to_string(normalized))

  let rms = t.rms_norm_init(4)
  let assert Ok(rms_out) = t.rms_norm_forward(rms, input)
  io.println("RMSNorm output:")
  io.println(t.to_string(rms_out))
}

// =============================================================================
// 6. Attention (Sprint 4)
// =============================================================================

fn attention_demo() {
  io.println("-- 6. Attention --")
  // Use identity-ish weights for clarity.
  let assert Ok(q) = t.matrix(2, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  let assert Ok(k) = t.matrix(2, 4, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  let assert Ok(v) =
    t.matrix(2, 4, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

  let assert Ok(attn_out) = t.scaled_dot_product_attention(q, k, v, None, False)
  io.println("SDPA output (q=k=one-hot, v=values):")
  io.println(t.to_string(attn_out))

  io.println("causal_mask(4):")
  io.println(t.to_string(t.causal_mask(4)))
}

// =============================================================================
// 7. RNN (Sprint 4)
// =============================================================================

fn rnn_demo() {
  io.println("-- 7. RNN cells --")
  let lstm = t.lstm_cell_init(2, 3)
  let h0 = t.zeros([3])
  let c0 = t.zeros([3])

  let inputs = [
    t.from_list([1.0, 0.5]),
    t.from_list([0.5, -0.5]),
    t.from_list([-0.5, 1.0]),
  ]

  let assert Ok(#(all_h, final_h, final_c)) =
    t.lstm_sequence(lstm, inputs, h0, c0)
  io.println(
    "LSTM ran "
    <> int.to_string(list.length(all_h))
    <> " timesteps with zero weights (output stays zero):",
  )
  io.println("final hidden: " <> t.to_string(final_h))
  io.println("final cell:   " <> t.to_string(final_c))
}

// =============================================================================
// 8. Conv (Sprint 4)
// =============================================================================

fn conv_demo() {
  io.println("-- 8. Conv1d --")
  // Sliding sum on [1,2,3,4,5] with kernel [1,1,1]
  let input = t.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let assert Ok(input_3d) = t.reshape(input, [1, 1, 5])
  let kernel = t.from_list([1.0, 1.0, 1.0])
  let assert Ok(kernel_3d) = t.reshape(kernel, [1, 1, 3])

  let cfg =
    conv.Conv1dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: 3,
      stride: 1,
      padding: 0,
      weight: kernel_3d,
      bias: None,
    )

  let assert Ok(out) = t.conv1d_forward(cfg, input_3d)
  io.println("conv1d([1..5], kernel=[1,1,1]) sliding sum:")
  io.println(t.to_string(out))
}

// =============================================================================
// 9. Mini training loop (Sprints 2/3/4 together)
// =============================================================================

fn training_loop_demo() {
  io.println("-- 9. Mini training loop --")
  io.println(
    "Task: learn y = 2x + 1 with a single Linear weight (manual grad).",
  )

  // Build a small dataset y = 2x + 1
  let xs = [
    t.from_list([1.0]),
    t.from_list([2.0]),
    t.from_list([3.0]),
    t.from_list([4.0]),
  ]
  let ys = [
    t.from_list([3.0]),
    t.from_list([5.0]),
    t.from_list([7.0]),
    t.from_list([9.0]),
  ]
  let assert Ok(dataset) = t.dataset_from_lists(xs, ys)
  io.println("dataset size: " <> int.to_string(t.dataset_len(dataset)))

  let loader = t.data_loader_new(dataset, 2, False, False)
  io.println("loader batches: " <> int.to_string(t.data_loader_len(loader)))

  let assert Ok(batches) = t.data_loader_batches(loader)
  io.println("\nWalking batches:")
  list.each(batches, fn(batch) {
    let dataloader.Batch(inp, tgt) = batch
    io.println(
      "  input: " <> t.to_string(inp) <> "  target: " <> t.to_string(tgt),
    )
  })

  // Set up Adam optimizer + cosine scheduler on a single "weight" param.
  let weight_param = optim.Param(name: "w", value: t.from_list([0.0]))
  let opt = t.adam(0.1)
  let sched = t.cosine_annealing_lr(0.1, 10, 0.001)

  io.println("\nOptimizer + scheduler:")
  io.println("  initial lr: " <> float.to_string(opt.lr))
  let #(sched1, opt1) = t.apply_to_optimizer(sched, opt)
  io.println("  after 1 cosine step lr: " <> float.to_string(opt1.lr))
  let #(_sched2, opt2) = t.apply_to_optimizer(sched1, opt1)
  io.println("  after 2 cosine steps lr: " <> float.to_string(opt2.lr))

  // Demonstrate a "gradient step" on the toy param: pretend grad = -0.5
  let fake_grad = optim.GradPair(name: "w", grad: t.from_list([-0.5]))
  let assert Ok(#(_opt3, params)) = t.step(opt, [weight_param], [fake_grad])
  case params {
    [optim.Param(_, v)] ->
      io.println("\nAdam step with grad=-0.5: weight = " <> t.to_string(v))
    _ -> Nil
  }
}
