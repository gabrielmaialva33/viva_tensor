//// Small regression benchmark for stable public API hot paths.

import gleam/int
import gleam/io
import viva_tensor as tensor
import viva_tensor/core/ffi

pub fn main() -> Nil {
  io.println("viva_tensor regression benchmark")
  time("broadcast_to [3] -> [4096,3]", benchmark_broadcast)
  time("softmax_axis [128,64] axis 1", benchmark_softmax_axis)
  time("matmul [32,32] @ [32,32]", benchmark_matmul)
}

fn time(label: String, work: fn() -> Nil) -> Nil {
  let started_at = ffi.now_microseconds()
  work()
  let duration_us = ffi.now_microseconds() - started_at
  io.println(label <> ": " <> int.to_string(duration_us) <> " us")
}

fn benchmark_broadcast() -> Nil {
  let input = tensor.from_list([1.0, 2.0, 3.0])
  case tensor.broadcast_to(input, [4096, 3]) {
    Ok(output) -> {
      let _ = tensor.to_list(output)
      Nil
    }
    Error(_) -> Nil
  }
}

fn benchmark_softmax_axis() -> Nil {
  let logits = tensor.fill([128, 64], 1.0)
  case tensor.softmax_axis(logits, 1) {
    Ok(output) -> {
      let _ = tensor.sum(output)
      Nil
    }
    Error(_) -> Nil
  }
}

fn benchmark_matmul() -> Nil {
  let a = tensor.fill([32, 32], 1.0)
  let b = tensor.fill([32, 32], 2.0)
  case tensor.matmul(a, b) {
    Ok(output) -> {
      let _ = tensor.sum(output)
      Nil
    }
    Error(_) -> Nil
  }
}
