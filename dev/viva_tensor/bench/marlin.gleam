import gleam/io

@external(erlang, "viva_tensor_zig", "marlin_w4a16_bench")
fn marlin_w4a16_bench(m: Int, n: Int, k: Int, groupsize: Int, iters: Int) -> Int

pub fn main() {
  io.println("Marlin W4A16 GEMM benchmark (kernel-only)")
  let _ = marlin_w4a16_bench(16, 4096, 4096, 128, 20)
  let _ = marlin_w4a16_bench(16, 4096, 11_008, 128, 20)
  let _ = marlin_w4a16_bench(64, 4096, 4096, 128, 20)
  let _ = marlin_w4a16_bench(256, 4096, 4096, 128, 20)
  Nil
}
