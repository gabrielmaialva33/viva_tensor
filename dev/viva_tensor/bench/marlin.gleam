import gleam/io
import gleam/list

@external(erlang, "viva_tensor_zig", "marlin_w4a16_bench")
fn marlin_w4a16_bench(m: Int, n: Int, k: Int, groupsize: Int, iters: Int) -> Int

pub fn main() {
  io.println("Marlin W4A16 GEMM benchmark (kernel-only, RTX 4090 SM89)")
  io.println("")
  io.println("Sweep K=N=4096 (square, transformer-typical) groupsize=128:")
  let ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  list.each(ms, fn(m) {
    let _ = marlin_w4a16_bench(m, 4096, 4096, 128, 50)
    Nil
  })
  io.println("")
  io.println(
    "Sweep K=4096 N=11008 (FFN gate-proj shape, TinyLlama-ish) groupsize=128:",
  )
  list.each([1, 4, 16, 64, 256], fn(m) {
    let _ = marlin_w4a16_bench(m, 11_008, 4096, 128, 30)
    Nil
  })
  Nil
}
