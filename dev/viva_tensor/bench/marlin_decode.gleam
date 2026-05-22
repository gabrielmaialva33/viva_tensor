//// Benchmark: TinyLlama-1.1B decode FP8 vs Marlin W4A16 on RTX 4090 (SM89).
////
//// Marlin sweet spot is M=8-32 (batched-M), so at M=1 single-token decode
//// it's expected to regress vs FP8. This bench measures the regression
//// magnitude as input to Phase C (batched-M decode integration).
////
//// Run: gleam run -m viva_tensor/bench/marlin_decode

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva_tensor as vt

const model_path = "tmp/tinyllama/model.safetensors"

const prompts = [
  "Hello", "The quick brown fox", "Once upon a time", "In a galaxy far away",
]

pub fn main() {
  io.println(
    "viva_tensor — TinyLlama FP8 vs Marlin W4A16 decode bench (RTX 4090)",
  )
  io.println("")

  case vt.load_model(model_path) {
    Error(e) -> {
      io.println("FP8 baseline load failed: " <> e)
      io.println("Ensure " <> model_path <> " exists. Aborting.")
    }
    Ok(fp8_handle) -> {
      case vt.load_model_with_format(model_path, vt.MarlinW4A16) {
        Ok(marlin_handle) -> run_comparison(fp8_handle, marlin_handle)
        Error(e) -> {
          io.println("Marlin load failed: " <> e)
          io.println("Running FP8-only smoke instead...")
          io.println("")
          run_fp8_only(fp8_handle)
        }
      }
    }
  }
}

fn run_comparison(fp8: vt.ModelHandle, marlin: vt.ModelHandle) -> Nil {
  io.println("Config: max_new_tokens=32, temperature=0.0 (argmax), seed=42")
  io.println("Each prompt: 1 warmup run + 2 measured runs, averaged.")
  io.println("")

  list.each(prompts, fn(p) {
    io.println("--- prompt: " <> p <> " ---")
    let fp8_result = bench_one(fp8, p, vt.FP8W8A16)
    let marlin_result = bench_one(marlin, p, vt.MarlinW4A16)
    print_comparison(fp8_result, marlin_result)
    io.println("")
  })
}

fn run_fp8_only(fp8: vt.ModelHandle) -> Nil {
  io.println("Config: max_new_tokens=32, temperature=0.0 (argmax), seed=42")
  io.println("")
  list.each(prompts, fn(p) {
    io.println("--- prompt: " <> p <> " ---")
    let #(avg_ms, sample) = bench_one(fp8, p, vt.FP8W8A16)
    io.println(
      "FP8:    "
      <> fmt_ms(avg_ms)
      <> " ms/tok  |  "
      <> fmt_tok_s(avg_ms)
      <> " tok/s  |  output: "
      <> truncate(sample, 60),
    )
    io.println("")
  })
}

fn bench_one(
  handle: vt.ModelHandle,
  prompt: String,
  format: vt.WeightFormat,
) -> #(Float, String) {
  let opts =
    vt.GenerateOpts(
      max_new_tokens: 32,
      temperature: 0.0,
      top_k: vt.TopKInfinity,
      top_p: 1.0,
      seed: 42,
      stop_on_eos: False,
      weight_format: format,
    )

  // Warmup (discarded)
  let _ = vt.generate(handle, prompt, opts)

  case vt.generate(handle, prompt, opts), vt.generate(handle, prompt, opts) {
    Ok(r1), Ok(r2) -> {
      let avg = { r1.ms_per_token +. r2.ms_per_token } /. 2.0
      #(avg, r1.text)
    }
    Error(e), _ -> #(-1.0, "<error: " <> e <> ">")
    _, Error(e) -> #(-1.0, "<error: " <> e <> ">")
  }
}

fn print_comparison(fp8: #(Float, String), marlin: #(Float, String)) -> Nil {
  let #(fp8_ms, fp8_text) = fp8
  let #(marlin_ms, marlin_text) = marlin

  io.println(
    "FP8:    "
    <> fmt_ms(fp8_ms)
    <> " ms/tok  |  "
    <> fmt_tok_s(fp8_ms)
    <> " tok/s  |  output: "
    <> truncate(fp8_text, 60),
  )
  io.println(
    "Marlin: "
    <> fmt_ms(marlin_ms)
    <> " ms/tok  |  "
    <> fmt_tok_s(marlin_ms)
    <> " tok/s  |  output: "
    <> truncate(marlin_text, 60),
  )

  case fp8_ms >. 0.0 && marlin_ms >. 0.0 {
    True -> {
      let speedup = fp8_ms /. marlin_ms
      let note = case speedup <. 1.0 {
        True -> " (Marlin slower at M=1 — expected, gains in Phase C)"
        False -> " (Marlin faster)"
      }
      io.println("Speedup: " <> fmt_ms(speedup) <> "x" <> note)
    }
    False -> io.println("Speedup: n/a (one side errored)")
  }
}

fn fmt_ms(v: Float) -> String {
  // Two decimals: round(v * 100) / 100
  let scaled = float.round(v *. 100.0)
  let whole = scaled / 100
  let frac = case scaled - whole * 100 {
    n if n < 0 -> -n
    n -> n
  }
  let frac_str = case frac < 10 {
    True -> "0" <> int.to_string(frac)
    False -> int.to_string(frac)
  }
  int.to_string(whole) <> "." <> frac_str
}

fn fmt_tok_s(ms_per_tok: Float) -> String {
  case ms_per_tok >. 0.0 {
    True -> int.to_string(float.round(1000.0 /. ms_per_tok))
    False -> "n/a"
  }
}

fn truncate(s: String, n: Int) -> String {
  case string.length(s) > n {
    True -> string.slice(s, 0, n) <> "..."
    False -> s
  }
}
