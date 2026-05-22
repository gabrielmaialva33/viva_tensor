//// Benchmark: TinyLlama-1.1B decode generate vs generate_batch.
////
//// Run: gleam run -m viva_tensor/bench/marlin_decode

import gleam/float
import gleam/int
import gleam/io
import viva_tensor as vt

const model_path = "tmp/tinyllama/model.safetensors"
const max_new_tokens = 32

pub fn main() {
  io.println("viva_tensor - TinyLlama batched decode bench")
  io.println("Config: max_new_tokens=32, temperature=0.0 (argmax), seed=42")
  io.println("")

  case vt.load_model(model_path) {
    Error(e) -> {
      io.println("FP8 baseline load failed: " <> e)
      io.println("Ensure " <> model_path <> " exists. Aborting.")
    }
    Ok(fp8_handle) -> {
      io.println("=== FP8 ===")
      run_sweep(fp8_handle, vt.FP8W8A16)
      io.println("")

      case vt.load_model_with_format(model_path, vt.MarlinW4A16) {
        Ok(marlin_handle) -> {
          io.println("=== Marlin ===")
          run_sweep(marlin_handle, vt.MarlinW4A16)
        }
        Error(e) -> {
          io.println("=== Marlin ===")
          io.println("load failed: " <> e)
        }
      }
    }
  }
}

fn run_sweep(handle: vt.ModelHandle, format: vt.WeightFormat) -> Nil {
  let opts =
    vt.GenerateOpts(
      max_new_tokens: max_new_tokens,
      temperature: 0.0,
      top_k: vt.TopKInfinity,
      top_p: 1.0,
      seed: 42,
      stop_on_eos: False,
      weight_format: format,
    )

  let baseline = bench_generate(handle, "Hello", opts)
  print_row(1, max_new_tokens, baseline, 1.0)

  case baseline {
    Ok(base) -> {
      let base_tps = tok_s(base)
      print_batch_row(handle, format, opts, 4, base_tps)
      print_batch_row(handle, format, opts, 16, base_tps)
    }
    Error(_) -> {
      print_batch_row(handle, format, opts, 4, 0.0)
      print_batch_row(handle, format, opts, 16, 0.0)
    }
  }
}

fn print_batch_row(
  handle: vt.ModelHandle,
  _format: vt.WeightFormat,
  opts: vt.GenerateOpts,
  batch: Int,
  base_tps: Float,
) -> Nil {
  let prompts = prompts_for_batch(batch)
  let total = batch * max_new_tokens
  let measured = bench_batch(handle, prompts, opts)
  let speedup = case measured, base_tps >. 0.0 {
    Ok(m), True -> tok_s(m) /. base_tps
    _, _ -> 0.0
  }
  print_row(batch, total, measured, speedup)
}

fn bench_generate(
  handle: vt.ModelHandle,
  prompt: String,
  opts: vt.GenerateOpts,
) -> Result(Measurement, String) {
  let _ = vt.generate(handle, prompt, opts)
  case timer_tc(fn() { vt.generate(handle, prompt, opts) }) {
    #(us, Ok(result)) -> Ok(Measurement(us: us, tokens: result.total_tokens))
    #(_, Error(e)) -> Error(e)
  }
}

fn bench_batch(
  handle: vt.ModelHandle,
  prompts: List(String),
  opts: vt.GenerateOpts,
) -> Result(Measurement, String) {
  let _ = vt.generate_batch(handle, prompts, opts)
  case timer_tc(fn() { vt.generate_batch(handle, prompts, opts) }) {
    #(us, results) -> {
      case total_tokens(results) {
        Ok(tokens) -> Ok(Measurement(us: us, tokens: tokens))
        Error(e) -> Error(e)
      }
    }
  }
}

fn total_tokens(results: List(Result(vt.GenerateResult, String))) -> Result(Int, String) {
  case results {
    [] -> Ok(0)
    [Ok(result), ..rest] -> {
      case total_tokens(rest) {
        Ok(n) -> Ok(result.total_tokens + n)
        Error(e) -> Error(e)
      }
    }
    [Error(e), ..] -> Error(e)
  }
}

type Measurement {
  Measurement(us: Int, tokens: Int)
}

fn print_row(
  batch: Int,
  total_tokens: Int,
  measured: Result(Measurement, String),
  speedup: Float,
) -> Nil {
  case measured {
    Ok(m) -> {
      io.println(
        "batch="
        <> int.to_string(batch)
        <> ":  "
        <> int.to_string(batch)
        <> " prompt"
        <> plural(batch)
        <> " x "
        <> int.to_string(max_new_tokens)
        <> " tok = "
        <> int.to_string(total_tokens)
        <> " tok in "
        <> fmt_ms(us_to_ms(m.us))
        <> " ms -> "
        <> fmt_float(tok_s(m))
        <> " tok/s ("
        <> fmt_float(speedup)
        <> "x vs batch=1)",
      )
    }
    Error(e) -> {
      io.println("batch=" <> int.to_string(batch) <> ": error: " <> e)
    }
  }
}

fn prompts_for_batch(batch: Int) -> List(String) {
  case batch {
    1 -> ["Hello"]
    4 -> [
      "Hello",
      "The quick brown fox",
      "Once upon a time",
      "In a galaxy far away",
    ]
    16 -> [
      "Hello",
      "The quick brown fox",
      "Once upon a time",
      "In a galaxy far away",
      "A small compiler",
      "The weather today",
      "Write a note",
      "Summarize this",
      "Explain tensors",
      "The next token",
      "A useful benchmark",
      "Brazil is",
      "CUDA kernels",
      "Machine learning",
      "Functional code",
      "Performance matters",
    ]
    _ -> ["Hello"]
  }
}

fn tok_s(m: Measurement) -> Float {
  case m.us > 0 {
    True -> int.to_float(m.tokens) *. 1_000_000.0 /. int.to_float(m.us)
    False -> 0.0
  }
}

fn us_to_ms(us: Int) -> Float {
  int.to_float(us) /. 1000.0
}

fn plural(n: Int) -> String {
  case n {
    1 -> ""
    _ -> "s"
  }
}

fn fmt_ms(v: Float) -> String {
  fmt_float(v)
}

fn fmt_float(v: Float) -> String {
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

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)
