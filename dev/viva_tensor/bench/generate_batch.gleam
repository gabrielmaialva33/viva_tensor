import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor as t
import viva_tensor/core/ffi

const tinyllama_path = "tmp/tinyllama/model.safetensors"

const prompt = "Hello"

pub fn main() -> Nil {
  case path_exists(tinyllama_path) {
    False ->
      io.println(
        "tmp/tinyllama fixture not found; skipping generate_batch bench",
      )
    True -> {
      case t.load_model(tinyllama_path) {
        Error(reason) -> io.println("load_model failed: " <> reason)
        Ok(model) -> {
          let opts =
            t.GenerateOpts(
              max_new_tokens: 16,
              temperature: 0.0,
              top_k: t.TopKInfinity,
              top_p: 1.0,
              seed: 42,
              stop_on_eos: True,
            )
          let prompts = list.repeat(prompt, 16)
          let #(seq_us, seq_tokens) = run_sequential(model, prompts, opts)
          let #(batch_us, batch_tokens) = run_batch(model, prompts, opts)
          let speedup = int.to_float(seq_us) /. int.to_float(batch_us)

          io.println("generate_batch TinyLlama bench")
          print_result("sequential", seq_us, seq_tokens)
          print_result("batch", batch_us, batch_tokens)
          io.println("speedup: " <> float.to_string(speedup) <> "x")
        }
      }
    }
  }
}

fn run_sequential(
  model: t.ModelHandle,
  prompts: List(String),
  opts: t.GenerateOpts,
) -> #(Int, Int) {
  let started_at = ffi.now_microseconds()
  let tokens =
    list.fold(prompts, 0, fn(total, one_prompt) {
      case t.generate(model, one_prompt, opts) {
        Ok(result) -> total + result.total_tokens
        Error(reason) -> {
          io.println("sequential error: " <> reason)
          total
        }
      }
    })
  #(ffi.now_microseconds() - started_at, tokens)
}

fn run_batch(
  model: t.ModelHandle,
  prompts: List(String),
  opts: t.GenerateOpts,
) -> #(Int, Int) {
  let started_at = ffi.now_microseconds()
  let results = t.generate_batch(model, prompts, opts)
  let tokens =
    list.fold(results, 0, fn(total, result) {
      case result {
        Ok(generation) -> total + generation.total_tokens
        Error(reason) -> {
          io.println("batch error: " <> reason)
          total
        }
      }
    })
  #(ffi.now_microseconds() - started_at, tokens)
}

fn print_result(label: String, us: Int, tokens: Int) -> Nil {
  let tok_s = int.to_float(tokens) *. 1_000_000.0 /. int.to_float(us)
  io.println(
    label
    <> ": "
    <> int.to_string(us)
    <> " us, "
    <> int.to_string(tokens)
    <> " tokens, "
    <> float.to_string(tok_s)
    <> " tok/s",
  )
}

@external(erlang, "viva_tensor_llm", "path_exists")
fn path_exists(path: String) -> Bool
