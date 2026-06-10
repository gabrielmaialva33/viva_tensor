//// Real end-to-end inference demo on Llama-3.2-1B-Instruct.
//// Loads the model through the public high-level API and runs a few
//// deterministic (argmax) generations, reporting throughput in tok/s.
////
//// Run: gleam run -m viva_tensor/infer_demo

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor as t

const model_path = "tmp/llama32_1b/model.safetensors"

const prompts = [
  "The capital of France is", "Q: What is 2 + 2? A:",
  "Once upon a time, in a kingdom of code,",
]

pub fn main() -> Nil {
  io.println("=== viva_tensor :: Llama-3.2-1B-Instruct (FP8 W8A16) ===\n")

  case t.load_model(model_path) {
    Error(reason) -> io.println("load_model failed: " <> reason)
    Ok(model) -> {
      let opts =
        t.GenerateOpts(
          max_new_tokens: 48,
          temperature: 0.0,
          top_k: t.TopKInfinity,
          top_p: 1.0,
          seed: 42,
          stop_on_eos: True,
          weight_format: t.FP8W8A16,
        )

      list.each(prompts, fn(prompt) {
        case t.generate(model, prompt, opts) {
          Error(reason) -> io.println("generate failed: " <> reason)
          Ok(gen) -> {
            let tok_s = case gen.ms_per_token >. 0.0 {
              True -> 1000.0 /. gen.ms_per_token
              False -> 0.0
            }
            io.println("PROMPT : " <> prompt)
            io.println("OUTPUT : " <> gen.text)
            io.println(
              "STATS  : "
              <> int.to_string(gen.total_tokens)
              <> " tok, "
              <> float.to_string(float_round(gen.ms_per_token, 2))
              <> " ms/tok, "
              <> float.to_string(float_round(tok_s, 1))
              <> " tok/s\n",
            )
          }
        }
      })
    }
  }
}

fn float_round(x: Float, places: Int) -> Float {
  let factor = int.to_float(pow10(places))
  int.to_float(float.round(x *. factor)) /. factor
}

fn pow10(n: Int) -> Int {
  case n {
    0 -> 1
    _ -> 10 * pow10(n - 1)
  }
}
