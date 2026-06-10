//// Real instruct chat on Llama-3.2-1B-Instruct, using the Llama-3 chat
//// template so the model actually answers (vs raw continuation). The model
//// handle prepends BOS, so the template omits <|begin_of_text|>.
////
//// Run: gleam run -m viva_tensor/chat_demo

import gleam/io
import gleam/list
import viva_tensor as t

const model_path = "tmp/llama32_1b/model.safetensors"

const questions = [
  "What is the capital of France? Answer in one word.",
  "Write a one-sentence bedtime story about a robot.",
]

fn chat_prompt(question: String) -> String {
  "<|start_header_id|>user<|end_header_id|>\n\n"
  <> question
  <> "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

pub fn main() -> Nil {
  io.println("=== viva_tensor :: Llama-3.2-1B-Instruct chat ===\n")

  case t.load_model(model_path) {
    Error(reason) -> io.println("load_model failed: " <> reason)
    Ok(model) -> {
      let opts =
        t.GenerateOpts(
          max_new_tokens: 64,
          temperature: 0.0,
          top_k: t.TopKInfinity,
          top_p: 1.0,
          seed: 42,
          stop_on_eos: True,
          weight_format: t.FP8W8A16,
        )

      list.each(questions, fn(q) {
        case t.generate(model, chat_prompt(q), opts) {
          Error(reason) -> io.println("generate failed: " <> reason)
          Ok(gen) -> {
            io.println("USER : " <> q)
            io.println("BOT  : " <> gen.text)
            io.println("")
          }
        }
      })
    }
  }
}
