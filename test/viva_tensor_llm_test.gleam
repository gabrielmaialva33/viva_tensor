//// Public LLM ModelHandle smoke test.

import gleam/io
import gleam/string
import gleeunit
import gleeunit/should
import viva_tensor as t

const tinyllama_path = "tmp/tinyllama/model.safetensors"

// Golden recaptured after the FP8 E4M3 encode fix (commit 59b9aa4); the prior
// golden was produced with the [256,448)-saturation bug and was incoherent.
const expected_hello = ", World!\n\n5. Node.js:\n\nNode.js is a popular server-side JavaScript runtime that allows you to write server-side applications in JavaScript. Here's an example of a Node.js server that listens"

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn tinyllama_model_handle_generate_hello_test() {
  case path_exists(tinyllama_path) {
    False -> {
      io.println(
        "tmp/tinyllama fixture not found; skipping ModelHandle LLM smoke test",
      )
      Nil
    }
    True -> {
      case t.load_model(tinyllama_path) {
        Error(_) -> {
          io.println(
            "ModelHandle load failed (likely NIF not loaded); skipping smoke test",
          )
          Nil
        }
        Ok(model) -> {
          let opts =
            t.GenerateOpts(
              max_new_tokens: 50,
              temperature: 0.0,
              top_k: t.TopKInfinity,
              top_p: 1.0,
              seed: 42,
              stop_on_eos: True,
              weight_format: t.FP8W8A16,
            )
          let assert Ok(result) = t.generate(model, "Hello", opts)

          result.text |> should.equal(expected_hello)
        }
      }
    }
  }
}

pub fn deterministic_sampling_with_seed_test() {
  case path_exists(tinyllama_path) {
    False -> {
      io.println(
        "tmp/tinyllama fixture not found; skipping ModelHandle sampling test",
      )
      Nil
    }
    True -> {
      case t.load_model(tinyllama_path) {
        Error(_) -> {
          io.println(
            "ModelHandle load failed (likely NIF not loaded); skipping sampling test",
          )
          Nil
        }
        Ok(model) -> {
          let opts_42 =
            t.GenerateOpts(
              max_new_tokens: 20,
              temperature: 0.8,
              top_k: t.TopK(40),
              top_p: 0.95,
              seed: 42,
              stop_on_eos: True,
              weight_format: t.FP8W8A16,
            )
          let opts_43 =
            t.GenerateOpts(
              max_new_tokens: 20,
              temperature: 0.8,
              top_k: t.TopK(40),
              top_p: 0.95,
              seed: 43,
              stop_on_eos: True,
              weight_format: t.FP8W8A16,
            )

          let assert Ok(first) = t.generate(model, "Hello", opts_42)
          let assert Ok(second) = t.generate(model, "Hello", opts_42)
          let assert Ok(third) = t.generate(model, "Hello", opts_43)

          first.tokens |> should.equal(second.tokens)
          { first.tokens == third.tokens } |> should.be_false()
        }
      }
    }
  }
}

pub fn generate_batch_matches_sequential_argmax_test() {
  case path_exists(tinyllama_path) {
    False -> {
      io.println(
        "tmp/tinyllama fixture not found; skipping generate_batch test",
      )
      Nil
    }
    True -> {
      case t.load_model(tinyllama_path) {
        Error(_) -> {
          io.println("ModelHandle load failed; skipping generate_batch test")
          Nil
        }
        Ok(model) -> {
          let opts = opts_argmax(8)
          let assert Ok(expected) = t.generate(model, "Hello", opts)
          let assert [Ok(first), Ok(second)] =
            t.generate_batch(model, ["Hello", "Hello"], opts)

          // Compare deterministic fields only; ms_per_token is wall-clock
          // timing and legitimately differs between batched and sequential.
          first.tokens |> should.equal(expected.tokens)
          first.text |> should.equal(expected.text)
          second.tokens |> should.equal(expected.tokens)
          second.text |> should.equal(expected.text)
        }
      }
    }
  }
}

pub fn generate_batch_empty_prompts_test() {
  case path_exists(tinyllama_path) {
    False -> {
      io.println(
        "tmp/tinyllama fixture not found; skipping generate_batch empty test",
      )
      Nil
    }
    True -> {
      case t.load_model(tinyllama_path) {
        Error(_) -> {
          io.println(
            "ModelHandle load failed; skipping generate_batch empty test",
          )
          Nil
        }
        Ok(model) ->
          t.generate_batch(model, [], opts_argmax(1)) |> should.equal([])
      }
    }
  }
}

pub fn generate_batch_isolates_prompt_errors_test() {
  case path_exists(tinyllama_path) {
    False -> {
      io.println(
        "tmp/tinyllama fixture not found; skipping generate_batch error test",
      )
      Nil
    }
    True -> {
      case t.load_model(tinyllama_path) {
        Error(_) -> {
          io.println(
            "ModelHandle load failed; skipping generate_batch error test",
          )
          Nil
        }
        Ok(model) -> {
          let opts = opts_argmax(8)
          let oversized_prompt = string.repeat("Hello ", 3000)
          let assert Ok(expected) = t.generate(model, "Hello", opts)
          let assert [Ok(first), Error(_)] =
            t.generate_batch(model, ["Hello", oversized_prompt], opts)

          first.tokens |> should.equal(expected.tokens)
          first.text |> should.equal(expected.text)
        }
      }
    }
  }
}

fn opts_argmax(max_new_tokens: Int) -> t.GenerateOpts {
  t.GenerateOpts(
    max_new_tokens: max_new_tokens,
    temperature: 0.0,
    top_k: t.TopKInfinity,
    top_p: 1.0,
    seed: 42,
    stop_on_eos: True,
    weight_format: t.FP8W8A16,
  )
}

// Qwen2 support (qkv bias + no BOS). Robust contains-check rather than a
// brittle full-text golden. Fixture lives under tmp/ (gitignored).
pub fn qwen2_generates_paris_test() {
  let qwen_path = "tmp/qwen25_05b/model.safetensors"
  case path_exists(qwen_path) {
    False -> {
      io.println("tmp/qwen25_05b fixture not found; skipping Qwen2 test")
      Nil
    }
    True -> {
      case t.load_model(qwen_path) {
        Error(_) -> {
          io.println("Qwen load failed (likely no NIF/GPU); skipping")
          Nil
        }
        Ok(model) -> {
          let assert Ok(result) =
            t.generate(model, "The capital of France is", opts_argmax(8))
          string.contains(result.text, "Paris") |> should.be_true
        }
      }
    }
  }
}

// Marlin W4A16 (4-bit) end-to-end. The packer transposes HF [out,in] -> [in,out]
// on read (weight_layout=1) and computes proper per-group (128) symmetric scales;
// a global scale collapses Llama outlier channels and produced garbage. Robust
// contains-check since 4-bit RTN drifts from FP8 over a long generation.
pub fn marlin_w4a16_generates_paris_test() {
  let path = "tmp/llama32_1b/model.safetensors"
  case path_exists(path) {
    False -> {
      io.println("tmp/llama32_1b fixture not found; skipping Marlin W4A16 test")
      Nil
    }
    True -> {
      case t.load_model_with_format(path, t.MarlinW4A16) {
        Error(_) -> {
          io.println("Marlin load failed (likely no NIF/GPU); skipping")
          Nil
        }
        Ok(model) -> {
          let opts =
            t.GenerateOpts(
              max_new_tokens: 8,
              temperature: 0.0,
              top_k: t.TopKInfinity,
              top_p: 1.0,
              seed: 42,
              stop_on_eos: True,
              weight_format: t.MarlinW4A16,
            )
          let assert Ok(result) =
            t.generate(model, "The capital of France is", opts)
          string.contains(result.text, "Paris") |> should.be_true
        }
      }
    }
  }
}

@external(erlang, "viva_tensor_llm", "path_exists")
fn path_exists(path: String) -> Bool
