//// Public LLM ModelHandle smoke test.

import gleam/io
import gleeunit
import gleeunit/should
import viva_tensor as t

const tinyllama_path = "tmp/tinyllama/model.safetensors"

const expected_hello = ", I am interested to bookmark this job for [company/brand name], please? I am interested in [job title/function] at [company name] in [city/state/country]?</s>"

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
      let assert Ok(model) = t.load_model(tinyllama_path)
      let opts =
        t.GenerateOpts(
          max_new_tokens: 50,
          temperature: 0.0,
          top_k: t.TopKInfinity,
          top_p: 1.0,
          seed: 42,
          stop_on_eos: True,
        )
      let assert Ok(result) = t.generate(model, "Hello", opts)

      result.text |> should.equal(expected_hello)
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
      let assert Ok(model) = t.load_model(tinyllama_path)
      let opts_42 =
        t.GenerateOpts(
          max_new_tokens: 20,
          temperature: 0.8,
          top_k: t.TopK(40),
          top_p: 0.95,
          seed: 42,
          stop_on_eos: True,
        )
      let opts_43 =
        t.GenerateOpts(
          max_new_tokens: 20,
          temperature: 0.8,
          top_k: t.TopK(40),
          top_p: 0.95,
          seed: 43,
          stop_on_eos: True,
        )

      let assert Ok(first) = t.generate(model, "Hello", opts_42)
      let assert Ok(second) = t.generate(model, "Hello", opts_42)
      let assert Ok(third) = t.generate(model, "Hello", opts_43)

      first.tokens |> should.equal(second.tokens)
      { first.tokens == third.tokens } |> should.be_false()
    }
  }
}

@external(erlang, "viva_tensor_llm", "path_exists")
fn path_exists(path: String) -> Bool
