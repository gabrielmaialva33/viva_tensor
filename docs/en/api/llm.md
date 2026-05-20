# LLM ModelHandle API

`viva_tensor` exposes a public `ModelHandle` API for Llama-family HuggingFace
models stored as SafeTensors. It packages the production TinyLlama decode path
into two calls: load the model once, then generate from the cached handle.

The API is designed for local BF16 HF checkpoints with the standard Llama
tensor names:

- `model.embed_tokens.weight`
- `model.layers.N.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.N.mlp.{gate,up,down}_proj.weight`
- `model.layers.N.{input,post_attention}_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

If `config.json` is present next to the SafeTensors file, `viva_tensor` reads
the hidden size, layer count, head count, KV head count, RMSNorm epsilon, RoPE
theta, intermediate size, and vocab size from it. Otherwise it infers what it
can from tensor shapes and uses the TinyLlama-compatible defaults.

## Gleam

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")

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
  result.text
}
```

`temperature: 0.0` uses the fused argmax decode-step NIF for byte-identical
reproducibility. `temperature > 0.0` uses fused top-k logits plus host
temperature, top-k, top-p, and seeded multinomial sampling.

For reproducible sampling:

```gleam
let opts =
  t.GenerateOpts(
    max_new_tokens: 20,
    temperature: 0.8,
    top_k: t.TopK(40),
    top_p: 0.95,
    seed: 42,
    stop_on_eos: True,
  )

let assert Ok(result) = t.generate(model, "Hello", opts)
```

## Erlang

```erlang
{ok, Model} = viva_tensor_llm:load(
    <<"tmp/tinyllama/model.safetensors">>,
    #{block_size => 16}
),

{ok, Result} = viva_tensor_llm:generate(
    Model,
    <<"Hello">>,
    #{max_new_tokens => 50, temperature => 0.0}
),

#{tokens := Tokens,
  text := Text,
  ms_per_token := MsPerToken,
  total_tokens := TotalTokens} = Result.
```

## Load Options

`viva_tensor_llm:load/2` accepts:

| Option | Default | Notes |
| :-- | :-- | :-- |
| `num_layers` | detected from SafeTensors / `config.json` | Number of decoder blocks to load. |
| `block_size` | `16` | FP8 blocked prepack size used by the decode-step path. |
| `tokenizer_path` | `<model>_tokenizer.json`, then sibling `tokenizer.json` fallback | HF tokenizer JSON. |

## Generation Options

`viva_tensor_llm:generate/3` accepts:

| Option | Default | Notes |
| :-- | :-- | :-- |
| `max_new_tokens` | `50` | Maximum generated tokens. |
| `temperature` | `0.0` | `0.0` keeps the argmax path and absolute reproducibility; values above zero enable sampling. |
| `top_k` | `infinity` | Sampling candidate cap. `infinity` uses up to 256 fused top-k logits; explicit values are capped at 256. |
| `top_p` | `1.0` | Nucleus sampling probability applied over the fused candidate set. |
| `seed` | `42` | Deterministic seed; the same prompt, model, and options reproduce the same sampled tokens. |
| `stop_on_eos` | `true` | Stop after emitting EOS. |

## Cached vs Per Call

The `ModelHandle` caches:

- tokenizer state
- BF16 embedding table as a native resource
- all layer weights prepacked with blocked FP8 scales
- fused QKV and gate-up packed weights
- final RMSNorm bytes
- packed `lm_head`
- RoPE frequency bytes
- model metadata from `config.json` and tensor shapes

Each `generate` call allocates fresh KV caches before prefill. KV cache
resources are mutable during decode, so they are intentionally per call to keep
one `ModelHandle` reusable across prompts.

## Performance

On the Round 7 TinyLlama-1.1B benchmark with an RTX 4090, the fused decode-step
path runs at about `2.23 ms/token`. The public handle API keeps the same hot
loop: generation still calls `nt_forward_decode_step/8` once per decoded token,
so it should stay within the same `2.23-2.30 ms/token` band when CUDA graph
cache warmup is comparable.
