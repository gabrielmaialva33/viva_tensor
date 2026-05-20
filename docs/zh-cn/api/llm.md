# LLM ModelHandle API

`viva_tensor` 为存储为 SafeTensors 的 Llama-family HuggingFace 模型暴露公共 `ModelHandle` API。
它把生产级 TinyLlama decode path 封装成两个调用：加载一次模型，然后从缓存 handle 生成。

该 API 面向具有标准 Llama tensor 名称的本地 BF16 HF checkpoints：

- `model.embed_tokens.weight`
- `model.layers.N.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.N.mlp.{gate,up,down}_proj.weight`
- `model.layers.N.{input,post_attention}_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

如果 `config.json` 位于 SafeTensors 文件旁边，`viva_tensor` 会从中读取 hidden size、layer count、
head count、KV head count、RMSNorm epsilon、RoPE theta、intermediate size 和 vocab size。
否则，它会尽量从 tensor shapes 推断，并使用 TinyLlama-compatible defaults。

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

`temperature: 0.0` 使用 fused argmax decode-step NIF，以获得字节一致的可复现性。
`temperature > 0.0` 使用 fused top-k logits，再在 host 上执行 temperature、top-k、top-p
和 seeded multinomial sampling。

可复现采样：

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

`viva_tensor_llm:load/2` 接受：

| Option | Default | Notes |
| :-- | :-- | :-- |
| `num_layers` | detected from SafeTensors / `config.json` | 要加载的 decoder blocks 数量。 |
| `block_size` | `16` | decode-step 路径使用的 FP8 blocked prepack size。 |
| `tokenizer_path` | `<model>_tokenizer.json`, then sibling `tokenizer.json` fallback | HF tokenizer JSON。 |

## Generation Options

`viva_tensor_llm:generate/3` 接受：

| Option | Default | Notes |
| :-- | :-- | :-- |
| `max_new_tokens` | `50` | 最大生成 tokens 数。 |
| `temperature` | `0.0` | `0.0` 保持 argmax 路径和绝对可复现性；大于零的值启用 sampling。 |
| `top_k` | `infinity` | Sampling candidate cap。`infinity` 最多使用 256 个 fused top-k logits；显式值会被限制到 256。 |
| `top_p` | `1.0` | 应用于 fused candidate set 的 nucleus sampling 概率。 |
| `seed` | `42` | 确定性 seed；相同 prompt、model 和 options 会复现相同 sampled tokens。 |
| `stop_on_eos` | `true` | 发出 EOS 后停止。 |

## Cached vs Per Call

`ModelHandle` 会缓存：

- tokenizer state
- 作为 native resource 的 BF16 或 F16 embedding table
- 所有使用 blocked FP8 scales prepacked 的 layer weights
- fused QKV 和 gate-up packed weights
- final RMSNorm bytes
- packed `lm_head`
- RoPE frequency bytes
- 来自 `config.json` 和 tensor shapes 的 model metadata

每次 `generate` 调用都会在 prefill 前分配新的 KV caches。KV cache resources 在 decode 期间是可变的，
因此它们有意按调用分配，以保持一个 `ModelHandle` 可跨 prompts 复用。

## 已测试模型

| Model | Status | Decode speed | Notes |
| :-- | :-- | --: | :-- |
| TinyLlama-1.1B-Chat-v1.0 | validated | `2.31 ms/token` | `head_dim=64`、GQA fast path、byte-level BPE tokenizer。 |
| Llama-3.2-1B-Instruct | validated | `2.47 ms/token` | sharded SafeTensors、tied embeddings / `lm_head`、Llama-3 tokenizer path。 |
| NousResearch/Llama-2-7b-chat-hf | validated | `113.18 ms/token` | sharded F16 SafeTensors、`head_dim=128`、无 GQA；覆盖 dynamic CUDA fallback path。 |

同一个公共 API 驱动两个模型：

```gleam
let assert Ok(model) = t.load_model("tmp/llama32_1b/model-00001-of-00002.safetensors")
let opts = t.default_generate_opts()
let assert Ok(result) = t.generate(model, "Hello", opts)
```

## 性能

在 RTX 4090 上，当前公共 handle API 已验证 TinyLlama-1.1B 为 `2.31 ms/token`，
Llama-3.2-1B-Instruct 为 `2.47 ms/token`，NousResearch/Llama-2-7b-chat-hf 为
`113.18 ms/token`。Llama-2-7B 运行功能正确且文本连贯，但更慢，因为它走当前的
`head_dim=128` dynamic path。最佳 TinyLlama FP8 W8A16 decode run 达到 `448 tok/s`，
高于本地 Ollama baseline 的 `352 tok/s`。

Generation 仍然对每个 decoded token 调用一次 `nt_forward_decode_step/8`。Prefill 当前也是
token-by-token；batched prefill path 属于未来工作。

## 限制

- **Phi-2 不是 drop-in target。** 它的架构和 tensor 命名与 `ModelHandle` 使用的
  Llama-family loader contract 不同。
- **Llama-2-7B 当前使用较慢的 dynamic attention path。**
  `NousResearch/Llama-2-7b-chat-hf` 验证了 sharded F16 loading 和 `head_dim=128`
  正确性，但 decode throughput 尚未优化。
- **尚无 batched prefill path。** Decode kernel 针对 `batch=1` 优化；batched prompt processing
  仍表示为重复的 decode-step calls。
- **LLM decode 不使用 True FP8xFP8。** 数值已验证路径是带 blocked FP8 weights 的 W8A16。
  量化 single-token activation 每个 token 只能节省几 KB，却可能影响已在 TinyLlama 和
  Llama-3.2-1B 上验证的 argmax/EOS 行为。
