# Llama-style 推理端到端

本指南演示使用公共 `viva_tensor.load_model` / `viva_tensor.generate` API，
在 TinyLlama-1.1B 上完成一次真实的 text-in / text-out forward pass。相同调用序列已在
Llama-3.2-1B-Instruct 上验证；模型特定差异从 `config.json` 和 SafeTensors metadata 加载。

## 前置条件

```
sudo apt install build-essential
# CUDA 12.x + driver 555+ for Ada SM89

# Project root:
make cutlass-libs     # builds CUTLASS + cuSPARSELt static archives
make zig              # builds the NIF .so

# Get TinyLlama-1.1B (chat-tuned, 4-bit-friendly):
mkdir -p tmp/tinyllama
cd tmp/tinyllama
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json
```

## 端到端运行

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")

  let opts =
    t.GenerateOpts(
      max_new_tokens: 20,
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

预期输出（使用 `block_size=16`、argmax sampling）：

```
Prompt:     "Hello"
Generated:  ", I am interested to bookmark this job for [company/brand name], please? I am"
Throughput: ~2.31 ms/token on TinyLlama-1.1B
Argmax token after BOS: 529 (matches HF transformers fp32 reference)
```

## Pipeline

```
Prompt text
   ↓ viva_tensor.generate
   ↓ viva_tensor_tokenizer_ffi:encode  (BPE, byte-fallback)
[token_ids]
   ↓ embed_row(EmbedTbl, token_id)     (bf16 row from SafeTensors)
hidden_state [hidden_size]
   ↓ ×22 transformer blocks:
   │     rmsnorm
   │     → Q/K/V proj (linear_fp8_w8a16)
   │     → RoPE rotation
   │     → GQA attention (32 Q heads / 4 KV heads)
   │     → KV cache append
   │     → O proj (linear_fp8_w8a16)
   │     → residual
   │     → rmsnorm
   │     → gate/up (linear_fp8_w8a16)
   │     → silu(gate)·up
   │     → down (linear_fp8_w8a16)
   │     → residual
hidden_state
   ↓ final rmsnorm + lm_head (linear_fp8_w8a16)
logits [vocab=32000]
   ↓ argmax or sample (temp/top-k/top-p)
next_token_id
   ↓ viva_tensor_tokenizer_ffi:decode
text
```

## `load_model` 做了什么

`viva_tensor.load_model(path)` 将较底层的 SafeTensors 和 prepack steps 封装到一个可复用的
`ModelHandle` 后面。内部每个 linear weight 遵循这个 shape：

```erlang
{ok, Header} = viva_tensor_safetensors_ffi:open_header(Path),
{ok, Bf16}   = viva_tensor_safetensors_ffi:read_tensor_bf16(
                 Header, <<"model.layers.0.self_attn.q_proj.weight">>),
Fp32         = viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16),
%% HF stores weight as [out, in]; viva_tensor prepack expects [in, out].
{ok, Trans}  = viva_tensor_safetensors_ffi:transpose_fp32(Fp32, OutF, InF),
{ok, {Resource, _, _, _}} =
    viva_tensor_zig:nt_prepack_fp8_blocked(Trans, [InF, OutF], 16).
```

对于 32000×2048 的 LM head，transpose 过去在纯 Erlang 中需要约 ~20 秒。
快速路径位于 `nif_transpose.c`，运行约 ~180 ms。

## 为什么是 block_size=16

| Per-channel only      | block_size=128    | **block_size=16**  | HF reference |
| :-------------------- | :---------------- | :----------------- | :----------- |
| Q proj ratio: 1.234×  | 1.150×            | **1.077×**         | 1.000×       |
| K proj ratio: —       | 1.108×            | **1.018×**         | 1.000×       |
| argmax token after BOS| 6763              | **529 ✅**         | 529          |

`block_size=16` 是能让 argmax token 与 HF transformers fp32 参考对齐的最小 block。
它是推荐的推理默认值。内存开销可以忽略（约 weight bytes 的 ~3%）。

## Sampling

设置 `temperature > 0.0` 并传入 `top_k`、`top_p` 和 `seed`：

```gleam
let opts =
  t.GenerateOpts(
    max_new_tokens: 30,
    temperature: 0.8,
    top_k: t.TopK(40),
    top_p: 0.95,
    seed: 42,
    stop_on_eos: True,
  )

let assert Ok(result) = t.generate(model, "Hello", opts)
```

使用 `seed` 可使运行在不同机器间可复现。

## KV cache

当前 driver 将每层 K/V cache 保持为 Erlang lists（每个 token 追加一个 binary）。
对于 pos≤512 的 TinyLlama，每个 cache row 是 512 bytes，22 层每 token 总传输约 ~1 MB，
可以忽略。对于更长上下文，cache 应迁移到 persistent device resource（作为未来工作跟踪；见
`bench/plans/INFERENCE_API_PLAN.md`）。

## 性能

在 RTX 4090 上，公共 `ModelHandle` 路径已验证 TinyLlama-1.1B 为 `2.31 ms/token`，
Llama-3.2-1B-Instruct 为 `2.47 ms/token`。最佳 TinyLlama FP8 W8A16 decode run 达到
`448 tok/s`，高于本地 Ollama baseline 的 `352 tok/s`。

## 后续方向

当前瓶颈是每个 linear 的 host↔device round-trips，而不是 GPU compute。下一次吞吐跃迁
（5.5 → ~11 tok/sec）需要 fused single-block NIF，使 hidden state 在整个 block 中保持
device-resident。该工作跟踪于
[`bench/plans/INFERENCE_API_PLAN.md`](../../../bench/plans/INFERENCE_API_PLAN.md)
和 debug runner。

## Advanced / Debug

历史 reference driver 仍适用于 bisect 单个 weights 和 kernels：

```bash
erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

仅用于 maintainer debugging。新的应用代码应使用 `viva_tensor.load_model` 和
`viva_tensor.generate`。

## 故障排查

| 现象                                  | 可能原因                                                                      |
| :----------------------------------- | :---------------------------------------------------------------------------- |
| `nif_not_loaded` on prepack          | NIF 未构建：运行 `make zig`。                                                  |
| `bad_lib: function not found`        | Erlang stub list 不匹配：重建 Gleam 项目（`gleam build`）。                    |
| Token diverges from HF reference     | 使用了 per-channel scales，而不是 `block_size=16`。切换到 `nt_prepack_fp8_blocked`。 |
| Spurious Inf in output FP16          | `cuBLASLt` path FP16 输出饱和；已通过将所有路径路由到 FP32 输出缓冲区修复。更新 .so。 |
| Slow load (~3 min for 22 layers)     | 回退到了 Erlang transpose：确认 `nt_transpose_fp32` 已注册。                   |
