# Llama-style inference end-to-end

This guide walks through a real text-in / text-out forward pass on
TinyLlama-1.1B using the public `viva_tensor.load_model` /
`viva_tensor.generate` API. The same call sequence is validated on
Llama-3.2-1B-Instruct; model-specific differences are loaded from
`config.json` and SafeTensors metadata.

## Prerequisites

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

## End-to-end run

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

Expected output (with `block_size=16`, argmax sampling):

```
Prompt:     "Hello"
Generated:  ", I am interested to bookmark this job for [company/brand name], please? I am"
Throughput: ~2.31 ms/token on TinyLlama-1.1B
Argmax token after BOS: 529 (matches HF transformers fp32 reference)
```

## The pipeline

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

## What `load_model` does

`viva_tensor.load_model(path)` wraps the lower-level SafeTensors and prepack
steps behind a reusable `ModelHandle`. Internally, each linear weight follows
this shape:

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

The transpose used to take ~20 seconds for the 32000×2048 LM head in pure
Erlang. The fast path lives in `nif_transpose.c` and runs in ~180 ms.

## Why block_size=16

| Per-channel only      | block_size=128    | **block_size=16**  | HF reference |
| :-------------------- | :---------------- | :----------------- | :----------- |
| Q proj ratio: 1.234×  | 1.150×            | **1.077×**         | 1.000×       |
| K proj ratio: —       | 1.108×            | **1.018×**         | 1.000×       |
| argmax token after BOS| 6763              | **529 ✅**         | 529          |

`block_size=16` was the smallest block that aligns the argmax token with
the HF transformers fp32 reference. It is the recommended default for
inference. Memory overhead is negligible (~3% of weight bytes).

## Sampling

Set `temperature > 0.0` and pass `top_k`, `top_p`, and `seed`:

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

Use `seed` to make the run reproducible across machines.

## KV cache

The current driver keeps the per-layer K/V cache as Erlang lists (one
binary appended per token). For TinyLlama at pos≤512 each cache row is
512 bytes and the total transfer per token is ~1 MB across 22 layers —
negligible. For longer contexts the cache should move to a persistent
device resource (tracked as future work; see
`bench/plans/INFERENCE_API_PLAN.md`).

## Performance

On RTX 4090, the public `ModelHandle` path has been validated at
`2.31 ms/token` for TinyLlama-1.1B and `2.47 ms/token` for
Llama-3.2-1B-Instruct. A best TinyLlama FP8 W8A16 decode run reaches
`448 tok/s`, ahead of the local Ollama baseline at `352 tok/s`.

## What's next

The current bottleneck is host↔device round-trips per linear, not GPU
compute. The next throughput jump (5.5 → ~11 tok/sec) needs a fused
single-block NIF that keeps the hidden state device-resident across the
whole block. This is tracked at
[`bench/plans/INFERENCE_API_PLAN.md`](../../../bench/plans/INFERENCE_API_PLAN.md)
and the debug runner.

## Advanced / Debug

The historical reference driver remains useful for bisecting individual
weights and kernels:

```bash
erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

Use it for maintainer debugging only. New application code should use
`viva_tensor.load_model` and `viva_tensor.generate`.

## Troubleshooting

| Symptom                              | Likely cause                                                                  |
| :----------------------------------- | :---------------------------------------------------------------------------- |
| `nif_not_loaded` on prepack          | NIF wasn't built — run `make zig`.                                            |
| `bad_lib: function not found`        | Erlang stub list mismatch — rebuild the Gleam project (`gleam build`).         |
| Token diverges from HF reference     | Using per-channel scales instead of `block_size=16`. Switch to `nt_prepack_fp8_blocked`. |
| Spurious Inf in output FP16          | `cuBLASLt` path FP16 output saturation — already fixed by routing all paths to FP32 output buffers. Update the .so. |
| Slow load (~3 min for 22 layers)     | Falling back to Erlang transpose — confirm `nt_transpose_fp32` is registered. |
