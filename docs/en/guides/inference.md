# Llama-style inference end-to-end

This guide walks through a real text-in / text-out forward pass on
TinyLlama-1.1B using `viva_tensor`. Same approach scales to other
Llama-architecture models (Llama-2 7B, 13B, etc) — the difference is
hidden_size / num_layers / intermediate_size, not the call sequence.

> The reference driver lives at `dev/llama_forward.erl`. The Gleam-side
> wrappers live under `viva_tensor` (re-exports of the prepack / linear
> NIFs). This guide shows the Erlang flow because that's where the
> current end-to-end driver lives, but the calls translate 1:1 to Gleam.

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

```erlang
erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

Expected output (with `block_size=16`, argmax sampling):

```
Prompt:     "Hello"
Generated:  ", I am interested to bookmark this job for [company/brand name], please? I am"
Throughput: ~5.6 tok/sec
Argmax token after BOS: 529 (matches HF transformers fp32 reference)
```

## The pipeline

```
Prompt text
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

## Weight loading

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

Replace the empty options map with a sampling config:

```erlang
{ok, _, Text} = llama_forward:run_generate_w8a16(
    22, <<"Hello">>, 30,
    #{temperature => 0.8, top_k => 40, top_p => 0.95, seed => 42},
    16).
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

Profile of a warm layer at 1 token / 22 layers / block_size=16 on
RTX 4090:

| Stage                | Time   | %    | Backend                                                |
| :------------------- | :----- | :--- | :----------------------------------------------------- |
| 7× linear FP8 GEMMs  | 4.6 ms | 83%  | CUTLASS / cuBLASLt — limited by NIF round-trips, not compute |
| RoPE Q+K             | 160 µs | 2.7% | Pure Erlang                                            |
| GQA softmax + attn   | 135 µs | 2.2% | Pure Erlang                                            |
| 2× RMSNorm           | 95 µs  | 1.5% | Pure Erlang                                            |
| fp16 encode (NIF)    | 42 µs  | 0.7% | C                                                      |
| silu·mul (NIF)       | 30 µs  | 0.5% | C                                                      |

Total per layer ≈ 6.0 ms (warm). The 7 linears average ~660 µs each;
pure cuBLAS for the same shape on a 4090 is 50–120 µs — the rest is BEAM
↔ NIF marshaling and PCIe round-trip overhead.

## What's next

The current bottleneck is host↔device round-trips per linear, not GPU
compute. The next throughput jump (5.5 → ~11 tok/sec) needs a fused
single-block NIF that keeps the hidden state device-resident across the
whole block. This is tracked at
[`bench/plans/INFERENCE_API_PLAN.md`](../../../bench/plans/INFERENCE_API_PLAN.md)
and [task #83 in the working journal](../../../dev/llama_forward.erl).

## Troubleshooting

| Symptom                              | Likely cause                                                                  |
| :----------------------------------- | :---------------------------------------------------------------------------- |
| `nif_not_loaded` on prepack          | NIF wasn't built — run `make zig`.                                            |
| `bad_lib: function not found`        | Erlang stub list mismatch — rebuild the Gleam project (`gleam build`).         |
| Token diverges from HF reference     | Using per-channel scales instead of `block_size=16`. Switch to `nt_prepack_fp8_blocked`. |
| Spurious Inf in output FP16          | `cuBLASLt` path FP16 output saturation — already fixed by routing all paths to FP32 output buffers. Update the .so. |
| Slow load (~3 min for 22 layers)     | Falling back to Erlang transpose — confirm `nt_transpose_fp32` is registered. |
