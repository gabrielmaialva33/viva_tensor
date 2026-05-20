# Inference API

`viva_tensor` exposes a stable inference surface for FP8 dense, INT8 2:4
sparse, INT4 2:4 sparse, and the fused SwiGLU FFN. The same opaque
`PackedWeight*` handle types are used across all paths so callers can mix
dtypes at the model level.

```gleam
import viva_tensor as t
```

> Native (CUDA + CUTLASS) is required. With `VIVA_NO_CUDA=1` the prepack
> calls return `Error(_)` and the linear calls degrade to `nif_not_loaded`
> on the BEAM. The pure-Gleam tensor API ([`tensor.md`](tensor.md))
> remains available.

## Packed weight handles

Each dtype has an opaque handle returned from its `prepack_*` call. They
carry the device-resident quantized weight plus the per-channel (or
per-block) scale buffer that the matching `linear_*` call expects.

| Handle                       | Backed by                       | Used by                              |
| :--------------------------- | :------------------------------ | :----------------------------------- |
| `PackedWeightFp8`            | `nt_prepack_fp8` / `_blocked`   | `linear_fp8`, `linear_fp8_w8a16`, `linear_gelu_fp8`, `linear_swiglu_fp8` |
| `PackedWeightInt8Sparse`     | `nt_prepack_int8_sparse`        | `linear_int8_sparse`                 |
| `PackedWeightInt4Sparse`     | `nt_prepack_int4_sparse`        | `linear_int4_sparse`                 |

Handles are reference-counted Erlang resources; the device buffer is
released when the BEAM GC's the handle. Caller code should NOT call
`cudaFree` directly — there is no public release API.

## FP8 dense (E4M3)

```gleam
let assert Ok(packed) = t.prepack_fp8_weight(weight)
let assert Ok(out)    = t.linear_fp8(input, packed, bias)
```

| Function                             | Output dtype | Notes                                                       |
| :----------------------------------- | :----------- | :---------------------------------------------------------- |
| `prepack_fp8_weight(weight)`         | `PackedWeightFp8` | Per-channel FP8 E4M3 scale; FP32 stored on device.       |
| `prepack_fp8_weight_blocked(w, blk)` | `PackedWeightFp8` | Per-block-K scale (typical `blk=16` or `128`). Closes the numerical gap on real LLM weights. |
| `linear_fp8(input, packed, bias)`    | Tensor (FP16) | CUTLASS dense FP8 GEMM, FP32 output buffer + host dequant.  |
| `linear_fp8_w8a16(input, packed, bias)` | Tensor (FP16) | FP16 input × FP8 weight via dequant kernel + cuBLAS FP16 GEMM. Eliminates the FP8-input quantization step. |
| `linear_gelu_fp8(input, packed, bias)` | Tensor (FP16) | cuBLASLt FP8 GEMM with fused BIAS+GELU epilogue.          |
| `linear_swiglu_fp8(input, gate_pk, up_pk, bias)` | Tensor (FP16) | Two FP8 GEMMs + fused silu·mul with per-channel dequant inside the kernel. |

### W8A16 vs W8A8

The default `linear_fp8` quantizes the input on the fly (per-row absmax /
448) and runs a true FP8×FP8 GEMM. For real LLM weights with mixed signs
this can cancel out ~50% of output channels through accumulator noise.
The `_w8a16` variant skips input quantization (input stays FP16) and is
recommended for inference. See [`guides/inference.md`](../guides/inference.md)
for the full diagnostic story.

### Block-wise scales

`prepack_fp8_weight_blocked(w, block_size)` emits one FP32 scale per
`block_size` weights along the K axis instead of one per output channel.
For TinyLlama-1.1B `block_size=16` brings the argmax token in line with
the HF transformers fp32 reference.

### Public LLM decode path

Application code should use `viva_tensor.load_model` and
`viva_tensor.generate`; see [`llm.md`](llm.md) for the `ModelHandle` contract.
Internally, `nt_embedding_table_new/3` uploads `embed_tokens.weight` once as a
device resident FP16 table. `nt_forward_decode_step/8` then takes a token id,
that embedding resource, the blocked layer records, final RMSNorm weights,
packed `lm_head`, KV cache resources, position, and RoPE frequencies. It
performs the embedding lookup, all transformer blocks, final RMSNorm, `lm_head`,
and argmax inside one NIF call per decoded token.

The historical dev harness still exposes this path for kernel debugging:

```sh
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
  -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

## INT8 2:4 sparse (cuSPARSELt)

```gleam
let assert Ok(packed) = t.prepack_int8_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int8_sparse(input, packed, bias)
```

Magnitude-pruned 2:4 weight stored in cuSPARSELt's compressed format.
Runs ~1320 TOPS on Ada SM89. Per-channel weight scale + per-row input
scale, dequanted on host after the int32 GEMM accumulator.

## INT4 2:4 sparse (CUTLASS Sm80)

```gleam
let assert Ok(packed) = t.prepack_int4_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int4_sparse(input, packed, bias)
```

INT4 magnitude pruning + CUTLASS m16n8k128 sparse Tensor Op. The host
prepack writes the ElementE metadata in `ColumnMajorInterleaved<2>`
layout that the kernel expects; correctness is validated via a built-in
`cutlass_int4_sparse_self_test`. Runs ~1854 TOPS.

## Sampling

A separate pure-Erlang module exposes the standard sampling primitives:

```erlang
%% dev/llama_sampling.erl — also used directly from Gleam via FFI helpers
sample(Logits, #{temperature => 0.8, top_k => 40, top_p => 0.95, seed => 42}).
```

| Function                  | Notes                                                   |
| :------------------------ | :------------------------------------------------------ |
| `argmax/1`                | `{TokenId, Logit}` from raw logits.                     |
| `softmax/1`               | Stable softmax (max-subtraction).                       |
| `sample/2`                | Multinomial with `temperature`, `top_k`, `top_p`, `seed`. Reproducible. |

## Tokenizer

```gleam
let assert Ok(tk) = viva_tensor_tokenizer_ffi.load("tmp/tinyllama/tokenizer.json")
let ids = viva_tensor_tokenizer_ffi.encode(tk, "Hello")
let text = viva_tensor_tokenizer_ffi.decode(tk, ids)
```

SentencePiece-style BPE with byte-fallback. encode/decode is bit-exact
vs HuggingFace `transformers` on TinyLlama-1.1B.

## SafeTensors loader

```gleam
let assert Ok(header) = viva_tensor_safetensors_ffi.open_header(path)
let assert Ok(bf16)   = viva_tensor_safetensors_ffi.read_tensor_bf16(header, name)
let fp32              = viva_tensor_safetensors_ffi.bf16_to_fp32_binary(bf16)
let assert Ok(trans)  = viva_tensor_safetensors_ffi.transpose_fp32(fp32, rows, cols)
```

Parses the JSON header via OTP 27's `json` module, reads tensor bytes,
and exposes a fast NIF-backed transpose (32×32 tiled, ~110× faster than
the pure-Erlang fallback).

## See also

- [`guides/inference.md`](../guides/inference.md) — full TinyLlama-1.1B
  end-to-end walkthrough.
- [`guides/ffi-architecture.md`](../guides/ffi-architecture.md) — the
  Gleam → Erlang → C/CUDA boundary contract.
- [`api/tensor.md`](tensor.md) — the pure-Gleam tensor API that does
  not require CUDA.
