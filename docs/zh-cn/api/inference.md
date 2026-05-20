# 推理 API

`viva_tensor` 为 FP8 dense、INT8 2:4 sparse、INT4 2:4 sparse 以及 fused SwiGLU FFN
暴露稳定推理表面。所有路径都使用相同的不透明 `PackedWeight*` handle 类型，因此调用方可以在模型层级混合 dtype。

```gleam
import viva_tensor as t
```

> 需要 Native (CUDA + CUTLASS)。设置 `VIVA_NO_CUDA=1` 时，prepack 调用返回 `Error(_)`，
> linear 调用在 BEAM 上降级为 `nif_not_loaded`。纯 Gleam 张量 API
> （[`tensor.md`](tensor.md)）仍然可用。

## Packed weight handles

每种 dtype 都有一个由其 `prepack_*` 调用返回的不透明 handle。它们携带驻留设备端的量化 weight，
以及匹配的 `linear_*` 调用所需的 per-channel（或 per-block）scale buffer。

| Handle                       | 后端来源                        | 使用方                               |
| :--------------------------- | :------------------------------ | :----------------------------------- |
| `PackedWeightFp8`            | `nt_prepack_fp8` / `_blocked`   | `linear_fp8`, `linear_fp8_w8a16`, `linear_gelu_fp8`, `linear_swiglu_fp8` |
| `PackedWeightInt8Sparse`     | `nt_prepack_int8_sparse`        | `linear_int8_sparse`                 |
| `PackedWeightInt4Sparse`     | `nt_prepack_int4_sparse`        | `linear_int4_sparse`                 |

Handles 是引用计数的 Erlang resources；当 BEAM GC 回收 handle 时，device buffer 会被释放。
调用方代码不应直接调用 `cudaFree`，因为没有公共 release API。

## FP8 dense (E4M3)

```gleam
let assert Ok(packed) = t.prepack_fp8_weight(weight)
let assert Ok(out)    = t.linear_fp8(input, packed, bias)
```

| 函数                                   | 输出 dtype     | 说明                                                        |
| :------------------------------------- | :------------- | :---------------------------------------------------------- |
| `prepack_fp8_weight(weight)`           | `PackedWeightFp8` | Per-channel FP8 E4M3 scale；FP32 存储在设备端。          |
| `prepack_fp8_weight_blocked(w, blk)`   | `PackedWeightFp8` | Per-block-K scale（典型 `blk=16` 或 `128`）。缩小真实 LLM weights 上的数值差距。 |
| `linear_fp8(input, packed, bias)`      | Tensor (FP16)  | CUTLASS dense FP8 GEMM，FP32 输出缓冲区 + host dequant。    |
| `linear_fp8_w8a16(input, packed, bias)` | Tensor (FP16) | 通过 dequant kernel + cuBLAS FP16 GEMM 执行 FP16 input × FP8 weight。消除 FP8-input quantization 步骤。 |
| `linear_gelu_fp8(input, packed, bias)` | Tensor (FP16)  | 带 fused BIAS+GELU epilogue 的 cuBLASLt FP8 GEMM。          |
| `linear_swiglu_fp8(input, gate_pk, up_pk, bias)` | Tensor (FP16) | 两个 FP8 GEMMs + fused silu·mul，并在 kernel 内做 per-channel dequant。 |

### W8A16 vs W8A8

默认 `linear_fp8` 会即时量化输入（per-row absmax / 448），并运行真正的 FP8×FP8 GEMM。
对于带混合符号的真实 LLM weights，这会通过 accumulator noise 抵消约 50% 的输出通道。
`_w8a16` variant 跳过输入量化（输入保持 FP16），推荐用于推理。完整诊断过程见
[`guides/inference.md`](../guides/inference.md)。

### Block-wise scales

`prepack_fp8_weight_blocked(w, block_size)` 沿 K 轴每 `block_size` 个 weights 发出一个 FP32 scale，
而不是每个输出通道一个。对于 TinyLlama-1.1B，`block_size=16` 使 argmax token 与
HF transformers fp32 参考对齐。

### 公共 LLM decode path

应用代码应使用 `viva_tensor.load_model` 和 `viva_tensor.generate`；`ModelHandle` 契约见
[`llm.md`](llm.md)。内部实现中，`nt_embedding_table_new/3` 会将 `embed_tokens.weight`
一次性上传为 device resident FP16 table。随后 `nt_forward_decode_step/8` 接收 token id、
该 embedding resource、blocked layer records、final RMSNorm weights、packed `lm_head`、
KV cache resources、position 和 RoPE frequencies。它在每个 decoded token 的一次 NIF 调用中完成
embedding lookup、所有 transformer blocks、final RMSNorm、`lm_head` 和 argmax。

历史 dev harness 仍暴露该路径用于 kernel 调试：

```sh
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
  -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

## INT8 2:4 sparse (cuSPARSELt)

```gleam
let assert Ok(packed) = t.prepack_int8_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int8_sparse(input, packed, bias)
```

Magnitude-pruned 2:4 weight 以 cuSPARSELt 的 compressed format 存储。
在 Ada SM89 上运行约 ~1320 TOPS。Per-channel weight scale + per-row input scale，
在 int32 GEMM accumulator 后于 host 上 dequant。

## INT4 2:4 sparse (CUTLASS Sm80)

```gleam
let assert Ok(packed) = t.prepack_int4_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int4_sparse(input, packed, bias)
```

INT4 magnitude pruning + CUTLASS m16n8k128 sparse Tensor Op。Host prepack 会写出 kernel
所期望的 `ColumnMajorInterleaved<2>` layout 中的 ElementE metadata；正确性通过内置
`cutlass_int4_sparse_self_test` 验证。运行约 ~1854 TOPS。

## Sampling

一个单独的纯 Erlang 模块暴露标准采样 primitives：

```erlang
%% dev/llama_sampling.erl — also used directly from Gleam via FFI helpers
sample(Logits, #{temperature => 0.8, top_k => 40, top_p => 0.95, seed => 42}).
```

| 函数                       | 说明                                                    |
| :------------------------- | :------------------------------------------------------ |
| `argmax/1`                 | 从 raw logits 得到 `{TokenId, Logit}`。                 |
| `softmax/1`                | 稳定 softmax（max-subtraction）。                       |
| `sample/2`                 | 带 `temperature`、`top_k`、`top_p`、`seed` 的 multinomial。可复现。 |

## Tokenizer

```gleam
let assert Ok(tk) = viva_tensor_tokenizer_ffi.load("tmp/tinyllama/tokenizer.json")
let ids = viva_tensor_tokenizer_ffi.encode(tk, "Hello")
let text = viva_tensor_tokenizer_ffi.decode(tk, ids)
```

SentencePiece-style BPE，带 byte-fallback。在 TinyLlama-1.1B 上，encode/decode 与
HuggingFace `transformers` 位级一致。

## SafeTensors loader

```gleam
let assert Ok(header) = viva_tensor_safetensors_ffi.open_header(path)
let assert Ok(bf16)   = viva_tensor_safetensors_ffi.read_tensor_bf16(header, name)
let fp32              = viva_tensor_safetensors_ffi.bf16_to_fp32_binary(bf16)
let assert Ok(trans)  = viva_tensor_safetensors_ffi.transpose_fp32(fp32, rows, cols)
```

通过 OTP 27 的 `json` 模块解析 JSON header，读取 tensor bytes，并暴露由 NIF 支持的快速 transpose
（32×32 tiled，比纯 Erlang fallback 快约 ~110×）。

## 另见

- [`guides/inference.md`](../guides/inference.md) — 完整 TinyLlama-1.1B 端到端 walkthrough。
- [`guides/ffi-architecture.md`](../guides/ffi-architecture.md) — Gleam → Erlang → C/CUDA 边界契约。
- [`api/tensor.md`](tensor.md) — 不需要 CUDA 的纯 Gleam 张量 API。
