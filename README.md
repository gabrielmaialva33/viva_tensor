<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:76B900,100:1A1A1A&height=200&section=header&text=viva_tensor&fontSize=64&fontColor=fff&animation=twinkling&fontAlignY=35&desc=FP8%20LLM%20inference%20in%20pure%20Gleam%20on%20the%20BEAM&descSize=18&descAlignY=55" width="100%"/>

[![Gleam](https://img.shields.io/badge/Gleam-FFAFF3?style=for-the-badge&logo=gleam&logoColor=black)](https://gleam.run/)
[![BEAM](https://img.shields.io/badge/BEAM-A90533?style=for-the-badge&logo=erlang&logoColor=white)](https://www.erlang.org/)
[![OTP](https://img.shields.io/badge/OTP_27+-4B275F?style=for-the-badge)](https://www.erlang.org/doc/design_principles/des_princ)
[![CUDA](https://img.shields.io/badge/CUDA_12+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![SM89](https://img.shields.io/badge/RTX_4090-Ada_SM89-76B900?style=for-the-badge)](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
[![Tests](https://img.shields.io/badge/tests-792_passing-00875A?style=for-the-badge)](./test)
[![Version](https://img.shields.io/badge/version-2.2.102-CD5C5C?style=for-the-badge)](./CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-228B22?style=for-the-badge)](./LICENSE)

**[🇧🇷 Português](docs/pt-br/README.md)** · **[🇺🇸 English](docs/en/README.md)** · **[🇨🇳 中文](docs/zh-cn/README.md)**

---

*"Tensors speak Gleam. Kernels burn silicon. The BEAM holds the soul."*

</div>

---

> [!IMPORTANT]
> **viva_tensor IS NOT A WRAPPER.**
> It is a **production-grade FP8 LLM inference engine** written from scratch:
> hand-tuned CUDA kernels, blocked W8A16 GEMV, full-token CUDA Graphs, and a
> public `ModelHandle` API — all driven from Gleam on the BEAM.
>
> It is **faster than Ollama on the same hardware**.

---

## 🎯 Overview

A tensor library for Gleam on the BEAM. Provides a pure-Gleam tensor API
for portability, an inference API for FP8 / INT4 pair-4:8 / INT8-2:4 sparse
linear layers, and a public LLM `ModelHandle` API for Llama-family
HuggingFace checkpoints.

The library works fully in pure BEAM (slow but portable) and transparently
upgrades to the native CUDA path when the NIF is loaded.

| Property            | Value                                                |
| :------------------ | :--------------------------------------------------- |
| **Language**        | Pure Gleam (type-safe functional)                    |
| **Runtime**         | BEAM / OTP 27+                                       |
| **Native backend**  | CUDA 12 + CUTLASS + cuSPARSELt (SM89 / Ada)          |
| **Tests**           | 792 passing                                          |
| **Decode**          | `448 tok/s` TinyLlama-1.1B (vs Ollama 352)           |
| **Public API**      | `viva_tensor.load_model` / `viva_tensor.generate`    |

---

## ⚡ Quick Start

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor.git && cd viva_tensor
gleam deps download

# Optional: native CUDA backend (RTX 4090 / Ada SM89)
make cutlass-libs    # CUTLASS + cuSPARSELt static archives
make zig             # the NIF shared object

gleam test           # 792 tests, all pass with NIF loaded
```

### Generate text in 4 lines of Gleam

```gleam
import viva_tensor as t

let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")
let assert Ok(result) = t.generate(model, "Hello", t.default_generate_opts())
result.text
```

<details>
<summary><strong>📋 Prerequisites</strong></summary>

| Tool                      | Version     | Required for          |
| :------------------------ | :---------- | :-------------------- |
| Gleam                     | `>= 1.14`   | Build / pure-Gleam    |
| Erlang/OTP                | `>= 27`     | BEAM runtime          |
| CUDA toolkit              | `>= 12.0`   | Native inference path |
| NVIDIA GPU                | Ada+ (SM89) | FP8 / Tensor Cores    |
| `make` + `zig` + `clang`  | recent      | NIF build pipeline    |

The pure-Gleam path needs only Gleam + Erlang/OTP.

</details>

---

## 🏗️ Architecture

```
   ┌─────────────────────────────────────────────────────────┐
   │                  Gleam application code                 │
   │       viva_tensor.load_model / .generate / .Tensor      │
   └────────────────────────┬────────────────────────────────┘
                            │
   ┌────────────────────────▼────────────────────────────────┐
   │            Erlang public API (viva_tensor_llm)          │
   │  SafeTensors loader · BPE tokenizer · sampling · KV     │
   └────────────────────────┬────────────────────────────────┘
                            │
   ┌────────────────────────▼────────────────────────────────┐
   │          NIF dispatch (viva_tensor_zig.so + .erl)       │
   │   PackedWeight · EmbeddingTable · KvCache · ModelHandle │
   └─────────┬──────────────────────────────┬────────────────┘
             │                              │
   ┌─────────▼──────────┐         ┌─────────▼────────────────┐
   │ Pure-Gleam tensors │         │      CUDA kernels        │
   │   (no GPU needed)  │         │ W8A16 GEMV · FlashAttn   │
   │                    │         │ Full-token CUDA Graph    │
   │                    │         │ CUTLASS FP8/INT4 sparse  │
   └────────────────────┘         └──────────────────────────┘
```

<details>
<summary><strong>📋 Core Modules</strong></summary>

| Module                              | Description                                       |
| :---------------------------------- | :------------------------------------------------ |
| `viva_tensor`                       | Public Gleam API: tensors, prepack, linear, LLM   |
| `viva_tensor_llm`                   | `load_model` / `generate` — opaque `ModelHandle`  |
| `viva_tensor_zig`                   | NIF dispatch (Erlang stubs)                       |
| `viva_tensor_safetensors_ffi`       | HF SafeTensors loader, sharded support, BF16/F16  |
| `viva_tensor_tokenizer_ffi`         | SentencePiece + byte-level BPE (GPT-2/Llama-3)    |
| `zig_src/cuda_block_forward.cu`     | RMSNorm, RoPE, GQA flash attn, SiLU, residual     |
| `zig_src/nif_forward_block.c`       | Decode-step orchestration, CUDA Graph capture     |
| `zig_src/cuda_fp8_cutlass.cu`       | CUTLASS FP8 dense GEMM                            |
| `zig_src/nif_prepack_int_sparse.c`  | INT4 pair-4:8 / INT8 2:4 sparse weight prepack    |

</details>

---

## 📊 Performance

> All numbers measured on **RTX 4090** (Ada SM89) + Intel i9-13900K (32 threads @ 5.80 GHz). Reproducible
> via [`bench/`](./bench/) harness.

### Text generation — TinyLlama-1.1B-Chat (FP8 W8A16)

| Runtime                                      | Decode speed     |
| :------------------------------------------- | ---------------: |
| **viva_tensor — best run**                   | **`448 tok/s`**  |
| **Ollama local baseline (same model)**       | `352 tok/s`      |
| `viva_tensor.generate` (warm)                | `2.31 ms/token`  |
| `viva_tensor.generate` Llama-3.2-1B-Instruct | `2.47 ms/token`  |

### Validated models

| Model                            | Status        | Path                  | Notes                                  |
| :------------------------------- | :------------ | :-------------------- | :------------------------------------- |
| TinyLlama-1.1B-Chat              | ✅ validated  | single safetensors    | byte-identical baseline, `2.31 ms/tok` |
| Llama-3.2-1B-Instruct (unsloth)  | ✅ validated  | single safetensors    | tied embeddings, byte-level BPE, `2.47 ms/tok` |
| NousResearch/Llama-2-7b-chat-hf  | ✅ validated  | sharded F16 (13.5GB)  | `head_dim=128` dynamic path, `113 ms/tok` |
| Phi-2                            | ⚠️ partial   | sharded folder        | sharded discovery OK, Phi arch ≠ Llama |

### Quantized GEMM kernels (RTX 4090)

| Kernel                                  | Peak performance    | Backend          |
| :-------------------------------------- | ------------------: | :--------------- |
| INT8 2:4 sparse (cuSPARSELt)            | `1320 TOPS`         | cuSPARSELt       |
| INT4 pair-4:8 sparse (CUTLASS Sm80)     | `1854 TOPS`         | CUTLASS          |
| FP8 dense (CUTLASS E4M3 W8A8)           | `~660 TFLOPS`       | CUTLASS          |
| FP8 W8A16 blocked GEMV (custom)         | decode-optimized    | hand-tuned CUDA  |

Full methodology + raw numbers in [bench/results/matmul_showdown.md](bench/results/matmul_showdown.md).

---

## 🧬 Design Principles

| Principle                    | Description                                                          |
| :--------------------------- | :------------------------------------------------------------------- |
| **Honest numerics**          | argmax tokens stay byte-identical to HF reference fp32               |
| **Pure-Gleam fallback**      | Every API works without CUDA, just slower                            |
| **Owned device memory**      | `PackedWeight`, `EmbeddingTable`, `KvCache` are Erlang resources     |
| **Single-token by default**  | Decode is `batch=1` first; batched prefill is future work            |
| **No magic kernels**         | Every `.cu` file is human-written, benchmarked, and committed        |

---

## 🛠️ Public API

### High-level: LLM inference

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")

  let opts =
    t.GenerateOpts(
      max_new_tokens: 50,
      temperature: 0.0,           // argmax — deterministic
      top_k: t.TopKInfinity,
      top_p: 1.0,
      seed: 42,
      stop_on_eos: True,
    )

  let assert Ok(result) = t.generate(model, "Hello", opts)
  result.text
}
```

### Reproducible sampling

```gleam
let sampling_opts =
  t.GenerateOpts(
    max_new_tokens: 30,
    temperature: 0.8,
    top_k: t.TopK(40),
    top_p: 0.95,
    seed: 42,
    stop_on_eos: True,
  )
```

Same `seed` → same token sequence across machines.

### Low-level: quantized linears

```gleam
let assert Ok(packed) = t.prepack_fp8_weight_blocked(weight, 16)
let assert Ok(output) = t.linear_fp8_w8a16(input, packed, bias)
```

Prepack once, run linear forwards many times — the FP8 weight + scales
live on the device for the lifetime of the `PackedWeight` resource.

---

## 🗺️ Roadmap

| Phase                                         | Status |
| :-------------------------------------------- | :----: |
| Pure-Gleam tensors                            | ✅     |
| CUDA backend (CUTLASS + cuSPARSELt)           | ✅     |
| FP8 / INT4-2:4 / INT8-2:4 sparse kernels      | ✅     |
| Public `ModelHandle` API                      | ✅     |
| Sharded SafeTensors loader                    | ✅     |
| Byte-level BPE + SentencePiece tokenizers     | ✅     |
| Weight-tied embeddings                        | ✅     |
| Full-token CUDA Graph capture                 | ✅     |
| Reproducible temperature/top-k/top-p sampling | ✅     |
| FP16 weight dtype (Llama-2-7B)                | 🔄     |
| Batched prefill                               | ⏳     |
| Speculative decoding                          | ⏳     |
| Hopper SM90 / Blackwell FP4 / NVFP4           | ⏳     |

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
make cutlass-libs && make zig
gleam test          # 792 should pass with NIF loaded
make test-no-nif    # 791 should pass without NIF
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📚 Documentation

| Language     | Link                                  |
| :----------- | :------------------------------------ |
| 🇧🇷 Português | [docs/pt-br/](docs/pt-br/README.md)   |
| 🇺🇸 English   | [docs/en/](docs/en/README.md)         |
| 🇨🇳 中文      | [docs/zh-cn/](docs/zh-cn/README.md)   |

### Guides

- **[Getting started](docs/en/guides/getting-started.md)** — install, first run.
- **[LLM inference end-to-end](docs/en/guides/inference.md)** — load → tokenize → decode → sample.
- **[FFI architecture](docs/en/guides/ffi-architecture.md)** — Gleam ↔ Erlang ↔ C/CUDA boundaries.
- **[Project structure](docs/en/guides/project-structure.md)** — repo layout.

### API reference

- **[LLM ModelHandle](docs/en/api/llm.md)** — `load_model`, `generate`, tested models.
- **[Inference API](docs/en/api/inference.md)** — prepack + linear FP8 / INT-sparse.
- **[Tensor API](docs/en/api/tensor.md)** — pure-Gleam tensor surface.

### Technical paper

- **[Honest paper](docs/en/paper.md)** — what works, what doesn't, why.

---

## 📜 What's new in 2.2.102

- **Public LLM API.** `viva_tensor.load_model(path)` accepts a HuggingFace
  Llama-family checkpoint (single file, sharded, or folder) and returns
  an opaque `ModelHandle`. `viva_tensor.generate(model, prompt, opts)`
  drives deterministic argmax or seeded temperature/top-k/top-p sampling.
- **Fast FP8 W8A16 decode.** Hand-tuned `vt_w8a16_mmv_blocked_k16` GEMV
  with `uint4` vectorized loads, full-token CUDA Graph capture with
  `cudaGraphExecUpdate`, persistent device-resident KV caches, and a
  cuBLASLt plan cache.
- **Multi-model validated.** TinyLlama-1.1B and Llama-3.2-1B-Instruct
  pass byte-identical and through the same public API.
- **Sharded SafeTensors.** Loads single `.safetensors`, HF
  `model.safetensors.index.json`, or any folder containing either.
- **Byte-level BPE.** GPT-2 / Llama-3 byte-encoded vocabularies decode
  back to readable text; SentencePiece (`▁`) still works as before.
- **Tied embeddings.** Detects `tie_word_embeddings` from `config.json`
  and reuses `embed_tokens` as `lm_head` when set.

Full evolution from `63 tok/s` baseline to `448 tok/s` across 14 rounds
of optimization is documented in [CHANGELOG.md](./CHANGELOG.md).

---

<div align="center">

**Star if you believe BEAM can do LLM inference ⭐**

[![GitHub stars](https://img.shields.io/github/stars/gabrielmaialva33/viva_tensor?style=social)](https://github.com/gabrielmaialva33/viva_tensor)

*Created by Gabriel Maia · MIT License*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A1A1A,100:76B900&height=100&section=footer" width="100%"/>

</div>
