# viva_tensor

[![Hex](https://img.shields.io/hexpm/v/viva_tensor.svg)](https://hex.pm/packages/viva_tensor)
[![HexDocs](https://img.shields.io/badge/hex-docs-blueviolet)](https://hexdocs.pm/viva_tensor)
[![Tests](https://img.shields.io/badge/tests-792%20passing-2E8B57)](./test)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

A tensor library for Gleam on the BEAM. Provides a pure-Gleam tensor API
with zero-copy views, automatic broadcasting, and an optional native
acceleration layer that delegates to Intel MKL, cuBLAS / cuBLASLt,
cuSPARSELt, and CUTLASS when available.

The library works fully in pure BEAM (slow but portable) and transparently
upgrades to the native paths when the NIF shared object is loaded.

## What's new in v3.0

- **Public LLM API.** `viva_tensor.load_model(path)` loads a HuggingFace
  Llama-family SafeTensors checkpoint into an opaque `ModelHandle`, and
  `viva_tensor.generate(model, prompt, opts)` runs deterministic argmax or
  seeded temperature/top-k/top-p sampling.
- **Fast FP8 W8A16 decode.** The fused blocked path reaches **448 tok/s**
  on TinyLlama-1.1B FP8 W8A16 decode on RTX 4090, ahead of the local Ollama
  baseline at **352 tok/s**. Current public-handle validation is
  `2.31 ms/token` for TinyLlama-1.1B and `2.47 ms/token` for
  Llama-3.2-1B-Instruct.
- **Validated model coverage.** TinyLlama-1.1B and Llama-3.2-1B-Instruct run
  through the same `ModelHandle` API with weight tying, byte-level BPE,
  sharded SafeTensors, RoPE/GQA shape metadata, and blocked FP8 weights.
- **Reproducible sampling.** Generation options include `temperature`,
  `top_k`, `top_p`, `seed`, `max_new_tokens`, and `stop_on_eos`.

## Install

```bash
gleam add viva_tensor
```

## Quick start

```gleam
import gleam/result
import viva_tensor as t

pub fn main() {
  let a = t.zeros([1000, 1000])
  let b = t.random_uniform([1000, 1000])

  use c <- result.try(t.matmul(a, b))
  Ok(t.mean(c))
}
```

If you want native acceleration on Linux + CUDA, build the NIF locally:

```bash
make zig-cpu       # CPU-only (Intel MKL + AVX2)
make zig-cuda      # full path (CUTLASS + cuSPARSELt, requires CUDA toolkit)
```

The pure-BEAM path keeps working with no NIF; the upgrade is transparent
once `priv/viva_tensor_zig.so` is in place.

## What's in the box

- **Core tensor ops.** Create, reshape, slice, broadcast, gather, scatter,
  einsum, linalg (solve / det / lu / qr / cholesky), 50+ activations and
  pooling primitives.
- **Neural network layers.** Conv1d/2d/3d, attention, RNN/GRU/LSTM,
  embeddings, normalisations, optimisers (SGD / Adam / AdamW), schedulers,
  autograd with a `Tape` API plus standalone backward functions.
- **Pre-baked transformer architectures.** Llama, BERT, GPT, and T5
  blocks ready to wire up. Mixture of Experts with top-k routing.
  Tokenizers: WordPiece, BPE, Unigram (Viterbi), Whitespace, Char,
  SentencePiece.
- **Data + IO.** Dataloader, vision transforms / augmentations,
  diffusion samplers (DDPM / DDIM), HuggingFace SafeTensors loader,
  ONNX JSON import / runtime.
- **Native acceleration.** Intel MKL on CPU; FP16 / FP8 / INT8 / INT4
  paths on CUDA Tensor Cores; 2:4 structured sparsity via cuSPARSELt
  and CUTLASS.
- **Inference API (`2.2.101+`).** `prepack_*` + `linear_*` /
  `linear_gelu_fp8` / `linear_swiglu_fp8` against opaque
  `PackedWeight*` handles that own their device memory across calls.
- **LLM API (`3.0.0+`).** `load_model` + `generate` package the production
  decode path behind an opaque `ModelHandle` for Llama-family HF checkpoints.

## Measured performance (RTX 4090 + Ryzen 24-core)

Numbers are kernel-only via `cudaEvent_t` for GPU and `time.perf_counter`
for CPU, averaged over 30 iterations with one warm-up. Reproducer scripts
live in `dev/viva_tensor/bench/`.

### Matmul throughput

| Path                                       |     2048² |     4096² |     8192² |
| :----------------------------------------- | --------: | --------: | --------: |
| Pure BEAM matmul                           |     ~0.02 GFLOPS | ~0.01 GFLOPS |              — |
| MKL CPU dense FP64 (24-core, AVX2)         |     ~150 GFLOPS  | ~575 GFLOPS  |              — |
| cuBLASLt FP16 (heuristic)                  | 108 TFLOPS | 102 TFLOPS | 282 TFLOPS |
| cuBLASLt FP16 (algo-sweep best of 16)      | 120 TFLOPS | 266 TFLOPS | 305 TFLOPS |
| CUTLASS FP8 + FP16 accum                   | 218 TFLOPS | 392 TFLOPS | 618 TFLOPS |
| cuSPARSELt INT8 2:4 sparse                 | 228 TFLOPS | 629 TFLOPS | 872 TFLOPS |
| CUTLASS INT8 2:4 sparse (cfg=28)           | 250 TFLOPS | 634 TFLOPS | 750 TFLOPS |
| CUTLASS INT4 2:4 sparse (cfg=22/28)        | 600 TFLOPS | 1074 TFLOPS | 1355 TFLOPS |

### Versus PyTorch / NumPy (same hardware, same shape)

At 4096²:

| Backend                                       |       TFLOPS |
| :-------------------------------------------- | -----------: |
| NumPy CPU FP32                                |          0.8 |
| PyTorch CPU FP32 (oneDNN + MKL)               |          1.2 |
| PyTorch GPU FP32 (cuBLAS)                     |         20.6 |
| PyTorch GPU FP16 (cuBLAS Tensor Core)         |        148.9 |
| PyTorch GPU BF16 (cuBLAS Tensor Core)         |        149.0 |
| PyTorch GPU FP8 E4M3 (`torch._scaled_mm`)     |        307.8 |
| viva_tensor cuBLASLt FP16 (algo-sweep)        |        265.6 |
| viva_tensor CUTLASS FP8 + FP16 accum          |        392.5 |
| viva_tensor cuSPARSELt INT8 2:4               |        628.0 |
| viva_tensor CUTLASS INT4 2:4                  |       1074.3 |

The FP8 win over PyTorch's `_scaled_mm` (392 vs 308 TFLOPS) comes from
the CUTLASS f16-accum path that bypasses the GeForce FP32-accum
half-rate cap; sparse paths have no `torch._scaled_mm` equivalent today.
Full methodology + raw numbers in
[bench/results/matmul_showdown.md](bench/results/matmul_showdown.md).

### Text generation

| Model / runtime                              | Decode speed |
| :------------------------------------------- | -----------: |
| TinyLlama-1.1B FP8 W8A16 best decode run     | 448 tok/s    |
| Ollama local baseline                        | 352 tok/s    |
| TinyLlama-1.1B via `ModelHandle`             | 2.31 ms/token |
| Llama-3.2-1B-Instruct via `ModelHandle`      | 2.47 ms/token |

## Inference API

Higher-level surface for the championship kernels, designed for actual
inference (the `cutlass_*_bench` NIFs are throughput probes — they
allocate and free GEMM tensors on every call). Prepack once, run linear
forwards many times.

For Llama-family models, prefer the public `ModelHandle` API:

```gleam
import viva_tensor as t

let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")
let opts = t.default_generate_opts()
let assert Ok(result) = t.generate(model, "Hello", opts)
```

```gleam
import gleam/option.{None}
import viva_tensor as t

// w has shape [in_features, out_features]
let assert Ok(packed) = t.prepack_fp8_weight(w)
let assert Ok(output) = t.linear_fp8(input, packed, None)
```

Variants:

- `prepack_fp8_weight` + `linear_fp8` / `linear_gelu_fp8` (FP8 dense)
- `prepack_int8_sparse_24_weight` + `linear_int8_sparse` (INT8 2:4 sparse)
- `prepack_int4_sparse_24_weight` + `linear_int4_sparse` (INT4 2:4 sparse)
- `linear_swiglu_fp8` (fused gate + up + silu*mul, Llama FFN building
  block)

FP8 linear is validated end-to-end against an FP32 reference matmul
across `K = 32 ... 4096`; relative L2 error stays bounded (5–13%
depending on `K`, on uniform random fixtures). Tighter bands need an
FP32 output buffer (a planned CUTLASS template change).

Numerical / known limitations are tracked in
[bench/plans/INFERENCE_API_PLAN.md](bench/plans/INFERENCE_API_PLAN.md).

## Architecture (text)

```
+-------------------------------------------------------------+
|   Gleam library                                             |
|   - src/viva_tensor.gleam            (public facade)        |
|   - src/viva_tensor/* (core, nn, optim, models, vision, …)  |
|   - test/                                                   |
|   - dev/  (benchmarks, examples)                            |
+----------------------------+--------------------------------+
                             |
                             | Erlang stubs (@external)
                             v
+-------------------------------------------------------------+
|   src/viva_tensor_zig.erl  (NIF wrapper)                    |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|   zig_src/  (native NIF, compiled with Zig + nvcc)          |
|   - nif_entry.c   (dispatch + resource types)               |
|   - nif_cpu_ops.c, nif_packed_weight.c, nif_prepack_*.c     |
|   - nif_linear_*.c, nif_linear_swiglu_fp8.cu                |
|   - cuda_fp8_cutlass.cu, cuda_fp16_bench.cu, …              |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|   Native backends                                           |
|   - Intel MKL (CPU dgemm/sgemm)                             |
|   - cuBLAS / cuBLASLt (FP32 / FP16 / FP8)                   |
|   - cuSPARSELt (FP8 / FP16 / INT8 2:4 sparse)               |
|   - CUTLASS (FP8 f16-accum, INT8/INT4 sparse)               |
+-------------------------------------------------------------+
```

## Documentation

- API guide: [docs/en/api.md](docs/en/api.md)
- Project structure: [docs/en/project-structure.md](docs/en/project-structure.md)
- FFI architecture: [docs/en/ffi-architecture.md](docs/en/ffi-architecture.md)
- Stability policy: [docs/en/stability.md](docs/en/stability.md)
- Technical paper: [docs/en/paper.md](docs/en/paper.md)
- Inference API roadmap: [bench/plans/INFERENCE_API_PLAN.md](bench/plans/INFERENCE_API_PLAN.md)
- HexDocs site: <https://hexdocs.pm/viva_tensor>
- Project site: <https://gabrielmaialva33.github.io/viva_tensor/>

## Build targets

```bash
make build           # pure Gleam build
make test            # gleam test (uses NIF if priv/viva_tensor_zig.so exists)
make test-no-nif     # runs gleam test with the NIF temporarily moved aside

make zig-cpu         # build NIF without CUDA (Intel MKL + AVX2 SIMD only)
make cutlass-libs    # compile CUTLASS / cuSPARSELt static libs via nvcc
make zig-cuda        # cutlass-libs + full NIF with CUDA paths

make bench           # write benchmarks/latest.txt
make docs            # gleam docs build
```

Override CUDA arch / paths via env variables:

```bash
make zig-cuda CUDA_ARCH=sm_89 NVCC=/usr/local/cuda/bin/nvcc \
              CUTLASS_INCLUDE=/usr/include
```

## Requirements

- Gleam 1.16.0+, OTP 28+
- Zig 0.15.2+ (for the NIF build)
- Intel MKL (CPU BLAS path) — `apt install intel-mkl`
- CUDA 13+ with cuBLAS / cuBLASLt (GPU)
- cuSPARSELt 0.8.1+ (sparse ops)
- CUTLASS 4.x headers (FP8 and INT4 sparse)

The CPU-only build (`make zig-cpu`) drops the CUDA / cuSPARSELt
requirements; everything else still works.

## API stability

`import viva_tensor as t` is the stable surface. Submodules under
`viva_tensor/core`, `viva_tensor/nn`, `viva_tensor/native`, etc. remain
internal until their contracts are documented and covered by
compatibility tests. See [docs/en/stability.md](docs/en/stability.md)
for the policy and `test/public_api_contract_test.gleam` for the
automated check.

## Third-party

Some kernels are derived from permissively licensed work (CUTLASS,
cuSPARSELt examples, ggml inspiration on the block-wise quantisation
side). Original notices are kept in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

[MIT](./LICENSE).
