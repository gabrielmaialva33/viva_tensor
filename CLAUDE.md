# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`viva_tensor` is a **pure-Gleam tensor library on the BEAM** with an optional
native CUDA/MKL backend driven by a Zig+C+CUDA NIF. It is **not** a wrapper —
the FP8 LLM inference path (`viva_tensor.load_model` / `viva_tensor.generate`)
is hand-tuned in `zig_src/` (CUTLASS, cuSPARSELt, custom GEMV, CUDA Graphs).

Pure-Gleam path works everywhere (slow). With `priv/viva_tensor_zig.so`
loaded, it transparently upgrades to native CUDA on Ada SM89 (RTX 4090).

- Version: `2.2.104` (see `gleam.toml`)
- Target: `erlang` (no JS target)
- Tests: ~792 with NIF loaded, ~791 without

## Build / test / verify

```bash
# Dependencies
gleam deps download

# Pure-Gleam build + tests
gleam build
gleam test                      # full suite
gleam test <name>_test          # single test file (gleeunit pattern)

# Quality gates run before commits/PRs (matches `make verify`)
gleam format --check src test dev
gleam check
gleam test
gleam docs build

# Format (mandatory before commit)
gleam format src test dev       # or: make fmt
```

### Native NIF (Zig + CUDA)

```bash
make cutlass-libs     # nvcc-builds libcutlass_fp8.a, libcusparselt_int8.a,
                      # libcutlass_int4_sparse.a (needs CUDA toolkit + cutlass headers)
make zig              # builds priv/viva_tensor_zig.so via `zig build`
make zig-cuda         # cutlass-libs + zig in one shot
make zig-cpu          # CPU-only NIF (MKL + SIMD, no CUDA) — for CI/non-GPU hosts
make zig-clean        # wipe NIF artifacts
make test-no-nif      # runs gleam test with priv/.so temporarily moved aside (Unix only)
```

Default CUDA arch is `sm_89` (Ada). Override with
`make cutlass-libs CUDA_ARCH=sm_86 CUTLASS_INCLUDE=/path/cutlass/include`.

### Benchmarks

```bash
make bench                          # gleam run -m viva_tensor/bench/full → bench/reports/
make bench-rtx                      # RTX-focused bench
gleam run -m viva_tensor/bench/regression
gleam run -m viva_tensor/bench/<kernel>   # int8, nf4, awq, flash, sparse, etc.
python3 bench/benchmark.py          # external Python comparison harness
```

## Architecture

Four layers, each owns a clear responsibility:

```
Gleam app  ─►  src/viva_tensor.gleam           (public API)
                src/viva_tensor/tensor.gleam   (public tensor surface)
                ─►  internal modules under src/viva_tensor/*  (marked internal in gleam.toml)
                    ─►  Erlang FFI wrappers (src/*.erl)
                        ─►  priv/viva_tensor_zig.so  (NIF, zig + CUDA)
                            ─►  pure-Gleam fallback if NIF not loaded
```

| Layer                                | Files                                       |
| :----------------------------------- | :------------------------------------------ |
| **Public Gleam API**                 | `src/viva_tensor.gleam`, `src/viva_tensor/tensor.gleam` |
| **Internal Gleam modules**           | `src/viva_tensor/{nn,quant,optim,models,native,core,io,...}/` |
| **Erlang FFI wrappers**              | `src/viva_tensor_*.erl` (one per concern: zig, blas, llm, safetensors, tokenizer, distributed, format, inference, nif, test) |
| **Native NIF source**                | `zig_src/` — Zig entrypoint + C bridges + `.cu` CUDA kernels (CUTLASS FP8/INT4-2:4, cuSPARSELt INT8, custom W8A16 GEMV, fused decode block) |
| **Native build**                     | `zig_src/build.zig` driven by Makefile      |

**Internal modules are listed explicitly in `gleam.toml` under `internal_modules`.**
When adding a new submodule, register it there if it shouldn't be part of the
public hexdocs surface.

### NIF contract (important)

- Every NIF registered in Rust/Zig/C **must** have a stub in the matching
  `.erl` module. Missing stub → `bad_lib` at load time. See `src/viva_tensor_zig.erl`
  for the canonical list of exported NIFs.
- Resource types (`PackedWeight`, `EmbeddingTable`, `KvCache`, `ModelHandle`)
  own device memory and must stay opaque to user code.
- The library always provides a pure-Gleam fallback for every public op. The
  NIF path is an optimization, not a requirement. Never call a NIF without
  a working fallback.

### LLM inference path

- Entry: `viva_tensor.load_model(path)` → opaque `ModelHandle`
- Loader: `src/viva_tensor_safetensors_ffi.erl` — accepts single `.safetensors`,
  HF `model.safetensors.index.json`, or any folder containing either.
  Supports BF16/F16/F32 and tied embeddings.
- Tokenizer: `src/viva_tensor_tokenizer_ffi.erl` — SentencePiece (`▁`) and
  byte-level BPE (GPT-2 / Llama-3).
- Decode kernel: `zig_src/cuda_block_forward.cu` (RMSNorm, RoPE, GQA flash
  attn, SiLU, residual) orchestrated by `zig_src/nif_forward_block.c` with
  full-token CUDA Graph capture + `cudaGraphExecUpdate`.
- Sampling: deterministic argmax or seeded `temperature`/`top_k`/`top_p`
  (reproducible across machines for fixed seed).
- Validated: TinyLlama-1.1B, Llama-3.2-1B-Instruct, Llama-2-7b-chat-hf (sharded F16).
  Phi-2 is partial (sharded discovery OK, arch ≠ Llama).

## Project conventions

These come from `.claude/rules/gleam.md`, `CONTRIBUTING.md`, and existing patterns:

- **Constructors must be unique within a module.** Name collision = compile error.
- Use `case` for pattern matching, never `match`.
- Pipes: `value |> function()` for chains, not nested `function(value)` calls.
- Use `use` for monadic binds (`Result`, `Option`).
- **`let assert` and `panic` are tests-only.** Production code: explicit `case`
  on every branch. Public API returns `Result` with a domain error type, not `String`.
- **gleam_json v3:** `json.parse(str, decoder)` with `gleam/dynamic/decode`.
  Never `json.decode` (removed).
- Error types are imported from their specific error module, not re-exported.
- Doc comments (`///`) on every public type and function.
- Never commit generated binaries, object files, archives, or local build
  outputs (`priv/*.so`, `zig_src/*.o`, `zig_src/lib*.a`, `bench/reports/`).
- After editing Gleam code, `gleam format` should leave the file unchanged
  (a hook runs it automatically in this environment).
- After editing NIF code (`zig_src/`), rebuild with `make zig` (or
  `make zig-cuda` for CUDA changes) and re-run `gleam test`.

## Public API stability

`src/viva_tensor.gleam` is the stable entry point. Two surfaces matter:

1. **Tensor API** — `Tensor`, `tensor()`, arithmetic, reductions, reshape,
   matmul, broadcast (`src/viva_tensor/tensor.gleam`).
2. **LLM API** — `load_model/1`, `generate/3`, `default_generate_opts/0`,
   `GenerateOpts`, `ModelHandle` (opaque). Erlang callers can use
   `viva_tensor_llm:load/2` and `viva_tensor_llm:generate/3` directly.
3. **Low-level inference** — `prepack_fp8_weight_blocked/2`,
   `linear_fp8_w8a16/3`. `PackedWeight` holds device-resident FP8 weight +
   scales for the lifetime of the resource.

Breaking changes to either surface require a major-version bump and a CHANGELOG entry.

## Dependencies

| Dep                | Floor              | Purpose                                |
| :----------------- | :----------------- | :------------------------------------- |
| `gleam_stdlib`     | `>= 0.44.0`        | core stdlib                            |
| `viva_telemetry`   | `>= 1.0.102`       | metrics / observability                |
| `gleam_json`       | `>= 3.0.0`         | `config.json`, tokenizer vocab         |
| `simplifile`       | `>= 2.0.0`         | file IO (loader)                       |
| `gleam_erlang`     | `>= 1.3.0`         | atom / dynamic interop                 |
| `viva_math`        | `>= 1.2.103`       | scalar activations (gelu/mish/softplus delegate here as of 2.2.103) |

Dev: `gleeunit`, `gleamy_bench`.

## Common pitfalls

- **Building without `cutlass-libs` first**: `make zig` (default) links against
  CUDA archives. On non-CUDA hosts, use `make zig-cpu` explicitly.
- **NIF not loading**: check `priv/viva_tensor_zig.so` exists and
  `build/dev/erlang/viva_tensor/priv/viva_tensor_zig.so` is symlinked/copied.
  `make zig` handles both paths.
- **CUDA tests fail in sandbox**: BEAM process needs access to `/dev/nvidia*`.
  Run outside restricted sandboxes for GPU paths.
- **Adding a new NIF**: register both in `zig_src/nif_entry.c` *and*
  `src/viva_tensor_zig.erl` (or matching `_ffi.erl`). Missing either side =
  `bad_lib` or `undef`.
- **Changing internal module visibility**: update `internal_modules` in
  `gleam.toml`, otherwise it leaks into hexdocs.

## Repo layout cheat-sheet

```
src/viva_tensor.gleam              Public API entry
src/viva_tensor/{tensor,axis,layout,named,quant,runtime,spec}.gleam
src/viva_tensor/core/              Tensor primitives, axis ops, broadcasting
src/viva_tensor/nn/                NN ops: activations, autograd, attention,
                                   conv, embedding, init, layers, losses, moe,
                                   norm, optim, pool, rnn, scheduler, transformer,
                                   flash_attention
src/viva_tensor/quant/             INT8 / NF4 / AWQ / FP8 quantization
src/viva_tensor/models/            Reference model impls
src/viva_tensor/io/                Safetensors, ONNX, HF loaders
src/viva_tensor/native/            NIF-backed wrappers
src/viva_tensor/observability/     Metrics, telemetry
src/viva_tensor/distributed/       Multi-device / multi-node
src/viva_tensor/{diffusion,vision,text,generate}/
src/viva_tensor_*.erl              FFI wrappers (one per concern)
zig_src/                           Zig + C + CUDA NIF sources
test/                              Gleam tests (gleeunit); fixtures/ has numpy goldens
bench/                             Bench harnesses (Gleam, Python, R)
dev/                               Reference Erlang/Gleam scripts (llama_forward etc.)
docs/{en,pt-br,zh-cn}/             Trilingual docs + landing pages
priv/                              Built NIF (.so/.dll) lives here
```

## Documentation

Trilingual docs live under `docs/{en,pt-br,zh-cn}/`. EN is canonical; PT-BR
and ZH-CN should mirror its structure (README + `api/` + `guides/` + `paper.md`
+ `reference/`). Pages registered in `gleam.toml` under `[[documentation.pages]]`
are bundled into hexdocs.
