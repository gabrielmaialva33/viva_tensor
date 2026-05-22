# Marlin W4A16 + Batched-M Decode Plan

Internal planning doc — drives v2.2.106. **Bump version only at the very end.**

## Phase B — Marlin integration into decode path

| Sub | Goal | Validation |
|---|---|---|
| **B1** | Port `Layer.pack()` (Python) → `viva_marlin_pack.c`. Includes `_perm` (4096-entry), `_scale_perm` (64-entry), `_scale_perm_single` (32-entry) precomputed tables, quantize+offset+clamp, tile 16×16 transpose, perm-shuffle, 8-nibble bitpack into uint32. | Standalone unit test (C): packs a small known weight via the C port, compares **byte-for-byte** against output of `tmp/marlin/marlin/__init__.py:Layer.pack()` for same input. Diff = 0 across all bytes of `B` and `s`. |
| **B2** | NIF wrapper `viva_marlin_w4a16_prepack(weight_fp16_iolist, scales_fp16_iolist, K, N, groupsize) → MarlinPackedResource`. Resource owns device `B`, `s`, `workspace`. Destructor frees device memory. Erlang stub in `viva_tensor_zig.erl`. | Erlang call from BEAM produces resource; `nt_to_list` of `B` and `s` matches B1 reference output. |
| **B3** | Public Gleam API `viva_tensor.prepack_marlin_w4a16(weight: Tensor, scales: Tensor, groupsize: Int) → Result(MarlinPacked, TensorError)`. Opaque type. | Gleam unit test: prepack a small FP16 weight, assert resource is valid, assert shape matches. |
| **B4** | Decode path swap: extend `GenerateOpts` with `weight_format: WeightFormat` (default `FP8W8A16`, new variant `MarlinW4A16`). In `nif_forward_block.c`, when `weight_format == MarlinW4A16`, dispatch `viva_marlin_w4a16_mm` instead of `vt_w8a16_mmv_blocked_k16` for QKV / O / gate / up / down linears. `BlockState` carries the per-format weight handle. | TinyLlama decode runs end-to-end via Marlin path. Output is **not** byte-identical to FP16 ref (Marlin is 4-bit lossy) — log perplexity loss but do not assert byte-equality. Add `perplexity_loss < 0.05` regression test. |
| **B5** | Bench: `dev/viva_tensor/bench/marlin_decode.gleam` — TinyLlama decode tok/s FP8 vs Marlin, same prompt, same opts. Report ms/token + tok/s for both. | Real number on RTX 4090. Target: ≥ FP8 throughput (Marlin's win is batch>1, but should not regress single-token). |

## Phase C — Batched-M decode

| Sub | Goal | Validation |
|---|---|---|
| **C1** | `BlockState` refactor: every scratch buffer that's `[hidden]` becomes `[max_batch_size, hidden]`. Add `cur_batch_size` field. Allocate up to `max_batch_size` at `with_block_state` time. | Single-prompt (`batch=1`) decode still produces byte-identical output to current main. |
| **C2** | KV cache becomes `[max_batch_size][num_layers][...]`. Each batch row owns its own KV slots. | Single-prompt path unchanged; smoke test passes. |
| **C3** | Argmax / sample vetorizado: outputs `[batch_size]` token vector per step. Each batch row drives its own next-token selection. | Two different prompts in a 2-batch call produce two different (but each individually correct) token streams. |
| **C4** | CUDA Graph cache key includes `batch_size`. Re-capture if batch size changes. | Graph capture works for `batch=1`, `batch=4`, `batch=16` without conflicts. |
| **C5** | `viva_tensor_llm:generate_batch/3` rewritten: instead of `spawn_monitor` per prompt, makes **one** call to a new `nt_forward_decode_step_batched(state, batch_size, ...)` that drives all M prompts through a single CUDA graph per step. | All 16 prompts of `generate_batch_test` produce the same outputs as today's per-process path. |
| **C6** | Bench: 5 runs of `generate_batch` on TinyLlama with 16 prompts. Target: 4–8× single-prompt throughput. Zero crashes. | Real numbers. Compare against current 1.55× honest baseline. |

## Release closeout (only after C6 is green)

1. Bump `gleam.toml` to `2.2.106`.
2. CHANGELOG.md: write 2.2.106 entry with B1–C6 highlights, perplexity loss, decode tok/s numbers.
3. `gleam format` + `gleam check` + `gleam test` (5 runs) + `make zig` + `make cutlass-libs` clean.
4. `git tag -a v2.2.106` + push + `gh release create`.

## Risk notes

- **B1 byte-identical check is non-negotiable.** If `pack()` port diverges by one bit, decode will silently produce garbage (no compile error, no runtime error — just bad tokens). Always validate against the Python reference in `tmp/marlin/marlin/__init__.py` before moving on.
- **Marlin constraints are hard:** `K % 128 == 0`, `N % 256 == 0`, `groupsize in {-1, 128}`. TinyLlama hidden=2048 (OK), inter=5632 (`% 256 == 0` ✓), Llama-3.2-1B hidden=2048, inter=8192 (✓). Reject other shapes at prepack time with a clear error.
- **Phase C must keep `batch=1` paths byte-identical.** Any divergence at `batch=1` = production regression.
