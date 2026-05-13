# Project Structure

`viva_tensor` follows the shape of mature Gleam packages rather than the shape
of the Gleam compiler monorepo.

The Gleam compiler repository is a large toolchain workspace with Rust crates,
language-server code, CLI packages, benchmark fixtures, package compiler tests,
and release tooling. That structure is useful for a compiler, but too broad for
a Hex package.

For a library, the pattern used across packages such as `gleam_stdlib`,
`gleam_json`, `gleam_http`, `gleam_erlang`, and community packages is smaller:
root metadata, `src/`, `test/`, optional `dev/` or `examples/`, and a
well-documented public module surface.

## Top-Level Directories

| Path | Purpose |
|:-----|:--------|
| `src/` | Packaged runtime code. This is what users compile as a dependency. |
| `src/viva_tensor.gleam` | Stable facade. User examples should prefer `import viva_tensor as t`. |
| `src/viva_tensor/` | Domain modules for tensor APIs, layout, axes, named tensors, quantization, neural-network helpers, and internal adapters. |
| `src/viva_tensor/core/` | Implementation core. Shape, tensor storage, errors, FFI wrappers, and layout math live here. |
| `src/viva_tensor/backend/` | Backend protocol and planning internals. |
| `src/viva_tensor/native/` | Optional native backend adapters and performance helpers for BLAS, CUDA, sparse kernels, and TFLOPS benchmarking. |
| `src/viva_tensor/observability/` | Telemetry and measurement helpers that support tests, benchmarks, and native-path diagnostics. |
| `src/viva_tensor/experimental/` | Research-oriented modules such as HDC, Horde, and LNS until they have stable package contracts. |
| `src/viva_tensor/quant/`, `nn/`, `optim/` | Domain-specific areas whose modules stay internal until their public contracts are stable. |
| `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl` | Erlang bridge modules required by the BEAM target. Keep these explicit and close to packaged source. |
| `test/` | Unit, behavior, public API contract, and no-NIF compatibility tests. |
| `dev/` | Development-only Gleam entrypoints for examples and benchmarks. These are intentionally outside `src/` so they do not become package API. |
| `bench/` | External benchmark scripts, reports, and comparison tooling. |
| `zig_src/` | Native C, CUDA, and Zig source for the optional NIF. |
| `priv/` | Runtime native artifacts loaded by Erlang when available. |
| `docs/` | Maintainer-authored guides and long-form documentation. |
| `.github/` | CI, issue templates, and pull request templates. |

## Public API Boundary

The root module is the stable user surface:

```gleam
import viva_tensor as t
```

Public companion modules are allowed when they represent a durable concept:

- `viva_tensor/layout`
- `viva_tensor/axis`
- `viva_tensor/named`

Implementation modules should remain listed in `gleam.toml` under
`internal_modules` until they have:

- documented shape and dtype contracts
- recoverable errors represented with `Result`
- tests through the root facade or a stable companion module
- no-NIF fallback behavior when native acceleration is optional
- generated documentation that is useful to package users

## File Naming

Use short domain names for stable modules: `layout`, `axis`, `named`.

Use explicit internal names for implementation modules: `core/tensor`,
`core/layout_math`, `backend/protocol`, `native/cuda`, `observability/metrics`,
`experimental/horde`, `quant/turboquant`.

Avoid putting benchmark, demo, or one-off experiment modules in `src/`. If a
module exists to run a measurement or show an idea, put it in `dev/` or `bench/`
until it becomes a supported runtime feature.

## FFI And Native Code

Gleam packages commonly keep target-specific FFI modules in `src/` with names
that match the package. `viva_tensor` keeps Erlang bridge modules in `src/` and
the native implementation in `zig_src/`.

The public Gleam API must not require the native library to be present unless a
function is explicitly documented as native-only. CI should continue to run the
no-NIF path.

## Documentation Pages

Generated HexDocs should include pages that help users understand the stable
package:

- API guide
- stability policy
- project structure
- technical paper

Native implementation notes, benchmark reports, and exploratory research can
stay in `bench/`, `docs/`, or future design notes without becoming public API.
