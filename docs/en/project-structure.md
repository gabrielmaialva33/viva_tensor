# Project Structure

`viva_tensor` is a Gleam package with a stable root facade, internal
implementation modules, and optional native acceleration. Keep this split clear
when adding features: package users should depend on the public Gleam contract,
not on the current native or planner internals.

## Package Layout

| Path                                                               | Purpose                                                                                                                                                                 |
|:-------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/viva_tensor.gleam`                                            | Stable public facade. User examples should prefer `import viva_tensor as t`.                                                                                            |
| `src/viva_tensor/axis.gleam`, `layout.gleam`, `named.gleam`        | Public companion modules for durable tensor concepts.                                                                                                                   |
| `src/viva_tensor/tensor.gleam`                                     | Internal tensor implementation used by the facade. It owns pure operations, native dispatch, shape behavior, and fallback paths.                                        |
| `src/viva_tensor/core/`                                            | Internal storage, shape, dtype, errors, layout math, and FFI wrappers.                                                                                                  |
| `src/viva_tensor/backend/`                                         | Backend protocol and capability descriptions used by planner-style selection code.                                                                                      |
| `src/viva_tensor/native/`                                          | Gleam-facing native helpers for BLAS, CUDA, sparse kernels, and TFLOPS/backend diagnostics.                                                                             |
| `src/viva_tensor/quant.gleam`                                      | Internal quantization entrypoint that re-exports the supported quantization modules.                                                                                    |
| `src/viva_tensor/quant/`                                           | Quantization implementations: compression, NF4, AWQ, Hadamard preprocessing, tensor-core layout helpers, and TurboQuant reference code.                                 |
| `src/viva_tensor/nn/`, `optim/`, `observability/`, `experimental/` | Internal domain modules until their contracts are stable enough for public API.                                                                                         |
| `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl`                  | Erlang bridge modules required by the BEAM target and NIF loading path.                                                                                                 |
| `zig_src/`                                                         | Native C, CUDA, and Zig implementation for the optional NIF.                                                                                                            |
| `priv/`                                                            | Runtime native artifacts loaded by Erlang when present.                                                                                                                 |
| `test/`                                                            | Unit, behavior, public API contract, and NIF/no-NIF compatibility tests.                                                                                                |
| `dev/`                                                             | Development-only Gleam examples and benchmark entrypoints. These modules are runnable with `gleam run -m ...` but are not supported package API.                        |
| `bench/`                                                           | External benchmark scripts, grouped by runtime or tool: `python/`, `r/`, `erlang/`, `cuda/`, `scripts/`, and `windows/`. Generated `data/` and `reports/` stay ignored. |
| `docs/`                                                            | Maintainer-authored guides and long-form documentation.                                                                                                                 |

## Public API Boundary

The package boundary is defined by `gleam.toml`: the root module and a small set
of companion modules are public, while `backend`, `core`, `native`, `quant`,
`tensor`, `nn`, `optim`, `observability`, and `experimental` are internal.

Promote a module out of `internal_modules` only when it has:

- documented shape and dtype behavior
- recoverable failures represented with `Result`
- tests through the root facade or a stable companion module
- a defined pure Gleam behavior when native acceleration is unavailable
- generated documentation that helps package users, not only maintainers

Benchmarks, demos, and research probes belong in `dev/` or `bench/` until they
become supported runtime features.

## Pure Gleam And NIF Fallback

Native acceleration is optional. Public tensor operations must continue to work
when the NIF is missing, fails to load, or returns an error.

The usual flow is:

1. The public facade delegates to internal tensor code.
2. Internal code validates shape/broadcasting behavior in Gleam.
3. If inputs are native tensors and a matching NIF operation exists, the native
   path is attempted through `core/ffi.gleam` and the Erlang bridge modules.
4. If the native path is unavailable or fails, the operation falls back to the
   pure Gleam implementation.

This contract is important for Hex users, CI portability, and development on
machines without CUDA, MKL, or a compiled `priv/viva_tensor_zig.*` artifact.
Functions that are truly native-only must say so explicitly in their docs and
tests.

The detailed FFI ownership and split contract lives in
[`FFI Architecture`](ffi-architecture.md). Keep `core/ffi.gleam` as the
forwarding facade until any `core/ffi/*` split modules are validated in Gleam
and migrated one disjoint resource family at a time.

## Backend Planner

Backend selection is split between small internal layers:

- `backend/protocol.gleam` defines backend types, availability checks, pure
  operations, local auto-selection, and distributed matmul hooks.
- `backend/capability.gleam` describes what the planner can reason about,
  including CPU, native, CUDA, and tensor-core capability records.
- `native/cuda.gleam` contains the higher-level acceleration planner for CUDA,
  MKL/native CPU, and CPU fallback. CUDA tensors stay on device until an API
  boundary requires conversion back to CPU tensors.
- `native/blas.gleam`, `native/sparse.gleam`, and `native/tflops.gleam` expose
  backend detection and diagnostics used by tests, benchmarks, and planner
  decisions.

Keep planner code descriptive rather than magical: record why a backend was
chosen, preserve CPU fallback, and do not let benchmark-only assumptions leak
into the stable facade.

## Quantization Layout

Quantization code is intentionally layered:

- `quant/compression.gleam`, `nf4.gleam`, and `awq.gleam` hold concrete
  quantization algorithms.
- `quant/hadamard.gleam` and `quant/turboquant.gleam` are pure Gleam reference
  paths for Hadamard-style preprocessing and low-bit experiments before native
  kernels exist.
- `quant/layout.gleam` documents tensor-core-oriented packing assumptions such
  as block and tile shapes.
- `zig_src/nif_quant.c` and CUDA files in `zig_src/` are the native landing zone
  once a quantization contract is locked down by the pure Gleam implementation
  and tests.

Prefer a readable reference implementation first. Move hot loops to NIF/CUDA
only after the Gleam contract, invalid-input behavior, and no-NIF fallback are
covered.

## Native Backend Locations

The native build is centered in `zig_src/build.zig`.

- MKL is wired from `zig_src/build.zig` and CPU/NIF code such as
  `zig_src/nif_entry.c` and `zig_src/nif_cpu_ops.c`.
- macOS Accelerate support lives in `zig_src/accelerate.c`.
- CUDA and sparse GPU work live in `zig_src/cuda_*.c`, `zig_src/cuda_*.cu`,
  `zig_src/nif_cuda_*.c`, `zig_src/nif_sparse.c`, and `zig_src/sage/`.
- NIF registration and shared declarations live in `zig_src/nif_entry.c` and
  `zig_src/viva_nif.h`.

Gleam modules should call native code only through the existing FFI wrappers and
bridge modules. Do not make public APIs depend on a specific native library
being installed unless the API is explicitly documented as native-only.
