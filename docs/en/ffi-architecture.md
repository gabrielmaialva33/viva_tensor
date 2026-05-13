# FFI Architecture

This page defines the ownership contract for `viva_tensor` FFI code. It is a
maintainer-facing contract, not a public API guarantee.

## Current Boundary

The supported call path is:

```text
public Gleam API
  -> src/viva_tensor/tensor.gleam or domain modules
  -> src/viva_tensor/core/ffi.gleam
  -> src/viva_tensor_ffi.erl, src/viva_tensor_nif.erl, or src/viva_tensor_zig.erl
  -> zig_src/
```

`src/viva_tensor/core/ffi.gleam` remains the single compatibility facade for
Gleam code. Existing callers should continue importing it directly until a split
is complete and validated.

## Ownership Rules

| Area | Owner | Rule |
|:-----|:------|:-----|
| Public API | `src/viva_tensor.gleam` and documented companions | Must not expose native-only requirements unless explicitly documented. |
| Tensor behavior | `src/viva_tensor/tensor.gleam` and domain modules | Owns fallback selection and tensor semantics. |
| FFI facade | `src/viva_tensor/core/ffi.gleam` | Owns stable internal wrapper names used by Gleam call sites. |
| FFI split modules | `src/viva_tensor/core/ffi/*` | May own grouped internal wrappers after import compatibility is validated. |
| Erlang bridge | `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl` | Owns BEAM module exports and NIF stubs. |
| Native implementation | `zig_src/` | Owns C, CUDA, and Zig implementation details. |
| Documentation | `docs/en/ffi-architecture.md` | Owns the split contract and migration rules. |

## Split Contract

Future FFI modules under `src/viva_tensor/core/ffi/` must be disjoint by backend
or resource family. Do not split by arbitrary operation names if that creates
duplicate ownership over the same resource type.

Recommended groups:

- `core/ffi/erlang_array.gleam`: `ErlangArray` and pure Erlang array helpers.
- `core/ffi/math.gleam`: thin Erlang `math` and `rand` wrappers.
- `core/ffi/native_tensor.gleam`: `NativeTensorRef` resource constructors,
  element-wise operations, reductions, matrix operations, mutation, and fused
  CPU kernels.
- `core/ffi/cuda.gleam`: CUDA tensor resource families.
- `core/ffi/research.gleam`: LNS, Horde, HDC, sparse, quantized, and
  experimental native resources until they graduate to a domain-specific owner.

Each split module must own its resource types and its private `@external`
bindings together. A wrapper must not live in one module while its opaque type
or matching external declaration lives in another module unless there is a
deliberate shared type module.

## Migration Rules

1. Add split modules first without changing existing call sites.
2. Validate that Gleam accepts `src/viva_tensor/core/ffi.gleam` and
   `src/viva_tensor/core/ffi/*.gleam` together in this package.
3. Move one disjoint group at a time and keep `core/ffi.gleam` as a forwarding
   facade.
4. Run formatting, type checking, no-NIF tests, and native-path tests after each
   group.
5. Only update `tensor.gleam` or the public facade when the change is purely
   mechanical and the old import path remains available.

## Fallback Requirements

Native acceleration is optional. New FFI wrappers must return recoverable
`Result` values for native failures unless they wrap deterministic Erlang
standard-library functions. Tensor-level code remains responsible for choosing
native execution and falling back to pure Gleam behavior.

No split module may require a compiled NIF at package load time. If the NIF is
missing, the package must still compile and the no-NIF path must remain testable.
