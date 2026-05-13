# viva_tensor — Pretty Repr API Design (Sprint 1)

Synthesizes findings from `repr-numpy.md`, `repr-pytorch.md`, and
`repr-current-state.md`. This is the spec we implement against.

## Goals

- Single `tensor.to_string(t)` that renders any `Tensor` variant
  (`Tensor | StridedTensor | NativeTensor`) and any `AcceleratedTensor`
  (`CudaFp16 | CudaFp32 | Cpu`) into a readable, column-aligned string.
- NumPy/PyTorch-grade column alignment, elision for large tensors,
  scientific notation for mixed magnitudes.
- Backend / storage annotations (PyTorch's "default = hide" rule).
- No silent GPU H2D copy: tensors above `threshold` show a summarized
  view (top/bottom `edgeitems` per axis) — same elision rule that NumPy
  uses, which already bounds materialization cost.
- Explicit `PrintOptions` record, no module-level mutable state.

## Public API

Location: new module `src/viva_tensor/internal/format.gleam`,
re-exported from `viva_tensor/tensor.gleam` and `viva_tensor.gleam`.

```gleam
pub type PrintOptions {
  PrintOptions(
    precision: Int,        // max fractional digits, default 4
    threshold: Int,        // elements above which we summarize, default 1000
    edgeitems: Int,        // items kept at each end per axis, default 3
    linewidth: Int,        // soft line budget, default 80
    suppress_small: Bool,  // force fixed notation, default False
    sci_mode: SciMode,     // Auto | Always | Never, default Auto
    nan_str: String,       // default "nan"
    inf_str: String,       // default "inf"
    sign: SignMode,        // Negative | AlwaysPositive, default Negative
  )
}

pub type SciMode { SciAuto SciAlways SciNever }
pub type SignMode { SignNegative SignAlwaysPositive }

pub fn default_print_options() -> PrintOptions
pub fn to_string(t: Tensor) -> String
pub fn to_string_with(t: Tensor, opts: PrintOptions) -> String
pub fn inspect(t: Tensor) -> String           // alias for to_string
pub fn accelerated_to_string(t: AcceleratedTensor) -> String
pub fn accelerated_to_string_with(t: AcceleratedTensor, opts: PrintOptions) -> String
```

## Rendering rules

### Prefix

| Type | Prefix |
|---|---|
| `Tensor` (dense) | `tensor(` |
| `StridedTensor` | `tensor(` |
| `NativeTensor` | `tensor(` |
| `AcceleratedTensor.Cpu` | `tensor(` (delegates to inner) |
| `AcceleratedTensor.CudaFp16` | `accelerated_tensor(` |
| `AcceleratedTensor.CudaFp32` | `accelerated_tensor(` |

### Body

NumPy-style recursive `_formatArray` over axis indices (not slicing):
- 1D: `[e1, e2, ..., eN]`
- 2D: `[[row0], [row1], ...]` with one `\n` between rows
- N-D: `axes_left - 1` blank lines between sibling sub-arrays
- Elision: when `total_size > threshold`, keep first `edgeitems` and
  last `edgeitems` per axis, separated by `"..."`
- Line wrap: `linewidth - indent_depth - 1` budget per line, break at
  separator

### Suffixes (only when non-default)

| Condition | Suffix |
|---|---|
| `StridedTensor` (not contiguous) | `, storage=strided` |
| `NativeTensor` | `, storage=native` |
| `AcceleratedTensor.Cpu` with `MklNative` backend | `, backend=mkl` |
| `AcceleratedTensor.CudaFp16` | `, device='cuda', dtype=fp16` |
| `AcceleratedTensor.CudaFp32` | `, device='cuda', dtype=fp32` |
| `size > threshold` (summarized) | `, shape=(D1, D2, ...)` |
| `numel == 0` (empty) | `, shape=(...)` |

Suffix list joined by `, `, wrapped onto a new line if the result
would exceed `linewidth`.

### Float two-pass algorithm

Pass 1 — decide formatting mode (PyTorch's heuristic, cleaner than NumPy's):
```
finite_nonzero = filter(values, fn(x) { is_finite(x) && x != 0.0 })
if finite_nonzero is empty:
  int_mode = False; sci_mode = False
else:
  int_mode = all(v == float.truncate(v))  // looks like an int
  abs_vals = map(finite_nonzero, fn(x) { float.absolute_value(x) })
  fmax = max(abs_vals); fmin = min(abs_vals)
  sci_mode = case opts.sci_mode {
    SciAlways -> True
    SciNever -> False
    SciAuto -> fmax /. fmin >. 1000.0 || fmax >. 1.0e8 || fmin <. 1.0e-4
  }
```

Pass 2 — render each finite value with chosen format, measure widths:
```
fmt = case #(int_mode, sci_mode) {
  #(True,  False) -> fixed_no_decimal      // "{:.0f}." => "1."
  #(True,  True ) -> sci_zero_decimal       // "{:.0e}"
  #(False, False) -> fixed_precision        // "{:.Pf}"
  #(False, True ) -> sci_precision          // "{:.Pe}"
}
max_width = max(map(rendered, string.length))
nan_str_width = string.length(opts.nan_str)
inf_str_width = string.length(opts.inf_str) + (if has_neg_inf then 1 else 0)
column_width = max(max_width, nan_str_width, inf_str_width)
```

Pass 3 — emit values left-padded with spaces to `column_width`.

### Erlang float rendering primitive

Use `float.to_string` from gleam_stdlib for shortest-roundtrip. For
precision-bound formatting, use an Erlang-side helper:
```erlang
%% returns float as a binary with N fractional digits
fmt_fixed(F, N) -> erlang:float_to_binary(F, [{decimals, N}]).
fmt_sci(F, N)   -> erlang:float_to_binary(F, [{scientific, N}]).
```
Wrapped behind FFI in `viva_tensor/internal/format_ffi.gleam`.

### Elision logic

```
need_summary = total_size(shape) > opts.threshold
if need_summary:
  axis_sizes = trim each axis to first/last edgeitems, summary_insert="..."
else:
  axis_sizes = full shape
```

NumPy's `_leading_trailing` recurses through axes; we do the same but
keep a flag `axis_elided[i]` so we emit `"..."` between first and last
chunks.

### Storage walker

The walker function the recurser calls to fetch element at index path
`[i0, i1, ..., ik]`:
- `Tensor(data, shape)` — compute flat index from path + shape strides
- `StridedTensor(storage, shape, strides, offset)` — `offset + sum(i_k * strides_k)`,
  read from `ErlangArray` at that index
- `NativeTensor(ref, shape)` — read via `ffi.nt_get_at(ref, flat_index)` (already exists? — see below)
- `AcceleratedTensor.CudaFp16` — only when materializing; do a single
  `ffi.ct16_to_list` at the start, then walk the dense list
- `AcceleratedTensor.CudaFp32` — same with `ffi.ct_to_list`
- `AcceleratedTensor.Cpu(inner, _)` — delegate to inner tensor walker

Open: confirm `ffi.nt_get_at` exists; if not, materialize via
`ffi.nt_to_list` once and walk the list.

## Worked example outputs

For test fixtures (see snapshot tests):

```
tensor([1, 22, 333, 4, 55])

tensor([1.0e-05, 1.0e+00, 2.0e+00, 3.0e+00, 4.0e+00])

tensor([[ 1.  ,  2.5 ,  3.  ,  4.25],
        [ 5.  ,  6.  ,  7.5 ,  8.  ],
        [ 9.  , 10.  , 11.5 , 12.  ]])

tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

tensor([   0,    1,    2, ...,   97,   98,   99], shape=(100,))

tensor([ 1. ,  nan,  inf, -inf,  2.5])

tensor([1.0, 2.0, 3.0], storage=native)

accelerated_tensor([1., 2., 3.], device='cuda', dtype=fp16)
```

## What we skip in v1

- `formatter` callable dispatch — Gleam has no untyped callable bag.
- `legacy` mode — we have no legacy.
- `floatmode` knob — pick `MaxprecEqual` as the only mode (precision is
  the only knob the user needs).
- `space` sign character — only `-` and `+`.
- Complex numbers, sparse, quantized — no Gleam-level dtype yet.
- Per-process printopts via process dict — explicit `PrintOptions`
  threaded through calls.
- `set_print_options` / `get_print_options` global — defer until users
  ask. Today: pass `PrintOptions` explicitly.

## What we do better than NumPy/PyTorch

1. Storage tag in suffix (`storage=strided` / `storage=native`) — NumPy
   has no zero-copy view distinction, PyTorch hides it.
2. Backend tag for accelerated tensors — distinct prefix
   (`accelerated_tensor(`) since the Gleam type IS different.
3. Deterministic test mode — `PrintOptions` with fixed `linewidth=80`
   and `sci_mode=SciNever` for stable snapshots in CI.
4. Bounded materialization — for `NativeTensor` and `CudaFp*` with
   `size > threshold`, only fetch `edgeitems` items per axis (≤162
   elements for default 3D), never the full tensor.
5. No silent panics — every FFI read returns `Result`; we render
   `tensor(<unreadable: nif_not_loaded>, shape=(3,4))` instead of
   crashing.

## Test plan

Snapshot tests in `test/format_test.gleam` covering:
1. 1D int small
2. 1D float small (no sci)
3. 1D float mixed (forces sci)
4. 2D 3x4 floats
5. 2D 100x100 (forces elision)
6. 3D 2x3x4 ints
7. NaN + Inf inline
8. StridedTensor view (suffix annotation)
9. NativeTensor (graceful when NIF absent: should still produce a
   string, falling back to header-only)
10. AcceleratedTensor.Cpu / CudaFp16 / CudaFp32 prefix + suffix
11. Empty tensor
12. Custom `PrintOptions` (precision=2, linewidth=40)

## Implementation order

1. `internal/format_ffi.gleam` — Erlang `float_to_binary` wrappers
2. `internal/format.gleam` — main module with `PrintOptions`, walker,
   two-pass formatter, recurser
3. Re-export from `tensor.gleam` and `viva_tensor.gleam`
4. `accelerated_to_string` in `internal/format.gleam` (delegating to
   the dense walker after download or summary materialization)
5. `test/format_test.gleam` snapshot fixtures
6. Wire `accelerated_to_string` through `viva_tensor/native/cuda.gleam`
   facade if needed

Estimated effort: 2-3 days for the full sprint, including tests.
