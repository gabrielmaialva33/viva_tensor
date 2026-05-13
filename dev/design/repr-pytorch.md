# PyTorch tensor repr — research notes for viva_tensor

Source under study: `tmp/pytorch/torch/_tensor_str.py` (726 lines).
Entry point: `torch.Tensor.__repr__` at `_tensor.py:568`, which calls
`torch._tensor_str._str(self, ...)` at `_tensor.py:574`.

## 1. Algorithm summary

PyTorch's pretty-printer is essentially NumPy's with metadata bolted on.
`Tensor.__repr__` (and `__str__`, both routed through `_tensor_str._str`)
wraps a NumPy-style multi-line block inside `tensor(...)` and then appends
trailing `key=value` suffixes for anything that differs from defaults.
Like NumPy, it picks a single column width per tensor by scanning all
finite values with `_Formatter` (`_tensor_str.py:127`), elides huge
tensors via `summarize`/`get_summarized_data` (`_tensor_str.py:383`,
`_tensor_str.py:303`), and breaks lines at `linewidth` characters.
Unlike NumPy, PyTorch inlines `dtype=`, `device=`, `layout=`, `grad_fn=`,
`requires_grad=True`, `names=`, `size=` (for empty/meta/sparse), and
`nnz=` so the printed value round-trips meaningfully without losing
backend or autograd state. PyTorch also has no separate `repr` vs `str`
distinction — both forms produce the same annotated output (see Section 2
for the entry-point chain). The summarization rule is identical to
NumPy's: when `numel() > threshold` (default 1000), each axis longer
than `2 * edgeitems` is collapsed to first-N, `...`, last-N
(`_tensor_str.py:271`, `_tensor_str.py:347`).

## 2. Annotation rules — exact source references

Entry point: `torch._tensor_str._str` (`_tensor_str.py:712`) → `_str_intern`
(`_tensor_str.py:415`). Both `Tensor.__repr__` (`_tensor.py:568`) and the
default `Tensor.__str__` (inherited / re-routed via `__repr__`) produce
the same string. There is no shorter `str` form.

Suffixes are appended into `suffixes: list[str]` then joined by
`_add_suffixes` (`_tensor_str.py:393`), which also handles line-wrapping
relative to `PRINT_OPTS.linewidth`.

- `device='...'` — appended at `_tensor_str.py:454` when **any** of:
  - `self.device.type != torch._C._get_default_device()` (i.e. not CPU
    when CPU is the default), OR
  - device is CUDA but the *index* differs from the current device
    (`torch.cuda.current_device() != self.device.index`), OR
  - device type is `"mps"` (always shown).
  See the comment block at `_tensor_str.py:438` ("Note [Print tensor
  device]").

- `dtype=...` — emitted in several branches. The gating predicate is
  `has_default_dtype` (`_tensor_str.py:469`), which is True only for
  `torch.get_default_dtype()`, the matching complex default,
  `torch.int64`, and `torch.bool`. Specific sites:
  - sparse COO: `_tensor_str.py:485` (only when not default)
  - sparse CSR/CSC/BSR/BSC: `_tensor_str.py:521`
  - quantized: `_tensor_str.py:563`
  - meta / FakeTensor: `_tensor_str.py:608` (only when not default)
  - empty tensor: `_tensor_str.py:617` (always, because elements can't
    pin the dtype down)
  - dense non-empty: `_tensor_str.py:626` (only when not default)

- `requires_grad=True` / `grad_fn=<...>` — mutually exclusive. Computed
  at `_tensor_str.py:643`–`_tensor_str.py:666`:
  - If the tensor has a non-`None` `grad_fn` (non-leaf in an autograd
    graph), append `grad_fn=<ClassName>` where `ClassName` is
    `type(grad_fn).__name__` (special-cased to `grad_fn.name().rsplit("::", 1)[-1]`
    for `CppFunction`).
  - Else if `inp.requires_grad` is True (leaf with grad enabled), append
    `requires_grad=True`.
  - Accessing `grad_fn` can itself raise; the `try/except RuntimeError`
    at `_tensor_str.py:649` falls back to `grad_fn=<Invalid>`.

- `layout=...` — appended at `_tensor_str.py:641` whenever
  `self.layout != torch.strided`. Covers sparse COO/CSR/CSC/BSR/BSC,
  `mkldnn`, `jagged`, etc.

- `size=(...)` — appended for sparse (`_tensor_str.py:480`,
  `_tensor_str.py:514`), quantized (`_tensor_str.py:561`), meta / Fake
  (`_tensor_str.py:606`), empty non-1D (`_tensor_str.py:614`), and when
  `edgeitems == 0` (`_tensor_str.py:623`).

- `nnz=...` — sparse / sparse-compressed, when not meta
  (`_tensor_str.py:483`, `_tensor_str.py:518`).

- `quantization_scheme=`, `scale=`, `zero_point=`, `axis=` — quantized
  branch (`_tensor_str.py:564`–`_tensor_str.py:579`).

- `names=...` — appended at `_tensor_str.py:671` whenever
  `self.has_names()` is True.

- `tangent=...` — appended at `_tensor_str.py:674` if forward-mode AD
  unpacking returned a non-`None` tangent.

- Wrapper prefixes: nested tensors use prefix `"nested_tensor("`
  (`_tensor_str.py:420`); functional tensors use `"_to_functional_tensor("`
  (`_tensor_str.py:585`); `Parameter` wrapping adds `Parameter(...)` at
  `_tensor_str.py:686`; functorch wrappers handled in
  `_functorch_wrapper_str_intern` (`_tensor_str.py:692`).

## 3. viva_tensor recommendation

Ground truth from `src/viva_tensor/core/tensor.gleam:35`:
`Tensor` is a 3-variant sum (`Dense | Strided | Native`).
`AcceleratedTensor` from `src/viva_tensor/native/cuda.gleam:112` is a
separate type with `CudaFp16 | CudaFp32 | Cpu` variants and an
`AccelerationBackend` field (`Rtx4090Fp16 | Rtx4090Fp32 | MklNative |
CpuFallback`). No formal dtype (everything is `Float`), no autograd
state on the tensor itself, no `requires_grad` flag.

Recommendation: stay close to PyTorch's "default = hide it" rule —
annotate only when the value is non-default or operationally important.
Don't reserve names we can't honor yet (no `dtype=`, no `grad_fn=`).

| viva_tensor distinction | Show in repr? | Format |
|---|---|---|
| Variant: `Dense` (default) | no | (implicit when no annotation) |
| Variant: `Strided` | yes | `, storage=strided` |
| Variant: `Native` | yes | `, storage=native` |
| `AcceleratedTensor.Cpu` | yes (different type) | `, device='cpu'` |
| `AcceleratedTensor.CudaFp16` | yes | `, device='cuda', dtype=fp16` |
| `AcceleratedTensor.CudaFp32` | yes | `, device='cuda', dtype=fp32` |
| `AccelerationBackend` (Rtx4090Fp16 / MklNative / CpuFallback) | optional | `, backend=mkl` only when not the auto-selected default for the device |
| Shape | only for empty / `edgeitems=0` / summarized | `, size=(2, 3, 4)` |
| Strided view offset/strides differ from contiguous | optional debug-only | `, strides=(6, 1), offset=2` behind a `verbose=True` flag |
| NaN / Inf present | no extra suffix | rendered inline as `nan` / `inf` like NumPy/PyTorch |
| Autograd / `requires_grad` | **no** | open question until autograd surface stabilizes — see hard requirement |
| Named axes | yes (when present) | `, names=('batch', 'channel')` mirroring PyTorch |
| `tangent` / forward-mode | no | not applicable yet |

Prefix choice: `tensor(...)` for `Tensor`, `accelerated_tensor(...)`
for `AcceleratedTensor`. Avoid PyTorch's bare `tensor(` for the
accelerated type because the variant is part of the value's identity in
Gleam pattern matching, and the user will want to see it.

Open question: should `Strided` show its `strides` and `offset` even
when contiguous? Probably no — match PyTorch which never prints strides
even though they're always present. Surface them via a separate
`debug_repr` helper.

## 4. Float formatting heuristic (`_Formatter`)

Source: `_tensor_str.py:127`–`_tensor_str.py:225`. Decision tree
condensed:

```text
# init: int_mode=True, sci_mode=False, max_width=1
view = tensor.reshape(-1)                                   # :135

if not floating_dtype:                                       # :138
    for v in view: max_width = max(max_width, len(str(v)))   # :140
    return

# floating path
nonzero_finite = mask_select(view, isfinite & nonzero)       # :153
if nonzero_finite.numel() == 0: return                       # :156   (all 0/inf/nan)

abs_vals  = to_double(nonzero_finite.abs())                  # :167
fmin, fmax = abs_vals.min(), abs_vals.max()                  # :168-169

int_mode  = all(v == ceil(v) for v in nonzero_finite)        # :171-174
sci_mode  = PRINT_OPTS.sci_mode if set                       # :177-181
            else (fmax/fmin > 1000) or fmax > 1e8 or fmin < 1e-4

# width pass: pick format string, scan all values for longest
if int_mode and not sci_mode: fmt = "{:.0f}"  (+1 for "."  ) # :188
if int_mode and     sci_mode: fmt = "{:.<P>e}"               # :184
if not int_mode and not sci: fmt = "{:.<P>f}"                # :199
if not int_mode and     sci: fmt = "{:.<P>e}"                # :195
max_width = max(len(fmt.format(v)) for v in nonzero_finite)
```

At render time, `format(value)` (`_tensor_str.py:217`) right-pads with
spaces to `max_width` for column alignment. `int_mode` formatting
appends `.` to mark "this is a float that happens to be integral" so
e.g. `tensor([1., 2., 3.])` is unambiguous.

Note: `int_mode`/`sci_mode` are computed from the **non-zero finite**
subset only, so a tensor like `[0.0, 1e10, nan]` will pick `sci_mode`
based on `1e10` and render `nan` / `0.` inline.

## 5. Edge cases PyTorch handles that NumPy doesn't

- **Complex tensors** — two `_Formatter`s, one for real, one for imag;
  joined as `a+bj` with sign fix-up (`_tensor_str.py:235`–`_tensor_str.py:264`,
  `_tensor_str.py:387`). viva_tensor: **later** — no complex type yet.

- **Sparse COO / CSR / CSC / BSR / BSC** — prints `indices=tensor(...),
  values=tensor(...)` blocks instead of dense data, plus `size=`,
  `nnz=`, `layout=` suffixes (`_tensor_str.py:476`–`_tensor_str.py:557`).
  viva_tensor: **later** — we have CUDA int8 sparse paths in zig_src but
  no Gleam-level sparse `Tensor` variant.

- **Quantized tensors** — dequantizes for the data block, then appends
  `quantization_scheme=`, `scale=`, `zero_point=`, `axis=`
  (`_tensor_str.py:559`–`_tensor_str.py:582`). viva_tensor: **later** —
  applies once we expose int8/fp8 dtypes from the NIF layer.

- **Named tensors** — strips names before formatting data
  (`_tensor_str.py:367`) but reattaches `names=(...)` suffix
  (`_tensor_str.py:671`). viva_tensor: **now-ish** — `src/viva_tensor/named.gleam`
  exists, so the repr should already surface axis names.

- **Meta / FakeTensor** — no data to print, emits `...` plus `size=` and
  `dtype=` (`_tensor_str.py:600`–`_tensor_str.py:612`). viva_tensor:
  **later** — we don't have a meta variant yet, but `Native` with a
  null ref could need similar treatment if we add lazy alloc.

- **functorch wrappers** (BatchedTensor, GradTrackingTensor,
  FunctionalTensor) — entirely different rendering with `lvl=`, `bdim=`,
  nested `value=` (`_tensor_str.py:692`–`_tensor_str.py:710`).
  viva_tensor: **not applicable** — no vmap/grad transform stack.

- **Nested (jagged) tensors** — prefix `nested_tensor(`, body is `[\n  t1,\n  t2\n]`
  (`_tensor_str.py:584`–`_tensor_str.py:596`). viva_tensor: **later**.

- **Negative-bit / conjugate-bit / zerotensor** — resolved before
  printing (`_tensor_str.py:377`, `_tensor_str.py:391`,
  `_tensor_str.py:374`). viva_tensor: **not applicable** — we don't
  carry view-level lazy negation flags.

## 6. `set_printoptions` interaction

Single module-level `PRINT_OPTS = __PrinterOptions()` dataclass
(`_tensor_str.py:14`, `_tensor_str.py:22`) with fields `precision=4`,
`threshold=1000`, `edgeitems=3`, `linewidth=80`, `sci_mode=None`. The
free function `set_printoptions` (`_tensor_str.py:25`) mutates it in
place; `printoptions(**kwargs)` (`_tensor_str.py:104`) is a context
manager that saves and restores. There are three profile presets
(`"default"`, `"short"`, `"full"`) at `_tensor_str.py:68`–`_tensor_str.py:84`.

`sci_mode` is the only ternary: `None` means "let `_Formatter` decide
per-tensor"; `True` / `False` force the choice
(`_tensor_str.py:177`–`_tensor_str.py:181`).

Recommendation for viva_tensor: copy this exact surface
(`set_print_options`, `get_print_options`, `print_options` block) but
defer the implementation behind a single `PrintOptions` record passed
explicitly into a `tensor.to_string(t, opts)` function, since Gleam has
no module-level mutable state. Provide a `default_print_options()`
constructor with the same numeric defaults.

## Open questions

- Should `repr` of an `AcceleratedTensor.CudaFp16` materialize values to
  host memory eagerly (PyTorch always does — `_tensor_str.py:461` copies
  XLA/IPU/lazy tensors to CPU before formatting), or render a placeholder
  like `cuda_tensor(<on-device>, shape=(...), dtype=fp16)` to avoid
  surprise H2D copies? PyTorch's choice favors UX over performance;
  viva_tensor could go either way.
- Do we want a separate, terser `to_string` for log lines vs the full
  `inspect`-quality repr? PyTorch deliberately collapses both into one
  output — worth deciding before we ship a second function.
- For `Strided` views, should the repr indicate "this is a view of a
  larger storage" (e.g. `, view=True`)? PyTorch doesn't, but our zero-copy
  storage sharing is a load-bearing invariant worth surfacing in debug
  flows.
