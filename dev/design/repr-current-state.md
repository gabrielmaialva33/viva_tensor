# viva_tensor — Current State of Tensor Display & Errors

Audit snapshot (May 2026) of how tensors stringify and how errors are constructed today.
Scope: `src/viva_tensor/**`. All line refs verified against working tree.

---

## 1. Display surface today

There is **no tensor `to_string` / `repr` / `pretty` / `inspect` / `debug` / `format` helper anywhere in `src/`**.
A grep for the usual suspects produced zero hits inside any tensor module:

- No `to_string(t: Tensor)` in `src/viva_tensor/tensor.gleam` (4958 lines).
- No `to_string(t: Tensor)` in `src/viva_tensor/core/tensor.gleam` (734 lines).
- No `to_string` / `repr` / `pretty` for `AcceleratedTensor` in `src/viva_tensor/native/cuda.gleam` (741 lines).
- The only `to_string` helpers that exist are:
  - `src/viva_tensor/core/error.gleam:61` — `to_string(error: TensorError)` (errors, not tensors).
  - `src/viva_tensor/core/error.gleam:90` — `shape_to_string(shape: List(Int))` (shape only).
  - `src/viva_tensor/tensor.gleam:4948` — private `shape_to_string(shp: List(Int))` (duplicates the one above).
  - `src/viva_tensor/axis.gleam:142` — `to_string(a: Axis)` (named-axis label only).
  - `src/viva_tensor/quant/layout.gleam:134` — `format_name(format: QuantFormat)`.

What users actually print today is the **raw List(Float)**:

- Tests rely exclusively on `tensor.to_list(t) |> should.equal([...])` (see §5).
- Demo / benchmark output (e.g. `src/viva_tensor/quant/nf4.gleam:357-460`, `src/viva_tensor/optim/blackwell.gleam:527-577`, `src/viva_tensor/nn/flash_attention.gleam:401-455`, `src/viva_tensor/optim/rtx4090.gleam:433-463`, `src/viva_tensor/optim/pool.gleam:188-223`) hand-builds strings with `int.to_string` / `float.to_string` — nobody prints a tensor.
- `src/viva_tensor/nn/autograd.gleam:461` rolls its own `[a, b, c]` shape printer for grad-debugging messages.

Accessors available right now (what a future `repr` can lean on):

| Accessor | Location | Notes |
|---|---|---|
| `shape(t)` | `src/viva_tensor/tensor.gleam:341`, `src/viva_tensor/core/tensor.gleam:238` | List(Int) |
| `size(t)` | `src/viva_tensor/tensor.gleam:391`, `src/viva_tensor/core/tensor.gleam:287` | Int |
| `rank(t)` | `src/viva_tensor/tensor.gleam:401`, `src/viva_tensor/core/tensor.gleam:292` | Int |
| `dim(t, axis)` | `src/viva_tensor/tensor.gleam:450`, `src/viva_tensor/core/tensor.gleam:297` | Result(Int, _) |
| `rows(t)` / `cols(t)` | `src/viva_tensor/tensor.gleam:458,466`; `src/viva_tensor/core/tensor.gleam:305,313` | Int |
| `is_native(t)` | `src/viva_tensor/tensor.gleam:279`, `src/viva_tensor/core/tensor.gleam:665` | Bool |
| `is_contiguous(t)` | `src/viva_tensor/tensor.gleam:3444`, `src/viva_tensor/core/tensor.gleam:545` | Bool |
| `layout(t)` | `src/viva_tensor/tensor.gleam:406` | `layout.TensorLayout` — full metadata bundle (storage, device, dtype, shape, strides, offset, size, rank, contiguous) |
| `to_list(t)` | `src/viva_tensor/tensor.gleam:1841`, `src/viva_tensor/core/tensor.gleam:281` | List(Float) — materializes Native/Strided |
| `try_to_list(t)` | `src/viva_tensor/tensor.gleam:350`, `src/viva_tensor/core/tensor.gleam:247` | Result(List(Float), _) |
| `to_list2d(t)` | `src/viva_tensor/tensor.gleam:1846` | 2D only |
| `native_ref(t)` | `src/viva_tensor/tensor.gleam:271`, `src/viva_tensor/core/tensor.gleam:673` | Result(NativeTensorRef, _) |
| `accelerated_shape(t)` | `src/viva_tensor/native/cuda.gleam:437` | List(Int), no download |
| `backend(t)` | `src/viva_tensor/native/cuda.gleam:428` | `AccelerationBackend` |
| `fp16_available()` | `src/viva_tensor/native/cuda.gleam:60` | Bool, global |

---

## 2. Tensor variant catalog

### `viva_tensor/core/tensor.gleam` — `pub opaque type Tensor` (lines 35–49)

| Variant | Line | Fields | Storage | Display-relevant fields |
|---|---|---|---|---|
| `Dense` | 37 | `data: List(Float)`, `shape: List(Int)` | List | Direct float data, easy to walk |
| `Strided` | 39–44 | `storage: ErlangArray`, `shape: List(Int)`, `strides: List(Int)`, `offset: Int` | Erlang `:array` | Needs stride-walk via `flat_to_multi` to materialize; offset matters |
| `Native` | 48 | `ref: NativeTensorRef`, `shape: List(Int)` | NIF C buffer | Opaque ref — must call `ffi.nt_to_list` to read; can fail |

### `viva_tensor/tensor.gleam` — `pub type Tensor` (lines 40–49) — **non-opaque**

| Variant | Line | Fields | Storage | Notes |
|---|---|---|---|---|
| `Tensor` | 41 | `data: List(Float)`, `shape: List(Int)` | List | Same role as `Dense` in core, but exposed publicly |
| `StridedTensor` | 42–47 | `storage: ErlangArray`, `shape`, `strides`, `offset: Int` | Erlang `:array` | Mirrors core `Strided` |
| `NativeTensor` | 48 | `ref: NativeTensorRef`, `shape: List(Int)` | NIF | Mirrors core `Native` |

> Duplication: every variant exists twice (core vs public) with different constructor names. Any new repr must handle both type families or live behind a shared trait/helper.

### `viva_tensor/native/cuda.gleam` — `AcceleratedTensor` (lines 112–116) and friends

| Variant | Line | Fields | Notes for display |
|---|---|---|---|
| `CudaFp16` | 113 | `ref: CudaTensor16`, `shape: List(Int)`, `backend: AccelerationBackend` | GPU FP16; `ffi.ct16_to_list` to download |
| `CudaFp32` | 114 | `ref: CudaTensor`, `shape: List(Int)`, `backend: AccelerationBackend` | GPU FP32; `ffi.ct_to_list` to download |
| `Cpu` | 115 | `tensor: tensor.Tensor`, `backend: AccelerationBackend` | Falls through to the public `Tensor` printer |

Backend tag (`AccelerationBackend`, line 102): `Rtx4090Fp16 | Rtx4090Fp32 | MklNative | CpuFallback`.

---

## 3. TensorError variant catalog

Source: `src/viva_tensor/core/error.gleam:20-56` (type) and `:61-87` (`to_string`).

| Variant | Line (def) | Payload | `to_string` template (line) |
|---|---|---|---|
| `ShapeMismatch(expected, got)` | 25 | `List(Int), List(Int)` | `"Shape mismatch: expected <shape>, got <shape>"` (63–67) |
| `InvalidShape(reason)` | 31 | `String` | `"Invalid shape: " <> reason` (69) |
| `DimensionError(reason)` | 37 | `String` | `"Dimension error: " <> reason` (71) |
| `BroadcastError(shape_a, shape_b)` | 43 | `List(Int), List(Int)` | `"Cannot broadcast shapes <a> and <b>"` (73–77) |
| `IndexOutOfBounds(index, size)` | 49 | `Int, Int` | `"Index <i> out of bounds for size <n>"` (79–83) |
| `DtypeError(reason)` | 55 | `String` | `"Dtype error: " <> reason` (85) |

Construction census across `src/` (regex `Error(`):

- **402** total `Error(...)` sites.
- **~111** carry a literal string (`Error("...")` or `XError("..."`).
- **~202** use a structured `TensorError` constructor (still about half of those wrap a `DimensionError(String)` as a junk drawer).
- `DimensionError("...")` is by far the most common, used as the catch-all for "dimensionality", "axis OOB", "wrong arity", "empty tensor", and even "backend not available" / "NIF failure".

Notes:

- Three variants (`InvalidShape`, `DimensionError`, `DtypeError`) collapse to a free-form `String` payload, which means they lose all structure at the boundary — callers cannot pattern-match on the *kind* of dimension error.
- `IndexOutOfBounds` reports `size` but **never** the shape or the failing axis, so the user has no idea *which dimension* blew up.
- `ShapeMismatch` reports two shapes but no operator name — same message whether `add`, `matmul`, or `concat` triggered it.

---

## 4. Top 5 unhelpful error messages

### `src/viva_tensor/tensor.gleam:1383`
- **Current:** `Error(DimensionError("Invalid axis index"))`
- **Why bad:** No axis value, no tensor rank, no operation name. Triggered inside `sum_axis_with_keepdims`; the caller has zero context to fix the bug. Same message is reused on lines 1447, 1667, 1698 across `mean_axis`, `argmax_axis`, etc.
- **Better:** `AxisOutOfBounds(operation: "sum_axis", axis: -1, rank: 3)` rendering as `"sum_axis: axis -1 is out of bounds for tensor of rank 3 (valid: 0..2)"`.

### `src/viva_tensor/core/tensor.gleam:460` (and `:481`, `:499`)
- **Current:** `Error(error.DimensionError("Tensor is not 2D"))`
- **Why bad:** Operation name absent and the actual shape is never echoed. The same string is duplicated 3× in `get2d`, `get_row`, `get_col` (also in `tensor.gleam:1860`, 3488).
- **Better:** `RankMismatch(operation: "get_row", expected_rank: 2, got_shape: [3, 4, 5])` → `"get_row requires a 2D tensor, got rank 3 with shape [3, 4, 5]"`.

### `src/viva_tensor/tensor.gleam:754`
- **Current:** `Error(DimensionError("Expected [m,k], [k,n], and [n] bias"))`
- **Why bad:** Tells the user the *expected* placeholder shapes but never prints the shapes they actually passed. For a linear layer with 3 inputs that's the *only* useful debug info.
- **Better:** `LinearShapeMismatch(weight: [16, 32], input: [4, 64], bias: [32])` → `"linear_relu: input [4, 64] is not compatible with weight [16, 32] (expected input cols == weight rows)"`.

### `src/viva_tensor/tensor.gleam:2288`
- **Current:** `Error(DimensionError("Slice dimensions must match tensor rank"))`
- **Why bad:** Doesn't say what the rank is or what was passed. Same shape information loss in `tensor.gleam:2294,2298,2312` ("Invalid slice start", "Invalid slice length", "Slice bounds exceed tensor shape").
- **Better:** `SliceArityMismatch(tensor_shape: [4, 5, 6], start: [0, 1], lengths: [2, 3])` → `"slice: start/length must have rank 3 (tensor shape [4, 5, 6]); got start=[0, 1], length=[2, 3]"`.

### `src/viva_tensor/native/cuda.gleam:506`
- **Current:** `Error(DimensionError("Output, lhs, and rhs must use the same backend"))`
- **Why bad:** Three backends are involved but none of them is named. User can't tell which tensor is on FP16 vs FP32 vs CPU.
- **Better:** `BackendMismatch(operation: "matmul_accelerated_into", out: Rtx4090Fp32, lhs: Rtx4090Fp16, rhs: MklNative)` → `"matmul_accelerated_into: backend mismatch — out=Rtx4090Fp32, lhs=Rtx4090Fp16, rhs=MklNative"`.

Bonus offender worth flagging:

- `src/viva_tensor/core/ffi.gleam:462,470,481,492` all return `Error("nif_not_loaded")` as a bare string — this `Result(_, String)` never flows into a `TensorError`, so `cuda` and `tensor` layers wrap it with another `DimensionError(reason)` losing the type. See `cuda.gleam:444` (`sync`) and `cuda.gleam:539,603` (`unsupported_activation` → silently downgraded to `DimensionError(String)` at `:613`/`:639`).

---

## 5. Test surface

| Test file | Lines | What it asserts | Snapshot-test candidate? |
|---|---|---|---|
| `test/tensor_core_test.gleam` | 787 | Mostly `tensor.shape(...) \|> should.equal([...])` + `core_tensor.to_list(...) \|> should.equal([...])` (eye/diag/eye/strided layout) at `:32-104`, `:48`, `:53-80`, `:97-150`. | High — eye/diag/identity outputs are perfect repr fixtures. |
| `test/viva_tensor_test.gleam` | 1682 | `t.to_list(...) \|> should.equal([...])` and `t.shape(...) \|> should.equal([...])` everywhere (`:24-188` zeros/ones/full/linspace/logspace/identity/diag, then through the file). | High — linspace, logspace, identity, diag outputs would be ideal "small canonical tensor" snapshots. |
| `test/shape_test.gleam` | 474 | `tensor.shape(r) \|> should.equal([...])` and `tensor.to_list(r) \|> should.equal([...])` across slice/concat/stack (`:18-214`). | Medium — slice/transpose/concat results give multi-dim test data. |
| `test/ops_test.gleam` | 526 | `core_tensor.to_list(...) \|> should.equal([...])` for ops (`:41-156`+). | Medium — easy small tensors. |
| `test/autograd_test.gleam` | 395 | `tensor.to_list(...) \|> should.equal([...])` for forward/backward checks (`:100-378`); has a private `assert_list_close` helper for fuzzy float compare. | Lower — gradient outputs, not great for visual snapshots. |
| `test/metrics_test.gleam` | 105 | Float-only assertions on metric results; no tensor display. | No. |
| `test/turboquant_test.gleam` | 87 | `tensor.shape(recovered) \|> should.equal([4])` + length-of-list checks (`:77-78`). | No. |
| `test/rubin_readiness_test.gleam` | 89 | Shape + value list checks (`:62`, `:74`). | No. |
| `test/public_api_contract_test.gleam` | 315 | API existence + minimal Ok/Err contract checks; one `string.contains` check at `:1520` of viva_tensor_test for descriptions (not tensors). | No. |
| `test/layout_math_test.gleam` | 29 | Stride math only. | No. |
| `test/bench.gleam` | 292 | Bench harness only. | No. |

**Zero tests today compare a tensor *string representation*.** Every assertion is structural (`shape` + `to_list`). Introducing a snapshot test for `repr` would be a clean greenfield — no existing string-format expectation to break.

---

## 6. Recommendations

1. **Ship a single `tensor.to_string(t: Tensor) -> String` (and matching `inspect` / `format` overloads) on the public `viva_tensor/tensor.gleam`**, then re-export from `core/tensor.gleam`. Make it handle all six variants (Dense/Strided/Native × public/core) plus `AcceleratedTensor`. Without this, the only debugging path is `to_list` → eyeball flat list, which is unusable past rank 2.
2. **Pretty-print should never *silently* force a GPU download.** For `Native` and `CudaFp*`, default to a header-only repr (`Tensor[shape=[1024, 1024], dtype=f32, device=cuda:0]`) and require an explicit `to_string_full` / opt-in flag to materialize. Avoids surprising perf cliffs in `io.debug` calls.
3. **Replace the four string-payload error variants (`InvalidShape`, `DimensionError`, `DtypeError`, plus `Error("nif_not_loaded")`) with structured variants** carrying `operation`, `tensor_shape`, `axis`, `expected`, `got`. Today ~half of the 402 `Error(...)` sites lose information by stuffing context into a free-form string. Suggested new variants: `AxisOutOfBounds`, `RankMismatch`, `EmptyTensor`, `BackendUnavailable`, `BackendMismatch`, `NifNotLoaded`.
4. **Add `operation: String` (or a `pub type Operation` enum) to every error construction.** Right now identical messages (`"Tensor is not 2D"`, `"Invalid axis index"`) come from a dozen functions and there's no way for a user to know which one. A constant + grep is enough; no runtime cost.
5. **De-duplicate `shape_to_string`.** It's reimplemented privately at `src/viva_tensor/tensor.gleam:4948` and at `src/viva_tensor/nn/autograd.gleam:461`. Promote `error.shape_to_string` to a `viva_tensor/internal/format.gleam` and call it everywhere — also the natural home for `dtype_to_string`, `device_to_string`, `backend_to_string`.
6. **Treat the existing `layout(t) -> TensorLayout` (tensor.gleam:406) as the canonical metadata bundle the new repr consumes.** It already exposes storage, device, dtype, shape, strides, offset, size, rank, contiguous. The header line of any repr should be one render of this struct — no duplicated case-on-variant.
7. **Add snapshot tests for repr.** None exist today; eye/diag/identity/linspace/logspace (from `viva_tensor_test.gleam:47-85`) are small, deterministic, multi-dim, and would lock in the format. Use birdie or simple string equality — there's nothing to migrate.

---

## Heads-up (out of scope, but worth raising)

- **Duplicate `Tensor` type families.** `src/viva_tensor/core/tensor.gleam` (opaque) and `src/viva_tensor/tensor.gleam` (non-opaque) define two parallel tensor types with the same three storage flavors and different constructor names (`Dense`/`Strided`/`Native` vs `Tensor`/`StridedTensor`/`NativeTensor`). Errors and repr have to handle both, and any client code that imports both modules ends up with two incompatible `Tensor` types in scope. Worth resolving before the repr lands so we don't have to write the printer twice. (Already flagged in prior observations 23924 and 34009.)
- **`Result(_, String)` from `core/ffi.gleam:462-492`** leaks raw `"nif_not_loaded"` strings into the higher layers, which then rewrap them in `DimensionError(reason)` — losing the type. Same pattern at `cuda.gleam:539`, `:603` (`"unsupported_activation"`). Promote these to a real variant when reworking errors.
- **No `let assert` in `src/`** — checked with `grep -rn 'let assert' src/`, returned zero hits. Good news: nothing is silently panicking in a hot path.
