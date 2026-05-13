# NumPy Array Pretty-Printing — Reference for viva_tensor

Reference: `tmp/numpy/numpy/_core/arrayprint.py` (1788 lines).

This document captures, in implementer-ready detail, how NumPy renders
`array_repr` / `array_str` / `array2string`, so we can port an
equivalent into viva_tensor without re-deriving the algorithm.

---

## 1. Algorithm in plain English

1. Entry point is `array2string` (`arrayprint.py:644`). It merges the
   caller's overrides with the global `format_options` snapshot
   (`arrayprint.py:793-797`) and short-circuits empty arrays with
   `"[]"` (`arrayprint.py:807-808`). It then delegates to
   `_array2string` (`arrayprint.py:606`).
2. `_array2string` decides whether the array exceeds `threshold`
   (default 1000). If so it calls `_leading_trailing` to keep only the
   first/last `edgeitems` (default 3) along **every** axis and remembers
   `summary_insert = "..."` (`arrayprint.py:614-618`,
   `arrayprint.py:438-455`). The dtype-specific formatter is chosen
   once, from the **trimmed** data — that is how widths stay consistent
   even when elided (`arrayprint.py:621`).
3. The formatter (e.g. `FloatingFormat.fillFormat`,
   `arrayprint.py:1013-1102`) does a **first pass** over the finite
   values to compute per-column widths: `pad_left` (chars left of the
   decimal), `pad_right` (chars right of the decimal), `exp_size`
   (exponent width when in sci mode), and the effective `precision`.
   Every element produced by `__call__` is guaranteed to render with the
   same total width, so columns align by construction — there is **no
   second alignment pass**.
4. `_formatArray` (`arrayprint.py:854`) is recursive over axes via a
   local `recurser(index, hanging_indent, curr_width)`. For
   `axes_left == 0` it returns `format_function(a[index])`. For
   `axes_left == 1` (the innermost row) it walks the elements,
   appending each to a `line` buffer; `_extendLine_pretty` wraps the
   line when adding the next element would push past
   `linewidth - len(']')` (`arrayprint.py:813-852`). For
   `axes_left > 1` it inserts `axes_left - 1` newlines between sibling
   sub-arrays — this is why a 3D array shows a blank line between 2D
   slices but only one newline between rows of a 2D slice
   (`arrayprint.py:939`).
5. Elision per axis is independent: `show_summary = summary_insert and
   2*edge_items < a_len` (`arrayprint.py:884`). When triggered, the
   recurser emits indices `0..edgeitems-1`, then `summary_insert`, then
   the trailing `edgeitems` items via negative indices
   (`arrayprint.py:906-933` for rows, `941-961` for non-leaf axes).
6. Line-budget bookkeeping: the top-level call passes
   `next_line_prefix = " " + " " * len(prefix)` (`arrayprint.py:624-626`).
   Every recursion adds one space to `hanging_indent` to align with the
   opening `[` and subtracts 1 from `curr_width` to account for the
   closing `]` (`arrayprint.py:877-881`). The effective per-element
   budget is `curr_width - max(len(sep.rstrip()), len(']'))`
   (`arrayprint.py:901-903`).
7. Scientific vs fixed for floats is decided in `fillFormat`
   (`arrayprint.py:1017-1030`). Let `abs_non_zero` be the absolute
   value of finite non-zero entries; let `max_val` / `min_val` be its
   max/min. Pick scientific iff `max_val >= exp_cutoff_max` **or**
   (`not suppress_small` **and** (`min_val < 1e-4` **or**
   `max_val/min_val > 1000`)). `exp_cutoff_max` is `1e8` on legacy
   `<= 2.2`, otherwise `10**min(8, finfo(dtype).precision)`.
8. Float digit generation uses Dragon4 via `dragon4_positional` /
   `dragon4_scientific` (C extension). `floatmode` controls the
   `unique`/`trim` flags: `'fixed'` -> non-unique with `precision`
   exact digits; `'unique'` -> shortest unique; `'maxprec'` ->
   shortest unique up to `precision`; `'maxprec_equal'` -> all elements
   share the per-column maximum width (`arrayprint.py:1040-1087`).
9. NaN/Inf strings are padded to the same total width as a number so
   columns still align (`arrayprint.py:1095-1102`,
   `arrayprint.py:1104-1116`). Width is `pad_left + pad_right + 1`
   (the `+1` is the decimal point).
10. `array_repr` (`arrayprint.py:1601-1652`) wraps the
    `array2string` output with `"array("` / `")"`, then appends
    `shape=...` and/or `dtype=...` only when not implied. The wrap
    point also handles whether the extras go inline or on a new line.

---

## 2. Tunable options

Sourced from `set_printoptions` docstring (`arrayprint.py:123-309`) and
`_make_options_dict` (`arrayprint.py:57-119`). Defaults from the
`format_options` ContextVar in `numpy/_core/printoptions.py`.

| Option        | Default          | Type                  | Effect                                                                                                              | viva_tensor |
|---------------|------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------|-------------|
| `precision`   | `8`              | int                   | Max fractional digits for floats; interpretation depends on `floatmode` (`arrayprint.py:1005,1052,1083`).            | mirror      |
| `threshold`   | `1000`           | int                   | Total elements above which the array is summarized (`arrayprint.py:614`).                                           | mirror      |
| `edgeitems`   | `3`              | int                   | Items kept at each end of each axis when summarizing (`arrayprint.py:438-455`).                                     | mirror      |
| `linewidth`   | `75`             | int                   | Soft line budget; `_extendLine` wraps when exceeded (`arrayprint.py:813-824`).                                      | mirror (default 80) |
| `suppress`    | `False`          | bool                  | If true, force fixed notation and zero-out tiny numbers (`arrayprint.py:1028-1029`).                                | mirror      |
| `nanstr`      | `"nan"`          | str                   | Rendering for NaN (`arrayprint.py:1108-1110`).                                                                      | mirror      |
| `infstr`      | `"inf"`          | str                   | Rendering for Inf (`arrayprint.py:1112-1113`).                                                                      | mirror      |
| `sign`        | `'-'`            | `'-' \| '+' \| ' '`   | Sign char for positive floats/ints (`arrayprint.py:76-77`, `1090-1092`).                                            | mirror (only `-` and `+` initially; skip space-pad) |
| `formatter`   | `None`           | dict[str, callable]   | Per-kind custom formatters; keys like `'all'`, `'float_kind'`, etc. (`arrayprint.py:504-523`).                      | **skip** v1 (Gleam has no untyped callable bag) |
| `floatmode`   | `'maxprec_equal'`| `'fixed'\|'unique'\|'maxprec'\|'maxprec_equal'` | Controls Dragon4 `unique`/`trim` flags (`arrayprint.py:1042,1066-1086`).                            | mirror, default `maxprec_equal` |
| `legacy`      | `False`          | str \| False          | Selects bug-compatible behavior with old NumPy versions (`arrayprint.py:79-102`).                                   | **skip** — we have no legacy |
| `override_repr` | `None`         | callable              | Bypasses everything (`arrayprint.py:1606-1608`).                                                                    | **skip**    |
| `suppress_small` (param only) | n/a | bool                | `array2string` per-call override of `suppress` (`arrayprint.py:644`).                                               | mirror as call kwarg |
| `separator` (param only) | `' '`   | str                   | Between elements; `array_repr` overrides to `', '` (`arrayprint.py:1623`).                                          | mirror      |
| `prefix` / `suffix` (param) | `""` | str               | Width reserved for the surrounding `array(...)` wrapper (`arrayprint.py:670-680`).                                  | mirror — we use `tensor(` prefix |

---

## 3. Worked examples

All examples assume defaults: `precision=8`, `threshold=1000`,
`edgeitems=3`, `linewidth=75`, `suppress=False`, `sign='-'`,
`floatmode='maxprec_equal'`, `legacy=False`.

### (a) 1D length-5 ints

```python
>>> np.array([1, 22, 333, 4, 55])
array([  1,  22, 333,   4,  55])
```

Key choice: `IntegerFormat` (`arrayprint.py:1312-1329`) measures the
max string length across all elements (`len("333") = 3`) and builds
`'{:3d}'`, so every column is right-justified to that width.

### (b) 1D length-5 floats, mixed magnitudes (forces sci notation)

```python
>>> np.array([1e-5, 1.0, 2.0, 3.0, 4.0])
array([1.e-05, 1.e+00, 2.e+00, 3.e+00, 4.e+00])
```

Key choice: `min_val = 1e-5 < 1e-4` triggers `exp_format = True`
(`arrayprint.py:1028`). All values render in scientific with shared
`exp_size = 2` and `pad_left = 1`. With `floatmode='maxprec_equal'`,
`trim='k'` keeps the trailing decimal, but `precision` collapses to
`max(len(frac_part)) = 0` since no element needs fractional digits
beyond the leading `1.` — hence `1.e-05` not `1.00000000e-05`.

### (c) 2D 3x4 floats

```python
>>> np.array([[1.0, 2.5, 3.0, 4.25],
...           [5.0, 6.0, 7.5, 8.0],
...           [9.0, 10.0, 11.5, 12.0]])
array([[ 1.  ,  2.5 ,  3.  ,  4.25],
       [ 5.  ,  6.  ,  7.5 ,  8.  ],
       [ 9.  , 10.  , 11.5 , 12.  ]])
```

Key choice: One **global** `pad_left=2` (longest int part is `"10"`)
and `pad_right=2` (longest frac part is `"25"`). `floatmode=maxprec_equal`
forces every cell to that width by appending zeros, then trimming to
shared `precision`. Rows align because every cell is fixed-width.

### (d) 2D 100x100 (elision via threshold)

```python
>>> np.arange(10000).reshape(100, 100)
array([[   0,    1,    2, ...,   97,   98,   99],
       [ 100,  101,  102, ...,  197,  198,  199],
       [ 200,  201,  202, ...,  297,  298,  299],
       ...,
       [9700, 9701, 9702, ..., 9797, 9798, 9799],
       [9800, 9801, 9802, ..., 9897, 9898, 9899],
       [9900, 9901, 9902, ..., 9997, 9998, 9999]], shape=(100, 100))
```

Key choice: `a.size = 10000 > threshold = 1000` triggers
`_leading_trailing` on **both** axes with `edgeitems=3`
(`arrayprint.py:614-616`, `438-455`). The recurser emits `0..2`,
`"..."`, then `-3..-1` along each axis. Column width derives from the
post-trim corner data (max value `9999`, width 4). The trailing
`shape=(100, 100)` is appended by `array_repr` because legacy > 210
and size > threshold (`arrayprint.py:1629-1632`).

### (e) 3D 2x3x4 ints

```python
>>> np.arange(24).reshape(2, 3, 4)
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
```

Key choice: At the outermost axis, `axes_left = 3 > 1`, so the
recurser inserts `axes_left - 1 = 2` newlines between siblings
(`arrayprint.py:939`) — that is the blank line separating the two 2D
panels. Inside each panel, `axes_left = 2` gives 1 newline between
rows. `hanging_indent` grows by one space per recursion to align with
each `[`.

### (f) Array containing NaN and Inf

```python
>>> np.array([1.0, np.nan, np.inf, -np.inf, 2.5])
array([ 1. ,  nan,  inf, -inf,  2.5])
```

Key choice: `isfinite` filters NaN/Inf out of the width pass, but the
non-finite branch `(arrayprint.py:1095-1102)` then **bumps** `pad_left`
so that `nanstr`/`infstr` plus a possible `-` sign fit in the column
width. NaN/Inf are rendered with leading-space padding so total width
equals `pad_left + pad_right + 1` (`arrayprint.py:1114-1116`). Since
one Inf is negative, `neginf = True` adds the `-` to the budget.

---

## 4. Gleam translation notes

### 4.1 Direct one-for-one mappings

| NumPy / Python                              | Gleam                                                        |
|---------------------------------------------|--------------------------------------------------------------|
| `str.rjust(n)` / `'{:3d}'.format(x)`        | `string.pad_start(str, to: n, with: " ")`                    |
| `str.ljust(n)`                              | `string.pad_end(str, to: n, with: " ")`                      |
| `len(s)`                                    | `string.length(s)` (note: grapheme count, fine for ASCII)    |
| `str.rstrip()`                              | `string.trim_end(s)`                                         |
| `"\n".join(parts)`                          | `string.join(parts, "\n")`                                   |
| `repr(int)` for non-neg                     | `int.to_string(i)`                                           |
| `max(iter)` / `min(iter)`                   | `list.fold` with `int.max` / `int.min` (or `float.max`)      |
| Default ContextVar `format_options`         | Module-level `Config` record passed explicitly; default via `config.default()` |

### 4.2 Need a replacement

| NumPy uses                                  | viva_tensor approach                                         |
|---------------------------------------------|--------------------------------------------------------------|
| Python `repr(float)` (shortest-round-trip)  | `float.to_string` (Erlang's `~p`-style). Round-trips IEEE 754, **but** always emits at least one fractional digit and never uses scientific until `\|x\| < 1e-4` or `\|x\| >= 1e15`. We must wrap it to enforce our own sci/fixed decision. |
| Dragon4 with `precision`, `unique`, `trim`  | We don't have Dragon4 on the BEAM. For `floatmode='unique'` use `float.to_string` directly. For other modes, build a small helper that formats with explicit digit count via `float_to_binary/2` (Erlang `[{decimals, N}]` or `[{scientific, N}]`). **Open question:** does Erlang's `io_lib_format` give shortest-unique? Per OTP docs, `~p` does for floats, but `float_to_binary/1` defaults to `~p` semantics — confirm before relying on it. |
| Dragon4's `pad_left` / `pad_right` arg      | Post-process the formatted string: split on `"."`, pad each half with `string.pad_start` / `pad_end`. |
| `numpy.signbit(x)` (distinguishes `-0`)     | Gleam `float` does not expose IEEE sign bit. Use a Rust NIF (`f64::is_sign_negative`) — already feasible in viva_burn. |
| `_recursive_guard` (cycle detection in object arrays) | Not needed: viva_tensor has no object dtype.       |
| `isfinite` / `isnan` / `isinf` on whole array | Add to `viva_tensor/internal/float_classify.gleam` (NIF-backed). |
| Python `**kwargs` formatter dispatch        | A static `case dtype` match. Custom formatters via `Config` field `formatter: Option(fn(BackendValue) -> String)`. |

### 4.3 Genuinely hard — the float two-pass

Pseudocode for picking sci-vs-fixed and computing column widths. This
is the heart of `FloatingFormat.fillFormat` (`arrayprint.py:1013-1102`):

```
fn fill_format(finite_vals, precision, floatmode, suppress, sign, dtype):
  // PASS 1: decide mode
  abs_nz = filter(finite_vals, fn(x) { x != 0.0 }) |> map(abs)
  exp_format = case abs_nz {
    [] -> False
    _ -> {
      let max_v = max(abs_nz)
      let min_v = min(abs_nz)
      let cutoff = pow(10.0, min(8, dtype_precision(dtype)))
      max_v >=. cutoff || (!suppress && (min_v <. 0.0001 || max_v /. min_v >. 1000.0))
    }
  }
  // PASS 2: format all values once to measure widths
  let raw_strs = map(finite_vals, fn(x) { format_one(x, exp_format, precision, floatmode) })
  let (int_parts, frac_parts, exp_parts) = split_parts(raw_strs, exp_format)
  pad_left   = max(map(int_parts, string.length))
  pad_right  = max(map(frac_parts, string.length))
  exp_size   = if exp_format { max(map(exp_parts, string.length)) - 1 } else { -1 }
  // bump pad_left for NaN/Inf
  if has_non_finite { pad_left = max(pad_left, nanstr_len - (pad_right + 1), infstr_len - (pad_right + 1)) }
  return FloatFormatter(exp_format, precision, pad_left, pad_right, exp_size, ...)
```

Notes:
- **Pass 1** never touches strings — only magnitudes.
- **Pass 2** renders every (finite) element so we can measure. This is
  O(N) extra string allocation, but N is bounded by `edgeitems * 2 ^ ndim`
  after elision, so it's tiny in practice.
- `floatmode='maxprec_equal'` forces a third normalization where
  `precision := pad_right` to equalize trailing digits
  (`arrayprint.py:1082-1084`).

### 4.4 Recursion shape

`_formatArray` recurses on **axis index**, not on slicing the array
(`arrayprint.py:863`). Translation: pass a `List(Int)` index path into
the recurser; index into the storage with the existing
`Tensor.read_at_indices(index)` helper. No need to materialize sliced
sub-tensors. This avoids allocating intermediate tensors during repr.

### 4.5 Line-budget invariant

```
remaining = linewidth - len(suffix) - depth   // one char per opening [
elem_budget = remaining - max(len(separator.rstrip()), 1)   // 1 for closing ]
wrap when: len(current_line) + len(next_word) > elem_budget
```

In Gleam, keep a `BuildBuffer { lines: List(String), current: String, indent: String, budget: Int }`
record threaded through the recursion. Avoid `String <> String <> ...`
in tight loops — use `string_tree` builder.

---

## 5. What viva_tensor can do better

1. **Backend tag in the prefix.** Replace `array(` with
   `tensor[backend=Native, shape=(3, 4), dtype=Float32](` so the user
   immediately sees whether the tensor lives in `Native` (BEAM term),
   `Dense` (binary), `CudaFp16`, etc. Mirrors the heterogeneous storage
   reality of viva_tensor — NumPy never had this problem.
2. **Stride / contiguity hint** for non-contiguous views. When
   `tensor.strides != contiguous_strides(shape)`, append
   `, view=non-contiguous` (or show explicit `strides=(...)`). Helps
   debug surprising performance cliffs; NumPy hides this in repr.
3. **Quantization / retired-block coloring.** For tensors that have
   regions marked as quantized or retired (RuRA/Bloom hot-set), emit a
   sentinel like `<q8>` or a dim sentinel `·` in place of the value
   inside elided sections, so a reader can see structure even on
   summarized arrays. Optional ANSI color when `IO.ansi_enabled()`.
4. **Per-row dtype hint when heterogeneous.** If a logical tensor
   internally stores rows under different dtypes (e.g., a CSR with
   mixed `Float16` / `Float32` chunks), annotate the row break:
   `# row 0..15 :: Float16`. NumPy assumes homogeneous dtype.
5. **Deterministic colorless mode for tests.** Expose a
   `Config.test_mode = True` switch that disables ANSI, fixes
   `linewidth = 80`, and locks `floatmode = 'fixed'`. Avoids the
   classic "tests pass locally, fail in CI" pretty-printer flake.
6. **Pluggable summarization strategy.** NumPy's `_leading_trailing`
   takes the head and tail of each axis. We can also offer
   `summary = 'extremes'` (show argmax/argmin entries) or
   `summary = 'random'` (uniformly-sampled `edgeitems`) — useful when
   inspecting a tensor whose interesting values are not at the edges.

---

## 6. Open questions

- **Open question:** Does Erlang's `float_to_binary(F, [short])` (OTP
  ≥ 25) produce the shortest-round-trip decimal? If yes, that replaces
  Dragon4's `unique=True` mode for free. Need to verify against IEEE
  half / float / double in `viva_burn` before relying on it.
- **Open question:** NumPy's `_FloatingFormat` uses `np.finfo(dtype).precision`
  to set `exp_cutoff_max` post-legacy 2.2 (`arrayprint.py:1024-1026`).
  We need an equivalent `finfo`-like table for our supported dtypes
  (`Float16`, `Float32`, `Float64`, plus any bfloat16). What is the
  canonical "decimal precision" we expose? Suggest: `Float16 -> 3`,
  `Float32 -> 7`, `Float64 -> 15`, `Bfloat16 -> 2`.
- **Open question:** Should `printoptions` be a process-dict
  (`erlang:put/2`) or an explicit `Config` threaded through callers?
  NumPy's ContextVar approach is concurrency-safe; the BEAM analog is
  a per-process dictionary. Likely we want both: a global default
  module + `with_config(cfg, fn)` for scoped overrides — but the
  API shape is undecided.
- **Open question:** For complex floats we are unlikely to ship in v1;
  confirm before designing `ComplexFloatingFormat` (`arrayprint.py:1341-1371`).
  If yes, note that NumPy uses two `FloatingFormat`s with separate
  `sign` ('-' for real, '+' for imag) and stitches `j` before the
  trailing whitespace of the imaginary part.

---

## 7. Source-line index (quick jump table)

| Concern                              | Lines                          |
|--------------------------------------|--------------------------------|
| `_make_options_dict`                 | `arrayprint.py:57-119`         |
| `set_printoptions` docstring         | `arrayprint.py:123-309`        |
| `printoptions` context manager       | `arrayprint.py:397-435`        |
| `_leading_trailing` (elision)        | `arrayprint.py:438-455`        |
| `_get_formatdict` / dispatch         | `arrayprint.py:476-524`        |
| `_get_format_function` (dtype mux)   | `arrayprint.py:526-572`        |
| `_array2string` (top-level)          | `arrayprint.py:606-631`        |
| `array2string` (public)              | `arrayprint.py:644-810`        |
| `_extendLine` (wrap logic)           | `arrayprint.py:813-824`        |
| `_extendLine_pretty` (multi-line)    | `arrayprint.py:827-852`        |
| `_formatArray` recurser              | `arrayprint.py:854-976`        |
| `FloatingFormat.__init__`            | `arrayprint.py:987-1011`       |
| `FloatingFormat.fillFormat`          | `arrayprint.py:1013-1102`      |
| `FloatingFormat.__call__`            | `arrayprint.py:1104-1136`      |
| `IntegerFormat`                      | `arrayprint.py:1312-1329`      |
| `BoolFormat`                         | `arrayprint.py:1331-1338`      |
| `ComplexFloatingFormat`              | `arrayprint.py:1341-1371`      |
| `StructuredVoidFormat`               | `arrayprint.py:1464-1498`      |
| `array_repr` (wrapper)               | `arrayprint.py:1601-1707`      |
| `array_str`                          | `arrayprint.py:1717-1780`      |
