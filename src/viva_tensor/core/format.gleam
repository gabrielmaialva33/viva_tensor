//// Pretty-printing for tensors.
////
//// Renders `Tensor` and `AcceleratedTensor` values into NumPy/PyTorch
//// style multi-line strings with column alignment, scientific notation
//// for mixed magnitudes, and elision for large tensors. Internal —
//// `viva_tensor` re-exports `to_string` and `inspect`.

import gleam/float
import gleam/int
import gleam/list
import gleam/string
import viva_tensor/core/ffi
import viva_tensor/core/format_ffi
import viva_tensor/native/cuda.{
  type AcceleratedTensor, type AccelerationBackend, Cpu, CpuFallback, CudaFp16,
  CudaFp32, MklNative, Rtx4090Fp16, Rtx4090Fp32,
}
import viva_tensor/tensor.{
  type Tensor, NativeTensor, StridedTensor, Tensor as DenseTensor,
}

// =============================================================================
// Public types
// =============================================================================

/// User-tunable options for pretty-printing. Mirrors a subset of
/// NumPy/PyTorch print options that translate cleanly to the BEAM.
pub type PrintOptions {
  PrintOptions(
    precision: Int,
    threshold: Int,
    edgeitems: Int,
    linewidth: Int,
    suppress_small: Bool,
    sci_mode: SciMode,
    nan_str: String,
    inf_str: String,
    sign: SignMode,
  )
}

/// Whether to force scientific notation, force fixed notation, or let
/// the formatter pick based on the value range.
pub type SciMode {
  SciAuto
  SciAlways
  SciNever
}

/// Sign formatting for non-negative values. `SignNegative` writes
/// negatives as `-x` and leaves positives bare. `SignAlwaysPositive`
/// writes positives as `+x` so columns align under a unified sign
/// column.
pub type SignMode {
  SignNegative
  SignAlwaysPositive
}

// =============================================================================
// Defaults
// =============================================================================

/// Default print options matching NumPy/PyTorch's reasonable defaults
/// adapted for viva_tensor's heterogeneous storage.
pub fn default_print_options() -> PrintOptions {
  PrintOptions(
    precision: 4,
    threshold: 1000,
    edgeitems: 3,
    linewidth: 80,
    suppress_small: False,
    sci_mode: SciAuto,
    nan_str: "nan",
    inf_str: "inf",
    sign: SignNegative,
  )
}

// =============================================================================
// Public entry points
// =============================================================================

/// Render a `Tensor` with default print options.
pub fn to_string(t: Tensor) -> String {
  to_string_with(t, default_print_options())
}

/// Render a `Tensor` with caller-supplied print options.
pub fn to_string_with(t: Tensor, opts: PrintOptions) -> String {
  let prefix = "tensor("
  let suffixes = tensor_suffixes(t)
  render_tensor(t, opts, prefix, suffixes)
}

/// Alias for `to_string` — matches NumPy/PyTorch's `inspect` /
/// `__repr__` convention.
pub fn inspect(t: Tensor) -> String {
  to_string(t)
}

/// Render an `AcceleratedTensor` with default print options. Uses the
/// `accelerated_tensor(` prefix and includes the device/dtype suffix.
pub fn accelerated_to_string(t: AcceleratedTensor) -> String {
  accelerated_to_string_with(t, default_print_options())
}

/// Render an `AcceleratedTensor` with caller-supplied options.
///
/// `AcceleratedTensor.Cpu` delegates to the underlying `Tensor`
/// printer, optionally tagging the backend. `CudaFp16`/`CudaFp32`
/// materialize values via the existing FFI download primitives only
/// when the total size is at or below `opts.threshold`; larger tensors
/// emit a header-only repr to avoid surprise H2D copies.
pub fn accelerated_to_string_with(
  t: AcceleratedTensor,
  opts: PrintOptions,
) -> String {
  case t {
    Cpu(inner, backend) -> {
      let suffixes =
        list.append(tensor_suffixes(inner), [accel_backend_suffix(backend)])
        |> list.filter(fn(s) { s != "" })
      render_tensor(inner, opts, "tensor(", suffixes)
    }

    CudaFp16(ref, shape, backend) -> {
      let total = total_size(shape)
      case total <= opts.threshold {
        True -> render_cuda_fp16(ref, shape, backend, opts)
        False ->
          header_only("accelerated_tensor(", shape, [
            "device='cuda'",
            "dtype=fp16",
            accel_backend_suffix(backend),
          ])
      }
    }

    CudaFp32(ref, shape, backend) -> {
      let total = total_size(shape)
      case total <= opts.threshold {
        True -> render_cuda_fp32(ref, shape, backend, opts)
        False ->
          header_only("accelerated_tensor(", shape, [
            "device='cuda'",
            "dtype=fp32",
            accel_backend_suffix(backend),
          ])
      }
    }
  }
}

// =============================================================================
// Suffix construction
// =============================================================================

fn tensor_suffixes(t: Tensor) -> List(String) {
  case t {
    DenseTensor(_, _) -> []
    StridedTensor(_, _, _, _) -> ["storage=strided"]
    NativeTensor(_, _) -> ["storage=native"]
  }
}

fn accel_backend_suffix(backend: AccelerationBackend) -> String {
  case backend {
    Rtx4090Fp16 -> ""
    Rtx4090Fp32 -> ""
    MklNative -> "backend=mkl"
    CpuFallback -> "backend=cpu"
  }
}

// =============================================================================
// CUDA materialization shims
// =============================================================================

fn render_cuda_fp16(
  ref: ffi.CudaTensor16Ref,
  shape: List(Int),
  backend: AccelerationBackend,
  opts: PrintOptions,
) -> String {
  case ffi.ct16_to_list(ref) {
    Ok(data) -> {
      let inner = DenseTensor(data: data, shape: shape)
      let suffixes =
        [
          "device='cuda'",
          "dtype=fp16",
          accel_backend_suffix(backend),
        ]
        |> list.filter(fn(s) { s != "" })
      render_tensor(inner, opts, "accelerated_tensor(", suffixes)
    }
    Error(reason) -> unreadable("accelerated_tensor(", shape, reason)
  }
}

fn render_cuda_fp32(
  ref: ffi.CudaTensorRef,
  shape: List(Int),
  backend: AccelerationBackend,
  opts: PrintOptions,
) -> String {
  case ffi.ct_to_list(ref) {
    Ok(data) -> {
      let inner = DenseTensor(data: data, shape: shape)
      let suffixes =
        [
          "device='cuda'",
          "dtype=fp32",
          accel_backend_suffix(backend),
        ]
        |> list.filter(fn(s) { s != "" })
      render_tensor(inner, opts, "accelerated_tensor(", suffixes)
    }
    Error(reason) -> unreadable("accelerated_tensor(", shape, reason)
  }
}

// =============================================================================
// Headers for degraded paths (size > threshold or unreadable native)
// =============================================================================

fn header_only(
  prefix: String,
  shape: List(Int),
  raw_suffixes: List(String),
) -> String {
  let suffixes =
    [shape_suffix(shape), ..raw_suffixes]
    |> list.filter(fn(s) { s != "" })
  prefix <> "<...>" <> suffix_block(suffixes) <> ")"
}

fn unreadable(prefix: String, shape: List(Int), reason: String) -> String {
  prefix
  <> "<unreadable: "
  <> reason
  <> ">"
  <> suffix_block([shape_suffix(shape)])
  <> ")"
}

fn shape_suffix(shape: List(Int)) -> String {
  "shape=" <> render_shape(shape)
}

fn render_shape(shape: List(Int)) -> String {
  let parts = list.map(shape, int.to_string)
  case parts {
    [single] -> "(" <> single <> ",)"
    _ -> "(" <> string.join(parts, ", ") <> ")"
  }
}

fn suffix_block(suffixes: List(String)) -> String {
  case suffixes {
    [] -> ""
    _ -> ", " <> string.join(suffixes, ", ")
  }
}

// =============================================================================
// Main rendering pipeline
// =============================================================================

fn render_tensor(
  t: Tensor,
  opts: PrintOptions,
  prefix: String,
  suffixes: List(String),
) -> String {
  let shape = tensor.shape(t)

  case shape {
    [] -> prefix <> "<scalar>" <> suffix_block(suffixes) <> ")"

    _ -> {
      let total = total_size(shape)
      case total == 0 {
        True -> {
          let s = [shape_suffix(shape), ..suffixes]
          prefix <> "[]" <> suffix_block(s) <> ")"
        }

        False -> {
          // Materialize once into a flat list for walking.
          case materialize(t) {
            Ok(values) -> {
              let summarize = total > opts.threshold
              let summary_suffixes = case summarize {
                True -> [shape_suffix(shape), ..suffixes]
                False -> suffixes
              }
              let formatter = build_formatter(values, opts)
              let body =
                render_body(
                  values,
                  shape,
                  formatter,
                  opts,
                  summarize,
                  string.length(prefix),
                )
              prefix <> body <> suffix_block(summary_suffixes) <> ")"
            }
            Error(reason) -> unreadable(prefix, shape, reason)
          }
        }
      }
    }
  }
}

// =============================================================================
// Materialization
// =============================================================================

fn materialize(t: Tensor) -> Result(List(Float), String) {
  case t {
    DenseTensor(data, _) -> Ok(data)

    StridedTensor(_, _, _, _) ->
      // Use the existing public to_list which handles strides.
      Ok(tensor.to_list(t))

    NativeTensor(ref, _) -> ffi.nt_to_list(ref)
  }
}

// =============================================================================
// Body rendering — recursive _formatArray
// =============================================================================

fn render_body(
  values: List(Float),
  shape: List(Int),
  formatter: Formatter,
  opts: PrintOptions,
  summarize: Bool,
  prefix_len: Int,
) -> String {
  let strides = contiguous_strides(shape)
  let rank = list.length(shape)
  recurse(
    values,
    shape,
    strides,
    formatter,
    opts,
    summarize,
    [],
    rank,
    prefix_len,
  )
}

fn recurse(
  values: List(Float),
  shape: List(Int),
  strides: List(Int),
  formatter: Formatter,
  opts: PrintOptions,
  summarize: Bool,
  path: List(Int),
  axes_left: Int,
  prefix_len: Int,
) -> String {
  let depth = list.length(path)
  let indent = prefix_len + depth + 1
  // The "+1" accounts for the outer `(` of `tensor(`; the opening `[`
  // sits at column `prefix_len`, and children render starting at
  // column `prefix_len + depth + 1`.

  let dim_index = depth
  let dim_size = case list_at(shape, dim_index) {
    Ok(d) -> d
    Error(_) -> 0
  }

  let indices = expand_indices(dim_size, summarize, opts.edgeitems)

  case axes_left {
    1 ->
      render_innermost(values, strides, formatter, opts, path, indices, indent)

    _ ->
      render_outer(
        values,
        shape,
        strides,
        formatter,
        opts,
        summarize,
        path,
        axes_left,
        indices,
        indent,
        prefix_len,
      )
  }
}

fn render_innermost(
  values: List(Float),
  strides: List(Int),
  formatter: Formatter,
  opts: PrintOptions,
  path: List(Int),
  indices: List(IndexEntry),
  indent: Int,
) -> String {
  let prefix = "["
  let suffix = "]"
  let budget = opts.linewidth - indent - 1

  let parts =
    list.map(indices, fn(entry) {
      case entry {
        Elision -> "..."
        Idx(i) -> {
          let full_path = list.append(path, [i])
          let value = read_value(values, strides, full_path)
          formatter.render(value)
        }
      }
    })

  // Wrap at linewidth: greedily fill each line then break.
  let separator = ", "
  let wrapped = wrap_line(parts, separator, budget, string.repeat(" ", indent))
  prefix <> wrapped <> suffix
}

fn render_outer(
  values: List(Float),
  shape: List(Int),
  strides: List(Int),
  formatter: Formatter,
  opts: PrintOptions,
  summarize: Bool,
  path: List(Int),
  axes_left: Int,
  indices: List(IndexEntry),
  indent: Int,
  prefix_len: Int,
) -> String {
  let separator_newlines = string.repeat("\n", axes_left - 1)
  let separator = "," <> separator_newlines <> string.repeat(" ", indent)

  let parts =
    list.map(indices, fn(entry) {
      case entry {
        Elision -> "..."
        Idx(i) -> {
          let new_path = list.append(path, [i])
          recurse(
            values,
            shape,
            strides,
            formatter,
            opts,
            summarize,
            new_path,
            axes_left - 1,
            prefix_len,
          )
        }
      }
    })

  "[" <> string.join(parts, separator) <> "]"
}

// =============================================================================
// Index expansion (elision)
// =============================================================================

type IndexEntry {
  Idx(Int)
  Elision
}

fn expand_indices(
  dim_size: Int,
  summarize: Bool,
  edgeitems: Int,
) -> List(IndexEntry) {
  case summarize && dim_size > 2 * edgeitems {
    True -> {
      let head =
        list.range(0, edgeitems - 1)
        |> list.map(Idx)
      let tail =
        list.range(dim_size - edgeitems, dim_size - 1)
        |> list.map(Idx)
      list.flatten([head, [Elision], tail])
    }
    False ->
      list.range(0, dim_size - 1)
      |> list.map(Idx)
  }
}

// =============================================================================
// Value read via flat index + contiguous strides
// =============================================================================

fn read_value(
  values: List(Float),
  strides: List(Int),
  path: List(Int),
) -> Float {
  let flat = flat_index(path, strides)
  case list_at_float(values, flat) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn flat_index(path: List(Int), strides: List(Int)) -> Int {
  list.zip(path, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
}

fn contiguous_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(_, strides) =
    list.fold(reversed, #(1, []), fn(acc, dim) {
      let #(running, out) = acc
      #(running * dim, [running, ..out])
    })
  strides
}

// =============================================================================
// Two-pass float formatter
// =============================================================================

type Formatter {
  Formatter(width: Int, render: fn(Float) -> String)
}

fn build_formatter(values: List(Float), opts: PrintOptions) -> Formatter {
  let finite_nonzero =
    values
    |> list.filter(fn(v) { format_ffi.is_finite(v) && v != 0.0 })

  case looks_integral(values) {
    True -> build_int_formatter(values, opts)
    False -> build_float_formatter(values, finite_nonzero, opts)
  }
}

fn looks_integral(values: List(Float)) -> Bool {
  list.all(values, fn(v) {
    case format_ffi.is_finite(v) {
      False -> True
      True -> v == float_truncate(v)
    }
  })
}

fn float_truncate(v: Float) -> Float {
  int.to_float(float.truncate(v))
}

fn build_int_formatter(values: List(Float), opts: PrintOptions) -> Formatter {
  // Render each finite value as "N." and each non-finite as nan/inf.
  let strs =
    list.map(values, fn(v) {
      case classify(v) {
        Finite -> {
          let n = float.truncate(v)
          int.to_string(n) <> "."
        }
        Nan -> opts.nan_str
        PosInf -> opts.inf_str
        NegInf -> "-" <> opts.inf_str
      }
    })
  let max_w = max_width(strs)
  let opts_local = opts
  Formatter(width: max_w, render: fn(v: Float) -> String {
    let raw = case classify(v) {
      Finite -> int.to_string(float.truncate(v)) <> "."
      Nan -> opts_local.nan_str
      PosInf -> opts_local.inf_str
      NegInf -> "-" <> opts_local.inf_str
    }
    string.pad_start(raw, to: max_w, with: " ")
  })
}

fn build_float_formatter(
  values: List(Float),
  finite_nonzero: List(Float),
  opts: PrintOptions,
) -> Formatter {
  let sci_mode = decide_sci_mode(finite_nonzero, opts)
  let precision = opts.precision

  let strs =
    list.map(values, fn(v) {
      case classify(v) {
        Finite -> render_float(v, sci_mode, precision)
        Nan -> opts.nan_str
        PosInf -> opts.inf_str
        NegInf -> "-" <> opts.inf_str
      }
    })

  let max_w = max_width(strs)
  let opts_local = opts

  Formatter(width: max_w, render: fn(v: Float) -> String {
    let raw = case classify(v) {
      Finite -> render_float(v, sci_mode, precision)
      Nan -> opts_local.nan_str
      PosInf -> opts_local.inf_str
      NegInf -> "-" <> opts_local.inf_str
    }
    string.pad_start(raw, to: max_w, with: " ")
  })
}

fn render_float(v: Float, sci_mode: Bool, precision: Int) -> String {
  case sci_mode {
    True -> format_ffi.fmt_sci(v, precision)
    False -> format_ffi.fmt_fixed(v, precision)
  }
}

fn decide_sci_mode(finite_nonzero: List(Float), opts: PrintOptions) -> Bool {
  case opts.sci_mode {
    SciAlways -> True
    SciNever -> False
    SciAuto ->
      case finite_nonzero {
        [] -> False
        _ -> {
          let abs_vals = list.map(finite_nonzero, float.absolute_value)
          let fmin = list.fold(abs_vals, 1.0e308, float.min)
          let fmax = list.fold(abs_vals, 0.0, float.max)
          let ratio_trigger = case fmin >. 0.0 {
            True -> fmax /. fmin >. 1000.0
            False -> False
          }
          fmax >=. 1.0e8 || fmin <. 1.0e-4 || ratio_trigger
        }
      }
  }
}

type Classification {
  Finite
  Nan
  PosInf
  NegInf
}

fn classify(v: Float) -> Classification {
  case format_ffi.is_finite(v) {
    True -> Finite
    False ->
      case format_ffi.is_nan(v) {
        True -> Nan
        False ->
          case v >. 0.0 {
            True -> PosInf
            False -> NegInf
          }
      }
  }
}

fn max_width(strs: List(String)) -> Int {
  list.fold(strs, 1, fn(acc, s) { int.max(acc, string.length(s)) })
}

// =============================================================================
// Line wrapping for innermost rows
// =============================================================================

fn wrap_line(
  parts: List(String),
  separator: String,
  budget: Int,
  indent: String,
) -> String {
  case parts {
    [] -> ""
    [first, ..rest] -> wrap_loop(rest, separator, budget, indent, first, first)
  }
}

fn wrap_loop(
  remaining: List(String),
  separator: String,
  budget: Int,
  indent: String,
  current_line: String,
  acc: String,
) -> String {
  case remaining {
    [] -> acc
    [next, ..rest] -> {
      let candidate = current_line <> separator <> next
      case string.length(candidate) > budget {
        True -> {
          // Break: new line with indent then next element.
          let new_line = indent <> next
          let new_acc = acc <> "," <> "\n" <> new_line
          wrap_loop(rest, separator, budget, indent, new_line, new_acc)
        }
        False -> {
          let new_acc = acc <> separator <> next
          wrap_loop(rest, separator, budget, indent, candidate, new_acc)
        }
      }
    }
  }
}

// =============================================================================
// Misc helpers
// =============================================================================

fn total_size(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, d) { acc * d })
}

fn list_at(xs: List(Int), idx: Int) -> Result(Int, Nil) {
  list_at_loop(xs, idx)
}

fn list_at_loop(xs: List(Int), idx: Int) -> Result(Int, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at_loop(rest, n - 1)
  }
}

fn list_at_float(xs: List(Float), idx: Int) -> Result(Float, Nil) {
  list_at_float_loop(xs, idx)
}

fn list_at_float_loop(xs: List(Float), idx: Int) -> Result(Float, Nil) {
  case xs, idx {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at_float_loop(rest, n - 1)
  }
}
