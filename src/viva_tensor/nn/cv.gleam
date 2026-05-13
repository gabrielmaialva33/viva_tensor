//// Computer-vision ops for VIVA tensor.
////
//// Pure-Gleam, no NIF, no autograd. Mirrors the conventions established in
//// `viva_tensor/nn/conv`, `viva_tensor/nn/pool` and `viva_tensor/nn/norm`:
//// values carry config + learnable params, forward passes return
//// `Result(Tensor, TensorError)`, padding/stride formulas match PyTorch.
////
//// Included:
////   - `max_pool_2d_with_indices` / `max_unpool_2d_forward`
////       Output shape (pool):
////         H_out = (H_in + 2*P - K) / S + 1
////         W_out = (W_in + 2*P - K) / S + 1
////       Output shape (unpool): caller-provided `(H_in, W_in)`.
////   - `nms`
////       Returns a `List(Int)` of kept box indices, sorted by descending score.
////   - `roi_align`
////       Output shape: `[K, C, output_h, output_w]`.
////   - `batched_matmul`
////       Output shape: `[max(Ba, Bb), M, N]` for `[Ba, M, K] @ [Bb, K, N]`,
////       with `B == 1` broadcasting on either side.
////   - `batch_norm_2d_init` / `batch_norm_2d_forward`
////       Normalizes over `[B, H, W]` per channel `C`. Output shape:
////       same as input `[B, C, H, W]`.

import gleam/dict
import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/core/ffi.{type ErlangArray}
import viva_tensor/tensor.{type Tensor, Tensor}

// ---------------------------------------------------------------------------
// MaxPool2d with indices / MaxUnpool2d
// ---------------------------------------------------------------------------

/// Configuration for `max_unpool_2d_forward`.
///
/// Mirrors the matching pool's config: the kernel/stride/padding used to
/// produce the indices in the first place. They're not consulted by the
/// unpool math (the stored flat indices fully encode source positions), but
/// kept on the config for symmetry with PyTorch's `nn.MaxUnpool2d`.
pub type MaxUnpool2dConfig {
  MaxUnpool2dConfig(kernel_size: Int, stride: Int, padding: Int)
}

/// Run a 2D max-pool and return both the pooled output and the flat index
/// of the argmax inside each window.
///
/// `input` shape `[N, C, H_in, W_in]`. Output:
/// - `pooled` shape `[N, C, H_out, W_out]` with
///   `H_out = (H_in + 2*padding - kernel_size) / stride + 1` and analogously
///   for `W_out`.
/// - `indices` shape `[N, C, H_out, W_out]`, each entry is a *flat* index
///   into the `[H_in, W_in]` spatial plane of the source channel, encoded as
///   a `Float` so it fits in `Tensor`. Out-of-bound (fully-padded) windows
///   get index `-1.0`.
///
/// Padding is zero-padding. Ties inside a window pick the first (row-major)
/// occurrence.
pub fn max_pool_2d_with_indices(
  input: Tensor,
  kernel_size: Int,
  stride: Int,
  padding: Int,
) -> Result(#(Tensor, Tensor), TensorError) {
  case tensor.shape(input) {
    [n, c, h_in, w_in] -> {
      case kernel_size <= 0 || stride <= 0 {
        True ->
          Error(InvalidShape(
            "max_pool_2d_with_indices: kernel_size and stride must be positive, got kernel="
            <> int.to_string(kernel_size)
            <> " stride="
            <> int.to_string(stride),
          ))
        False -> {
          let padded_h = h_in + 2 * padding
          let padded_w = w_in + 2 * padding
          let out_h = { padded_h - kernel_size } / stride + 1
          let out_w = { padded_w - kernel_size } / stride + 1
          case out_h <= 0 || out_w <= 0 {
            True ->
              Error(InvalidShape(
                "max_pool_2d_with_indices: invalid output dims ("
                <> int.to_string(out_h)
                <> ", "
                <> int.to_string(out_w)
                <> ")",
              ))
            False -> {
              let arr = ffi.list_to_array(tensor.to_list(input))
              let #(values, indices) =
                pool_with_indices_compute(
                  arr,
                  n,
                  c,
                  h_in,
                  w_in,
                  out_h,
                  out_w,
                  kernel_size,
                  stride,
                  padding,
                )
              let val_tensor = Tensor(data: values, shape: [n, c, out_h, out_w])
              let idx_tensor =
                Tensor(data: indices, shape: [n, c, out_h, out_w])
              Ok(#(val_tensor, idx_tensor))
            }
          }
        }
      }
    }
    other ->
      Error(InvalidShape(
        "max_pool_2d_with_indices: input must have shape [N, C, H, W], got "
        <> shape_to_string(other),
      ))
  }
}

fn pool_with_indices_compute(
  arr: ErlangArray,
  n: Int,
  c: Int,
  h_in: Int,
  w_in: Int,
  out_h: Int,
  out_w: Int,
  kernel_size: Int,
  stride: Int,
  padding: Int,
) -> #(List(Float), List(Float)) {
  let cells =
    list.range(0, n - 1)
    |> list.flat_map(fn(b) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ch) {
        let base = b * c * h_in * w_in + ch * h_in * w_in
        list.range(0, out_h - 1)
        |> list.flat_map(fn(oh) {
          let h_start = oh * stride - padding
          list.range(0, out_w - 1)
          |> list.map(fn(ow) {
            let w_start = ow * stride - padding
            max_in_window(arr, base, h_in, w_in, h_start, w_start, kernel_size)
          })
        })
      })
    })
  let values = list.map(cells, fn(p) { p.0 })
  let indices = list.map(cells, fn(p) { int.to_float(p.1) })
  #(values, indices)
}

/// Walk a `kernel_size x kernel_size` window, return `#(max_value, flat_idx)`.
/// `flat_idx` is into the `[H_in, W_in]` plane. Windows that lie entirely in
/// the zero-padding return `(0.0, -1)`.
fn max_in_window(
  arr: ErlangArray,
  base: Int,
  h_in: Int,
  w_in: Int,
  h_start: Int,
  w_start: Int,
  kernel_size: Int,
) -> #(Float, Int) {
  scan_window_rows(
    arr,
    base,
    h_in,
    w_in,
    h_start,
    w_start,
    kernel_size,
    0,
    None2,
  )
}

type MaxAcc {
  None2
  Some2(value: Float, idx: Int)
}

fn scan_window_rows(
  arr: ErlangArray,
  base: Int,
  h_in: Int,
  w_in: Int,
  h_start: Int,
  w_start: Int,
  kernel_size: Int,
  ky: Int,
  acc: MaxAcc,
) -> #(Float, Int) {
  case ky >= kernel_size {
    True ->
      case acc {
        None2 -> #(0.0, -1)
        Some2(v, i) -> #(v, i)
      }
    False -> {
      let acc2 =
        scan_window_cols(
          arr,
          base,
          h_in,
          w_in,
          h_start,
          w_start,
          kernel_size,
          ky,
          0,
          acc,
        )
      scan_window_rows(
        arr,
        base,
        h_in,
        w_in,
        h_start,
        w_start,
        kernel_size,
        ky + 1,
        acc2,
      )
    }
  }
}

fn scan_window_cols(
  arr: ErlangArray,
  base: Int,
  h_in: Int,
  w_in: Int,
  h_start: Int,
  w_start: Int,
  kernel_size: Int,
  ky: Int,
  kx: Int,
  acc: MaxAcc,
) -> MaxAcc {
  case kx >= kernel_size {
    True -> acc
    False -> {
      let y = h_start + ky
      let x = w_start + kx
      let acc2 = case y >= 0 && y < h_in && x >= 0 && x < w_in {
        False -> acc
        True -> {
          let flat = y * w_in + x
          let v = ffi.array_get(arr, base + flat)
          case acc {
            None2 -> Some2(v, flat)
            Some2(mv, _) ->
              case v >. mv {
                True -> Some2(v, flat)
                False -> acc
              }
          }
        }
      }
      scan_window_cols(
        arr,
        base,
        h_in,
        w_in,
        h_start,
        w_start,
        kernel_size,
        ky,
        kx + 1,
        acc2,
      )
    }
  }
}

/// Reverse a `max_pool_2d_with_indices` forward pass.
///
/// `input` shape `[N, C, H_out, W_out]` (the pooled values), `indices` shape
/// `[N, C, H_out, W_out]` (flat indices into `[H_in, W_in]` from the
/// matching pool forward), `output_size = #(H_in, W_in)`.
///
/// Output shape: `[N, C, H_in, W_in]`. Cells whose flat index appears in
/// `indices` carry the corresponding pooled value; every other cell is `0`.
/// Negative indices are treated as "no source" and contribute nothing
/// (matches the `(0.0, -1)` sentinel emitted by fully-padded windows).
///
/// `config.kernel_size` / `stride` / `padding` are not consulted by the math
/// — they live on the config for parity with PyTorch's `nn.MaxUnpool2d` and
/// future debug checks.
pub fn max_unpool_2d_forward(
  _config: MaxUnpool2dConfig,
  input: Tensor,
  indices: Tensor,
  output_size: #(Int, Int),
) -> Result(Tensor, TensorError) {
  let #(h_in, w_in) = output_size
  case tensor.shape(input), tensor.shape(indices) {
    [n, c, h_out, w_out], [ni, ci, hi, wi] -> {
      case n == ni && c == ci && h_out == hi && w_out == wi {
        False ->
          Error(
            ShapeMismatch(expected: [n, c, h_out, w_out], got: [
              ni,
              ci,
              hi,
              wi,
            ]),
          )
        True ->
          case h_in <= 0 || w_in <= 0 {
            True ->
              Error(InvalidShape(
                "max_unpool_2d: output_size must be positive, got ("
                <> int.to_string(h_in)
                <> ", "
                <> int.to_string(w_in)
                <> ")",
              ))
            False -> {
              let values = tensor.to_list(input)
              let idxs = tensor.to_list(indices)
              let plane = h_in * w_in
              let out = unpool_fill(n, c, h_out, w_out, plane, values, idxs)
              Ok(Tensor(data: out, shape: [n, c, h_in, w_in]))
            }
          }
      }
    }
    other_in, _ ->
      Error(InvalidShape(
        "max_unpool_2d: input must have shape [N, C, H_out, W_out], got "
        <> shape_to_string(other_in),
      ))
  }
}

fn unpool_fill(
  n: Int,
  c: Int,
  h_out: Int,
  w_out: Int,
  plane: Int,
  values: List(Float),
  idxs: List(Float),
) -> List(Float) {
  let per_chan = h_out * w_out
  list.range(0, n - 1)
  |> list.flat_map(fn(b) {
    list.range(0, c - 1)
    |> list.flat_map(fn(ch) {
      let start = { b * c + ch } * per_chan
      let chan_vals = list.take(list.drop(values, start), per_chan)
      let chan_idxs = list.take(list.drop(idxs, start), per_chan)
      let scatter =
        list.fold(list.zip(chan_idxs, chan_vals), dict.new(), fn(acc, pair) {
          let #(idx_f, v) = pair
          let idx = float.truncate(idx_f)
          case idx >= 0 && idx < plane {
            True -> dict.insert(acc, idx, v)
            False -> acc
          }
        })
      list.range(0, plane - 1)
      |> list.map(fn(i) {
        case dict.get(scatter, i) {
          Ok(v) -> v
          Error(_) -> 0.0
        }
      })
    })
  })
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

/// Greedy Non-Maximum Suppression.
///
/// `boxes` shape `[N, 4]`, rows are `[x1, y1, x2, y2]` (no specific
/// coordinate convention enforced — caller's responsibility to keep it
/// consistent across boxes). `scores` shape `[N]`. Returns the indices into
/// the original `boxes` of the boxes kept after suppression, sorted by
/// descending score.
///
/// Algorithm: sort by descending score; pop the top, add to `kept`, drop any
/// remaining whose IoU with the popped box exceeds `iou_threshold`.
///
/// `iou_threshold` is consumed as-is; callers commonly pass `0.5`. Values
/// outside `[0, 1]` are not validated — `0.0` keeps only the single
/// highest-scoring non-overlapping cluster, `1.0` keeps every box.
pub fn nms(
  boxes: Tensor,
  scores: Tensor,
  iou_threshold: Float,
) -> Result(List(Int), TensorError) {
  case tensor.shape(boxes) {
    [n, 4] -> {
      case tensor.shape(scores) {
        [ns] if ns == n -> {
          let box_rows = chunk_by(tensor.to_list(boxes), 4)
          let score_list = tensor.to_list(scores)
          let indexed =
            list.index_map(score_list, fn(s, i) { #(i, s) })
            |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
          let order = list.map(indexed, fn(p) { p.0 })
          let kept = nms_loop(order, box_rows, iou_threshold, [])
          Ok(list.reverse(kept))
        }
        other -> Error(ShapeMismatch(expected: [n], got: other))
      }
    }
    other ->
      Error(InvalidShape(
        "nms: boxes must have shape [N, 4], got " <> shape_to_string(other),
      ))
  }
}

fn nms_loop(
  order: List(Int),
  box_rows: List(List(Float)),
  iou_threshold: Float,
  kept: List(Int),
) -> List(Int) {
  case order {
    [] -> kept
    [top, ..rest] -> {
      let top_box = list_at_list(box_rows, top)
      let remaining =
        list.filter(rest, fn(j) {
          let other = list_at_list(box_rows, j)
          iou(top_box, other) <=. iou_threshold
        })
      nms_loop(remaining, box_rows, iou_threshold, [top, ..kept])
    }
  }
}

fn iou(a: List(Float), b: List(Float)) -> Float {
  case a, b {
    [ax1, ay1, ax2, ay2], [bx1, by1, bx2, by2] -> {
      let inter_x1 = float.max(ax1, bx1)
      let inter_y1 = float.max(ay1, by1)
      let inter_x2 = float.min(ax2, bx2)
      let inter_y2 = float.min(ay2, by2)
      let iw = float.max(0.0, inter_x2 -. inter_x1)
      let ih = float.max(0.0, inter_y2 -. inter_y1)
      let inter = iw *. ih
      let area_a = float.max(0.0, ax2 -. ax1) *. float.max(0.0, ay2 -. ay1)
      let area_b = float.max(0.0, bx2 -. bx1) *. float.max(0.0, by2 -. by1)
      let union = area_a +. area_b -. inter
      case union <=. 0.0 {
        True -> 0.0
        False -> inter /. union
      }
    }
    _, _ -> 0.0
  }
}

fn list_at_list(xs: List(List(Float)), idx: Int) -> List(Float) {
  case xs, idx {
    [], _ -> []
    [x, ..], 0 -> x
    [_, ..rest], i -> list_at_list(rest, i - 1)
  }
}

// ---------------------------------------------------------------------------
// ROIAlign
// ---------------------------------------------------------------------------

/// Configuration for `roi_align`.
///
/// - `output_h`, `output_w`: spatial size of the aligned output.
/// - `spatial_scale`: multiplicative factor mapping ROI coordinates from
///   image space into feature-map space. Set to `1.0` when ROIs are already
///   in feature coordinates.
/// - `sampling_ratio`: number of bilinear samples per output bin per axis.
///   `sampling_ratio = 2` ⇒ 2x2 = 4 samples per bin (TorchVision default).
///   Must be ≥ 1.
pub type RoiAlignConfig {
  RoiAlignConfig(
    output_h: Int,
    output_w: Int,
    spatial_scale: Float,
    sampling_ratio: Int,
  )
}

/// Bilinear ROIAlign, matching TorchVision's `aligned=False` semantics.
///
/// `features` shape `[N, C, H, W]`. `rois` shape `[K, 5]`, each row is
/// `[batch_index, x1, y1, x2, y2]` in image coordinates (scaled by
/// `spatial_scale` internally). Output shape: `[K, C, output_h, output_w]`.
///
/// Per output bin, samples `sampling_ratio x sampling_ratio` points with
/// bilinear interpolation and averages them. Out-of-bound samples
/// (after scaling) contribute `0.0`. `batch_index` is clamped into
/// `[0, N-1]` — see the module header for the simplifications relative to
/// full TorchVision ROIAlign.
pub fn roi_align(
  config: RoiAlignConfig,
  features: Tensor,
  rois: Tensor,
) -> Result(Tensor, TensorError) {
  case tensor.shape(features) {
    [n, c, h, w] -> {
      case tensor.shape(rois) {
        [k, 5] -> {
          case
            config.output_h <= 0
            || config.output_w <= 0
            || config.sampling_ratio <= 0
          {
            True ->
              Error(InvalidShape(
                "roi_align: output_h, output_w and sampling_ratio must be positive",
              ))
            False -> {
              let feat = ffi.list_to_array(tensor.to_list(features))
              let roi_rows = chunk_by(tensor.to_list(rois), 5)
              let out =
                roi_align_compute(
                  feat,
                  roi_rows,
                  n,
                  c,
                  h,
                  w,
                  config.output_h,
                  config.output_w,
                  config.spatial_scale,
                  config.sampling_ratio,
                )
              Ok(
                Tensor(data: out, shape: [
                  k,
                  c,
                  config.output_h,
                  config.output_w,
                ]),
              )
            }
          }
        }
        other ->
          Error(InvalidShape(
            "roi_align: rois must have shape [K, 5], got "
            <> shape_to_string(other),
          ))
      }
    }
    other ->
      Error(InvalidShape(
        "roi_align: features must have shape [N, C, H, W], got "
        <> shape_to_string(other),
      ))
  }
}

fn roi_align_compute(
  feat: ErlangArray,
  roi_rows: List(List(Float)),
  n: Int,
  c: Int,
  h: Int,
  w: Int,
  out_h: Int,
  out_w: Int,
  spatial_scale: Float,
  sampling_ratio: Int,
) -> List(Float) {
  let ratio_f = int.to_float(sampling_ratio)
  let inv_samples = 1.0 /. { ratio_f *. ratio_f }
  list.flat_map(roi_rows, fn(row) {
    case row {
      [bi_f, x1, y1, x2, y2] -> {
        let bi_raw = float.truncate(bi_f)
        let bi = case bi_raw < 0 {
          True -> 0
          False ->
            case bi_raw >= n {
              True -> n - 1
              False -> bi_raw
            }
        }
        let sx1 = x1 *. spatial_scale
        let sy1 = y1 *. spatial_scale
        let sx2 = x2 *. spatial_scale
        let sy2 = y2 *. spatial_scale
        let roi_w = float.max(sx2 -. sx1, 1.0)
        let roi_h = float.max(sy2 -. sy1, 1.0)
        let bin_w = roi_w /. int.to_float(out_w)
        let bin_h = roi_h /. int.to_float(out_h)
        list.flat_map(list.range(0, c - 1), fn(ch) {
          let base = bi * c * h * w + ch * h * w
          list.flat_map(list.range(0, out_h - 1), fn(oh) {
            list.map(list.range(0, out_w - 1), fn(ow) {
              roi_bin_value(
                feat,
                base,
                h,
                w,
                sx1,
                sy1,
                bin_w,
                bin_h,
                oh,
                ow,
                sampling_ratio,
                ratio_f,
                inv_samples,
              )
            })
          })
        })
      }
      _ -> []
    }
  })
}

fn roi_bin_value(
  feat: ErlangArray,
  base: Int,
  h: Int,
  w: Int,
  sx1: Float,
  sy1: Float,
  bin_w: Float,
  bin_h: Float,
  oh: Int,
  ow: Int,
  sampling_ratio: Int,
  ratio_f: Float,
  inv_samples: Float,
) -> Float {
  let bin_x_start = sx1 +. int.to_float(ow) *. bin_w
  let bin_y_start = sy1 +. int.to_float(oh) *. bin_h
  let sub_w = bin_w /. ratio_f
  let sub_h = bin_h /. ratio_f
  let sum =
    sum_range(0, sampling_ratio - 1, fn(iy) {
      let y = bin_y_start +. { int.to_float(iy) +. 0.5 } *. sub_h
      sum_range(0, sampling_ratio - 1, fn(ix) {
        let x = bin_x_start +. { int.to_float(ix) +. 0.5 } *. sub_w
        bilinear_sample(feat, base, h, w, y, x)
      })
    })
  sum *. inv_samples
}

fn bilinear_sample(
  feat: ErlangArray,
  base: Int,
  h: Int,
  w: Int,
  y: Float,
  x: Float,
) -> Float {
  case y <. -1.0 || y >. int.to_float(h) || x <. -1.0 || x >. int.to_float(w) {
    True -> 0.0
    False -> {
      let yc = clamp_float(y, 0.0, int.to_float(h - 1))
      let xc = clamp_float(x, 0.0, int.to_float(w - 1))
      let y0 = float.truncate(yc)
      let x0 = float.truncate(xc)
      let y1 = case y0 + 1 >= h {
        True -> h - 1
        False -> y0 + 1
      }
      let x1 = case x0 + 1 >= w {
        True -> w - 1
        False -> x0 + 1
      }
      let dy = yc -. int.to_float(y0)
      let dx = xc -. int.to_float(x0)
      let v00 = ffi.array_get(feat, base + y0 * w + x0)
      let v01 = ffi.array_get(feat, base + y0 * w + x1)
      let v10 = ffi.array_get(feat, base + y1 * w + x0)
      let v11 = ffi.array_get(feat, base + y1 * w + x1)
      let top = v00 *. { 1.0 -. dx } +. v01 *. dx
      let bot = v10 *. { 1.0 -. dx } +. v11 *. dx
      top *. { 1.0 -. dy } +. bot *. dy
    }
  }
}

fn clamp_float(v: Float, lo: Float, hi: Float) -> Float {
  case v <. lo {
    True -> lo
    False ->
      case v >. hi {
        True -> hi
        False -> v
      }
  }
}

// ---------------------------------------------------------------------------
// Batched matmul
// ---------------------------------------------------------------------------

/// Batched 2-D matrix multiplication.
///
/// `a` shape `[Ba, M, K]`, `b` shape `[Bb, K, N]`. Output shape
/// `[max(Ba, Bb), M, N]`. Broadcasts the batch dim when either `Ba == 1` or
/// `Bb == 1`; otherwise requires `Ba == Bb`. Inner contraction is the usual
/// `K`-dim sum.
pub fn batched_matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [ba, m, ka], [bb, kb, n] -> {
      case ka == kb {
        False -> Error(ShapeMismatch(expected: [bb, ka, n], got: [bb, kb, n]))
        True ->
          case batch_dim(ba, bb) {
            Error(e) -> Error(e)
            Ok(out_b) -> {
              let a_arr = ffi.list_to_array(tensor.to_list(a))
              let b_arr = ffi.list_to_array(tensor.to_list(b))
              let out = bmm_compute(a_arr, b_arr, out_b, m, ka, n, ba, bb)
              Ok(Tensor(data: out, shape: [out_b, m, n]))
            }
          }
      }
    }
    sa, sb ->
      Error(InvalidShape(
        "batched_matmul: inputs must be 3D [B, M, K] and [B, K, N], got "
        <> shape_to_string(sa)
        <> " and "
        <> shape_to_string(sb),
      ))
  }
}

fn batch_dim(ba: Int, bb: Int) -> Result(Int, TensorError) {
  case ba == bb {
    True -> Ok(ba)
    False ->
      case ba == 1 {
        True -> Ok(bb)
        False ->
          case bb == 1 {
            True -> Ok(ba)
            False -> Error(ShapeMismatch(expected: [ba], got: [bb]))
          }
      }
  }
}

fn bmm_compute(
  a_arr: ErlangArray,
  b_arr: ErlangArray,
  out_b: Int,
  m: Int,
  k: Int,
  n: Int,
  ba: Int,
  bb: Int,
) -> List(Float) {
  list.range(0, out_b - 1)
  |> list.flat_map(fn(batch) {
    let ai = case ba == 1 {
      True -> 0
      False -> batch
    }
    let bi = case bb == 1 {
      True -> 0
      False -> batch
    }
    let a_base = ai * m * k
    let b_base = bi * k * n
    list.range(0, m - 1)
    |> list.flat_map(fn(i) {
      list.range(0, n - 1)
      |> list.map(fn(j) {
        sum_range(0, k - 1, fn(p) {
          let av = ffi.array_get(a_arr, a_base + i * k + p)
          let bv = ffi.array_get(b_arr, b_base + p * n + j)
          av *. bv
        })
      })
    })
  })
}

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

/// 2D Batch Normalization layer. Normalizes over the `[B, H, W]` axes per
/// channel `C`.
///
/// Training mode: uses per-channel batch mean/var and updates running stats
/// via EMA. Eval mode: uses running stats and leaves them untouched.
///
/// Parameter shapes: all of `scale`, `bias`, `running_mean`, `running_var`
/// are 1D tensors of length `C` (`num_features`).
pub type BatchNorm2d {
  BatchNorm2d(
    scale: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: Float,
    eps: Float,
  )
}

/// Initialize a `BatchNorm2d` with `scale = ones([C])`, `bias = zeros([C])`,
/// `running_mean = zeros([C])`, `running_var = ones([C])`, `momentum = 0.1`,
/// and `eps = 1.0e-5`. `C = num_features`.
///
/// Formula (training):
///   `mu  = mean(x, axes=[B, H, W])`
///   `var = var(x,  axes=[B, H, W])`
///   `y   = (x - mu) / sqrt(var + eps) * scale + bias`
///   `running = (1 - momentum) * running + momentum * batch_stat`
///
/// Formula (eval):
///   `y = (x - running_mean) / sqrt(running_var + eps) * scale + bias`.
pub fn batch_norm_2d_init(num_features: Int) -> BatchNorm2d {
  BatchNorm2d(
    scale: tensor.ones([num_features]),
    bias: tensor.zeros([num_features]),
    running_mean: tensor.zeros([num_features]),
    running_var: tensor.ones([num_features]),
    momentum: 0.1,
    eps: 1.0e-5,
  )
}

/// Forward pass for `BatchNorm2d`.
///
/// `input` shape `[B, C, H, W]`. Output shape: same as input.
///
/// In `training` mode, computes per-channel mean/variance across the
/// `[B, H, W]` axes, normalizes, applies per-channel `scale` / `bias`, and
/// updates `running_mean` / `running_var` via exponential moving average:
///   `running = (1 - momentum) * running + momentum * batch_stat`.
///
/// In eval mode (`training = False`), normalizes using `running_mean` /
/// `running_var` directly and returns the layer unchanged.
///
/// Errors:
/// - `InvalidShape` if the input is not 4D or batch is non-positive.
/// - `ShapeMismatch` if `C` does not match `num_features`.
pub fn batch_norm_2d_forward(
  layer: BatchNorm2d,
  input: Tensor,
  training: Bool,
) -> Result(#(BatchNorm2d, Tensor), TensorError) {
  let scale_shape = tensor.shape(layer.scale)
  use num_features <- result.try(last_dim(scale_shape))
  case tensor.shape(input) {
    [b, c, h, w] -> {
      case c == num_features {
        False ->
          Error(
            ShapeMismatch(expected: [b, num_features, h, w], got: [
              b,
              c,
              h,
              w,
            ]),
          )
        True ->
          case b <= 0 || h <= 0 || w <= 0 {
            True ->
              Error(InvalidShape("batch_norm_2d: B, H, W must be positive"))
            False -> batch_norm_2d_apply(layer, input, b, c, h, w, training)
          }
      }
    }
    other ->
      Error(InvalidShape(
        "batch_norm_2d: input must have shape [B, C, H, W], got "
        <> shape_to_string(other),
      ))
  }
}

fn batch_norm_2d_apply(
  layer: BatchNorm2d,
  input: Tensor,
  b: Int,
  c: Int,
  h: Int,
  w: Int,
  training: Bool,
) -> Result(#(BatchNorm2d, Tensor), TensorError) {
  use data <- result.try(tensor.try_to_list(input))
  use scale_data <- result.try(tensor.try_to_list(layer.scale))
  use bias_data <- result.try(tensor.try_to_list(layer.bias))
  use running_mean_data <- result.try(tensor.try_to_list(layer.running_mean))
  use running_var_data <- result.try(tensor.try_to_list(layer.running_var))

  let plane = h * w
  let arr = ffi.list_to_array(data)

  // Per-channel mean/var over [B, H, W].
  let batch_mean = channel_means(arr, b, c, plane)
  let batch_var = channel_variances(arr, b, c, plane, batch_mean)

  let #(use_mean, use_var) = case training {
    True -> #(batch_mean, batch_var)
    False -> #(running_mean_data, running_var_data)
  }

  let out =
    list.range(0, b - 1)
    |> list.flat_map(fn(bi) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ci) {
        let mu = list_at(use_mean, ci)
        let var = list_at(use_var, ci)
        let s = list_at(scale_data, ci)
        let bias_v = list_at(bias_data, ci)
        let denom = safe_sqrt(var +. layer.eps)
        let base = { bi * c + ci } * plane
        list.range(0, plane - 1)
        |> list.map(fn(off) {
          let x = ffi.array_get(arr, base + off)
          { x -. mu } /. denom *. s +. bias_v
        })
      })
    })

  let out_tensor = Tensor(data: out, shape: [b, c, h, w])

  let updated_layer = case training {
    False -> layer
    True -> {
      let new_mean = ema_update(running_mean_data, batch_mean, layer.momentum)
      let new_var = ema_update(running_var_data, batch_var, layer.momentum)
      BatchNorm2d(
        scale: layer.scale,
        bias: layer.bias,
        running_mean: Tensor(data: new_mean, shape: [c]),
        running_var: Tensor(data: new_var, shape: [c]),
        momentum: layer.momentum,
        eps: layer.eps,
      )
    }
  }

  Ok(#(updated_layer, out_tensor))
}

fn channel_means(arr: ErlangArray, b: Int, c: Int, plane: Int) -> List(Float) {
  let n = int.to_float(b * plane)
  list.range(0, c - 1)
  |> list.map(fn(ci) {
    let sum =
      sum_range(0, b - 1, fn(bi) {
        let base = { bi * c + ci } * plane
        sum_range(0, plane - 1, fn(off) { ffi.array_get(arr, base + off) })
      })
    sum /. n
  })
}

fn channel_variances(
  arr: ErlangArray,
  b: Int,
  c: Int,
  plane: Int,
  means: List(Float),
) -> List(Float) {
  let n = int.to_float(b * plane)
  list.index_map(means, fn(mu, ci) {
    let sum =
      sum_range(0, b - 1, fn(bi) {
        let base = { bi * c + ci } * plane
        sum_range(0, plane - 1, fn(off) {
          let d = ffi.array_get(arr, base + off) -. mu
          d *. d
        })
      })
    sum /. n
  })
}

fn ema_update(
  running: List(Float),
  batch: List(Float),
  momentum: Float,
) -> List(Float) {
  let one_minus = 1.0 -. momentum
  list.map(list.zip(running, batch), fn(p) {
    let #(r, b) = p
    one_minus *. r +. momentum *. b
  })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sum_range(start: Int, end: Int, f: fn(Int) -> Float) -> Float {
  sum_range_acc(start, end, f, 0.0)
}

fn sum_range_acc(
  start: Int,
  end: Int,
  f: fn(Int) -> Float,
  acc: Float,
) -> Float {
  case start > end {
    True -> acc
    False -> sum_range_acc(start + 1, end, f, acc +. f(start))
  }
}

fn chunk_by(data: List(Float), n: Int) -> List(List(Float)) {
  case n <= 0 {
    True -> []
    False -> chunk_by_loop(data, n, [])
  }
}

fn chunk_by_loop(
  data: List(Float),
  n: Int,
  acc: List(List(Float)),
) -> List(List(Float)) {
  case data {
    [] -> list.reverse(acc)
    _ -> {
      let chunk = list.take(data, n)
      let rest = list.drop(data, n)
      chunk_by_loop(rest, n, [chunk, ..acc])
    }
  }
}

fn last_dim(shape: List(Int)) -> Result(Int, TensorError) {
  case list.last(shape) {
    Ok(d) -> Ok(d)
    Error(_) -> Error(InvalidShape("expected non-empty shape, got []"))
  }
}

fn safe_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn list_at(xs: List(Float), idx: Int) -> Float {
  case xs, idx {
    [], _ -> 0.0
    [x, ..], 0 -> x
    [_, ..rest], i -> list_at(rest, i - 1)
  }
}

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> join_with(list.map(shape, int.to_string), ", ") <> "]"
}

fn join_with(parts: List(String), sep: String) -> String {
  case parts {
    [] -> ""
    [s] -> s
    [s, ..rest] -> s <> sep <> join_with(rest, sep)
  }
}
