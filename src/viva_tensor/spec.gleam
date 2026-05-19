//// Tensor specifications for runtime planning.

import gleam/int
import gleam/list
import viva_tensor/layout
import viva_tensor/tensor

pub type TensorSpec {
  TensorSpec(
    shape: List(Int),
    dtype: layout.TensorDtype,
    device: layout.TensorDevice,
    storage: layout.TensorStorage,
    memory_layout: layout.TensorMemoryLayout,
    rank: Int,
    size: Int,
  )
}

pub fn tensor_spec(t: tensor.Tensor) -> TensorSpec {
  from_layout(tensor.layout(t))
}

pub fn from_layout(metadata: layout.TensorLayout) -> TensorSpec {
  TensorSpec(
    shape: metadata.shape,
    dtype: metadata.dtype,
    device: metadata.device,
    storage: metadata.storage,
    memory_layout: case metadata.contiguous {
      True -> layout.RowMajor
      False -> layout.StridedLayout
    },
    rank: metadata.rank,
    size: metadata.size,
  )
}

pub fn spec_from_parts(
  shape shape: List(Int),
  dtype dtype: layout.TensorDtype,
  device device: layout.TensorDevice,
  storage storage: layout.TensorStorage,
  memory_layout memory_layout: layout.TensorMemoryLayout,
) -> TensorSpec {
  TensorSpec(
    shape: shape,
    dtype: dtype,
    device: device,
    storage: storage,
    memory_layout: memory_layout,
    rank: list.length(shape),
    size: list.fold(shape, 1, fn(acc, dim) { acc * dim }),
  )
}

pub fn dtype_name(dtype: layout.TensorDtype) -> String {
  case dtype {
    layout.Float64 -> "f64"
    layout.Float32 -> "f32"
    layout.Float16 -> "f16"
    layout.BFloat16 -> "bf16"
    layout.Float8E4M3 -> "fp8_e4m3"
    layout.Int8 -> "int8"
    layout.Int4 -> "int4"
    layout.SparseFloat16 -> "sparse_f16"
  }
}

pub fn device_name(device: layout.TensorDevice) -> String {
  case device {
    layout.BeamCpu -> "beam_cpu"
    layout.NativeCpu -> "native_cpu"
    layout.CudaDevice(index) -> "cuda:" <> int.to_string(index)
  }
}

pub fn storage_name(storage: layout.TensorStorage) -> String {
  case storage {
    layout.DenseStorage -> "dense"
    layout.StridedStorage -> "strided"
    layout.NativeStorage -> "native"
  }
}

pub fn memory_layout_name(memory_layout: layout.TensorMemoryLayout) -> String {
  case memory_layout {
    layout.RowMajor -> "row_major"
    layout.ColumnMajor -> "column_major"
    layout.StridedLayout -> "strided"
    layout.PackedFp8Layout -> "packed_fp8"
    layout.PackedSparse24Layout -> "packed_sparse24"
  }
}

pub fn spec_key(spec: TensorSpec) -> String {
  device_name(spec.device)
  <> "|"
  <> dtype_name(spec.dtype)
  <> "|"
  <> storage_name(spec.storage)
  <> "|"
  <> memory_layout_name(spec.memory_layout)
  <> "|"
  <> shape_key(spec.shape)
}

fn shape_key(shape: List(Int)) -> String {
  case shape {
    [] -> "scalar"
    [dim] -> int.to_string(dim)
    [dim, ..rest] -> int.to_string(dim) <> "x" <> shape_key(rest)
  }
}
