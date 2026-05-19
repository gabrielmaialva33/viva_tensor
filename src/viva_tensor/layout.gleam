//// Tensor layout metadata.
////
//// A tensor is not just its values. To interpret tensor storage correctly, every
//// backend needs the same small contract: shape, strides, offset, dtype, device,
//// and storage representation.

/// How the tensor's payload is represented.
pub type TensorStorage {
  DenseStorage
  StridedStorage
  NativeStorage
}

/// Where the tensor payload lives.
pub type TensorDevice {
  BeamCpu
  NativeCpu
  CudaDevice(Int)
}

/// Element type for this tensor value.
pub type TensorDtype {
  Float64
  Float32
  Float16
  BFloat16
  Float8E4M3
  Int8
  Int4
  SparseFloat16
}

/// Logical memory layout used by runtime planning.
pub type TensorMemoryLayout {
  RowMajor
  ColumnMajor
  StridedLayout
  PackedFp8Layout
  PackedSparse24Layout
}

/// Canonical metadata for interpreting tensor storage.
pub type TensorLayout {
  TensorLayout(
    storage: TensorStorage,
    device: TensorDevice,
    dtype: TensorDtype,
    shape: List(Int),
    strides: List(Int),
    offset: Int,
    size: Int,
    rank: Int,
    contiguous: Bool,
  )
}
