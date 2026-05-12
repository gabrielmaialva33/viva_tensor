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
}

/// Element type for this tensor value.
pub type TensorDtype {
  Float64
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
