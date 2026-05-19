import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as t

pub fn main() {
  gleeunit.main()
}

pub fn tensor_spec_from_tensor_preserves_existing_layout_test() {
  let tensor = t.ones([2, 3])
  let spec = t.tensor_spec(tensor)

  let _ = accept_spec(spec)
  spec.shape |> should.equal([2, 3])
  spec.dtype |> should.equal(t.Float64)
  spec.device |> should.equal(t.BeamCpu)
  spec.storage |> should.equal(t.DenseStorage)
  spec.memory_layout |> should.equal(t.RowMajor)
  t.spec_key(spec) |> should.equal("2x3:float64:beam_cpu:dense:row_major")
}

pub fn spec_from_parts_covers_runtime_metadata_variants_test() {
  let column_major =
    t.spec_from_parts(
      shape: [16, 128],
      dtype: t.Float16,
      device: t.CudaDevice(0),
      storage: t.DenseStorage,
      memory_layout: t.ColumnMajor,
    )
  let strided =
    t.spec_from_parts(
      shape: [2, 3],
      dtype: t.Float32,
      device: t.NativeCpu,
      storage: t.StridedStorage,
      memory_layout: t.StridedLayout,
    )
  let packed_fp8 =
    t.spec_from_parts(
      shape: [64, 128],
      dtype: t.Float8E4M3,
      device: t.CudaDevice(1),
      storage: t.NativeStorage,
      memory_layout: t.PackedFp8Layout,
    )
  let packed_sparse =
    t.spec_from_parts(
      shape: [64, 128],
      dtype: t.Int4,
      device: t.CudaDevice(1),
      storage: t.NativeStorage,
      memory_layout: t.PackedSparse24Layout,
    )

  column_major.memory_layout |> should.equal(t.ColumnMajor)
  strided.storage |> should.equal(t.StridedStorage)
  packed_fp8.dtype |> should.equal(t.Float8E4M3)
  packed_sparse.dtype |> should.equal(t.Int4)
  packed_sparse.memory_layout |> should.equal(t.PackedSparse24Layout)

  t.dtype_name(t.Float64) |> should.equal("float64")
  t.dtype_name(t.Float32) |> should.equal("float32")
  t.dtype_name(t.Float16) |> should.equal("float16")
  t.dtype_name(t.BFloat16) |> should.equal("bfloat16")
  t.dtype_name(t.Float8E4M3) |> should.equal("float8_e4m3")
  t.dtype_name(t.Int8) |> should.equal("int8")
  t.dtype_name(t.Int4) |> should.equal("int4")
  t.dtype_name(t.SparseFloat16) |> should.equal("sparse_float16")

  t.device_name(t.BeamCpu) |> should.equal("beam_cpu")
  t.device_name(t.NativeCpu) |> should.equal("native_cpu")
  t.device_name(t.CudaDevice(2)) |> should.equal("cuda:2")
}

pub fn spec_key_is_deterministic_and_layout_sensitive_test() {
  let row_major =
    t.spec_from_parts(
      shape: [4, 8],
      dtype: t.Float32,
      device: t.BeamCpu,
      storage: t.DenseStorage,
      memory_layout: t.RowMajor,
    )
  let column_major =
    t.spec_from_parts(
      shape: [4, 8],
      dtype: t.Float32,
      device: t.BeamCpu,
      storage: t.DenseStorage,
      memory_layout: t.ColumnMajor,
    )

  t.spec_key(row_major) |> should.equal("4x8:float32:beam_cpu:dense:row_major")
  t.spec_key(row_major)
  |> should.equal(t.spec_key(row_major))
  let same_spec_key = t.spec_key(row_major) == t.spec_key(column_major)
  same_spec_key |> should.be_false()
}

pub fn plan_runtime_supports_core_operation_variants_test() {
  let spec =
    t.spec_from_parts(
      shape: [32, 64],
      dtype: t.Float32,
      device: t.BeamCpu,
      storage: t.DenseStorage,
      memory_layout: t.RowMajor,
    )

  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeElementwise))
  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeBroadcast))
  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeReduction))
  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeSoftmax))
  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeMatmul(32, 16, 64)))
  assert_runtime_plan_contract(t.plan_runtime(spec, t.RuntimeLinear(8, 64, 16)))
}

pub fn runtime_cache_key_is_stable_for_same_spec_and_op_test() {
  let spec =
    t.spec_from_parts(
      shape: [8, 128],
      dtype: t.Float8E4M3,
      device: t.CudaDevice(0),
      storage: t.NativeStorage,
      memory_layout: t.PackedFp8Layout,
    )
  let op = t.RuntimeLinear(batch: 8, in_features: 128, out_features: 64)
  let first = t.plan_runtime(spec, op)
  let second = t.plan_runtime(spec, op)
  let matmul = t.plan_runtime(spec, t.RuntimeMatmul(m: 8, n: 64, k: 128))

  let _ = accept_op(op)
  t.cache_key(first) |> should.equal(first.cache_key)
  t.cache_key(first)
  |> should.equal(t.cache_key(second))
  let same_cache_key = t.cache_key(first) == t.cache_key(matmul)
  same_cache_key |> should.be_false()
  assert_runtime_plan_contract(first)
}

fn assert_runtime_plan_contract(plan: t.RuntimePlan) {
  list.any(plan.fallbacks, fn(backend) { backend == t.BackendPureGleam })
  |> should.be_true()
  list.contains(plan.fallbacks, plan.selected)
  |> should.be_true()
  let empty_reason = plan.reason == ""
  empty_reason |> should.be_false()
  let empty_cache_key = plan.cache_key == ""
  empty_cache_key |> should.be_false()
  t.cache_key(plan) |> should.equal(plan.cache_key)
  list.all(plan.rejected, fn(rejection) { rejection.reason != "" })
  |> should.be_true()
}

fn accept_spec(_spec: t.TensorSpec) {
  Nil
}

fn accept_op(_op: t.RuntimeOp) {
  Nil
}
