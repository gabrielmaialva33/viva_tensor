//// Runtime planning primitives.

import gleam/int
import gleam/list
import gleam/result
import viva_tensor/layout
import viva_tensor/spec as tensor_spec

pub type RuntimeOp {
  RuntimeElementwise
  RuntimeBroadcast
  RuntimeReduction
  RuntimeSoftmax
  RuntimeMatmul(m: Int, n: Int, k: Int)
  RuntimeLinear(batch: Int, in_features: Int, out_features: Int)
}

pub type RuntimeBackendSet(backend) {
  RuntimeBackendSet(
    pure_gleam: backend,
    zig_simd: backend,
    mkl: backend,
    cuda_fp32: backend,
    cuda_fp16: backend,
    cuda_int8: backend,
    cuda_sparse: backend,
  )
}

pub type RuntimeRejection(backend) {
  RuntimeRejection(backend: backend, reason: String)
}

pub type RuntimePlan(backend) {
  RuntimePlan(
    spec: tensor_spec.TensorSpec,
    operation: RuntimeOp,
    selected: backend,
    fallbacks: List(backend),
    rejected: List(RuntimeRejection(backend)),
    reason: String,
    cache_key: String,
  )
}

pub fn plan_runtime(
  spec: tensor_spec.TensorSpec,
  operation: RuntimeOp,
  available: List(backend),
  backends: RuntimeBackendSet(backend),
) -> RuntimePlan(backend) {
  let candidates = candidates_for(spec, operation, backends)
  let selected =
    candidates
    |> list.find(fn(candidate) { list.contains(available, candidate) })
    |> result.unwrap(backends.pure_gleam)

  RuntimePlan(
    spec: spec,
    operation: operation,
    selected: selected,
    fallbacks: candidates,
    rejected: rejected_backends(selected, available, candidates, backends),
    reason: reason_for(spec, operation),
    cache_key: cache_key_for(spec, operation),
  )
}

pub fn cache_key(plan: RuntimePlan(backend)) -> String {
  plan.cache_key
}

pub fn cache_key_for(
  spec: tensor_spec.TensorSpec,
  operation: RuntimeOp,
) -> String {
  tensor_spec.spec_key(spec) <> "|" <> op_key(operation)
}

fn candidates_for(
  spec: tensor_spec.TensorSpec,
  operation: RuntimeOp,
  backends: RuntimeBackendSet(backend),
) -> List(backend) {
  case operation {
    RuntimeLinear(_, in_features, out_features) ->
      linear_candidates(spec, in_features, out_features, backends)
    RuntimeMatmul(m, n, k) -> matmul_candidates(m, n, k, backends)
    RuntimeElementwise -> [backends.zig_simd, backends.mkl, backends.pure_gleam]
    RuntimeBroadcast -> [backends.zig_simd, backends.pure_gleam]
    RuntimeReduction -> [backends.zig_simd, backends.mkl, backends.pure_gleam]
    RuntimeSoftmax -> [backends.pure_gleam]
  }
}

fn linear_candidates(
  spec: tensor_spec.TensorSpec,
  in_features: Int,
  out_features: Int,
  backends: RuntimeBackendSet(backend),
) -> List(backend) {
  let tensor_core_aligned = in_features % 16 == 0 && out_features % 16 == 0
  case spec.dtype, spec.memory_layout, tensor_core_aligned {
    layout.Int4, layout.PackedSparse24Layout, True -> [
      backends.cuda_sparse,
      backends.cuda_int8,
      backends.cuda_fp16,
      backends.mkl,
      backends.pure_gleam,
    ]
    layout.Int8, layout.PackedSparse24Layout, True -> [
      backends.cuda_int8,
      backends.cuda_sparse,
      backends.cuda_fp16,
      backends.mkl,
      backends.pure_gleam,
    ]
    layout.Float8E4M3, layout.PackedFp8Layout, True -> [
      backends.cuda_fp16,
      backends.cuda_fp32,
      backends.mkl,
      backends.pure_gleam,
    ]
    layout.Float16, _, True -> [
      backends.cuda_fp16,
      backends.cuda_fp32,
      backends.mkl,
      backends.pure_gleam,
    ]
    _, _, _ -> [
      backends.cuda_fp32,
      backends.mkl,
      backends.zig_simd,
      backends.pure_gleam,
    ]
  }
}

fn matmul_candidates(
  m: Int,
  n: Int,
  k: Int,
  backends: RuntimeBackendSet(backend),
) -> List(backend) {
  case m % 16 == 0 && n % 16 == 0 && k % 16 == 0 {
    True -> [
      backends.cuda_sparse,
      backends.cuda_fp16,
      backends.cuda_int8,
      backends.cuda_fp32,
      backends.mkl,
      backends.zig_simd,
      backends.pure_gleam,
    ]
    False -> [
      backends.cuda_fp32,
      backends.mkl,
      backends.zig_simd,
      backends.pure_gleam,
    ]
  }
}

fn rejected_backends(
  selected: backend,
  available: List(backend),
  candidates: List(backend),
  backends: RuntimeBackendSet(backend),
) -> List(RuntimeRejection(backend)) {
  all_backends(backends)
  |> list.filter(fn(backend) { backend != selected })
  |> list.map(fn(backend) {
    RuntimeRejection(
      backend: backend,
      reason: case
        list.contains(candidates, backend),
        list.contains(available, backend)
      {
        False, _ -> "Backend is not part of the runtime plan for this spec."
        True, False -> "Backend is not available in this VM."
        True, True -> "A higher-priority backend was selected."
      },
    )
  })
}

fn all_backends(backends: RuntimeBackendSet(backend)) -> List(backend) {
  [
    backends.cuda_sparse,
    backends.cuda_fp16,
    backends.cuda_int8,
    backends.cuda_fp32,
    backends.mkl,
    backends.zig_simd,
    backends.pure_gleam,
  ]
}

fn reason_for(spec: tensor_spec.TensorSpec, operation: RuntimeOp) -> String {
  case operation, spec.dtype, spec.memory_layout {
    RuntimeLinear(_, _, _), layout.Int4, layout.PackedSparse24Layout ->
      "INT4 2:4 packed linear prefers the sparse Tensor Core path."
    RuntimeLinear(_, _, _), layout.Int8, layout.PackedSparse24Layout ->
      "INT8 2:4 packed linear prefers sparse INT8 Tensor Cores."
    RuntimeLinear(_, _, _), layout.Float8E4M3, layout.PackedFp8Layout ->
      "FP8 packed linear prefers the FP8/CUTLASS-capable CUDA path."
    RuntimeMatmul(_, _, _), _, _ ->
      "Matmul planning is shape-driven and prefers Tensor Core aligned CUDA."
    _, _, _ -> "Runtime planning selected the first available stable backend."
  }
}

fn op_key(operation: RuntimeOp) -> String {
  case operation {
    RuntimeElementwise -> "elementwise"
    RuntimeBroadcast -> "broadcast"
    RuntimeReduction -> "reduction"
    RuntimeSoftmax -> "softmax"
    RuntimeMatmul(m, n, k) ->
      "matmul:"
      <> int.to_string(m)
      <> "x"
      <> int.to_string(n)
      <> "x"
      <> int.to_string(k)
    RuntimeLinear(batch, in_features, out_features) ->
      "linear:"
      <> int.to_string(batch)
      <> "x"
      <> int.to_string(in_features)
      <> "x"
      <> int.to_string(out_features)
  }
}
