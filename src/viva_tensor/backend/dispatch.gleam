//// Internal backend dispatch planning.

import gleam/list
import gleam/result
import viva_tensor/native/tflops

pub type BackendSet(backend) {
  BackendSet(
    pure_gleam: backend,
    zig_simd: backend,
    mkl: backend,
    cuda_fp32: backend,
    cuda_fp16: backend,
    cuda_int8: backend,
    cuda_sparse: backend,
  )
}

pub type OperationKind {
  Elementwise
  Broadcast
  Reduction
  Softmax
  Matmul(m: Int, n: Int, k: Int)
}

pub type Capability(backend, device, dtype, operation) {
  Capability(
    backend: backend,
    available: Bool,
    device: device,
    dtypes: List(dtype),
    operations: List(operation),
    reason: String,
  )
}

pub type Rejection(backend) {
  Rejection(backend: backend, reason: String)
}

pub type Plan(operation, backend) {
  Plan(
    operation: operation,
    selected: backend,
    fallbacks: List(backend),
    rejected: List(Rejection(backend)),
    reason: String,
  )
}

pub fn capabilities(
  backends: BackendSet(backend),
  beam_cpu: device,
  native_cpu: device,
  cuda: device,
  float64: dtype,
  float32: dtype,
  float16: dtype,
  int8: dtype,
  sparse_float16: dtype,
  elementwise: operation,
  broadcast: operation,
  reduction: operation,
  softmax: operation,
  matmul: operation,
  zig_loaded: Bool,
  detected: List(tflops.Backend),
) -> List(Capability(backend, device, dtype, operation)) {
  [
    Capability(
      backend: backends.pure_gleam,
      available: True,
      device: beam_cpu,
      dtypes: [float64],
      operations: [elementwise, broadcast, reduction, softmax, matmul],
      reason: "Always available fallback.",
    ),
    Capability(
      backend: backends.zig_simd,
      available: zig_loaded,
      device: native_cpu,
      dtypes: [float64],
      operations: [elementwise, reduction, matmul],
      reason: "Portable SIMD NIF for CPU hot paths.",
    ),
    Capability(
      backend: backends.mkl,
      available: zig_loaded,
      device: native_cpu,
      dtypes: [float64, float32],
      operations: [matmul],
      reason: "Native BLAS path exposed through the loaded Zig NIF.",
    ),
    Capability(
      backend: backends.cuda_fp32,
      available: list.contains(detected, tflops.CudaFP32),
      device: cuda,
      dtypes: [float32],
      operations: [matmul],
      reason: "CUDA FP32/cuBLAS dense matrix multiplication.",
    ),
    Capability(
      backend: backends.cuda_fp16,
      available: list.contains(detected, tflops.CudaFP16),
      device: cuda,
      dtypes: [float16],
      operations: [matmul],
      reason: "CUDA FP16 Tensor Core dense matrix multiplication.",
    ),
    Capability(
      backend: backends.cuda_int8,
      available: list.contains(detected, tflops.CudaINT8),
      device: cuda,
      dtypes: [int8],
      operations: [matmul],
      reason: "CUDA INT8 IMMA Tensor Core matrix multiplication.",
    ),
    Capability(
      backend: backends.cuda_sparse,
      available: list.contains(detected, tflops.CudaSparse),
      device: cuda,
      dtypes: [sparse_float16],
      operations: [matmul],
      reason: "CUDA 2:4 sparse Tensor Core matrix multiplication.",
    ),
  ]
}

pub fn available_backends(
  capabilities: List(Capability(backend, device, dtype, operation)),
) -> List(backend) {
  capabilities
  |> list.filter(fn(capability) { capability.available })
  |> list.map(fn(capability) { capability.backend })
}

pub fn is_available(
  backend: backend,
  capabilities: List(Capability(backend, device, dtype, operation)),
) -> Bool {
  capabilities
  |> list.any(fn(capability) {
    capability.backend == backend && capability.available
  })
}

pub fn plan_backend(
  operation: operation,
  kind: OperationKind,
  available: List(backend),
  backends: BackendSet(backend),
  nif_loaded: Bool,
) -> Plan(operation, backend) {
  case kind {
    Matmul(m, n, k) ->
      plan_matmul(operation, m, n, k, available, backends, nif_loaded)
    Elementwise ->
      plan_first_available(
        operation,
        kind,
        available,
        backends,
        [backends.zig_simd, backends.mkl, backends.pure_gleam],
        "Element-wise ops prefer SIMD, then native CPU, then pure Gleam.",
        "Backend does not support stable element-wise dispatch.",
      )
    Broadcast ->
      plan_first_available(
        operation,
        kind,
        available,
        backends,
        [backends.zig_simd, backends.pure_gleam],
        "Broadcasting preserves views and only needs native compute when materialized.",
        "Backend does not support stable broadcast dispatch.",
      )
    Reduction ->
      plan_first_available(
        operation,
        kind,
        available,
        backends,
        [backends.zig_simd, backends.mkl, backends.pure_gleam],
        "Reductions prefer SIMD/native CPU and fall back to pure Gleam.",
        "Backend does not support stable reduction dispatch.",
      )
    Softmax ->
      plan_first_available(
        operation,
        kind,
        available,
        backends,
        [backends.pure_gleam],
        "Softmax currently uses the stable Gleam implementation.",
        "Softmax currently only has stable pure Gleam dispatch.",
      )
  }
}

fn plan_matmul(
  operation: operation,
  m: Int,
  n: Int,
  k: Int,
  available: List(backend),
  backends: BackendSet(backend),
  nif_loaded: Bool,
) -> Plan(operation, backend) {
  let tensor_core_aligned = m % 16 == 0 && n % 16 == 0 && k % 16 == 0
  let candidates = case tensor_core_aligned {
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
  let reason = case nif_loaded {
    True ->
      case tensor_core_aligned {
        True -> "Matmul dimensions are Tensor Core aligned; CUDA is preferred."
        False ->
          "Matmul dimensions are not Tensor Core aligned; dense CUDA/CPU fallback is preferred."
      }
    False -> "Native NIF is not loaded; pure Gleam fallback is selected."
  }

  plan_first_available(
    operation,
    Matmul(m, n, k),
    available,
    backends,
    candidates,
    reason,
    "Backend is not part of the stable matmul dispatch path for this shape.",
  )
}

fn plan_first_available(
  operation: operation,
  kind: OperationKind,
  available: List(backend),
  backends: BackendSet(backend),
  candidates: List(backend),
  reason: String,
  unsupported_reason: String,
) -> Plan(operation, backend) {
  let selected = select_backend(available, candidates, backends.pure_gleam)

  Plan(
    operation: operation,
    selected: selected,
    fallbacks: candidates,
    rejected: backend_rejections(
      kind,
      selected,
      available,
      candidates,
      backends,
      unsupported_reason,
    ),
    reason: reason,
  )
}

fn select_backend(
  available: List(backend),
  candidates: List(backend),
  fallback: backend,
) -> backend {
  candidates
  |> list.find(fn(candidate) { list.contains(available, candidate) })
  |> result.unwrap(fallback)
}

fn backend_rejections(
  kind: OperationKind,
  selected: backend,
  available: List(backend),
  candidates: List(backend),
  backends: BackendSet(backend),
  unsupported_reason: String,
) -> List(Rejection(backend)) {
  all_backends(backends)
  |> list.filter(fn(backend) { backend != selected })
  |> list.map(fn(backend) {
    Rejection(
      backend: backend,
      reason: rejection_reason(
        kind,
        backend,
        available,
        candidates,
        backends,
        unsupported_reason,
      ),
    )
  })
}

fn rejection_reason(
  kind: OperationKind,
  backend: backend,
  available: List(backend),
  candidates: List(backend),
  backends: BackendSet(backend),
  unsupported_reason: String,
) -> String {
  case list.contains(candidates, backend), list.contains(available, backend) {
    False, _ ->
      operation_specific_rejection(kind, backend, backends, unsupported_reason)
    True, False -> "Backend is not available in this VM."
    True, True -> "A higher-priority backend was selected."
  }
}

fn operation_specific_rejection(
  kind: OperationKind,
  backend: backend,
  backends: BackendSet(backend),
  fallback_reason: String,
) -> String {
  case kind {
    Matmul(_, _, _) if backend == backends.cuda_sparse ->
      "Sparse Tensor Core dispatch requires an explicit sparse tensor."
    Matmul(_, _, _) if backend == backends.cuda_int8 ->
      "INT8 Tensor Core dispatch requires explicit quantized tensors."
    Matmul(m, n, k) if backend == backends.cuda_fp16 ->
      case m % 16 == 0 && n % 16 == 0 && k % 16 == 0 {
        True -> fallback_reason
        False -> "FP16 Tensor Core matmul requires dimensions aligned to 16."
      }
    _ -> fallback_reason
  }
}

fn all_backends(backends: BackendSet(backend)) -> List(backend) {
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
