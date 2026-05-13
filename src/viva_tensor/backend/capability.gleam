//// Hardware capability profiles for current and future accelerator targets.
////
//// These records describe what the planner can reason about without implying
//// that an experimental target is available on the current VM.

import gleam/list
import viva_tensor/native/tflops

pub type HardwareGeneration {
  GenerationBeamCpu
  GenerationNativeCpu
  GenerationAda
  GenerationBlackwell
  GenerationRubin
  GenerationVera
  GenerationRubinCpx
}

pub type HardwareFeature {
  FeatureCuda
  FeatureTensorCores
  FeatureSparseTensorCores
  FeatureInt8Imma
  FeatureNvfp4
  FeatureBlockScaledMma
  FeatureUnifiedCpuGpuMemory
  FeatureContextPaging
  FeatureHadamardPreprocess
  FeatureExperimentalInt2
}

pub type HardwareProfile {
  HardwareProfile(
    name: String,
    generation: HardwareGeneration,
    available: Bool,
    memory_bytes: Int,
    memory_bandwidth_gbps: Int,
    fp4_pflops: Int,
    nvlink_c2c_gbps: Int,
    preferred_micro_block: Int,
    features: List(HardwareFeature),
    reason: String,
  )
}

pub fn hardware_profiles(
  zig_loaded: Bool,
  backends: List(tflops.Backend),
) -> List(HardwareProfile) {
  let cuda_available =
    list.any(backends, fn(backend) {
      backend == tflops.CudaFP32
      || backend == tflops.CudaFP16
      || backend == tflops.CudaINT8
      || backend == tflops.CudaSparse
    })

  [
    HardwareProfile(
      name: "BEAM CPU",
      generation: GenerationBeamCpu,
      available: True,
      memory_bytes: 0,
      memory_bandwidth_gbps: 0,
      fp4_pflops: 0,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 1,
      features: [],
      reason: "Portable pure Gleam fallback.",
    ),
    HardwareProfile(
      name: "Native CPU NIF",
      generation: GenerationNativeCpu,
      available: zig_loaded,
      memory_bytes: 0,
      memory_bandwidth_gbps: 0,
      fp4_pflops: 0,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 1,
      features: [FeatureHadamardPreprocess],
      reason: "Loaded NIF path for CPU SIMD, MKL, and future native preprocessing.",
    ),
    HardwareProfile(
      name: "RTX 4090 / Ada",
      generation: GenerationAda,
      available: cuda_available,
      memory_bytes: 24_000_000_000,
      memory_bandwidth_gbps: 1008,
      fp4_pflops: 0,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 16,
      features: [FeatureCuda, FeatureTensorCores, FeatureInt8Imma],
      reason: "Current local CUDA development target.",
    ),
    HardwareProfile(
      name: "Blackwell B200",
      generation: GenerationBlackwell,
      available: False,
      memory_bytes: 192_000_000_000,
      memory_bandwidth_gbps: 8000,
      fp4_pflops: 10,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 16,
      features: [
        FeatureCuda,
        FeatureTensorCores,
        FeatureSparseTensorCores,
        FeatureNvfp4,
        FeatureBlockScaledMma,
      ],
      reason: "Known target profile; no Blackwell-specific runtime detection is wired yet.",
    ),
    HardwareProfile(
      name: "Rubin R100",
      generation: GenerationRubin,
      available: False,
      memory_bytes: 288_000_000_000,
      memory_bandwidth_gbps: 22_000,
      fp4_pflops: 50,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 16,
      features: [
        FeatureCuda,
        FeatureTensorCores,
        FeatureSparseTensorCores,
        FeatureNvfp4,
        FeatureBlockScaledMma,
        FeatureHadamardPreprocess,
        FeatureExperimentalInt2,
      ],
      reason: "Future target profile; kept unavailable until the runtime can detect Rubin hardware/toolchains.",
    ),
    HardwareProfile(
      name: "Vera CPU",
      generation: GenerationVera,
      available: False,
      memory_bytes: 1_500_000_000_000,
      memory_bandwidth_gbps: 1200,
      fp4_pflops: 0,
      nvlink_c2c_gbps: 1800,
      preferred_micro_block: 16,
      features: [FeatureUnifiedCpuGpuMemory],
      reason: "Future coherent CPU memory target for Vera Rubin systems.",
    ),
    HardwareProfile(
      name: "Rubin CPX",
      generation: GenerationRubinCpx,
      available: False,
      memory_bytes: 0,
      memory_bandwidth_gbps: 0,
      fp4_pflops: 0,
      nvlink_c2c_gbps: 0,
      preferred_micro_block: 16,
      features: [FeatureContextPaging, FeatureCuda],
      reason: "Future long-context inference target; no stable dispatch path yet.",
    ),
  ]
}
