/**
 * nif_entry.c - Erlang NIF entry point for viva_tensor
 *
 * Contains only: nif_load, nif_funcs[], ERL_NIF_INIT
 *
 * All NIF implementations are in separate files:
 *   - nif_platform.c     — BLAS detection, CPU topology
 *   - nif_tensor_core.c  — NativeTensor + Quant resources, helpers, constructors, accessors
 *   - nif_cpu_ops.c      — Element-wise, reductions, matmul, activations, in-place, fused
 *   - nif_specialized.c  — Resonance/LNS, Horde physics, HDC
 *   - nif_cuda_tensors.c — CudaTensor (FP32), CudaTensor16 (FP16), CudaInt8Tensor
 *   - nif_sparse_quant.c — Sparse, INT8 sparse, quantization, SageAttention
 *
 * Shared types, macros, and declarations are in viva_nif.h.
 */

#include "viva_nif.h"

/* =========================================================================
 * NIF Init
 * ========================================================================= */

static int nif_load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
  (void)priv;
  (void)info;

  /* Detect CPU topology once at NIF load (MKL-style runtime init) */
  detect_cpu_topology();

#ifdef _WIN32
  /* Windows: Configure MKL threads for maximum performance */
  {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int ncpus = sysinfo.dwNumberOfProcessors;
    mkl_set_num_threads(ncpus > 0 ? ncpus : 16);
    fprintf(stderr, "[viva_tensor] Intel MKL (Windows), %d threads\n", ncpus > 0 ? ncpus : 16);
  }
#elif !defined(USE_MKL_DIRECT)
  /* Linux without direct MKL: detect best BLAS backend dynamically */
  detect_blas_backend();

  /* Auto-tune thread count based on matrix size heuristics */
  if (g_set_threads && g_cpu_info.optimal_threads > 0) {
    blas_set_threads(g_cpu_info.optimal_threads);
  }
#else
  /* Linux with MKL directly linked — maximum performance tuning */
  {
    int phys = g_cpu_info.physical_cores;
    int logical = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = phys > 0 ? phys : (logical > 0 ? logical : 16);

    /* 1. Set thread count to physical cores (HT hurts BLAS) */
    mkl_set_num_threads(threads);

    /* 2. Disable MKL_DYNAMIC — don't let MKL reduce thread count */
    mkl_set_dynamic(0);

    /* 3. Set env for thread affinity if not already set.
     *    compact = pack threads on same socket, reduces cross-NUMA traffic.
     *    granularity=fine = bind to logical CPU, no migration. */
    if (!getenv("KMP_AFFINITY")) {
      setenv("KMP_AFFINITY", "granularity=fine,compact,1,0", 0);
    }

    /* 4. Flush denormals to zero (DAZ+FTZ) — avoids 100x penalty on subnormals.
     *    Standard practice for BLAS/ML workloads. */
    #if defined(__x86_64__) || defined(_M_X64)
    {
      unsigned int mxcsr = __builtin_ia32_stmxcsr();
      mxcsr |= (1 << 6)  /* DAZ - Denormals Are Zero */
             | (1 << 15); /* FTZ - Flush To Zero */
      __builtin_ia32_ldmxcsr(mxcsr);
    }
    #endif

    fprintf(stderr, "[viva_tensor] Intel MKL direct, %d threads (%d physical cores), compact affinity, DAZ+FTZ\n", threads, phys);
  }
#endif

  TENSOR_RESOURCE = enif_open_resource_type(
      env, NULL, "NativeTensor", tensor_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!TENSOR_RESOURCE)
    return -1;

  LNS_RESOURCE = enif_open_resource_type(env, NULL, "LnsTensor", lns_destructor,
                                         ERL_NIF_RT_CREATE, NULL);
  if (!LNS_RESOURCE)
    return -1;

  HORDE_RESOURCE = enif_open_resource_type(env, NULL, "Horde", horde_destructor,
                                           ERL_NIF_RT_CREATE, NULL);
  if (!HORDE_RESOURCE)
    return -1;

  /* QuantInt8Tensor — INT8 quantized (4x compression) */
  QINT8_RESOURCE = enif_open_resource_type(env, NULL, "QuantInt8Tensor",
                                            qint8_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!QINT8_RESOURCE)
    return -1;

  /* QuantNF4Tensor — NF4 quantized (8x compression) */
  QNF4_RESOURCE = enif_open_resource_type(env, NULL, "QuantNF4Tensor",
                                           qnf4_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!QNF4_RESOURCE)
    return -1;

  HDC_RESOURCE = enif_open_resource_type(env, NULL, "HdcVector", hdc_destructor,
                                         ERL_NIF_RT_CREATE, NULL);
  if (!HDC_RESOURCE)
    return -1;

#ifndef _WIN32
  /* CudaTensor — persistent FP32 GPU memory */
  CUDA_TENSOR_RESOURCE = enif_open_resource_type(
      env, NULL, "CudaTensor", cuda_tensor_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!CUDA_TENSOR_RESOURCE)
    return -1;

  /* CudaTensor16 — persistent FP16 GPU memory (Tensor Cores) */
  CUDA_TENSOR16_RESOURCE = enif_open_resource_type(
      env, NULL, "CudaTensor16", cuda_tensor16_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!CUDA_TENSOR16_RESOURCE)
    return -1;

  /* CudaInt8Tensor — persistent INT8 GPU memory (IMMA Tensor Cores) */
  CUDA_INT8_TENSOR_RESOURCE = enif_open_resource_type(
      env, NULL, "CudaInt8Tensor", cuda_int8_tensor_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!CUDA_INT8_TENSOR_RESOURCE)
    return -1;

  /* SparseTensor — 2:4 structured sparsity (cuSPARSELt) */
  SPARSE_TENSOR_RESOURCE = enif_open_resource_type(
      env, NULL, "SparseTensor", sparse_tensor_destructor, ERL_NIF_RT_CREATE, NULL);
  if (!SPARSE_TENSOR_RESOURCE)
    return -1;
#endif

  return 0;
}

static ErlNifFunc nif_funcs[] = {
    /* Legacy list-based API */
    {"nif_simd_dot", 2, nif_simd_dot, 0},
    {"nif_simd_sum", 1, nif_simd_sum, 0},
    {"nif_simd_scale", 2, nif_simd_scale, 0},
    {"nif_simd_add", 2, nif_simd_add, 0},
    {"nif_simd_mul", 2, nif_simd_mul, 0},
    {"nif_simd_matmul", 5, nif_simd_matmul, 0},
    {"nif_simd_available", 0, nif_simd_available, 0},
    {"nif_backend_info", 0, nif_backend_info, 0},
    {"cpu_topology", 0, nif_cpu_topology, 0},

    /* NIF Resource API — constructors */
    {"nt_zeros", 1, nt_zeros, 0},
    {"nt_ones", 1, nt_ones, 0},
    {"nt_fill", 2, nt_fill, 0},
    {"nt_from_list", 2, nt_from_list, 0},

    /* NIF Resource API — accessors */
    {"nt_to_list", 1, nt_to_list, 0},
    {"nt_shape", 1, nt_shape, 0},
    {"nt_size", 1, nt_size, 0},

    /* NIF Resource API — element-wise ops */
    {"nt_add", 2, nt_add, 0},
    {"nt_sub", 2, nt_sub, 0},
    {"nt_mul", 2, nt_mul, 0},
    {"nt_scale", 2, nt_scale, 0},
    {"nt_negate", 1, nt_negate, 0},

    /* NIF Resource API — reductions */
    {"nt_dot", 2, nt_dot, 0},
    {"nt_sum", 1, nt_sum, 0},
    {"nt_max", 1, nt_max, 0},
    {"nt_min", 1, nt_min, 0},

    /* NIF Resource API — matrix ops */
    {"nt_matmul", 5, nt_matmul_blas, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_blas", 5, nt_matmul_blas, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_inplace", 6, nt_matmul_inplace, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_cuda", 5, nt_matmul_cuda, ERL_NIF_DIRTY_JOB_CPU_BOUND},
#ifndef _WIN32
    {"nt_matmul_cuda_fp32", 5, nt_matmul_cuda_fp32, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_int8_tc", 5, nt_matmul_int8_tc, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_int8_tc_available", 0, nt_int8_tc_available, 0},
    {"nt_matmul_fp16_tc", 5, nt_matmul_fp16_tc, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_fp16_tc_available", 0, nt_fp16_tc_available, 0},
    {"nt_matmul_int8_lt", 5, nt_matmul_int8_lt, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_int8_lt_available", 0, nt_int8_lt_available, 0},
#endif
    {"nt_transpose", 1, nt_transpose, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* NIF Resource API — activations */
    {"nt_relu", 1, nt_relu, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_sigmoid", 1, nt_sigmoid, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_exp", 1, nt_exp_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_log", 1, nt_log_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* In-place mutation */
    {"nt_add_mut", 2, nt_add_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_scale_mut", 2, nt_scale_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_negate_mut", 1, nt_negate_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_relu_mut", 1, nt_relu_mut, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Retro / fused kernels */
    {"nt_saturn_blend", 3, nt_saturn_blend, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_fused_linear_relu", 6, nt_fused_linear_relu_nif,
     ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Resonance kernels — LNS f64 */
    {"nt_resonance_mul", 2, nt_resonance_mul, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_resonance_power", 2, nt_resonance_power, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* LNS (True Log-Number System) — f32 via IADD, 8x throughput */
    {"lns_from_f64", 1, lns_from_f64, 0},
    {"lns_to_f64", 1, lns_to_f64, 0},
    {"lns_mul", 2, lns_mul, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_mul_corrected", 2, lns_mul_corrected, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_div", 2, lns_div, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_sqrt", 1, lns_sqrt, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"lns_rsqrt", 1, lns_rsqrt, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Horde — SoA Physics, 10K+ entities at 60fps */
    {"horde_create", 2, horde_create, 0},
    {"horde_set_positions", 2, horde_set_positions, 0},
    {"horde_set_velocities", 2, horde_set_velocities, 0},
    {"horde_integrate", 2, horde_integrate_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_dampen", 2, horde_dampen_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_wrap", 2, horde_wrap_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"horde_get_positions", 1, horde_get_positions, 0},
    {"horde_get_velocities", 1, horde_get_velocities, 0},
    {"horde_count", 1, horde_count_nif, 0},
    {"horde_kinetic_energy", 1, horde_kinetic_energy_nif, 0},

    /* HDC — Hyperdimensional Computing, one-shot learning */
    {"hdc_create", 1, hdc_create_nif, 0},
    {"hdc_random", 2, hdc_random_nif, 0},
    {"hdc_bind", 2, hdc_bind_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_similarity", 2, hdc_similarity_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_permute", 2, hdc_permute_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"hdc_dim", 1, hdc_dim_nif, 0},

#ifndef _WIN32
    /* CudaTensor (FP32 GPU) */
    {"ct_from_list", 2, ct_from_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct_to_list", 1, ct_to_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct_shape", 1, ct_shape, 0},
    {"ct_matmul", 5, ct_matmul, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct_matmul_inplace", 6, ct_matmul_inplace_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},

    /* CudaTensor16 (FP16 GPU) */
    {"ct16_from_list", 2, ct16_from_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct16_to_list", 1, ct16_to_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct16_shape", 1, ct16_shape, 0},
    {"ct16_matmul", 5, ct16_matmul, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_inplace", 6, ct16_matmul_inplace_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_bench", 7, ct16_matmul_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct_int8_matmul_bench", 7, ct_int8_matmul_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_lt_32f_bench", 7, ct16_matmul_lt_32f_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_relu", 6, ct16_matmul_fused_relu_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_gelu", 6, ct16_matmul_fused_gelu_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_relu_bench", 7, ct16_matmul_fused_relu_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_gelu_bench", 7, ct16_matmul_fused_gelu_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_batched_bench", 5, ct16_matmul_batched_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"fp8_matmul_lt_tn_bench", 4, fp8_matmul_lt_tn_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cutlass_fp8_f16acc_bench", 4, cutlass_fp8_f16acc_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cutlass_fp8_f32acc_bench", 4, cutlass_fp8_f32acc_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cutlass_int8_sparse_bench", 5, cutlass_int8_sparse_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cutlass_int8_sparse_bench_ex", 6, cutlass_int8_sparse_bench_ex_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cusparselt_int8_sparse_bench", 5, cusparselt_int8_sparse_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cusparselt_fp8_sparse_bench", 4, cusparselt_fp8_sparse_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cusparselt_fp16_sparse_bench", 4, cusparselt_fp16_sparse_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"cutlass_int4_sparse_bench", 6, cutlass_int4_sparse_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_relu_tn_bench", 7, ct16_matmul_fused_relu_tn_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_fused_gelu_tn_bench", 7, ct16_matmul_fused_gelu_tn_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_available", 0, ct16_available, 0},

    /* Async CUDA */
    {"cuda_sync", 0, nif_cuda_sync, 0},
    {"ct_matmul_async", 5, ct_matmul_async, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct16_matmul_async", 5, ct16_matmul_async, ERL_NIF_DIRTY_JOB_IO_BOUND},

    /* CudaInt8Tensor (INT8 GPU) */
    {"ct_int8_available", 0, ct_int8_available, 0},
    {"ct_int8_from_list", 2, ct_int8_from_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct_int8_to_list", 1, ct_int8_to_list, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"ct_int8_shape", 1, ct_int8_shape, 0},
    {"ct_int8_matmul", 5, ct_int8_matmul, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct_int8_matmul_inplace", 6, ct_int8_matmul_inplace_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"ct_int8_matmul_async", 5, ct_int8_matmul_async, ERL_NIF_DIRTY_JOB_IO_BOUND},

    /* SparseTensor — 2:4 Sparsity with cuSPARSELt */
    {"sparse_from_ct16", 1, sparse_from_ct16, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"sparse_shape", 1, sparse_shape, 0},
    {"sparse_compression_ratio", 1, sparse_compression_ratio, 0},
    {"sparse_matmul", 5, sparse_matmul_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_matmul_inplace", 6, sparse_matmul_inplace_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_matmul_bench", 7, sparse_matmul_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_matmul_bench_tn", 7, sparse_matmul_bench_tn_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_available", 0, sparse_available, 0},

    /* SparseTensor INT8 (2:4 cuSPARSELt) */
    {"sparse_from_int8", 1, sparse_from_int8, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"sparse_matmul_int8", 5, sparse_matmul_int8_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_matmul_int8_inplace", 6, sparse_matmul_int8_inplace_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"sparse_matmul_int8_bench", 7, sparse_matmul_int8_bench_nif, ERL_NIF_DIRTY_JOB_IO_BOUND},

    /* SageAttention - INT8 QK^T + FP8 */
    {"sage_available", 0, nif_sage_available, 0},
    {"sage_fp8_available", 0, nif_sage_fp8_available, 0},
    {"sage_quant_int8", 2, nif_sage_quant_int8, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"sage_softmax", 2, nif_sage_softmax, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"sage_attention", 8, nif_sage_attention, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"sage_attention_ct", 8, sage_attention_ct, ERL_NIF_DIRTY_JOB_IO_BOUND},
#endif

    /* Fused Quantized Matmul */
    {"nt_matmul_int8", 6, nt_matmul_int8, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_nf4", 7, nt_matmul_nf4, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_quantize_int8", 1, nt_quantize_int8, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    /* Resource-based quantized tensors */
    {"nt_to_qint8", 1, nt_to_qint8, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_qint8", 5, nt_matmul_qint8, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_to_qnf4", 2, nt_to_qnf4, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"nt_matmul_qnf4", 5, nt_matmul_qnf4, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"qint8_scale", 1, qint8_scale, 0},
    {"qint8_shape", 1, qint8_shape, 0},
    {"qnf4_info", 1, qnf4_info, 0},
};

ERL_NIF_INIT(viva_tensor_zig, nif_funcs, nif_load, NULL, NULL, NULL)
