/**
 * viva_nif.h - Shared types, macros, and declarations for viva_tensor NIF
 *
 * All NIF source files include this header. It provides:
 *   - Common system includes
 *   - Aligned allocation macros (AVX-512 / hugepage / prefault)
 *   - Struct definitions for all resource types
 *   - Extern declarations for resource type globals
 *   - Helper function declarations (make_ok, make_error, etc.)
 *   - Zig SIMD extern declarations
 *   - CUDA extern declarations
 */

#ifndef VIVA_NIF_H
#define VIVA_NIF_H

/* _GNU_SOURCE must be defined BEFORE any includes for pthread_setaffinity_np */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#include "erl_nif.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <mkl.h>
#include <malloc.h>
#define BLAS_BACKEND_MKL 1
#else
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#ifdef USE_MKL_DIRECT
#include <mkl.h>
#endif
#endif

/* =========================================================================
 * Aligned Allocation (64-byte for AVX-512 / cache-line)
 * ========================================================================= */

#define TENSOR_ALIGN 64

#ifdef _WIN32
static inline void *aligned_tensor_alloc(size_t size) {
  return _aligned_malloc(size, TENSOR_ALIGN);
}
static inline void aligned_tensor_free(void *ptr) { _aligned_free(ptr); }
#else
#define HUGEPAGE_THRESHOLD (2 * 1024 * 1024)
#define PREFAULT_THRESHOLD (256 * 1024)
static inline void *aligned_tensor_alloc(size_t size) {
  size_t aligned_size = (size + TENSOR_ALIGN - 1) & ~(TENSOR_ALIGN - 1);
  void *ptr = aligned_alloc(TENSOR_ALIGN, aligned_size);
  if (ptr && aligned_size >= PREFAULT_THRESHOLD) {
    if (aligned_size >= HUGEPAGE_THRESHOLD) {
#ifdef MADV_HUGEPAGE
      madvise(ptr, aligned_size, MADV_HUGEPAGE);
#endif
    }
#ifdef MADV_POPULATE_WRITE
    madvise(ptr, aligned_size, MADV_POPULATE_WRITE);
#else
    memset(ptr, 0, aligned_size);
#endif
  }
  return ptr;
}
static inline void aligned_tensor_free(void *ptr) { free(ptr); }
#endif

/* =========================================================================
 * BLAS Backend Types (Linux dynamic loading)
 * ========================================================================= */

#ifndef _WIN32
typedef enum {
  BLAS_MKL = 1,
  BLAS_OPENBLAS_TUNED = 2,
  BLAS_OPENBLAS = 3,
  BLAS_ZIG_GEMM = 4
} BlasBackend;

typedef void (*dgemm_fn)(const int Order, const int TransA, const int TransB,
                         const int M, const int N, const int K,
                         const double alpha, const double *A, const int lda,
                         const double *B, const int ldb,
                         const double beta, double *C, const int ldc);
typedef void (*set_threads_fn)(int);

/* Defined in nif_platform.c */
extern BlasBackend g_blas_backend;
extern void *g_blas_handle;
extern dgemm_fn g_dgemm;
extern set_threads_fn g_set_threads;
extern const char *g_blas_name;
extern int g_blas_detected;
#endif

/* =========================================================================
 * CPU Topology
 * ========================================================================= */

typedef struct {
  int logical_cpus;
  int physical_cores;
  int sockets;
  int l1_cache_kb;
  int l2_cache_kb;
  int l3_cache_kb;
  int has_avx2;
  int has_avx512;
  int has_hybrid;
  int p_cores;
  int e_cores;
  int threads_per_core;
  int optimal_threads;
} CpuTopology;

/* Defined in nif_platform.c */
extern CpuTopology g_cpu_info;
extern int g_cpu_detected;

/* Platform detection functions (nif_platform.c) */
void detect_cpu_topology(void);
void detect_blas_backend(void);
void blas_dgemm(int M, int N, int K, double alpha,
                const double *A, int lda,
                const double *B, int ldb,
                double beta, double *C, int ldc);
void blas_set_threads(int n);

/* Exported to Zig (nif_platform.c) */
int vt_get_optimal_threads(void);
int vt_get_physical_cores(void);
int vt_get_logical_cpus(void);
int vt_get_l2_cache_kb(void);
int vt_get_l3_cache_kb(void);
int vt_is_hybrid_cpu(void);
int vt_has_avx512(void);

#ifdef _WIN32
int vt_set_thread_affinity(void* thread_handle, int core_id);
#else
int vt_set_thread_affinity_self(int core_id);
#endif

/* =========================================================================
 * NativeTensor - Core data structure
 * ========================================================================= */

typedef struct {
  double *data;
  int *shape;
  int *strides;
  int ndim;
  int size;
  int owns_data;
} NativeTensor;

extern ErlNifResourceType *TENSOR_RESOURCE;

/* NativeTensor lifecycle (nif_tensor_core.c) */
void tensor_destructor(ErlNifEnv *env, void *obj);
NativeTensor *alloc_tensor(int ndim, const int *shape);
NativeTensor *alloc_tensor_uninit(int ndim, const int *shape);
NativeTensor *get_tensor(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_tensor_term(ErlNifEnv *env, NativeTensor *t);

/* =========================================================================
 * QuantInt8Tensor - INT8 quantized (4x compression)
 * ========================================================================= */

typedef struct {
  int8_t *data;
  double scale;
  int *shape;
  int ndim;
  int size;
} QuantInt8Tensor;

extern ErlNifResourceType *QINT8_RESOURCE;

void qint8_destructor(ErlNifEnv *env, void *obj);
QuantInt8Tensor *get_qint8(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_qint8_term(ErlNifEnv *env, QuantInt8Tensor *t);

/* =========================================================================
 * QuantNF4Tensor - NF4 quantized (8x compression)
 * ========================================================================= */

typedef struct {
  uint8_t *indices;
  double *scales;
  int *shape;
  int ndim;
  int size;
  int block_size;
  int num_blocks;
  int packed_size;
} QuantNF4Tensor;

extern ErlNifResourceType *QNF4_RESOURCE;

void qnf4_destructor(ErlNifEnv *env, void *obj);
QuantNF4Tensor *get_qnf4(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_qnf4_term(ErlNifEnv *env, QuantNF4Tensor *t);

/* =========================================================================
 * LnsTensor - Log-Number System (f32 via IADD)
 * ========================================================================= */

typedef struct {
  float *data;
  int *shape;
  int ndim;
  int size;
} LnsTensor;

extern ErlNifResourceType *LNS_RESOURCE;

void lns_destructor(ErlNifEnv *env, void *obj);
LnsTensor *alloc_lns(int ndim, const int *shape);
LnsTensor *get_lns(ErlNifEnv *env, ERL_NIF_TERM term);

/* =========================================================================
 * Horde - SoA Physics Engine
 * ========================================================================= */

typedef struct {
  double *positions;
  double *velocities;
  double *accelerations;
  int entity_count;
  int dims;
} Horde;

extern ErlNifResourceType *HORDE_RESOURCE;

void horde_destructor(ErlNifEnv *env, void *obj);
Horde *get_horde(ErlNifEnv *env, ERL_NIF_TERM term);

/* =========================================================================
 * HdcVector - Hyperdimensional Computing
 * ========================================================================= */

typedef struct {
  uint64_t *data;
  int words;
  int dim;
} HdcVector;

extern ErlNifResourceType *HDC_RESOURCE;

void hdc_destructor(ErlNifEnv *env, void *obj);
HdcVector *get_hdc(ErlNifEnv *env, ERL_NIF_TERM term);

/* =========================================================================
 * GPU Tensor Types (Linux only)
 * ========================================================================= */

#ifndef _WIN32

typedef struct {
  float *d_data;
  int *shape;
  int ndim;
  int size;
} CudaTensor;

extern ErlNifResourceType *CUDA_TENSOR_RESOURCE;

void cuda_tensor_destructor(ErlNifEnv *env, void *obj);
CudaTensor *alloc_cuda_tensor(int ndim, const int *shape);
CudaTensor *get_cuda_tensor(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_cuda_tensor_term(ErlNifEnv *env, CudaTensor *t);

typedef struct {
  uint16_t *d_data;
  uint16_t *d_data_t;
  float *d_acc;
  int *shape;
  int ndim;
  int size;
} CudaTensor16;

extern ErlNifResourceType *CUDA_TENSOR16_RESOURCE;

void cuda_tensor16_destructor(ErlNifEnv *env, void *obj);
CudaTensor16 *alloc_cuda_tensor16(int ndim, const int *shape);
CudaTensor16 *get_cuda_tensor16(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_cuda_tensor16_term(ErlNifEnv *env, CudaTensor16 *t);

typedef struct {
  int8_t *d_data;
  int8_t *d_data_t;
  int32_t *d_acc;
  int *shape;
  int ndim;
  int size;
} CudaInt8Tensor;

extern ErlNifResourceType *CUDA_INT8_TENSOR_RESOURCE;

void cuda_int8_tensor_destructor(ErlNifEnv *env, void *obj);
CudaInt8Tensor *alloc_cuda_int8_tensor(int ndim, const int *shape);
CudaInt8Tensor *get_cuda_int8_tensor(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_cuda_int8_tensor_term(ErlNifEnv *env, CudaInt8Tensor *t);

/* SparseTensor - 2:4 Structured Sparsity */
typedef struct {
  void* d_compressed;
  void* d_workspace;
  size_t compressed_size;
  size_t workspace_size;
  int64_t rows;
  int64_t cols;
  int dtype;
  char mat_descr_storage[1024];
  char matmul_descr_storage[1024];
  char alg_sel_storage[1024];
  char plan_storage[1024];
} SparseTensorInternal;

extern ErlNifResourceType *SPARSE_TENSOR_RESOURCE;

void sparse_tensor_destructor(ErlNifEnv *env, void *obj);
SparseTensorInternal *get_sparse_tensor(ErlNifEnv *env, ERL_NIF_TERM term);
ERL_NIF_TERM make_sparse_tensor_term(ErlNifEnv *env, SparseTensorInternal *t);

#endif /* !_WIN32 */

/* =========================================================================
 * Helper Functions (nif_tensor_core.c)
 * ========================================================================= */

int parse_shape(ErlNifEnv *env, ERL_NIF_TERM list, int *out_shape, int *out_ndim);
double *list_to_doubles(ErlNifEnv *env, ERL_NIF_TERM list, unsigned *out_len);
ERL_NIF_TERM doubles_to_list(ErlNifEnv *env, const double *arr, unsigned len);
ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM value);
ERL_NIF_TERM make_ok_nil(ErlNifEnv *env);
ERL_NIF_TERM make_error(ErlNifEnv *env, const char *reason);
double get_number(ErlNifEnv *env, ERL_NIF_TERM term, int *ok);

/* FP16 conversion (float_to_half in nif_cpu_ops.c, f16_to_f32 in nif_cuda_tensors.c) */
uint16_t float_to_half(float f);
#ifndef _WIN32
float f16_to_f32(uint16_t h);
#endif

/* =========================================================================
 * Zig SIMD Extern Declarations
 * ========================================================================= */

extern double vt_simd_dot(const double *a, const double *b, size_t len);
extern double vt_simd_sum(const double *data, size_t len);
extern void vt_simd_scale(const double *data, double scalar, double *result, size_t len);
extern void vt_simd_add(const double *a, const double *b, double *result, size_t len);
extern void vt_simd_mul(const double *a, const double *b, double *result, size_t len);
extern void vt_simd_sub(const double *a, const double *b, double *result, size_t len);
extern void vt_simd_negate(const double *data, double *result, size_t len);
extern void vt_simd_relu(const double *data, double *result, size_t len);
extern double vt_simd_max(const double *data, size_t len);
extern double vt_simd_min(const double *data, size_t len);
extern void vt_simd_exp(const double *data, double *result, size_t len);
extern void vt_simd_sigmoid(const double *data, double *result, size_t len);
extern void vt_simd_log(const double *data, double *result, size_t len);

/* In-place mutation ops */
extern void vt_simd_add_mut(double *a, const double *b, size_t len);
extern void vt_simd_scale_mut(double *a, double scalar, size_t len);
extern void vt_simd_negate_mut(double *a, size_t len);
extern void vt_simd_relu_mut(double *a, size_t len);

/* Retro/fused kernels */
extern void vt_saturn_blend(const double *texture, const double *shade,
                            double bias, double *result, size_t len);
extern void vt_resonance_mul(const double *a, const double *b, double *result, size_t len);
extern void vt_resonance_power(const double *data, double exponent, double *result, size_t len);

/* LNS f32 */
extern void vt_lns_mul_f32(const float *a, const float *b, float *result, size_t len);
extern void vt_lns_mul_corrected_f32(const float *a, const float *b, float *result, size_t len);
extern void vt_lns_div_f32(const float *a, const float *b, float *result, size_t len);
extern void vt_lns_sqrt_f32(const float *data, float *result, size_t len);
extern void vt_lns_rsqrt_f32(const float *data, float *result, size_t len);

/* Horde SoA Physics */
extern void vt_horde_integrate(double *positions, const double *velocities, double dt, size_t count);
extern void vt_horde_dampen(double *velocities, double friction, size_t count);
extern void vt_horde_accelerate(double *velocities, const double *accelerations, double dt, size_t count);
extern void vt_horde_wrap(double *positions, double max_bound, size_t count);
extern void vt_horde_gravity_2d(double *accelerations, double gravity, size_t entity_count);
extern double vt_horde_kinetic_energy(const double *velocities, size_t count);

/* HDC */
extern void vt_hdc_bind(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t len);
extern uint64_t vt_hdc_hamming(const uint64_t *a, const uint64_t *b, size_t len);
extern double vt_hdc_similarity(const uint64_t *a, const uint64_t *b, size_t len, size_t dim);
extern void vt_hdc_bundle(const uint64_t *inputs, size_t n_vectors, size_t words, uint64_t *result);
extern void vt_hdc_permute(const uint64_t *input, uint64_t *output, size_t words, size_t shift);
extern void vt_hdc_random(uint64_t *output, size_t words, uint64_t seed);
extern void vt_hdc_weighted_bundle(const uint64_t *inputs, const double *weights,
                                   size_t n_vectors, size_t words, uint64_t *result);

/* Fused Quantized Matmul */
extern void vt_matmul_int8(const double *a, const int8_t *b_quant, double b_scale,
                           size_t m, size_t n, size_t k, double *c);
extern void vt_matmul_int8_blocked(const double *a, const int8_t *b_quant,
                                    const double *b_scales, size_t m, size_t n,
                                    size_t k, size_t block_size, double *c);
extern void vt_matmul_nf4(const double *a, const uint8_t *b_indices,
                           const double *b_scales, size_t m, size_t n, size_t k,
                           size_t block_size, double *c);
extern double vt_quantize_int8(const double *data, int8_t *output, size_t len);
extern void vt_quantize_nf4(const double *data, uint8_t *output, double *scales,
                            size_t len, size_t block_size);

/* =========================================================================
 * CUDA Extern Declarations (from cuda_gemm.c, cuda_sparselt.c, cuda_sage.c)
 * ========================================================================= */

#ifndef _WIN32

/* cuda_gemm.c - Core CUDA operations */
extern int cuda_init(void);
extern int cuda_available(void);
extern int cuda_dgemm(int M, int N, int K, double alpha, const double *A, int lda,
                      const double *B, int ldb, double beta, double *C, int ldc);
extern int cuda_sgemm(int M, int N, int K, float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc);
extern void cuda_cleanup(void);

/* CudaTensor GPU memory */
extern float* cuda_tensor_alloc(size_t num_elements);
extern void cuda_tensor_free(void *d_ptr);
extern int cuda_tensor_upload(float *d_dst, const float *h_src, size_t num_elements);
extern int cuda_tensor_download(float *h_dst, const float *d_src, size_t num_elements);
extern int cuda_sgemm_gpu(int M, int N, int K, float alpha, const float *d_A, int lda,
                          const float *d_B, int ldb, float beta, float *d_C, int ldc);
extern int cuda_sgemm_gpu_inplace(int M, int N, int K, float alpha, const float *d_A, int lda,
                                   const float *d_B, int ldb, float beta, float *d_C, int ldc);

/* INT8 Tensor Cores */
extern int cuda_int8_available(void);
extern int cuda_fp16_available(void);
extern int8_t* cuda_tensor_alloc_int8(size_t num_elements);
extern int32_t* cuda_tensor_alloc_int32(size_t num_elements);
extern int cuda_tensor_upload_int8(int8_t *d_dst, const int8_t *h_src, size_t num_elements);
extern int cuda_tensor_download_int32(int32_t *h_dst, const int32_t *d_src, size_t num_elements);
extern int cuda_igemm(int M, int N, int K, int32_t alpha, const int8_t *A, int lda,
                      const int8_t *B, int ldb, int32_t beta, int32_t *C, int ldc);
extern int cuda_igemm_gpu(int M, int N, int K, int32_t alpha, const int8_t *d_A, int lda,
                          const int8_t *d_B, int ldb, int32_t beta, int32_t *d_C, int ldc);

/* FP16 Tensor Cores */
extern uint16_t* cuda_tensor_alloc_fp16(size_t num_elements);
extern int cuda_tensor_upload_fp16(uint16_t *d_dst, const uint16_t *h_src, size_t num_elements);
extern int cuda_tensor_download_fp16(uint16_t *h_dst, const uint16_t *d_src, size_t num_elements);
extern int cuda_hgemm(int M, int N, int K, float alpha, const uint16_t *A, int lda,
                      const uint16_t *B, int ldb, float beta, float *C, int ldc);
extern int cuda_hgemm_gpu(int M, int N, int K, float alpha, const uint16_t *d_A, int lda,
                          const uint16_t *d_B, int ldb, float beta, float *d_C, int ldc);
extern int cuda_hgemm_gpu_pure16(int M, int N, int K,
                                  const uint16_t *d_A, int lda,
                                  const uint16_t *d_B, int ldb,
                                  uint16_t *d_C, int ldc);
extern int cuda_hgemm_gpu_pure16_async(int M, int N, int K,
                                        const uint16_t *d_A, int lda,
                                        const uint16_t *d_B, int ldb,
                                        uint16_t *d_C, int ldc);
extern int cuda_hgemm_lt_gpu_tn(int M, int N, int K,
                                  const uint16_t *d_A,
                                  const uint16_t *d_B_T,
                                  uint16_t *d_C);

/* cublasLt */
extern int cuda_int8_lt_available(void);
extern int cublaslt_init(void);
extern int cuda_igemm_lt(int M, int N, int K, float alpha, const int8_t *A, int lda,
                         const int8_t *B, int ldb, float beta, int32_t *C, int ldc);
extern int cuda_igemm_lt_gpu(int M, int N, int K, const int8_t *d_A, int lda,
                              const int8_t *d_B, int ldb, int32_t *d_C, int ldc);
extern int cuda_igemm_lt_gpu_tn(int M, int N, int K, const int8_t *d_A,
                                 const int8_t *d_B_T, int32_t *d_C);
extern int cuda_igemm_lt_gpu_async(int M, int N, int K, const int8_t *d_A, int lda,
                                    const int8_t *d_B, int ldb, int32_t *d_C, int ldc);
extern int cuda_hgemm_lt_32f(int M, int N, int K,
                              const uint16_t *d_A, const uint16_t *d_B, uint16_t *d_C);
extern int cuda_hgemm_fused_relu(int M, int N, int K,
                                  const uint16_t *d_A, const uint16_t *d_B, uint16_t *d_C);
extern int cuda_hgemm_fused_gelu(int M, int N, int K,
                                  const uint16_t *d_A, const uint16_t *d_B, uint16_t *d_C);
extern int cuda_hgemm_fused_relu_tn(int M, int N, int K,
                                     const uint16_t *d_A, const uint16_t *d_B_T, uint16_t *d_C);
extern int cuda_hgemm_fused_gelu_tn(int M, int N, int K,
                                     const uint16_t *d_A, const uint16_t *d_B_T, uint16_t *d_C);
extern int cuda_hgemm_batched(int M, int N, int K, int batch_count,
                               const uint16_t *d_A, const uint16_t *d_B, uint16_t *d_C);
extern int cuda_fp8gemm_lt_gpu_tn(int M, int N, int K,
                                   const uint8_t *d_A, const uint8_t *d_B_T, uint16_t *d_C);

/* FP8 */
extern uint8_t* cuda_tensor_alloc_fp8(size_t num_elements);
extern int cuda_tensor_upload_fp8(uint8_t *d_dst, const uint8_t *h_src, size_t num_elements);
extern int cuda_tensor_download_fp8(uint8_t *h_dst, const uint8_t *d_src, size_t num_elements);
extern int cuda_fp8gemm_gpu(int M, int N, int K, float alpha, const uint8_t *d_A, int lda,
                            const uint8_t *d_B, int ldb, float beta, float *d_C, int ldc);
extern int cuda_fp8gemm_gpu_async(int M, int N, int K, float alpha, const uint8_t *d_A, int lda,
                                  const uint8_t *d_B, int ldb, float beta, float *d_C, int ldc);
extern void float_to_fp8_e4m3_batch(uint8_t *dst, const float *src, size_t n);
extern void fp8_e4m3_to_float_batch(float *dst, const uint8_t *src, size_t n);

/* CUTLASS */
extern int cutlass_fp8_gemm_f16acc(int M, int N, int K,
                                    const void *d_A, const void *d_B, void *d_C);
extern int cutlass_fp8_gemm_f32acc(int M, int N, int K,
                                    const void *d_A, const void *d_B, void *d_C);
extern int cutlass_int8_sparse_gemm_bench(int M, int N, int K, int iters);
extern int cutlass_int8_sparse_gemm_bench_b(int M, int N, int K, int iters);
extern int cutlass_int8_sparse_gemm_bench_c(int M, int N, int K, int iters);
extern int cutlass_int8_sparse_gemm_bench_d(int M, int N, int K, int iters);
extern int cutlass_int8_sparse_gemm_bench_ex(int M, int N, int K, int iters, int config, int split_k);
extern void cutlass_int8_sparse_info(int *out_sparse, int *out_elements_per_e, int *out_sizeof_e);
extern int cutlass_int4_sparse_gemm_bench(int M, int N, int K, int iters, int config, int split_k);
extern void cutlass_int4_sparse_info(int *out_sparse, int *out_elements_per_e, int *out_sizeof_e);

/* cuSPARSELt */
extern int cusparselt_int8_sparse_bench(int M, int N, int K, int iters, int mode);
extern int cusparselt_fp8_sparse_bench(int M, int N, int K, int iters);
extern int cusparselt_fp16_sparse_bench(int M, int N, int K, int iters);

/* cuSPARSELt tensor operations (cuda_sparselt.c) */
extern int cusparselt_available(void);
extern int sparse_tensor_create_fp16(const uint16_t* d_dense, int64_t rows, int64_t cols,
                                      SparseTensorInternal* out_sparse);
extern int sparse_tensor_create_int8(const int8_t* d_dense, int64_t rows, int64_t cols,
                                      SparseTensorInternal* out_sparse);
extern void sparse_tensor_free(SparseTensorInternal* sparse);
extern int sparse_matmul_fp16(SparseTensorInternal* sparse, const uint16_t* d_B,
                               uint16_t* d_C, int64_t N, float alpha, float beta);
extern int sparse_matmul_fp16_bench(SparseTensorInternal* sparse, const uint16_t* d_B,
                                     uint16_t* d_C, int64_t N, int iters);
extern int sparse_matmul_fp16_bench_tn(SparseTensorInternal* sparse, const uint16_t* d_B,
                                        uint16_t* d_C, int64_t N, int iters);
extern int sparse_matmul_int8(SparseTensorInternal* sparse, const int8_t* d_B,
                               int8_t* d_C, int64_t N, float alpha, float beta);
extern int sparse_matmul_int8_bench(SparseTensorInternal* sparse, const int8_t* d_B,
                                     int8_t* d_C, int64_t N, int iters);

/* SageAttention (cuda_sage.c) */
extern int sage_init(void);
extern int sage_available(void);
extern int sage_fp8_available(void);
extern int quant_int8_per_block_cpu(int8_t *out, float *scales, const float *in, size_t n, size_t block_size);
extern int dequant_int8_per_block_cpu(float *out, const int8_t *in, const float *scales, size_t n, size_t block_size);
extern int softmax_cpu(float *out, const float *in, size_t batch, size_t dim);
extern int sage_attention_cpu(float *O, const float *Q, const float *K, const float *V,
                              int batch, int heads, int seq_q, int seq_k, int head_dim, float sm_scale);
extern int sage_attention_gpu(float *d_O, const float *d_Q, const float *d_K, const float *d_V,
                              int batch, int heads, int seq_q, int seq_k, int head_dim, float sm_scale);

/* Async GEMM + sync */
extern int cuda_sgemm_gpu_async(int M, int N, int K, float alpha, const float *d_A, int lda,
                                 const float *d_B, int ldb, float beta, float *d_C, int ldc);
extern int cuda_hgemm_gpu_async(int M, int N, int K, float alpha, const void *d_A, int lda,
                                 const void *d_B, int ldb, float beta, float *d_C, int ldc);
extern void cuda_explicit_sync(void);

#endif /* !_WIN32 */

/* =========================================================================
 * NIF Function Declarations
 *
 * All NIF functions share the same signature:
 *   ERL_NIF_TERM func(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
 * ========================================================================= */

#define NIF_FUNC_DECL(name) \
  ERL_NIF_TERM name(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

/* nif_tensor_core.c — constructors + accessors */
NIF_FUNC_DECL(nt_zeros);
NIF_FUNC_DECL(nt_ones);
NIF_FUNC_DECL(nt_fill);
NIF_FUNC_DECL(nt_from_list);
NIF_FUNC_DECL(nt_to_list);
NIF_FUNC_DECL(nt_shape);
NIF_FUNC_DECL(nt_size);

/* nif_cpu_ops.c — element-wise, reductions, matmul, activations, in-place, fused */
NIF_FUNC_DECL(nt_add);
NIF_FUNC_DECL(nt_sub);
NIF_FUNC_DECL(nt_mul);
NIF_FUNC_DECL(nt_scale);
NIF_FUNC_DECL(nt_negate);
NIF_FUNC_DECL(nt_dot);
NIF_FUNC_DECL(nt_sum);
NIF_FUNC_DECL(nt_max);
NIF_FUNC_DECL(nt_min);
NIF_FUNC_DECL(nt_matmul_blas);
NIF_FUNC_DECL(nt_matmul_inplace);
NIF_FUNC_DECL(nt_matmul_cuda);
NIF_FUNC_DECL(nt_transpose);
NIF_FUNC_DECL(nt_relu);
NIF_FUNC_DECL(nt_sigmoid);
NIF_FUNC_DECL(nt_exp_nif);
NIF_FUNC_DECL(nt_log_nif);
NIF_FUNC_DECL(nt_add_mut);
NIF_FUNC_DECL(nt_scale_mut);
NIF_FUNC_DECL(nt_negate_mut);
NIF_FUNC_DECL(nt_relu_mut);
NIF_FUNC_DECL(nt_saturn_blend);
NIF_FUNC_DECL(nt_fused_linear_relu_nif);
#ifndef _WIN32
NIF_FUNC_DECL(nt_matmul_cuda_fp32);
NIF_FUNC_DECL(nt_matmul_int8_tc);
NIF_FUNC_DECL(nt_int8_tc_available);
NIF_FUNC_DECL(nt_matmul_fp16_tc);
NIF_FUNC_DECL(nt_fp16_tc_available);
NIF_FUNC_DECL(nt_matmul_int8_lt);
NIF_FUNC_DECL(nt_int8_lt_available);
#endif

/* nif_specialized.c — Resonance, LNS, Horde, HDC */
NIF_FUNC_DECL(nt_resonance_mul);
NIF_FUNC_DECL(nt_resonance_power);
NIF_FUNC_DECL(lns_from_f64);
NIF_FUNC_DECL(lns_to_f64);
NIF_FUNC_DECL(lns_mul);
NIF_FUNC_DECL(lns_mul_corrected);
NIF_FUNC_DECL(lns_div);
NIF_FUNC_DECL(lns_sqrt);
NIF_FUNC_DECL(lns_rsqrt);
NIF_FUNC_DECL(horde_create);
NIF_FUNC_DECL(horde_set_positions);
NIF_FUNC_DECL(horde_set_velocities);
NIF_FUNC_DECL(horde_integrate_nif);
NIF_FUNC_DECL(horde_dampen_nif);
NIF_FUNC_DECL(horde_wrap_nif);
NIF_FUNC_DECL(horde_get_positions);
NIF_FUNC_DECL(horde_get_velocities);
NIF_FUNC_DECL(horde_count_nif);
NIF_FUNC_DECL(horde_kinetic_energy_nif);
NIF_FUNC_DECL(hdc_create_nif);
NIF_FUNC_DECL(hdc_random_nif);
NIF_FUNC_DECL(hdc_bind_nif);
NIF_FUNC_DECL(hdc_similarity_nif);
NIF_FUNC_DECL(hdc_permute_nif);
NIF_FUNC_DECL(hdc_dim_nif);

#ifndef _WIN32
/* nif_cuda_tensors.c — CudaTensor FP32, FP16, INT8, async, bench */
NIF_FUNC_DECL(ct_from_list);
NIF_FUNC_DECL(ct_to_list);
NIF_FUNC_DECL(ct_shape);
NIF_FUNC_DECL(ct_matmul);
NIF_FUNC_DECL(ct_matmul_inplace_nif);
NIF_FUNC_DECL(ct16_from_list);
NIF_FUNC_DECL(ct16_to_list);
NIF_FUNC_DECL(ct16_shape);
NIF_FUNC_DECL(ct16_matmul);
NIF_FUNC_DECL(ct16_matmul_inplace_nif);
NIF_FUNC_DECL(ct16_matmul_bench_nif);
NIF_FUNC_DECL(ct_int8_matmul_bench_nif);
NIF_FUNC_DECL(ct16_matmul_lt_32f_bench_nif);
NIF_FUNC_DECL(ct16_matmul_fused_relu_nif);
NIF_FUNC_DECL(ct16_matmul_fused_gelu_nif);
NIF_FUNC_DECL(ct16_matmul_fused_relu_bench_nif);
NIF_FUNC_DECL(ct16_matmul_fused_gelu_bench_nif);
NIF_FUNC_DECL(ct16_matmul_batched_bench_nif);
NIF_FUNC_DECL(fp8_matmul_lt_tn_bench_nif);
NIF_FUNC_DECL(cutlass_fp8_f16acc_bench_nif);
NIF_FUNC_DECL(cutlass_fp8_f32acc_bench_nif);
NIF_FUNC_DECL(cutlass_int8_sparse_bench_nif);
NIF_FUNC_DECL(cutlass_int8_sparse_bench_ex_nif);
NIF_FUNC_DECL(cusparselt_int8_sparse_bench_nif);
NIF_FUNC_DECL(cusparselt_fp8_sparse_bench_nif);
NIF_FUNC_DECL(cusparselt_fp16_sparse_bench_nif);
NIF_FUNC_DECL(cutlass_int4_sparse_bench_nif);
NIF_FUNC_DECL(ct16_matmul_fused_relu_tn_bench_nif);
NIF_FUNC_DECL(ct16_matmul_fused_gelu_tn_bench_nif);
NIF_FUNC_DECL(ct16_available);
NIF_FUNC_DECL(nif_cuda_sync);
NIF_FUNC_DECL(ct_matmul_async);
NIF_FUNC_DECL(ct16_matmul_async);
NIF_FUNC_DECL(ct_int8_available);
NIF_FUNC_DECL(ct_int8_from_list);
NIF_FUNC_DECL(ct_int8_to_list);
NIF_FUNC_DECL(ct_int8_shape);
NIF_FUNC_DECL(ct_int8_matmul);
NIF_FUNC_DECL(ct_int8_matmul_inplace_nif);
NIF_FUNC_DECL(ct_int8_matmul_async);

/* nif_sparse_quant.c — sparse, INT8 sparse, legacy, quant, SageAttention */
NIF_FUNC_DECL(sparse_from_ct16);
NIF_FUNC_DECL(sparse_shape);
NIF_FUNC_DECL(sparse_compression_ratio);
NIF_FUNC_DECL(sparse_matmul_nif);
NIF_FUNC_DECL(sparse_matmul_inplace_nif);
NIF_FUNC_DECL(sparse_matmul_bench_nif);
NIF_FUNC_DECL(sparse_matmul_bench_tn_nif);
NIF_FUNC_DECL(sparse_available);
NIF_FUNC_DECL(sparse_from_int8);
NIF_FUNC_DECL(sparse_matmul_int8_nif);
NIF_FUNC_DECL(sparse_matmul_int8_inplace_nif);
NIF_FUNC_DECL(sparse_matmul_int8_bench_nif);
NIF_FUNC_DECL(nif_sage_available);
NIF_FUNC_DECL(nif_sage_fp8_available);
NIF_FUNC_DECL(nif_sage_quant_int8);
NIF_FUNC_DECL(nif_sage_softmax);
NIF_FUNC_DECL(nif_sage_attention);
NIF_FUNC_DECL(sage_attention_ct);
#endif

/* nif_sparse_quant.c — legacy list-based + fused quantized (cross-platform) */
NIF_FUNC_DECL(nif_simd_dot);
NIF_FUNC_DECL(nif_simd_sum);
NIF_FUNC_DECL(nif_simd_scale);
NIF_FUNC_DECL(nif_simd_add);
NIF_FUNC_DECL(nif_simd_mul);
NIF_FUNC_DECL(nif_simd_matmul);
NIF_FUNC_DECL(nif_simd_available);
NIF_FUNC_DECL(nif_backend_info);
NIF_FUNC_DECL(nif_cpu_topology);
NIF_FUNC_DECL(nt_matmul_int8);
NIF_FUNC_DECL(nt_matmul_nf4);
NIF_FUNC_DECL(nt_quantize_int8);
NIF_FUNC_DECL(nt_to_qint8);
NIF_FUNC_DECL(nt_matmul_qint8);
NIF_FUNC_DECL(nt_to_qnf4);
NIF_FUNC_DECL(nt_matmul_qnf4);
NIF_FUNC_DECL(qint8_scale);
NIF_FUNC_DECL(qint8_shape);
NIF_FUNC_DECL(qnf4_info);

#undef NIF_FUNC_DECL

#endif /* VIVA_NIF_H */
