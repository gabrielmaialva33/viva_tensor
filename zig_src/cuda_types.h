/**
 * cuda_types.h - Shared CUDA/cuBLAS types, constants, and function pointer typedefs
 *
 * Single source of truth for CUDA types used across cuda_gemm.c, cuda_sage.c,
 * cuda_sparselt.c. All CUDA files include this instead of redefining types locally.
 */

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#ifndef _WIN32

#include <stdint.h>
#include <stddef.h>

/* === CUDA Runtime Types === */

typedef int cudaError_t;
typedef void* cudaStream_t;

#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

/* CUDA runtime function pointers */
typedef cudaError_t (*cuda_malloc_fn)(void**, size_t);
typedef cudaError_t (*cuda_free_fn)(void*);
typedef cudaError_t (*cuda_memcpy_fn)(void*, const void*, size_t, int);
typedef cudaError_t (*cuda_device_sync_fn)(void);
typedef const char* (*cuda_get_error_fn)(cudaError_t);

/* === cuBLAS Types === */

typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cudaDataType_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef int cublasMath_t;

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

/* CUDA data types */
#define CUDA_R_8I       3
#define CUDA_R_32I      10
#define CUDA_R_32F      0
#define CUDA_R_16F      2
#define CUDA_R_16BF     14
#define CUDA_R_8F_E4M3  28
#define CUDA_R_8F_E5M2  29

/* cuBLAS compute types */
#define CUBLAS_COMPUTE_16F           64
#define CUBLAS_COMPUTE_32F           68
#define CUBLAS_COMPUTE_64F           70
#define CUBLAS_COMPUTE_32I           72
#define CUBLAS_COMPUTE_32F_FAST_16F  74
#define CUBLAS_COMPUTE_32F_FAST_16BF 75
#define CUBLAS_COMPUTE_32F_FAST_TF32 77

/* cuBLAS GEMM algorithms */
#define CUBLAS_GEMM_DEFAULT          -1
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99

/* cuBLAS math modes */
#define CUBLAS_DEFAULT_MATH 0
#define CUBLAS_TENSOR_OP_MATH 1
#define CUBLAS_TF32_TENSOR_OP_MATH 3

/* FP8 types */
typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

/* === cuBLAS Function Pointer Types === */

typedef cublasStatus_t (*cublas_create_fn)(cublasHandle_t*);
typedef cublasStatus_t (*cublas_destroy_fn)(cublasHandle_t);

typedef cublasStatus_t (*cublas_dgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const double*, const double*, int,
    const double*, int,
    const double*, double*, int);

typedef cublasStatus_t (*cublas_sgemm_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int);

typedef cublasStatus_t (*cublas_gemm_ex_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t);

typedef cublasStatus_t (*cublas_set_workspace_fn)(cublasHandle_t, void *, size_t);
typedef cublasStatus_t (*cublas_set_stream_fn)(cublasHandle_t, void *);
typedef cublasStatus_t (*cublas_set_math_mode_fn)(cublasHandle_t, cublasMath_t);

typedef cublasStatus_t (*cublas_gemm_strided_batched_ex_fn)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, cudaDataType_t, int, long long int,
    const void*, cudaDataType_t, int, long long int,
    const void*, void*, cudaDataType_t, int, long long int,
    int, cublasComputeType_t, cublasGemmAlgo_t);

/* === Exported Globals (defined in cuda_gemm.c) === */

extern cublasHandle_t g_cublas_ctx;
extern cuda_malloc_fn g_cuda_malloc;
extern cuda_free_fn g_cuda_free;
extern cuda_memcpy_fn g_cuda_memcpy;
extern cuda_device_sync_fn g_cuda_sync;
extern cublas_sgemm_fn g_cublas_sgemm;
extern cublas_gemm_ex_fn g_cublas_gemm_ex;

#endif /* !_WIN32 */
#endif /* CUDA_TYPES_H */
