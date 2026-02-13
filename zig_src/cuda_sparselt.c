/** cuda_sparselt.c - cuSPARSELt 2:4 structured sparsity wrapper via dlopen */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef _WIN32

#include <dlfcn.h>

/* =========================================================================
 * cuSPARSELt Types — Correct 512-byte aligned structs (v0.8.0)
 * ========================================================================= */

typedef struct { __attribute__((aligned(16))) uint8_t data[512]; } cusparseLtHandle_t;
typedef struct { __attribute__((aligned(16))) uint8_t data[512]; } cusparseLtMatDescriptor_t;
typedef struct { __attribute__((aligned(16))) uint8_t data[512]; } cusparseLtMatmulDescriptor_t;
typedef struct { __attribute__((aligned(16))) uint8_t data[512]; } cusparseLtMatmulAlgSelection_t;
typedef struct { __attribute__((aligned(16))) uint8_t data[512]; } cusparseLtMatmulPlan_t;

typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
} cusparseStatus_t;

typedef enum {
    CUSPARSE_ORDER_COL = 0,
    CUSPARSE_ORDER_ROW = 1
} cusparseOrder_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
    CUSPARSELT_SPARSITY_50_PERCENT = 0
} cusparseLtSparsity_t;

typedef enum {
    CUSPARSELT_MATMUL_ALG_DEFAULT = 0
} cusparseLtMatmulAlg_t;

typedef enum {
    CUSPARSELT_PRUNE_SPMMA_TILE = 0,
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1
} cusparseLtPruneAlg_t;

/* Note: cudaDataType enum values must match CUDA's definitions */
#define SPLT_CUDA_R_32F  0
#define SPLT_CUDA_R_16F  2
#define SPLT_CUDA_R_8I   3
#define SPLT_CUDA_R_32I  10
#define SPLT_CUDA_R_16BF 14

typedef enum {
    CUSPARSE_COMPUTE_32I = 0,
    CUSPARSE_COMPUTE_16F = 1,
    CUSPARSE_COMPUTE_32F = 2
} cusparseComputeType;

typedef void* cudaStream_t;

/* =========================================================================
 * cuSPARSELt Function Pointer Types
 * ========================================================================= */

typedef cusparseStatus_t (*cusparseLtInit_fn)(cusparseLtHandle_t* handle);
typedef cusparseStatus_t (*cusparseLtDestroy_fn)(const cusparseLtHandle_t* handle);
typedef const char* (*cusparseLtGetErrorString_fn)(cusparseStatus_t status);
typedef const char* (*cusparseLtGetErrorName_fn)(cusparseStatus_t status);

typedef cusparseStatus_t (*cusparseLtStructuredDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment,
    int valueType,  /* cudaDataType as int */
    cusparseOrder_t order,
    cusparseLtSparsity_t sparsity);

typedef cusparseStatus_t (*cusparseLtDenseDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatDescriptor_t* matDescr,
    int64_t rows, int64_t cols, int64_t ld,
    uint32_t alignment,
    int valueType,  /* cudaDataType as int */
    cusparseOrder_t order);

typedef cusparseStatus_t (*cusparseLtMatDescriptorDestroy_fn)(
    const cusparseLtMatDescriptor_t* matDescr);

typedef cusparseStatus_t (*cusparseLtMatmulDescriptorInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseOperation_t opA,
    cusparseOperation_t opB,
    const cusparseLtMatDescriptor_t* matA,
    const cusparseLtMatDescriptor_t* matB,
    const cusparseLtMatDescriptor_t* matC,
    const cusparseLtMatDescriptor_t* matD,
    cusparseComputeType computeType);

typedef cusparseStatus_t (*cusparseLtMatmulAlgSelectionInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulAlgSelection_t* algSelection,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    cusparseLtMatmulAlg_t alg);

typedef cusparseStatus_t (*cusparseLtMatmulAlgSelectionDestroy_fn)(
    const cusparseLtMatmulAlgSelection_t* algSelection);

typedef cusparseStatus_t (*cusparseLtMatmulPlanInit_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulPlan_t* plan,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const cusparseLtMatmulAlgSelection_t* algSelection);

typedef cusparseStatus_t (*cusparseLtMatmulPlanDestroy_fn)(
    const cusparseLtMatmulPlan_t* plan);

typedef cusparseStatus_t (*cusparseLtSpMMAPrune_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    cusparseLtPruneAlg_t pruneAlg,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtSpMMAPrune2_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatDescriptor_t* sparseMatDescr,
    int isSparseA,
    cusparseOperation_t op,
    const void* d_in,
    void* d_out,
    cusparseLtPruneAlg_t pruneAlg,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtSpMMAPruneCheck_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtSpMMACompressedSize_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    size_t* compressedSize,
    size_t* compressBufferSize);

typedef cusparseStatus_t (*cusparseLtSpMMACompressedSize2_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatDescriptor_t* sparseMatDescr,
    size_t* compressedSize,
    size_t* compressedBufferSize);

typedef cusparseStatus_t (*cusparseLtSpMMACompress_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    const void* d_dense,
    void* d_compressed,
    void* d_compressBuffer,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtSpMMACompress2_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatDescriptor_t* sparseMatDescr,
    int isSparseA,
    cusparseOperation_t op,
    const void* d_dense,
    void* d_compressed,
    void* d_compressedBuffer,
    cudaStream_t stream);

typedef cusparseStatus_t (*cusparseLtMatmulGetWorkspace_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    size_t* workspaceSize);

typedef cusparseStatus_t (*cusparseLtMatmul_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* d_A,
    const void* d_B,
    const void* beta,
    const void* d_C,
    void* d_D,
    void* workspace,
    cudaStream_t* streams,
    int32_t numStreams);

typedef cusparseStatus_t (*cusparseLtMatmulSearch_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulPlan_t* plan,
    const void* alpha,
    const void* d_A,
    const void* d_B,
    const void* beta,
    const void* d_C,
    void* d_D,
    void* workspace,
    cudaStream_t* streams,
    int32_t numStreams);

typedef cusparseStatus_t (*cusparseLtMatmulAlgSetAttribute_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulAlgSelection_t* algSelection,
    int attribute,  /* cusparseLtMatmulAlgAttribute_t */
    const void* data,
    size_t dataSize);

typedef cusparseStatus_t (*cusparseLtMatmulAlgGetAttribute_fn)(
    const cusparseLtHandle_t* handle,
    const cusparseLtMatmulAlgSelection_t* algSelection,
    int attribute,  /* cusparseLtMatmulAlgAttribute_t */
    void* data,
    size_t dataSize);

/* cusparseLtMatmulAlgAttribute_t values */
#define CUSPARSELT_MATMUL_ALG_CONFIG_ID     0
#define CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID 1
#define CUSPARSELT_MATMUL_SEARCH_ITERATIONS 2
#define CUSPARSELT_MATMUL_SPLIT_K           3
#define CUSPARSELT_MATMUL_SPLIT_K_MODE      4
#define CUSPARSELT_MATMUL_SPLIT_K_BUFFERS   5

/* cusparseLtMatmulDescAttribute_t values */
#define CUSPARSELT_MATMUL_ACTIVATION_RELU           0
#define CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND 1
#define CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD 2
#define CUSPARSELT_MATMUL_ACTIVATION_GELU           3
#define CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING   4
#define CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING      5
#define CUSPARSELT_MATMUL_BETA_VECTOR_SCALING       6
#define CUSPARSELT_MATMUL_BIAS_STRIDE               7
#define CUSPARSELT_MATMUL_BIAS_POINTER               8
#define CUSPARSELT_MATMUL_ACTIVATION_ABS             9
#define CUSPARSELT_MATMUL_ACTIVATION_LEAKYRELU       10
#define CUSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA 11
#define CUSPARSELT_MATMUL_ACTIVATION_SIGMOID          12
#define CUSPARSELT_MATMUL_ACTIVATION_TANH             13
#define CUSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA       14
#define CUSPARSELT_MATMUL_ACTIVATION_TANH_BETA        15
#define CUSPARSELT_MATMUL_SPARSE_MAT_POINTER          16

/* cusparseLtMatmulDescSetAttribute function pointer */
typedef cusparseStatus_t (*cusparseLtMatmulDescSetAttribute_fn)(
    const cusparseLtHandle_t* handle,
    cusparseLtMatmulDescriptor_t* matmulDescr,
    int attribute,
    const void* data,
    size_t dataSize);

/* CUDA stream functions (from libcudart) */
typedef int (*cuda_stream_create_fn)(void** stream, unsigned int flags);
typedef int (*cuda_stream_destroy_fn)(void* stream);
typedef int (*cuda_stream_sync_fn)(void* stream);
#define cudaStreamNonBlocking 1

/* CUDA Graph types and functions — for zero-overhead kernel replay */
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
#define cudaStreamCaptureModeGlobal 0
typedef int (*cuda_stream_begin_capture_fn)(void* stream, int mode);
typedef int (*cuda_stream_end_capture_fn)(void* stream, cudaGraph_t* graph);
typedef int (*cuda_graph_instantiate_fn)(cudaGraphExec_t* graphExec, cudaGraph_t graph,
                                         unsigned long long flags);
typedef int (*cuda_graph_launch_fn)(cudaGraphExec_t graphExec, void* stream);
typedef int (*cuda_graph_destroy_fn)(cudaGraph_t graph);
typedef int (*cuda_graph_exec_destroy_fn)(cudaGraphExec_t graphExec);

/* =========================================================================
 * Global State
 * ========================================================================= */

static void* g_cusparselt_lib = NULL;
static int g_cusparselt_available = -1;

/* Persistent handle — correctly typed */
static cusparseLtHandle_t g_cusparselt_handle;

/* Function pointers */
static cusparseLtInit_fn g_cusparseLtInit = NULL;
static cusparseLtDestroy_fn g_cusparseLtDestroy = NULL;
static cusparseLtStructuredDescriptorInit_fn g_cusparseLtStructuredDescriptorInit = NULL;
static cusparseLtDenseDescriptorInit_fn g_cusparseLtDenseDescriptorInit = NULL;
static cusparseLtMatDescriptorDestroy_fn g_cusparseLtMatDescriptorDestroy = NULL;
static cusparseLtMatmulDescriptorInit_fn g_cusparseLtMatmulDescriptorInit = NULL;
static cusparseLtMatmulAlgSelectionInit_fn g_cusparseLtMatmulAlgSelectionInit = NULL;
static cusparseLtMatmulAlgSelectionDestroy_fn g_cusparseLtMatmulAlgSelectionDestroy = NULL;
static cusparseLtMatmulPlanInit_fn g_cusparseLtMatmulPlanInit = NULL;
static cusparseLtMatmulPlanDestroy_fn g_cusparseLtMatmulPlanDestroy = NULL;
static cusparseLtSpMMAPrune_fn g_cusparseLtSpMMAPrune = NULL;
static cusparseLtSpMMAPrune2_fn g_cusparseLtSpMMAPrune2 = NULL;
static cusparseLtSpMMAPruneCheck_fn g_cusparseLtSpMMAPruneCheck = NULL;
static cusparseLtSpMMACompressedSize_fn g_cusparseLtSpMMACompressedSize = NULL;
static cusparseLtSpMMACompressedSize2_fn g_cusparseLtSpMMACompressedSize2 = NULL;
static cusparseLtSpMMACompress_fn g_cusparseLtSpMMACompress = NULL;
static cusparseLtSpMMACompress2_fn g_cusparseLtSpMMACompress2 = NULL;
static cusparseLtMatmulGetWorkspace_fn g_cusparseLtMatmulGetWorkspace = NULL;
static cusparseLtMatmul_fn g_cusparseLtMatmul = NULL;
static cusparseLtMatmulSearch_fn g_cusparseLtMatmulSearch = NULL;
static cusparseLtGetErrorString_fn g_cusparseLtGetErrorString = NULL;
static cusparseLtGetErrorName_fn g_cusparseLtGetErrorName = NULL;
static cusparseLtMatmulAlgSetAttribute_fn g_cusparseLtMatmulAlgSetAttribute = NULL;
static cusparseLtMatmulAlgGetAttribute_fn g_cusparseLtMatmulAlgGetAttribute = NULL;
static cusparseLtMatmulDescSetAttribute_fn g_cusparseLtMatmulDescSetAttribute = NULL;

/* CUDA stream functions */
static cuda_stream_create_fn g_cuda_stream_create = NULL;
static cuda_stream_destroy_fn g_cuda_stream_destroy = NULL;
static cuda_stream_sync_fn g_cuda_stream_sync = NULL;

/* CUDA Graph functions */
static cuda_stream_begin_capture_fn g_cuda_stream_begin_capture = NULL;
static cuda_stream_end_capture_fn g_cuda_stream_end_capture = NULL;
static cuda_graph_instantiate_fn g_cuda_graph_instantiate = NULL;
static cuda_graph_launch_fn g_cuda_graph_launch = NULL;
static cuda_graph_destroy_fn g_cuda_graph_destroy = NULL;
static cuda_graph_exec_destroy_fn g_cuda_graph_exec_destroy = NULL;

/* Dedicated non-blocking stream for sparse ops (avoids default stream sync overhead) */
static cudaStream_t g_sparse_stream = NULL;

/* Global workspace for matmul (128 MiB, pre-allocated) */
#define SPARSELT_WORKSPACE_SIZE (128 * 1024 * 1024)
static void* g_sparselt_workspace = NULL;

/* Cached matmul plan for repeated executions with same dimensions */
typedef struct {
    int64_t M, N, K;
    cusparseLtMatDescriptor_t matA_desc;  /* Fresh structured descriptor for A */
    cusparseLtMatDescriptor_t matB_desc;
    cusparseLtMatDescriptor_t matC_desc;
    cusparseLtMatmulDescriptor_t matmul_desc;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    int valid;
    int compute_is_32f;  /* 1 if COMPUTE_32F (float alpha/beta), 0 if COMPUTE_16F (__fp16) */
    void* d_workspace;     /* Per-plan GPU workspace (for split-K buffers) */
    size_t workspace_size; /* Workspace size from cusparseLtMatmulGetWorkspace */
} SparseMatmulPlanCache;

static SparseMatmulPlanCache g_sparse_fp16_cache = { .valid = 0 };
static SparseMatmulPlanCache g_sparse_fp16_tn_cache = { .valid = 0 };  /* TN layout cache */

/* CUDA types + globals from cuda_gemm.c */
#include "cuda_types.h"

extern int cuda_available(void);
extern void* cuda_tensor_alloc_fp16(size_t num_elements);
extern void cuda_tensor_free(void* ptr);

/* =========================================================================
 * cuSPARSELt Initialization
 * ========================================================================= */

int cusparselt_init(void) {
    if (g_cusparselt_available >= 0) return g_cusparselt_available;

    if (!cuda_available()) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: CUDA not available\n");
        g_cusparselt_available = 0;
        return 0;
    }

    /* Try to load cuSPARSELt library — multiple paths */
    const char* paths[] = {
        "libcusparseLt.so",
        "libcusparseLt.so.0",
        "/usr/local/cuda/lib64/libcusparseLt.so",
        NULL  /* Sentinel — will try Python pip path below */
    };

    for (int i = 0; paths[i]; i++) {
        g_cusparselt_lib = dlopen(paths[i], RTLD_LAZY);
        if (g_cusparselt_lib) break;
    }

    /* Try Python pip nvidia package path */
    if (!g_cusparselt_lib) {
        /* Find via glob-like search in common pip locations */
        const char* pip_paths[] = {
            "/home/gabriel-maia/.asdf/installs/python/3.14.3/lib/python3.14/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0",
            NULL
        };
        for (int i = 0; pip_paths[i]; i++) {
            g_cusparselt_lib = dlopen(pip_paths[i], RTLD_LAZY);
            if (g_cusparselt_lib) break;
        }
    }

    if (!g_cusparselt_lib) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: library not found (install: pip install nvidia-cusparselt)\n");
        g_cusparselt_available = 0;
        return 0;
    }

    /* Load all function pointers */
    #define LOAD_FN(name) \
        g_##name = (name##_fn)dlsym(g_cusparselt_lib, #name); \
        if (!g_##name) { \
            fprintf(stderr, "[viva_tensor] cuSPARSELt: failed to load %s\n", #name); \
            dlclose(g_cusparselt_lib); \
            g_cusparselt_lib = NULL; \
            g_cusparselt_available = 0; \
            return 0; \
        }

    LOAD_FN(cusparseLtInit);
    LOAD_FN(cusparseLtDestroy);
    LOAD_FN(cusparseLtStructuredDescriptorInit);
    LOAD_FN(cusparseLtDenseDescriptorInit);
    LOAD_FN(cusparseLtMatDescriptorDestroy);
    LOAD_FN(cusparseLtMatmulDescriptorInit);
    LOAD_FN(cusparseLtMatmulAlgSelectionInit);
    LOAD_FN(cusparseLtMatmulPlanInit);
    LOAD_FN(cusparseLtMatmulPlanDestroy);
    LOAD_FN(cusparseLtSpMMAPrune);
    LOAD_FN(cusparseLtSpMMAPruneCheck);
    LOAD_FN(cusparseLtSpMMACompressedSize);
    LOAD_FN(cusparseLtSpMMACompress);
    LOAD_FN(cusparseLtMatmulGetWorkspace);
    LOAD_FN(cusparseLtMatmul);
    LOAD_FN(cusparseLtMatmulSearch);

    /* v2 APIs — optional (available in v0.6+) */
    g_cusparseLtSpMMACompressedSize2 = (cusparseLtSpMMACompressedSize2_fn)
        dlsym(g_cusparselt_lib, "cusparseLtSpMMACompressedSize2");
    g_cusparseLtSpMMACompress2 = (cusparseLtSpMMACompress2_fn)
        dlsym(g_cusparselt_lib, "cusparseLtSpMMACompress2");
    g_cusparseLtSpMMAPrune2 = (cusparseLtSpMMAPrune2_fn)
        dlsym(g_cusparselt_lib, "cusparseLtSpMMAPrune2");
    g_cusparseLtMatmulAlgSelectionDestroy = (cusparseLtMatmulAlgSelectionDestroy_fn)
        dlsym(g_cusparselt_lib, "cusparseLtMatmulAlgSelectionDestroy");
    g_cusparseLtGetErrorString = (cusparseLtGetErrorString_fn)
        dlsym(g_cusparselt_lib, "cusparseLtGetErrorString");
    g_cusparseLtGetErrorName = (cusparseLtGetErrorName_fn)
        dlsym(g_cusparselt_lib, "cusparseLtGetErrorName");
    g_cusparseLtMatmulAlgSetAttribute = (cusparseLtMatmulAlgSetAttribute_fn)
        dlsym(g_cusparselt_lib, "cusparseLtMatmulAlgSetAttribute");
    g_cusparseLtMatmulAlgGetAttribute = (cusparseLtMatmulAlgGetAttribute_fn)
        dlsym(g_cusparselt_lib, "cusparseLtMatmulAlgGetAttribute");
    g_cusparseLtMatmulDescSetAttribute = (cusparseLtMatmulDescSetAttribute_fn)
        dlsym(g_cusparselt_lib, "cusparseLtMatmulDescSetAttribute");

    #undef LOAD_FN

    /* Initialize persistent handle */
    memset(&g_cusparselt_handle, 0, sizeof(g_cusparselt_handle));
    cusparseStatus_t status = g_cusparseLtInit(&g_cusparselt_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: init failed with status %d\n", status);
        dlclose(g_cusparselt_lib);
        g_cusparselt_lib = NULL;
        g_cusparselt_available = 0;
        return 0;
    }

    /* Pre-allocate global workspace (32 MiB) */
    if (g_cuda_malloc(&g_sparselt_workspace, SPARSELT_WORKSPACE_SIZE) != 0) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt: workspace alloc failed\n");
        g_cusparseLtDestroy(&g_cusparselt_handle);
        dlclose(g_cusparselt_lib);
        g_cusparselt_lib = NULL;
        g_cusparselt_available = 0;
        return 0;
    }

    /* Create dedicated non-blocking CUDA stream for sparse ops.
     * Non-blocking avoids implicit sync with default stream = less pipeline stalls. */
    {
        void* cudart = dlopen("libcudart.so", RTLD_LAZY);
        if (!cudart) cudart = dlopen("libcudart.so.12", RTLD_LAZY);
        if (cudart) {
            g_cuda_stream_create = (cuda_stream_create_fn)dlsym(cudart, "cudaStreamCreateWithFlags");
            g_cuda_stream_destroy = (cuda_stream_destroy_fn)dlsym(cudart, "cudaStreamDestroy");
            g_cuda_stream_sync = (cuda_stream_sync_fn)dlsym(cudart, "cudaStreamSynchronize");
            if (g_cuda_stream_create) {
                int err = g_cuda_stream_create(&g_sparse_stream, cudaStreamNonBlocking);
                if (err != 0) {
                    g_sparse_stream = NULL;  /* Fall back to default stream */
                }
            }

            /* CUDA Graph functions for zero-overhead kernel replay */
            g_cuda_stream_begin_capture = (cuda_stream_begin_capture_fn)
                dlsym(cudart, "cudaStreamBeginCapture");
            g_cuda_stream_end_capture = (cuda_stream_end_capture_fn)
                dlsym(cudart, "cudaStreamEndCapture");
            g_cuda_graph_instantiate = (cuda_graph_instantiate_fn)
                dlsym(cudart, "cudaGraphInstantiateWithFlags");
            g_cuda_graph_launch = (cuda_graph_launch_fn)
                dlsym(cudart, "cudaGraphLaunch");
            g_cuda_graph_destroy = (cuda_graph_destroy_fn)
                dlsym(cudart, "cudaGraphDestroy");
            g_cuda_graph_exec_destroy = (cuda_graph_exec_destroy_fn)
                dlsym(cudart, "cudaGraphExecDestroy");
            /* Don't close cudart — already loaded globally */
        }
    }

    fprintf(stderr, "[viva_tensor] cuSPARSELt v0.8 loaded — 2:4 sparsity ready "
            "(%d MiB workspace, stream=%s)\n",
            SPARSELT_WORKSPACE_SIZE / (1024*1024),
            g_sparse_stream ? "dedicated" : "default");
    g_cusparselt_available = 1;
    return 1;
}

int cusparselt_available(void) {
    if (g_cusparselt_available < 0) cusparselt_init();
    return g_cusparselt_available;
}

/* =========================================================================
 * SparseTensor Data Structure
 * ========================================================================= */

/* NOTE: This struct layout MUST match the one in nif_entry.c exactly! */
typedef struct {
    void* d_compressed;         /* Compressed sparse data on GPU */
    void* d_workspace;          /* Reserved (unused — workspace is global) */
    size_t compressed_size;     /* Size of compressed data */
    size_t workspace_size;      /* Reserved */
    int64_t rows;               /* Original rows (M) */
    int64_t cols;               /* Original cols (K) */
    int dtype;                  /* Data type */
    /* cuSPARSELt descriptor storage — 1024 bytes each, cast to aligned types */
    char mat_descr_storage[1024];       /* cusparseLtMatDescriptor_t (512 bytes used) */
    char matmul_descr_storage[1024];    /* unused — plans cached globally */
    char alg_sel_storage[1024];         /* unused — plans cached globally */
    char plan_storage[1024];            /* unused — plans cached globally */
} SparseTensorInternal;

/* Cast helpers for aligned access */
#define SPARSE_MAT_DESCR(s) ((cusparseLtMatDescriptor_t*)(s)->mat_descr_storage)

/* =========================================================================
 * Sparse Tensor Creation — Prune + Compress
 * ========================================================================= */

int sparse_tensor_create_fp16(
    const uint16_t* d_dense,
    int64_t rows,
    int64_t cols,
    SparseTensorInternal* out_sparse
) {
    if (!cusparselt_available()) return -1;
    if (!d_dense || !out_sparse) return -2;

    /* Dimensions must be multiples of 16 for FP16 2:4 sparsity */
    if (rows % 16 != 0 || cols % 16 != 0) {
        fprintf(stderr, "[viva_tensor] SparseTensor: FP16 requires dims multiples of 16 (got %ldx%ld)\n",
                rows, cols);
        return -3;
    }

    cusparseStatus_t status;
    int64_t ld = cols;
    size_t dense_size = rows * cols * sizeof(uint16_t);

    /* Initialize output structure */
    memset(out_sparse, 0, sizeof(SparseTensorInternal));
    out_sparse->rows = rows;
    out_sparse->cols = cols;
    out_sparse->dtype = SPLT_CUDA_R_16F;

    cusparseLtMatDescriptor_t* matA = SPARSE_MAT_DESCR(out_sparse);

    /* Create sparse matrix descriptor (A) */
    status = g_cusparseLtStructuredDescriptorInit(
        &g_cusparselt_handle, matA,
        rows, cols, ld,
        16,                             /* 16-byte alignment */
        SPLT_CUDA_R_16F,               /* FP16 */
        CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: StructuredDescriptorInit failed: %d\n", status);
        return -4;
    }

    /* Allocate temporary buffer for pruned data */
    void* d_pruned;
    if (g_cuda_malloc(&d_pruned, dense_size) != 0) {
        g_cusparseLtMatDescriptorDestroy(matA);
        return -5;
    }

    /* Copy dense data (prune modifies in-place) */
    g_cuda_memcpy(d_pruned, d_dense, dense_size, cudaMemcpyDeviceToDevice);

    /* Use v2 Prune API if available (simpler, no matmul descriptor needed) */
    if (g_cusparseLtSpMMAPrune2) {
        status = g_cusparseLtSpMMAPrune2(
            &g_cusparselt_handle, matA,
            1,  /* isSparseA = true */
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_pruned, d_pruned,
            CUSPARSELT_PRUNE_SPMMA_STRIP,
            NULL
        );
    } else {
        /* Fallback: create a temporary matmul descriptor for v1 prune */
        cusparseLtMatDescriptor_t tmpB, tmpC;
        cusparseLtMatmulDescriptor_t tmpMatmul;

        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpB,
            cols, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpC,
            rows, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtMatmulDescriptorInit(&g_cusparselt_handle, &tmpMatmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matA, &tmpB, &tmpC, &tmpC,
            CUSPARSE_COMPUTE_16F);

        status = g_cusparseLtSpMMAPrune(
            &g_cusparselt_handle, &tmpMatmul,
            d_pruned, d_pruned,
            CUSPARSELT_PRUNE_SPMMA_STRIP, NULL
        );

        g_cusparseLtMatDescriptorDestroy(&tmpB);
        g_cusparseLtMatDescriptorDestroy(&tmpC);
    }

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: Prune failed: %d\n", status);
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -6;
    }

    /* Get compressed size using v2 API if available */
    size_t compress_buffer_size = 0;
    if (g_cusparseLtSpMMACompressedSize2) {
        status = g_cusparseLtSpMMACompressedSize2(
            &g_cusparselt_handle, matA,
            &out_sparse->compressed_size,
            &compress_buffer_size
        );
    } else {
        /* Fallback: create temp plan for v1 CompressedSize */
        cusparseLtMatDescriptor_t tmpB, tmpC;
        cusparseLtMatmulDescriptor_t tmpMatmul;
        cusparseLtMatmulAlgSelection_t tmpAlg;
        cusparseLtMatmulPlan_t tmpPlan;

        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpB,
            cols, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpC,
            rows, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtMatmulDescriptorInit(&g_cusparselt_handle, &tmpMatmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matA, &tmpB, &tmpC, &tmpC,
            CUSPARSE_COMPUTE_16F);
        g_cusparseLtMatmulAlgSelectionInit(&g_cusparselt_handle, &tmpAlg,
            &tmpMatmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        g_cusparseLtMatmulPlanInit(&g_cusparselt_handle, &tmpPlan, &tmpMatmul, &tmpAlg);

        status = g_cusparseLtSpMMACompressedSize(
            &g_cusparselt_handle, &tmpPlan,
            &out_sparse->compressed_size, &compress_buffer_size
        );

        g_cusparseLtMatmulPlanDestroy(&tmpPlan);
        if (g_cusparseLtMatmulAlgSelectionDestroy)
            g_cusparseLtMatmulAlgSelectionDestroy(&tmpAlg);
        g_cusparseLtMatDescriptorDestroy(&tmpB);
        g_cusparseLtMatDescriptorDestroy(&tmpC);
    }

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: CompressedSize failed: %d\n", status);
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -7;
    }

    /* Allocate compressed buffer */
    if (g_cuda_malloc(&out_sparse->d_compressed, out_sparse->compressed_size) != 0) {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -8;
    }

    /* Allocate compress buffer */
    void* d_compress_buffer = NULL;
    if (compress_buffer_size > 0) {
        if (g_cuda_malloc(&d_compress_buffer, compress_buffer_size) != 0) {
            g_cuda_free(out_sparse->d_compressed);
            g_cuda_free(d_pruned);
            g_cusparseLtMatDescriptorDestroy(matA);
            return -9;
        }
    }

    /* Compress using v2 API if available */
    if (g_cusparseLtSpMMACompress2) {
        status = g_cusparseLtSpMMACompress2(
            &g_cusparselt_handle, matA,
            1,  /* isSparseA = true */
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_pruned,
            out_sparse->d_compressed,
            d_compress_buffer,
            NULL
        );
    } else {
        /* Fallback: create temp plan for v1 Compress */
        cusparseLtMatDescriptor_t tmpB, tmpC;
        cusparseLtMatmulDescriptor_t tmpMatmul;
        cusparseLtMatmulAlgSelection_t tmpAlg;
        cusparseLtMatmulPlan_t tmpPlan;

        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpB,
            cols, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpC,
            rows, cols, cols, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW);
        g_cusparseLtMatmulDescriptorInit(&g_cusparselt_handle, &tmpMatmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matA, &tmpB, &tmpC, &tmpC,
            CUSPARSE_COMPUTE_16F);
        g_cusparseLtMatmulAlgSelectionInit(&g_cusparselt_handle, &tmpAlg,
            &tmpMatmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        g_cusparseLtMatmulPlanInit(&g_cusparselt_handle, &tmpPlan, &tmpMatmul, &tmpAlg);

        status = g_cusparseLtSpMMACompress(
            &g_cusparselt_handle, &tmpPlan,
            d_pruned, out_sparse->d_compressed,
            d_compress_buffer, NULL
        );

        g_cusparseLtMatmulPlanDestroy(&tmpPlan);
        if (g_cusparseLtMatmulAlgSelectionDestroy)
            g_cusparseLtMatmulAlgSelectionDestroy(&tmpAlg);
        g_cusparseLtMatDescriptorDestroy(&tmpB);
        g_cusparseLtMatDescriptorDestroy(&tmpC);
    }

    if (d_compress_buffer) g_cuda_free(d_compress_buffer);
    g_cuda_free(d_pruned);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] SparseTensor: Compress failed: %d\n", status);
        g_cuda_free(out_sparse->d_compressed);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -10;
    }

    g_cuda_sync();

    fprintf(stderr, "[viva_tensor] SparseTensor created: %ldx%ld -> %zu bytes (%.1f%% of dense)\n",
            rows, cols, out_sparse->compressed_size,
            100.0 * out_sparse->compressed_size / dense_size);

    return 0;
}

void sparse_tensor_free(SparseTensorInternal* sparse) {
    if (!sparse) return;

    if (sparse->d_compressed) {
        g_cuda_free(sparse->d_compressed);
        sparse->d_compressed = NULL;
    }
    if (sparse->d_workspace) {
        g_cuda_free(sparse->d_workspace);
        sparse->d_workspace = NULL;
    }

    if (g_cusparseLtMatDescriptorDestroy) {
        g_cusparseLtMatDescriptorDestroy(SPARSE_MAT_DESCR(sparse));
    }
}

/* =========================================================================
 * Cached Matmul Plan Management
 * ========================================================================= */

static void sparse_cache_invalidate(SparseMatmulPlanCache* cache) {
    if (!cache->valid) return;

    g_cusparseLtMatmulPlanDestroy(&cache->plan);
    if (g_cusparseLtMatmulAlgSelectionDestroy)
        g_cusparseLtMatmulAlgSelectionDestroy(&cache->alg_sel);
    g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
    g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
    g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);

    /* Free per-plan workspace if it was separately allocated (not the global one) */
    if (cache->d_workspace && cache->d_workspace != g_sparselt_workspace && g_cuda_free) {
        g_cuda_free(cache->d_workspace);
    }
    cache->d_workspace = NULL;
    cache->workspace_size = 0;

    cache->valid = 0;
}

/**
 * Create or retrieve a cached matmul plan for the given dimensions.
 * If dimensions match the cache, returns immediately (zero overhead).
 * Otherwise creates a new plan with algorithm search.
 *
 * opB: CUSPARSE_OPERATION_NON_TRANSPOSE (NN) or CUSPARSE_OPERATION_TRANSPOSE (TN)
 * For TN: B is N x K (transposed), compute C = A_sparse @ B^T
 */
static int sparse_get_or_create_plan(
    SparseMatmulPlanCache* cache,
    SparseTensorInternal* sparse,
    int64_t N,
    const void* d_A_compressed,
    const void* d_B,
    void* d_C,
    cusparseOperation_t opB
) {
    int64_t M = sparse->rows;
    int64_t K = sparse->cols;

    /* Cache hit? */
    if (cache->valid && cache->M == M && cache->N == N && cache->K == K) {
        return 0;
    }

    /* Cache miss — invalidate and rebuild */
    sparse_cache_invalidate(cache);

    cusparseStatus_t status;

    cache->M = M;
    cache->N = N;
    cache->K = K;

    /* Create FRESH structured A descriptor (can't reuse — cuSPARSELt internal state) */
    status = g_cusparseLtStructuredDescriptorInit(
        &g_cusparselt_handle, &cache->matA_desc,
        M, K, K, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: A structured descriptor failed: %d\n", status);
        return -1;
    }

    /* Create B descriptor — layout depends on operation:
     * NN: B is K x N, ld=N
     * TN: B is N x K, ld=K (stored as N x K, transposed during matmul) */
    int64_t B_rows = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? K : N;
    int64_t B_cols = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? N : K;
    int64_t B_ld   = B_cols;  /* Row-major: ld = number of columns */
    status = g_cusparseLtDenseDescriptorInit(
        &g_cusparselt_handle, &cache->matB_desc,
        B_rows, B_cols, B_ld, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: B descriptor failed: %d (opB=%d)\n", status, opB);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        return -2;
    }

    /* Create C descriptor: M x N dense */
    status = g_cusparseLtDenseDescriptorInit(
        &g_cusparselt_handle, &cache->matC_desc,
        M, N, N, 16, SPLT_CUDA_R_16F, CUSPARSE_ORDER_ROW
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: C descriptor failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        return -2;
    }

    /* COMPUTE_32F — the only compute type in cuSPARSELt v0.8+
     * (COMPUTE_16F was removed; FP16 always uses FP32 accumulation) */
    cache->compute_is_32f = 1;

    /* Zero out all opaque structs */
    memset(&cache->matmul_desc, 0, sizeof(cache->matmul_desc));
    memset(&cache->alg_sel, 0, sizeof(cache->alg_sel));
    memset(&cache->plan, 0, sizeof(cache->plan));

    /* Create matmul descriptor — opA always non-transpose, opB parameterized */
    status = g_cusparseLtMatmulDescriptorInit(
        &g_cusparselt_handle, &cache->matmul_desc,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        opB,
        &cache->matA_desc, &cache->matB_desc,
        &cache->matC_desc, &cache->matC_desc,
        CUSPARSE_COMPUTE_32F
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: MatmulDescInit failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -3;
    }

    /* Set sparse matrix pointer hint — gives search more flexibility to select
     * the best algorithm (recommended by NVIDIA docs). */
    if (g_cusparseLtMatmulDescSetAttribute && d_A_compressed) {
        g_cusparseLtMatmulDescSetAttribute(
            &g_cusparselt_handle, &cache->matmul_desc,
            CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
            &d_A_compressed, sizeof(d_A_compressed)
        );
    }

    /* Algorithm selection */
    status = g_cusparseLtMatmulAlgSelectionInit(
        &g_cusparselt_handle, &cache->alg_sel,
        &cache->matmul_desc, CUSPARSELT_MATMUL_ALG_DEFAULT
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: AlgSelInit failed: %d\n", status);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -4;
    }

    /* Thorough search — 100 iterations to find best algorithm + split-K config.
     * cusparseLtMatmulSearch auto-searches split-K factors, modes (ONE_KERNEL
     * vs TWO_KERNELS), and buffer counts on SM89. */
    if (g_cusparseLtMatmulAlgSetAttribute) {
        int search_iters = 100;
        g_cusparseLtMatmulAlgSetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
            &search_iters, sizeof(search_iters)
        );
    }

    /* Plan init */
    status = g_cusparseLtMatmulPlanInit(
        &g_cusparselt_handle, &cache->plan,
        &cache->matmul_desc, &cache->alg_sel
    );
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_plan: PlanInit failed: %d\n", status);
        if (g_cusparseLtMatmulAlgSelectionDestroy)
            g_cusparseLtMatmulAlgSelectionDestroy(&cache->alg_sel);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -4;
    }

    /* Query workspace size — may be larger than global workspace for split-K */
    cache->d_workspace = g_sparselt_workspace;
    cache->workspace_size = SPARSELT_WORKSPACE_SIZE;
    if (g_cusparseLtMatmulGetWorkspace) {
        size_t needed = 0;
        status = g_cusparseLtMatmulGetWorkspace(&g_cusparselt_handle, &cache->plan, &needed);
        if (status == CUSPARSE_STATUS_SUCCESS && needed > SPARSELT_WORKSPACE_SIZE) {
            /* Allocate larger per-plan workspace for split-K buffers */
            void* big_ws = NULL;
            if (g_cuda_malloc(&big_ws, needed) == 0) {
                cache->d_workspace = big_ws;
                cache->workspace_size = needed;
                fprintf(stderr, "[viva_tensor] sparse_plan: allocated %zu MiB workspace (split-K)\n",
                        needed / (1024*1024));
            }
        }
    }

    /* Sync GPU before algorithm search — ensure all prior CUDA operations complete */
    if (g_cuda_sync) g_cuda_sync();

    /* Algorithm search — runs multiple kernels, picks the fastest.
     * Also auto-searches split-K configs (factor, mode, buffers).
     * Uses dedicated non-blocking stream for less pipeline stalls.
     * Updates the plan IN-PLACE — do NOT destroy/recreate after! */
    if (d_A_compressed && d_B && d_C) {
        float alpha_f = 1.0f, beta_f = 0.0f;
        cudaStream_t* stream_ptr = g_sparse_stream ? &g_sparse_stream : NULL;
        int num_streams = g_sparse_stream ? 1 : 0;

        status = g_cusparseLtMatmulSearch(
            &g_cusparselt_handle, &cache->plan,
            &alpha_f, d_A_compressed, d_B,
            &beta_f, d_C, d_C,
            cache->d_workspace, stream_ptr, num_streams
        );
        if (status != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "[viva_tensor] sparse_plan: MatmulSearch failed: %d (using default alg)\n", status);
        }
    }

    /* Log selected algorithm + split-K info */
    if (g_cusparseLtMatmulAlgGetAttribute) {
        int alg_id = -1, max_alg_id = -1;
        int split_k = 0, split_k_mode = 0, split_k_buffers = 0;
        g_cusparseLtMatmulAlgGetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id));
        g_cusparseLtMatmulAlgGetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &max_alg_id, sizeof(max_alg_id));
        g_cusparseLtMatmulAlgGetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K, &split_k, sizeof(split_k));
        g_cusparseLtMatmulAlgGetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K_MODE, &split_k_mode, sizeof(split_k_mode));
        g_cusparseLtMatmulAlgGetAttribute(
            &g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &split_k_buffers, sizeof(split_k_buffers));
        fprintf(stderr, "[viva_tensor] sparse_plan: ready %ldx%ldx%ld (alg %d/%d, splitK=%d mode=%s bufs=%d)\n",
                M, N, K, alg_id, max_alg_id, split_k,
                split_k_mode == 0 ? "1kern" : split_k_mode == 1 ? "2kern" : "?",
                split_k_buffers);
    } else {
        fprintf(stderr, "[viva_tensor] sparse_plan: ready %ldx%ldx%ld\n", M, N, K);
    }
    cache->valid = 1;
    return 0;
}

/* =========================================================================
 * Sparse Matmul
 * ========================================================================= */

/**
 * Sparse GEMM: C = alpha * A_sparse @ B_dense + beta * C
 * Uses cached plan for zero descriptor overhead on hot path.
 * Async execution (no sync per call).
 */
int sparse_matmul_fp16(
    SparseTensorInternal* sparse,
    const uint16_t* d_B,
    uint16_t* d_C,
    int64_t N,
    float alpha,
    float beta
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;

    if (N % 16 != 0) {
        fprintf(stderr, "[viva_tensor] sparse_matmul: N must be multiple of 16 (got %ld)\n", N);
        return -3;
    }

    /* Get or create cached plan (NN layout) */
    int rc = sparse_get_or_create_plan(
        &g_sparse_fp16_cache, sparse, N,
        sparse->d_compressed, d_B, d_C,
        CUSPARSE_OPERATION_NON_TRANSPOSE
    );
    if (rc != 0) return rc;

    /* Execute sparse GEMM with cached plan.
     * Alpha/beta type must match compute type. */
    float alpha_f = alpha, beta_f = beta;
    __fp16 alpha_h = (__fp16)alpha, beta_h = (__fp16)beta;
    const void* alpha_ptr = g_sparse_fp16_cache.compute_is_32f ? (const void*)&alpha_f : (const void*)&alpha_h;
    const void* beta_ptr  = g_sparse_fp16_cache.compute_is_32f ? (const void*)&beta_f  : (const void*)&beta_h;

    cudaStream_t* stream_ptr = g_sparse_stream ? &g_sparse_stream : NULL;
    int num_streams = g_sparse_stream ? 1 : 0;

    cusparseStatus_t status = g_cusparseLtMatmul(
        &g_cusparselt_handle,
        &g_sparse_fp16_cache.plan,
        alpha_ptr,
        sparse->d_compressed,
        d_B,
        beta_ptr,
        d_C,
        d_C,
        g_sparse_fp16_cache.d_workspace,
        stream_ptr, num_streams
    );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "[viva_tensor] sparse_matmul: Matmul failed: %d\n", status);
        return -8;
    }

    /* No sync — async execution, caller syncs if needed */
    return 0;
}

/**
 * Sparse GEMM benchmark — CUDA Graph replay for zero-overhead kernel launch.
 * Captures 1 matmul into a CUDA graph, then replays it `iters` times.
 * Falls back to C-loop if CUDA graphs unavailable.
 */
int sparse_matmul_fp16_bench(
    SparseTensorInternal* sparse,
    const uint16_t* d_B,
    uint16_t* d_C,
    int64_t N,
    int iters
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;

    /* Get or create cached plan (NN layout) */
    int rc = sparse_get_or_create_plan(
        &g_sparse_fp16_cache, sparse, N,
        sparse->d_compressed, d_B, d_C,
        CUSPARSE_OPERATION_NON_TRANSPOSE
    );
    if (rc != 0) return rc;

    float alpha_f = 1.0f, beta_f = 0.0f;
    __fp16 alpha_h = (__fp16)1.0f, beta_h = (__fp16)0.0f;
    const void* alpha_ptr = g_sparse_fp16_cache.compute_is_32f ? (const void*)&alpha_f : (const void*)&alpha_h;
    const void* beta_ptr  = g_sparse_fp16_cache.compute_is_32f ? (const void*)&beta_f  : (const void*)&beta_h;

    /* Try CUDA Graph capture for zero-overhead replay */
    if (g_sparse_stream && g_cuda_stream_begin_capture && g_cuda_stream_end_capture &&
        g_cuda_graph_instantiate && g_cuda_graph_launch &&
        g_cuda_graph_destroy && g_cuda_graph_exec_destroy) {

        cudaGraph_t graph = NULL;
        cudaGraphExec_t graphExec = NULL;

        /* Capture 1 matmul call into graph */
        int err = g_cuda_stream_begin_capture(g_sparse_stream, cudaStreamCaptureModeGlobal);
        if (err == 0) {
            cusparseStatus_t status = g_cusparseLtMatmul(
                &g_cusparselt_handle,
                &g_sparse_fp16_cache.plan,
                alpha_ptr,
                sparse->d_compressed,
                d_B,
                beta_ptr,
                d_C,
                d_C,
                g_sparse_fp16_cache.d_workspace,
                &g_sparse_stream, 1
            );

            err = g_cuda_stream_end_capture(g_sparse_stream, &graph);
            if (err == 0 && graph && status == CUSPARSE_STATUS_SUCCESS) {
                err = g_cuda_graph_instantiate(&graphExec, graph, 0);
                if (err == 0 && graphExec) {
                    /* Replay graph `iters` times with near-zero launch overhead */
                    for (int i = 0; i < iters; i++) {
                        g_cuda_graph_launch(graphExec, g_sparse_stream);
                    }
                    g_cuda_graph_exec_destroy(graphExec);
                    g_cuda_graph_destroy(graph);
                    /* NO SYNC — caller must sync */
                    return 0;
                }
            }
            /* Cleanup on failure */
            if (graph) g_cuda_graph_destroy(graph);
        }
        /* Fall through to C-loop if graph capture failed */
    }

    /* Fallback: C-loop (still fast, just has per-call API overhead) */
    cudaStream_t* stream_ptr = g_sparse_stream ? &g_sparse_stream : NULL;
    int num_streams = g_sparse_stream ? 1 : 0;

    for (int i = 0; i < iters; i++) {
        cusparseStatus_t status = g_cusparseLtMatmul(
            &g_cusparselt_handle,
            &g_sparse_fp16_cache.plan,
            alpha_ptr,
            sparse->d_compressed,
            d_B,
            beta_ptr,
            d_C,
            d_C,
            g_sparse_fp16_cache.d_workspace,
            stream_ptr, num_streams
        );
        if (status != CUSPARSE_STATUS_SUCCESS) return -8;
    }

    /* NO SYNC — caller must sync if needed */
    return 0;
}

/**
 * Sparse GEMM TN benchmark — B is transposed (N x K instead of K x N).
 * Different memory access pattern may unlock better Tensor Core utilization.
 * For square matrices (M=N=K), B data is identical, only cuSPARSELt layout changes.
 */
int sparse_matmul_fp16_bench_tn(
    SparseTensorInternal* sparse,
    const uint16_t* d_B,
    uint16_t* d_C,
    int64_t N,
    int iters
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;

    /* Get or create cached plan (TN layout) */
    int rc = sparse_get_or_create_plan(
        &g_sparse_fp16_tn_cache, sparse, N,
        sparse->d_compressed, d_B, d_C,
        CUSPARSE_OPERATION_TRANSPOSE
    );
    if (rc != 0) return rc;

    float alpha_f = 1.0f, beta_f = 0.0f;
    const void* alpha_ptr = &alpha_f;
    const void* beta_ptr = &beta_f;

    cudaStream_t* stream_ptr = g_sparse_stream ? &g_sparse_stream : NULL;
    int num_streams = g_sparse_stream ? 1 : 0;

    for (int i = 0; i < iters; i++) {
        cusparseStatus_t status = g_cusparseLtMatmul(
            &g_cusparselt_handle,
            &g_sparse_fp16_tn_cache.plan,
            alpha_ptr,
            sparse->d_compressed,
            d_B,
            beta_ptr,
            d_C,
            d_C,
            g_sparse_fp16_tn_cache.d_workspace,
            stream_ptr, num_streams
        );
        if (status != CUSPARSE_STATUS_SUCCESS) return -8;
    }

    return 0;
}

/* =========================================================================
 * INT8 2:4 Sparse
 *
 * NOTE: cuSPARSELt v0.8 does NOT support INT8 structured sparse matmul
 * on Ada Lovelace (SM89). MatmulDescriptorInit returns NOT_SUPPORTED (10)
 * for all compute types with INT8 data. Compress2 also crashes (library bug).
 * The functions below detect this early and return -100 (not_supported).
 *
 * INT8 sparse may be supported on Hopper (SM90+) or in future cuSPARSELt.
 * ========================================================================= */

static int g_sparse_int8_supported = -1;  /* -1=unknown, 0=no, 1=yes */

/**
 * Check if INT8 sparse matmul is supported on this GPU + cuSPARSELt version.
 * Tests by creating a temporary matmul descriptor with INT8 types.
 */
static int sparse_int8_check_supported(void) {
    if (g_sparse_int8_supported >= 0) return g_sparse_int8_supported;
    if (!cusparselt_available()) { g_sparse_int8_supported = 0; return 0; }

    /* Create temporary INT8 descriptors to probe support */
    cusparseLtMatDescriptor_t tmpA, tmpB, tmpC;
    cusparseLtMatmulDescriptor_t tmpMatmul;
    int64_t N = 64;  /* Small test size */
    int have_A = 0, have_B = 0, have_C = 0;

    memset(&tmpA, 0, sizeof(tmpA));
    memset(&tmpB, 0, sizeof(tmpB));
    memset(&tmpC, 0, sizeof(tmpC));
    memset(&tmpMatmul, 0, sizeof(tmpMatmul));

    cusparseStatus_t s;
    s = g_cusparseLtStructuredDescriptorInit(&g_cusparselt_handle, &tmpA,
        N, N, N, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW,
        CUSPARSELT_SPARSITY_50_PERCENT);
    if (s != CUSPARSE_STATUS_SUCCESS) { g_sparse_int8_supported = 0; goto done_check; }
    have_A = 1;

    s = g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpB,
        N, N, N, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW);
    if (s != CUSPARSE_STATUS_SUCCESS) { g_sparse_int8_supported = 0; goto done_check; }
    have_B = 1;

    s = g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &tmpC,
        N, N, N, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW);
    if (s != CUSPARSE_STATUS_SUCCESS) { g_sparse_int8_supported = 0; goto done_check; }
    have_C = 1;

    /* Try all compute types */
    g_sparse_int8_supported = 0;
    int compute_types[] = {CUSPARSE_COMPUTE_32I, CUSPARSE_COMPUTE_32F, CUSPARSE_COMPUTE_16F};
    for (int i = 0; i < 3; i++) {
        memset(&tmpMatmul, 0, sizeof(tmpMatmul));
        s = g_cusparseLtMatmulDescriptorInit(&g_cusparselt_handle, &tmpMatmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &tmpA, &tmpB, &tmpC, &tmpC, compute_types[i]);
        if (s == CUSPARSE_STATUS_SUCCESS) {
            g_sparse_int8_supported = 1;
            break;
        }
    }

done_check:
    if (have_C) g_cusparseLtMatDescriptorDestroy(&tmpC);
    if (have_B) g_cusparseLtMatDescriptorDestroy(&tmpB);
    if (have_A) g_cusparseLtMatDescriptorDestroy(&tmpA);

    if (!g_sparse_int8_supported) {
        fprintf(stderr, "[viva_tensor] cuSPARSELt INT8 sparse: NOT SUPPORTED on this GPU/version "
                "(SM89/v0.8 only supports FP16/BF16)\n");
    }
    return g_sparse_int8_supported;
}

static SparseMatmulPlanCache g_sparse_int8_cache = { .valid = 0 };

int sparse_tensor_create_int8(
    const int8_t* d_dense,
    int64_t rows,
    int64_t cols,
    SparseTensorInternal* out_sparse
) {
    if (!cusparselt_available()) return -1;
    if (!d_dense || !out_sparse) return -2;

    /* Zero the struct early so destructor won't free garbage on error */
    memset(out_sparse, 0, sizeof(SparseTensorInternal));

    /* Early check: is INT8 sparse supported at all? */
    if (!sparse_int8_check_supported()) return -100;  /* Not supported */

    if (rows % 16 != 0 || cols % 16 != 0) {
        fprintf(stderr, "[viva_tensor] SparseTensor INT8: requires dims multiples of 16 (got %ldx%ld)\n",
                rows, cols);
        return -3;
    }

    cusparseStatus_t status;
    int64_t ld = cols;
    size_t dense_size = rows * cols * sizeof(int8_t);

    memset(out_sparse, 0, sizeof(SparseTensorInternal));
    out_sparse->rows = rows;
    out_sparse->cols = cols;
    out_sparse->dtype = SPLT_CUDA_R_8I;

    cusparseLtMatDescriptor_t* matA = SPARSE_MAT_DESCR(out_sparse);

    status = g_cusparseLtStructuredDescriptorInit(
        &g_cusparselt_handle, matA,
        rows, cols, ld, 16, SPLT_CUDA_R_8I,
        CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) return -4;

    void* d_pruned;
    if (g_cuda_malloc(&d_pruned, dense_size) != 0) {
        g_cusparseLtMatDescriptorDestroy(matA);
        return -5;
    }
    g_cuda_memcpy(d_pruned, d_dense, dense_size, cudaMemcpyDeviceToDevice);

    if (g_cusparseLtSpMMAPrune2) {
        status = g_cusparseLtSpMMAPrune2(&g_cusparselt_handle, matA,
            1, CUSPARSE_OPERATION_NON_TRANSPOSE, d_pruned, d_pruned,
            CUSPARSELT_PRUNE_SPMMA_STRIP, NULL);
    } else {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -6;  /* v1 prune needs matmul desc which fails for INT8 */
    }
    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -6;
    }

    size_t compress_buffer_size = 0;
    if (g_cusparseLtSpMMACompressedSize2) {
        status = g_cusparseLtSpMMACompressedSize2(&g_cusparselt_handle, matA,
            &out_sparse->compressed_size, &compress_buffer_size);
    } else {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -7;
    }
    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -7;
    }

    if (g_cuda_malloc(&out_sparse->d_compressed, out_sparse->compressed_size) != 0) {
        g_cuda_free(d_pruned);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -8;
    }

    void* d_compress_buffer = NULL;
    if (compress_buffer_size > 0) {
        if (g_cuda_malloc(&d_compress_buffer, compress_buffer_size) != 0) {
            g_cuda_free(out_sparse->d_compressed);
            g_cuda_free(d_pruned);
            g_cusparseLtMatDescriptorDestroy(matA);
            return -9;
        }
    }

    if (g_cusparseLtSpMMACompress2) {
        if (g_cuda_sync) g_cuda_sync();
        status = g_cusparseLtSpMMACompress2(&g_cusparselt_handle, matA,
            1, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_pruned, out_sparse->d_compressed, d_compress_buffer, NULL);
    } else {
        status = CUSPARSE_STATUS_NOT_SUPPORTED;
    }

    if (d_compress_buffer) g_cuda_free(d_compress_buffer);
    g_cuda_free(d_pruned);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        g_cuda_free(out_sparse->d_compressed);
        g_cusparseLtMatDescriptorDestroy(matA);
        return -10;
    }

    g_cuda_sync();
    fprintf(stderr, "[viva_tensor] SparseTensor INT8 created: %ldx%ld -> %zu bytes (%.1f%% of dense)\n",
            rows, cols, out_sparse->compressed_size,
            100.0 * out_sparse->compressed_size / dense_size);
    return 0;
}

static int sparse_get_or_create_plan_int8(
    SparseMatmulPlanCache* cache, SparseTensorInternal* sparse, int64_t N,
    const void* d_A_compressed, const void* d_B, void* d_C
) {
    int64_t M = sparse->rows, K = sparse->cols;

    if (cache->valid && cache->M == M && cache->N == N && cache->K == K) return 0;

    sparse_cache_invalidate(cache);
    cusparseStatus_t status;
    cache->M = M; cache->N = N; cache->K = K;

    status = g_cusparseLtStructuredDescriptorInit(&g_cusparselt_handle, &cache->matA_desc,
        M, K, K, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != 0) return -1;

    status = g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &cache->matB_desc,
        K, N, N, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW);
    if (status != 0) { g_cusparseLtMatDescriptorDestroy(&cache->matA_desc); return -2; }

    status = g_cusparseLtDenseDescriptorInit(&g_cusparselt_handle, &cache->matC_desc,
        M, N, N, 16, SPLT_CUDA_R_8I, CUSPARSE_ORDER_ROW);
    if (status != 0) {
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        return -2;
    }

    cache->compute_is_32f = 0;
    memset(&cache->matmul_desc, 0, sizeof(cache->matmul_desc));
    memset(&cache->alg_sel, 0, sizeof(cache->alg_sel));
    memset(&cache->plan, 0, sizeof(cache->plan));

    status = g_cusparseLtMatmulDescriptorInit(&g_cusparselt_handle, &cache->matmul_desc,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &cache->matA_desc, &cache->matB_desc, &cache->matC_desc, &cache->matC_desc,
        CUSPARSE_COMPUTE_32I);
    if (status != 0) {
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -3;
    }

    status = g_cusparseLtMatmulAlgSelectionInit(&g_cusparselt_handle, &cache->alg_sel,
        &cache->matmul_desc, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != 0) {
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -4;
    }

    if (g_cusparseLtMatmulAlgSetAttribute) {
        int search_iters = 20;
        g_cusparseLtMatmulAlgSetAttribute(&g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_SEARCH_ITERATIONS, &search_iters, sizeof(search_iters));
    }

    status = g_cusparseLtMatmulPlanInit(&g_cusparselt_handle, &cache->plan,
        &cache->matmul_desc, &cache->alg_sel);
    if (status != 0) {
        if (g_cusparseLtMatmulAlgSelectionDestroy) g_cusparseLtMatmulAlgSelectionDestroy(&cache->alg_sel);
        g_cusparseLtMatDescriptorDestroy(&cache->matA_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matB_desc);
        g_cusparseLtMatDescriptorDestroy(&cache->matC_desc);
        return -4;
    }

    if (g_cuda_sync) g_cuda_sync();

    if (d_A_compressed && d_B && d_C) {
        float alpha_f = 1.0f, beta_f = 0.0f;
        status = g_cusparseLtMatmulSearch(&g_cusparselt_handle, &cache->plan,
            &alpha_f, d_A_compressed, d_B, &beta_f, d_C, d_C,
            g_sparselt_workspace, NULL, 0);
        if (status != 0)
            fprintf(stderr, "[viva_tensor] sparse_plan_int8: MatmulSearch failed: %d\n", status);
    }

    if (g_cusparseLtMatmulAlgGetAttribute) {
        int alg_id = -1, max_alg_id = -1;
        g_cusparseLtMatmulAlgGetAttribute(&g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id));
        g_cusparseLtMatmulAlgGetAttribute(&g_cusparselt_handle, &cache->alg_sel,
            CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, &max_alg_id, sizeof(max_alg_id));
        fprintf(stderr, "[viva_tensor] sparse_plan_int8: ready %ldx%ldx%ld (alg %d/%d)\n",
                M, N, K, alg_id, max_alg_id);
    }
    cache->valid = 1;
    return 0;
}

int sparse_matmul_int8(
    SparseTensorInternal* sparse, const int8_t* d_B, int8_t* d_C,
    int64_t N, float alpha, float beta
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;
    if (N % 16 != 0) return -3;

    int rc = sparse_get_or_create_plan_int8(&g_sparse_int8_cache, sparse, N,
        sparse->d_compressed, d_B, d_C);
    if (rc != 0) return rc;

    float alpha_f = alpha, beta_f = beta;
    cusparseStatus_t status = g_cusparseLtMatmul(&g_cusparselt_handle,
        &g_sparse_int8_cache.plan, &alpha_f, sparse->d_compressed, d_B,
        &beta_f, d_C, d_C, g_sparselt_workspace, NULL, 0);
    return (status == 0) ? 0 : -8;
}

int sparse_matmul_int8_bench(
    SparseTensorInternal* sparse, const int8_t* d_B, int8_t* d_C,
    int64_t N, int iters
) {
    if (!cusparselt_available()) return -1;
    if (!sparse || !d_B || !d_C) return -2;

    int rc = sparse_get_or_create_plan_int8(&g_sparse_int8_cache, sparse, N,
        sparse->d_compressed, d_B, d_C);
    if (rc != 0) return rc;

    float alpha_f = 1.0f, beta_f = 0.0f;
    for (int i = 0; i < iters; i++) {
        cusparseStatus_t status = g_cusparseLtMatmul(&g_cusparselt_handle,
            &g_sparse_int8_cache.plan, &alpha_f, sparse->d_compressed, d_B,
            &beta_f, d_C, d_C, g_sparselt_workspace, NULL, 0);
        if (status != 0) return -8;
    }
    return 0;
}

#else /* _WIN32 */

/* Windows stubs */

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

int cusparselt_init(void) { return 0; }
int cusparselt_available(void) { return 0; }

int sparse_tensor_create_fp16(
    const uint16_t* d_dense, int64_t rows, int64_t cols,
    SparseTensorInternal* out_sparse
) {
    (void)d_dense; (void)rows; (void)cols; (void)out_sparse;
    return -1;
}

void sparse_tensor_free(SparseTensorInternal* sparse) {
    (void)sparse;
}

int sparse_matmul_fp16(
    SparseTensorInternal* sparse, const uint16_t* d_B, uint16_t* d_C,
    int64_t N, float alpha, float beta
) {
    (void)sparse; (void)d_B; (void)d_C; (void)N; (void)alpha; (void)beta;
    return -1;
}

int sparse_matmul_fp16_bench(
    SparseTensorInternal* sparse, const uint16_t* d_B, uint16_t* d_C,
    int64_t N, int iters
) {
    (void)sparse; (void)d_B; (void)d_C; (void)N; (void)iters;
    return -1;
}

int sparse_tensor_create_int8(
    const int8_t* d_dense, int64_t rows, int64_t cols,
    SparseTensorInternal* out_sparse
) {
    (void)d_dense; (void)rows; (void)cols; (void)out_sparse;
    return -1;
}

int sparse_matmul_int8(
    SparseTensorInternal* sparse, const int8_t* d_B, int8_t* d_C,
    int64_t N, float alpha, float beta
) {
    (void)sparse; (void)d_B; (void)d_C; (void)N; (void)alpha; (void)beta;
    return -1;
}

int sparse_matmul_int8_bench(
    SparseTensorInternal* sparse, const int8_t* d_B, int8_t* d_C,
    int64_t N, int iters
) {
    (void)sparse; (void)d_B; (void)d_C; (void)N; (void)iters;
    return -1;
}

#endif /* _WIN32 */
