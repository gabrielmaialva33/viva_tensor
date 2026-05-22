/**
 * CUDA Graphs vs loop launch — quantifies per-kernel launch overhead.
 *
 * Two paths run N identical kernels:
 *   1) `_loop` — N cudaLaunchKernel calls (status quo for any NIF that
 *                dispatches kernels one-at-a-time from BEAM)
 *   2) `_graph`— same N kernels captured into a cudaGraph then replayed
 *                via cudaGraphLaunch (overhead paid once, then replays
 *                are nearly free)
 *
 * We use a small `axpy`-style kernel (one fused mul-add per element) so
 * the work itself is short — that makes the launch overhead the dominant
 * cost and the graph speedup obvious. For a real production NIF, the same
 * pattern wraps the GEMM call chain (matmul + bias + activation + ...).
 *
 * CUTLASS GEMM doesn't compose cleanly with stream capture (it issues
 * cudaGetDeviceProperties etc. during the kernel call, which makes the
 * capture invalid). The result here generalises: any kernel that fits the
 * graph model gets the same overhead reduction.
 */

#include <cuda_runtime.h>

extern "C" {

__global__ static void axpy_kernel(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

static int launch_axpy(cudaStream_t stream, float *d_y, const float *d_x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    axpy_kernel<<<grid, block, 0, stream>>>(d_y, d_x, 2.0f, n);
    return (cudaPeekAtLastError() == cudaSuccess) ? 0 : -1;
}

/** Path 1: N kernel launches as N separate stream submits.
 *  Returns elapsed microseconds (kernel-only via CUDA events). */
int cuda_axpy_loop_bench(int n, int iters) {
    if (n <= 0 || iters <= 0)
        return -10;

    float *d_x = nullptr, *d_y = nullptr;
    if (cudaMalloc((void **)&d_x, n * sizeof(float)) != cudaSuccess)
        return -11;
    if (cudaMalloc((void **)&d_y, n * sizeof(float)) != cudaSuccess) {
        cudaFree(d_x);
        return -12;
    }
    cudaMemset(d_x, 0x3F, n * sizeof(float)); /* fill with ~0.74 */
    cudaMemset(d_y, 0, n * sizeof(float));

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cudaFree(d_x);
        cudaFree(d_y);
        return -13;
    }

    /* Warmup */
    launch_axpy(stream, d_y, d_x, n);
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    for (int i = 0; i < iters; ++i) {
        launch_axpy(stream, d_y, d_x, n);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_y);

    return (int)(elapsed_ms * 1000.0f);
}

/** Path 2: N kernel launches captured into a cudaGraph + replayed. */
int cuda_axpy_graph_bench(int n, int iters) {
    if (n <= 0 || iters <= 0)
        return -10;

    float *d_x = nullptr, *d_y = nullptr;
    if (cudaMalloc((void **)&d_x, n * sizeof(float)) != cudaSuccess)
        return -11;
    if (cudaMalloc((void **)&d_y, n * sizeof(float)) != cudaSuccess) {
        cudaFree(d_x);
        return -12;
    }
    cudaMemset(d_x, 0x3F, n * sizeof(float));
    cudaMemset(d_y, 0, n * sizeof(float));

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        cudaFree(d_x);
        cudaFree(d_y);
        return -13;
    }

    /* Warmup */
    launch_axpy(stream, d_y, d_x, n);
    cudaStreamSynchronize(stream);

    /* Capture one launch and replay it `iters` times. */
    cudaGraph_t graph;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        cudaStreamDestroy(stream);
        cudaFree(d_x);
        cudaFree(d_y);
        return -16;
    }
    launch_axpy(stream, d_y, d_x, n);
    cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
    if (end_err != cudaSuccess) {
        cudaStreamDestroy(stream);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1700 - (int)end_err;
    }

    cudaGraphExec_t exec;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_x);
        cudaFree(d_y);
        return -18;
    }

    /* Warmup the graph launcher */
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    for (int i = 0; i < iters; ++i) {
        cudaGraphLaunch(exec, stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_y);

    return (int)(elapsed_ms * 1000.0f);
}

} /* extern "C" */
