/**
 * nif_softmax.c - CPU SIMD scaffolding for softmax / layer_norm / gelu_exact
 *
 * This file lands the NIF *plumbing* for three new f64 operations:
 *
 *   - nt_softmax_axis(ref, axis)
 *     Compute softmax along an arbitrary axis using the log-sum-exp trick:
 *         shifted = x - max(x, axis)
 *         result  = exp(shifted) / sum(exp(shifted), axis)
 *     Returns a new contiguous NativeTensor of the same shape, f64.
 *
 *   - nt_layer_norm(ref, scale_ref, bias_ref, eps)
 *     Standard LayerNorm along the LAST dimension:
 *         mean = mean(x, last)
 *         var  = mean((x - mean)^2, last)
 *         y    = (x - mean) / sqrt(var + eps) * scale + bias
 *     scale and bias are 1D tensors with size == input.shape[-1].
 *
 *   - nt_gelu_exact(ref)
 *     Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
 *     Uses erf() from <math.h>. Element-wise, contiguous output.
 *
 * IMPORTANT — STATUS OF THIS FILE
 * --------------------------------
 * The kernel bodies are intentionally STUBBED. They parse arguments and
 * validate shapes (so the function table and ABI are stable), then return
 * make_error(env, "not_implemented"). The Gleam side already gates on
 * zig_is_loaded() so this is consistent with the "NIF absent" fallback
 * the tests exercise.
 *
 * A future implementer should:
 *   1. Replace each stubbed body with the algorithm described above.
 *   2. Use alloc_tensor / alloc_tensor_uninit to allocate outputs.
 *   3. The math is pure C using <math.h> (exp, log, erf, sqrt). No SIMD
 *      intrinsics required — the compiler auto-vectorizes these loops
 *      cleanly at -O3. Optional: emit @Vector vt_simd_* helpers and call
 *      them from here, mirroring vt_simd_exp / vt_simd_relu.
 *   4. CUDA is NOT in scope for this round. A future ct_softmax_axis /
 *      ct_layer_norm / ct_gelu_exact may live alongside the CudaTensor16
 *      kernels in nif_cuda_fp16.c.
 *
 * Tests in test/native_ops_test.gleam tolerate both "NIF absent" (the
 * Gleam wrapper returns Error("nif_not_loaded")) and "NIF loaded but stub"
 * (this file returns Error("not_implemented")). The round-trip tests skip
 * cleanly when zig_is_loaded() is false.
 */

#include "viva_nif.h"

/* =========================================================================
 * nt_softmax_axis(ref, axis) -> NativeTensorRef
 *
 * Pseudocode for the future implementer:
 *
 *   axis = normalize_axis(axis, ndim);
 *   axis_size  = shape[axis];
 *   inner_size = product(shape[axis+1..]);
 *   outer_size = size / (axis_size * inner_size);
 *
 *   for outer in 0..outer_size:
 *     for inner in 0..inner_size:
 *       // 1. find max along this slice
 *       m = -INFINITY;
 *       for i in 0..axis_size:
 *         idx = (outer * axis_size + i) * inner_size + inner;
 *         if (x[idx] > m) m = x[idx];
 *       // 2. compute shifted exp + sum
 *       s = 0.0;
 *       for i in 0..axis_size:
 *         idx = (outer * axis_size + i) * inner_size + inner;
 *         e = exp(x[idx] - m);
 *         y[idx] = e;
 *         s += e;
 *       // 3. normalize
 *       inv_s = 1.0 / s;
 *       for i in 0..axis_size:
 *         idx = (outer * axis_size + i) * inner_size + inner;
 *         y[idx] *= inv_s;
 * ========================================================================= */
ERL_NIF_TERM nt_softmax_axis(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *t = get_tensor(env, argv[0]);
    if (!t)
        return make_error(env, "invalid_tensor");

    int axis = 0;
    if (!enif_get_int(env, argv[1], &axis)) {
        return make_error(env, "invalid_axis");
    }

    /* Normalize negative axis. */
    if (axis < 0)
        axis += t->ndim;
    if (axis < 0 || axis >= t->ndim) {
        return make_error(env, "axis_out_of_range");
    }

    /* TODO: implement the log-sum-exp softmax. Until then, return a clean
   * not_implemented error so callers can fall back gracefully. */
    return make_error(env, "not_implemented");
}

/* =========================================================================
 * nt_layer_norm(ref, scale_ref, bias_ref, eps) -> NativeTensorRef
 *
 * Pseudocode for the future implementer:
 *
 *   last = shape[ndim - 1];
 *   rows = size / last;                      // product of all leading dims
 *   for r in 0..rows:
 *     base = r * last;
 *     // mean
 *     m = 0.0;
 *     for c in 0..last: m += x[base + c];
 *     m /= last;
 *     // variance (single-pass two-sum is fine; numerical stability not
 *     // critical at f64 for typical LayerNorm scales)
 *     v = 0.0;
 *     for c in 0..last:
 *       d = x[base + c] - m;
 *       v += d * d;
 *     v /= last;
 *     inv = 1.0 / sqrt(v + eps);
 *     // affine
 *     for c in 0..last:
 *       y[base + c] = (x[base + c] - m) * inv * scale[c] + bias[c];
 * ========================================================================= */
ERL_NIF_TERM nt_layer_norm(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *x = get_tensor(env, argv[0]);
    if (!x)
        return make_error(env, "invalid_tensor");
    NativeTensor *scale = get_tensor(env, argv[1]);
    if (!scale)
        return make_error(env, "invalid_scale");
    NativeTensor *bias = get_tensor(env, argv[2]);
    if (!bias)
        return make_error(env, "invalid_bias");

    int eps_ok = 0;
    double eps = get_number(env, argv[3], &eps_ok);
    if (!eps_ok)
        return make_error(env, "invalid_eps");
    (void)eps;

    /* Validate shapes: scale and bias must be 1D matching x's last dim. */
    if (x->ndim < 1)
        return make_error(env, "rank_too_low");
    int last = x->shape[x->ndim - 1];
    if (scale->ndim != 1 || scale->shape[0] != last) {
        return make_error(env, "scale_shape_mismatch");
    }
    if (bias->ndim != 1 || bias->shape[0] != last) {
        return make_error(env, "bias_shape_mismatch");
    }

    /* TODO: implement standard LayerNorm. Until then, not_implemented. */
    return make_error(env, "not_implemented");
}

/* =========================================================================
 * nt_gelu_exact(ref) -> NativeTensorRef
 *
 * y[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
 *
 * INV_SQRT2 = 0.7071067811865475 (= 1 / sqrt(2))
 *
 * Pseudocode:
 *   for i in 0..size:
 *     y[i] = 0.5 * x[i] * (1.0 + erf(x[i] * INV_SQRT2));
 * ========================================================================= */
ERL_NIF_TERM nt_gelu_exact(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    NativeTensor *t = get_tensor(env, argv[0]);
    if (!t)
        return make_error(env, "invalid_tensor");

    /* TODO: implement exact GELU using erf(). Until then, not_implemented. */
    return make_error(env, "not_implemented");
}
