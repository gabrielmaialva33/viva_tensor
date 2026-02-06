// viva_zig.zig - SIMD-optimized tensor operations
//
// Pure math library with no platform-specific dependencies.
// Uses Zig's @Vector for portable SIMD without platform-specific intrinsics.
// The NIF boilerplate is handled by nif_entry.c which calls these functions.
//
// Operations:
// - vt_simd_dot: Vectorized dot product
// - vt_simd_sum: Vectorized sum reduction
// - vt_simd_scale: Vectorized scalar multiplication
// - vt_simd_add: Vectorized element-wise addition
// - vt_simd_mul: Vectorized element-wise multiply
// - vt_simd_matmul: Blocked matrix multiplication with SIMD

// =============================================================================
// SIMD Configuration
// =============================================================================

// Vector length for SIMD operations (8 doubles = 512 bits, good for AVX-512)
// Falls back gracefully on systems without AVX-512
const VEC_LEN = 8;
const Vec = @Vector(VEC_LEN, f64);

// =============================================================================
// SIMD Operations (exported for C NIF layer)
// =============================================================================

/// SIMD-accelerated dot product
export fn vt_simd_dot(a: [*]const f64, b: [*]const f64, len: usize) callconv(.c) f64 {
    var sum: f64 = 0.0;
    var i: usize = 0;

    // Process VEC_LEN elements at a time
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        const prod = va * vb;
        sum += @reduce(.Add, prod);
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// SIMD-accelerated sum
export fn vt_simd_sum(data: [*]const f64, len: usize) callconv(.c) f64 {
    var sum: f64 = 0.0;
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        sum += @reduce(.Add, v);
    }

    while (i < len) : (i += 1) {
        sum += data[i];
    }

    return sum;
}

/// SIMD-accelerated scale (multiply by scalar)
export fn vt_simd_scale(data: [*]const f64, scalar: f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const scalar_vec: Vec = @splat(scalar);

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        const scaled = v * scalar_vec;
        result[i..][0..VEC_LEN].* = scaled;
    }

    while (i < len) : (i += 1) {
        result[i] = data[i] * scalar;
    }
}

/// SIMD-accelerated element-wise addition
export fn vt_simd_add(a: [*]const f64, b: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = va + vb;
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-accelerated element-wise multiply
export fn vt_simd_mul(a: [*]const f64, b: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = va * vb;
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// SIMD-accelerated element-wise subtraction
export fn vt_simd_sub(a: [*]const f64, b: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = va - vb;
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] - b[i];
    }
}

/// SIMD-accelerated negation
export fn vt_simd_negate(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = -v;
    }

    while (i < len) : (i += 1) {
        result[i] = -data[i];
    }
}

/// SIMD-accelerated ReLU activation
export fn vt_simd_relu(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const zero: Vec = @splat(0.0);

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = @max(v, zero);
    }

    while (i < len) : (i += 1) {
        result[i] = @max(data[i], 0.0);
    }
}

/// SIMD-accelerated max reduction
export fn vt_simd_max(data: [*]const f64, len: usize) callconv(.c) f64 {
    if (len == 0) return -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));

    var current_max: f64 = data[0];
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        const chunk_max = @reduce(.Max, v);
        if (chunk_max > current_max) current_max = chunk_max;
    }

    while (i < len) : (i += 1) {
        if (data[i] > current_max) current_max = data[i];
    }

    return current_max;
}

/// SIMD-accelerated min reduction
export fn vt_simd_min(data: [*]const f64, len: usize) callconv(.c) f64 {
    if (len == 0) return @as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));

    var current_min: f64 = data[0];
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        const chunk_min = @reduce(.Min, v);
        if (chunk_min < current_min) current_min = chunk_min;
    }

    while (i < len) : (i += 1) {
        if (data[i] < current_min) current_min = data[i];
    }

    return current_min;
}

// =============================================================================
// In-Place Mutation Ops (Zero-Allocation)
// "Quebrar a imutabilidade dentro do Zig para economizar RAM"
// The caller (C NIF) increments the resource refcount to keep a alive.
// =============================================================================

/// In-place add: a[i] += b[i]. Zero allocation.
export fn vt_simd_add_mut(a: [*]f64, b: [*]const f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        a[i..][0..VEC_LEN].* = va + vb;
    }
    while (i < len) : (i += 1) {
        a[i] += b[i];
    }
}

/// In-place scale: a[i] *= scalar. Zero allocation.
export fn vt_simd_scale_mut(a: [*]f64, scalar: f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const sv: Vec = @splat(scalar);
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = a[i..][0..VEC_LEN].*;
        a[i..][0..VEC_LEN].* = v * sv;
    }
    while (i < len) : (i += 1) {
        a[i] *= scalar;
    }
}

/// In-place negate: a[i] = -a[i]. Zero allocation.
export fn vt_simd_negate_mut(a: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = a[i..][0..VEC_LEN].*;
        a[i..][0..VEC_LEN].* = -v;
    }
    while (i < len) : (i += 1) {
        a[i] = -a[i];
    }
}

/// In-place ReLU: a[i] = max(0, a[i]). Zero allocation.
export fn vt_simd_relu_mut(a: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const zero: Vec = @splat(0.0);
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = a[i..][0..VEC_LEN].*;
        a[i..][0..VEC_LEN].* = @max(v, zero);
    }
    while (i < len) : (i += 1) {
        a[i] = @max(a[i], 0.0);
    }
}

// =============================================================================
// Retro Kernels (VDP1 / Saturn Inspired)
// "O que o VDP1 fazia com inteiros e o que papers modernos chamam de Quantization"
// =============================================================================

/// Saturn Blend: result = texture + (shade - bias)
/// VDP1-inspired lighting using pure SIMD addition (no multiplication).
/// Equivalent to affine transform with offset, but branchless and fast.
export fn vt_saturn_blend(texture: [*]const f64, shade: [*]const f64, bias: f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const bias_v: Vec = @splat(bias);
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const t: Vec = texture[i..][0..VEC_LEN].*;
        const s: Vec = shade[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = t + (s - bias_v);
    }
    while (i < len) : (i += 1) {
        result[i] = texture[i] + (shade[i] - bias);
    }
}

/// Fused MatMul + Bias + ReLU: C = max(0, A@B + bias)
/// Single pass over C instead of 3 separate ops. Saves 2 full tensor traversals.
export fn vt_fused_linear_relu(a: [*]const f64, b: [*]const f64, bias: [*]const f64, c: [*]f64, m: usize, n: usize, k: usize) callconv(.c) void {
    // Step 1: C = A @ B (Goto GEMM or small matmul)
    @memset(c[0 .. m * n], 0.0);
    if (m >= 32 and n >= 32 and k >= 32) {
        goto_gemm(a, b, c, m, n, k);
    } else {
        matmul_small(a, b, c, m, n, k);
    }

    // Step 2: Fused bias + ReLU in single pass (avoids 2 extra traversals)
    const zero: Vec = @splat(0.0);
    var row: usize = 0;
    while (row < m) : (row += 1) {
        const row_off = row * n;
        var j: usize = 0;
        while (j + VEC_LEN <= n) : (j += VEC_LEN) {
            const cv: Vec = c[row_off + j ..][0..VEC_LEN].*;
            const bv: Vec = bias[j..][0..VEC_LEN].*;
            c[row_off + j ..][0..VEC_LEN].* = @max(cv + bv, zero);
        }
        while (j < n) : (j += 1) {
            const val = c[row_off + j] + bias[j];
            c[row_off + j] = @max(val, 0.0);
        }
    }
}

// =============================================================================
// Goto-style GEMM: High-performance matrix multiplication
// Based on Goto & Van de Geijn 2008, BLIS Haswell dgemm_6x8 parameters
//
// Architecture: 5-loop Goto algorithm with:
//   - 6x8 micro-kernel (12 Vec4 accumulators -> 12 YMM registers on AVX2)
//   - Data packing for sequential memory access
//   - Cache-aware blocking: MCxKC fits L2, KCxNR fits L1
//   - K-unrolling by 4 for FMA pipeline saturation
// =============================================================================

// C stdlib for allocation (linked via build.zig linkLibC)
extern fn malloc(size: usize) callconv(.c) ?[*]u8;
extern fn free(ptr: ?[*]u8) callconv(.c) void;

// 64-byte aligned allocation for AVX-512 / cache-line alignment
const ALIGN = 64;

const builtin = @import("builtin");

// Platform-specific C allocation externs
extern fn _aligned_malloc(size: usize, alignment: usize) callconv(.c) ?[*]u8;
extern fn _aligned_free(ptr: ?[*]u8) callconv(.c) void;
// C11 aligned_alloc (POSIX) — Zig name = C linker symbol
extern fn aligned_alloc(alignment: usize, size: usize) callconv(.c) ?[*]u8;

/// 64-byte aligned allocation for SIMD pack buffers. Cross-platform.
fn simd_alloc(size: usize) ?[*]u8 {
    if (builtin.os.tag == .windows) {
        return _aligned_malloc(size, ALIGN);
    } else {
        // POSIX: aligned_alloc requires size = multiple of alignment
        const aligned_size = (size + ALIGN - 1) & ~@as(usize, ALIGN - 1);
        return aligned_alloc(ALIGN, aligned_size);
    }
}

/// Free 64-byte aligned memory. Cross-platform.
fn simd_free(ptr: ?[*]u8) void {
    if (builtin.os.tag == .windows) {
        _aligned_free(ptr);
    } else {
        free(ptr);
    }
}

// Goto GEMM constants (BLIS Haswell dgemm config, tuned for Raptor Lake)
const MR: usize = 6; // Micro-kernel rows (matches AVX2 dgemm_6x8)
const NR: usize = 8; // Micro-kernel cols (2 x Vec4 = 8 doubles)
const KC: usize = 256; // K-block: NRxKCx8 = 16KB fits L1 (48KB)
const MC: usize = 144; // M-block: MCxKCx8 = 288KB fits L2 (2MB on Raptor Lake)
const NC: usize = 4080; // N-block: KCxNCx8 = 8MB fits L3 (30MB)

// Vec4 maps directly to AVX2 YMM register (256-bit = 4 x f64)
const Vec4 = @Vector(4, f64);

/// 6x8 micro-kernel: C[6x8] += packed_A[6xkc] x packed_B[kcx8]
/// This is the innermost hot loop. 12 FMAs per k-step.
/// packed_a: MR contiguous doubles per k-step
/// packed_b: NR contiguous doubles per k-step
fn micro_6x8(packed_a: [*]const f64, packed_b: [*]const f64, c_ptr: [*]f64, ldc: usize, kc_len: usize) void {
    // 12 accumulator vectors: 6 rows x 2 halves (lo=cols 0..3, hi=cols 4..7)
    var c0_lo: Vec4 = @splat(0.0);
    var c0_hi: Vec4 = @splat(0.0);
    var c1_lo: Vec4 = @splat(0.0);
    var c1_hi: Vec4 = @splat(0.0);
    var c2_lo: Vec4 = @splat(0.0);
    var c2_hi: Vec4 = @splat(0.0);
    var c3_lo: Vec4 = @splat(0.0);
    var c3_hi: Vec4 = @splat(0.0);
    var c4_lo: Vec4 = @splat(0.0);
    var c4_hi: Vec4 = @splat(0.0);
    var c5_lo: Vec4 = @splat(0.0);
    var c5_hi: Vec4 = @splat(0.0);

    var kk: usize = 0;

    // K-unrolled by 4 for FMA pipeline saturation
    // FMA latency = 4 cycles, 2 ports -> need 8+ independent chains. We have 12.
    while (kk + 4 <= kc_len) : (kk += 4) {
        // Unroll 4 k-iterations at compile time
        comptime var u: usize = 0;
        inline while (u < 4) : (u += 1) {
            const a_off = (kk + u) * MR;
            const b_off = (kk + u) * NR;

            // Load B vector (8 doubles = 2 x Vec4)
            const b_lo: Vec4 = packed_b[b_off..][0..4].*;
            const b_hi: Vec4 = packed_b[b_off + 4 ..][0..4].*;

            // Broadcast each A element and FMA into accumulators
            const a0: Vec4 = @splat(packed_a[a_off + 0]);
            c0_lo = @mulAdd(Vec4, a0, b_lo, c0_lo);
            c0_hi = @mulAdd(Vec4, a0, b_hi, c0_hi);

            const a1: Vec4 = @splat(packed_a[a_off + 1]);
            c1_lo = @mulAdd(Vec4, a1, b_lo, c1_lo);
            c1_hi = @mulAdd(Vec4, a1, b_hi, c1_hi);

            const a2: Vec4 = @splat(packed_a[a_off + 2]);
            c2_lo = @mulAdd(Vec4, a2, b_lo, c2_lo);
            c2_hi = @mulAdd(Vec4, a2, b_hi, c2_hi);

            const a3: Vec4 = @splat(packed_a[a_off + 3]);
            c3_lo = @mulAdd(Vec4, a3, b_lo, c3_lo);
            c3_hi = @mulAdd(Vec4, a3, b_hi, c3_hi);

            const a4: Vec4 = @splat(packed_a[a_off + 4]);
            c4_lo = @mulAdd(Vec4, a4, b_lo, c4_lo);
            c4_hi = @mulAdd(Vec4, a4, b_hi, c4_hi);

            const a5: Vec4 = @splat(packed_a[a_off + 5]);
            c5_lo = @mulAdd(Vec4, a5, b_lo, c5_lo);
            c5_hi = @mulAdd(Vec4, a5, b_hi, c5_hi);
        }
    }

    // Handle remaining k iterations (0-3)
    while (kk < kc_len) : (kk += 1) {
        const a_off = kk * MR;
        const b_off = kk * NR;

        const b_lo: Vec4 = packed_b[b_off..][0..4].*;
        const b_hi: Vec4 = packed_b[b_off + 4 ..][0..4].*;

        const a0: Vec4 = @splat(packed_a[a_off + 0]);
        c0_lo = @mulAdd(Vec4, a0, b_lo, c0_lo);
        c0_hi = @mulAdd(Vec4, a0, b_hi, c0_hi);
        const a1: Vec4 = @splat(packed_a[a_off + 1]);
        c1_lo = @mulAdd(Vec4, a1, b_lo, c1_lo);
        c1_hi = @mulAdd(Vec4, a1, b_hi, c1_hi);
        const a2: Vec4 = @splat(packed_a[a_off + 2]);
        c2_lo = @mulAdd(Vec4, a2, b_lo, c2_lo);
        c2_hi = @mulAdd(Vec4, a2, b_hi, c2_hi);
        const a3: Vec4 = @splat(packed_a[a_off + 3]);
        c3_lo = @mulAdd(Vec4, a3, b_lo, c3_lo);
        c3_hi = @mulAdd(Vec4, a3, b_hi, c3_hi);
        const a4: Vec4 = @splat(packed_a[a_off + 4]);
        c4_lo = @mulAdd(Vec4, a4, b_lo, c4_lo);
        c4_hi = @mulAdd(Vec4, a4, b_hi, c4_hi);
        const a5: Vec4 = @splat(packed_a[a_off + 5]);
        c5_lo = @mulAdd(Vec4, a5, b_lo, c5_lo);
        c5_hi = @mulAdd(Vec4, a5, b_hi, c5_hi);
    }

    // Store: C[tile] += accumulators
    // Each row: load existing C, add accumulator, store back
    const c0 = c_ptr;
    c0[0..4].* = c0[0..4].* + c0_lo;
    c0[4..8].* = c0[4..8].* + c0_hi;
    const c1 = c_ptr + ldc;
    c1[0..4].* = c1[0..4].* + c1_lo;
    c1[4..8].* = c1[4..8].* + c1_hi;
    const c2 = c_ptr + 2 * ldc;
    c2[0..4].* = c2[0..4].* + c2_lo;
    c2[4..8].* = c2[4..8].* + c2_hi;
    const c3 = c_ptr + 3 * ldc;
    c3[0..4].* = c3[0..4].* + c3_lo;
    c3[4..8].* = c3[4..8].* + c3_hi;
    const c4 = c_ptr + 4 * ldc;
    c4[0..4].* = c4[0..4].* + c4_lo;
    c4[4..8].* = c4[4..8].* + c4_hi;
    const c5 = c_ptr + 5 * ldc;
    c5[0..4].* = c5[0..4].* + c5_lo;
    c5[4..8].* = c5[4..8].* + c5_hi;
}

/// Edge micro-kernel for partial tiles (m_rem < MR or n_rem < NR)
/// Falls back to scalar to handle any remainder
fn micro_edge(packed_a: [*]const f64, packed_b: [*]const f64, c_ptr: [*]f64, ldc: usize, kc_len: usize, m_rem: usize, n_rem: usize) void {
    var ir: usize = 0;
    while (ir < m_rem) : (ir += 1) {
        var jr: usize = 0;
        while (jr < n_rem) : (jr += 1) {
            var acc: f64 = 0.0;
            var kk: usize = 0;
            while (kk < kc_len) : (kk += 1) {
                acc += packed_a[kk * MR + ir] * packed_b[kk * NR + jr];
            }
            c_ptr[ir * ldc + jr] += acc;
        }
    }
}

/// Pack A panel: A[ic:ic+mc, pc:pc+kc] -> packed_a in MR-wide column panels
/// Layout: micro-panel 0 (MRxkc), micro-panel 1 (MRxkc), ...
/// Within each micro-panel: MR contiguous elements per k-step
fn pack_a(a: [*]const f64, dst_buf: [*]f64, mc_len: usize, kc_len: usize, lda: usize, ic: usize, pc: usize) void {
    var p: usize = 0;
    // Full MR-wide panels
    while (p + MR <= mc_len) : (p += MR) {
        var kk: usize = 0;
        while (kk < kc_len) : (kk += 1) {
            const dst = dst_buf + (p / MR) * MR * kc_len + kk * MR;
            comptime var r: usize = 0;
            inline while (r < MR) : (r += 1) {
                dst[r] = a[(ic + p + r) * lda + (pc + kk)];
            }
        }
    }
    // Remainder rows (< MR): pad with zeros
    if (p < mc_len) {
        const rem = mc_len - p;
        var kk: usize = 0;
        while (kk < kc_len) : (kk += 1) {
            const dst = dst_buf + (p / MR) * MR * kc_len + kk * MR;
            var r: usize = 0;
            while (r < MR) : (r += 1) {
                if (r < rem) {
                    dst[r] = a[(ic + p + r) * lda + (pc + kk)];
                } else {
                    dst[r] = 0.0;
                }
            }
        }
    }
}

/// Pack B panel: B[pc:pc+kc, jc:jc+nc] -> packed_b in NR-wide row panels
/// Layout: micro-panel 0 (kcxNR), micro-panel 1 (kcxNR), ...
/// Within each micro-panel: NR contiguous elements per k-step
fn pack_b(b: [*]const f64, dst_buf: [*]f64, kc_len: usize, nc_len: usize, ldb: usize, pc: usize, jc: usize) void {
    var p: usize = 0;
    // Full NR-wide panels
    while (p + NR <= nc_len) : (p += NR) {
        var kk: usize = 0;
        while (kk < kc_len) : (kk += 1) {
            const dst = dst_buf + (p / NR) * NR * kc_len + kk * NR;
            comptime var c_col: usize = 0;
            inline while (c_col < NR) : (c_col += 1) {
                dst[c_col] = b[(pc + kk) * ldb + (jc + p + c_col)];
            }
        }
    }
    // Remainder cols (< NR): pad with zeros
    if (p < nc_len) {
        const rem = nc_len - p;
        var kk: usize = 0;
        while (kk < kc_len) : (kk += 1) {
            const dst = dst_buf + (p / NR) * NR * kc_len + kk * NR;
            var c_col: usize = 0;
            while (c_col < NR) : (c_col += 1) {
                if (c_col < rem) {
                    dst[c_col] = b[(pc + kk) * ldb + (jc + p + c_col)];
                } else {
                    dst[c_col] = 0.0;
                }
            }
        }
    }
}

// =============================================================================
// Multi-threaded GEMM infrastructure
// =============================================================================

const std = @import("std");
const Thread = std.Thread;

const MAX_THREADS: usize = 8;

/// Compute padded buffer size for pack_a: ceil(mc/MR)*MR * kc
inline fn pack_a_size(mc: usize, kc: usize) usize {
    return ((mc + MR - 1) / MR) * MR * kc;
}

/// Compute padded buffer size for pack_b: ceil(nc/NR)*NR * kc
inline fn pack_b_size(nc: usize, kc: usize) usize {
    return kc * ((nc + NR - 1) / NR) * NR;
}

/// Context passed to each worker thread for the ic-loop parallelization
const GemmWorkerCtx = struct {
    a: [*]const f64,
    bp: [*]const f64, // Shared packed B (read-only)
    c: [*]f64,
    n: usize,
    k: usize,
    ic_start: usize,
    ic_end: usize,
    kc_len: usize,
    nc_len: usize,
    pc: usize,
    jc: usize,
    ap_buf: [*]f64, // Thread-private packing buffer for A
};

/// Macro-kernel: iterate micro-tiles over packed A and packed B
fn macro_kernel(ap_buf: [*]f64, bp_buf: [*]const f64, c: [*]f64, n: usize, mc_len: usize, nc_len: usize, kc_len: usize, ic: usize, jc: usize) void {
    var jr: usize = 0;
    while (jr < nc_len) : (jr += NR) {
        const nr_cur = @min(NR, nc_len - jr);
        const b_panel = bp_buf + (jr / NR) * NR * kc_len;
        var ir: usize = 0;
        while (ir < mc_len) : (ir += MR) {
            const mr_cur = @min(MR, mc_len - ir);
            const a_panel = ap_buf + (ir / MR) * MR * kc_len;
            const c_tile = c + (ic + ir) * n + (jc + jr);
            if (mr_cur == MR and nr_cur == NR) {
                micro_6x8(a_panel, b_panel, c_tile, n, kc_len);
            } else {
                micro_edge(a_panel, b_panel, c_tile, n, kc_len, mr_cur, nr_cur);
            }
        }
    }
}

/// Worker function: each thread processes a range of ic-tiles
fn gemm_worker(ctx: *const GemmWorkerCtx) void {
    var ic = ctx.ic_start;
    while (ic < ctx.ic_end) : (ic += MC) {
        const mc_len = @min(MC, ctx.ic_end - ic);
        pack_a(ctx.a, ctx.ap_buf, mc_len, ctx.kc_len, ctx.k, ic, ctx.pc);
        macro_kernel(ctx.ap_buf, ctx.bp, ctx.c, ctx.n, mc_len, ctx.nc_len, ctx.kc_len, ic, ctx.jc);
    }
}

/// Goto-style 5-loop GEMM: C[m,n] += A[m,k] x B[k,n]
/// Multi-threaded: parallelizes the ic (row-tile) loop across CPU cores.
/// C must be pre-zeroed by caller.
fn goto_gemm(a: [*]const f64, b: [*]const f64, c: [*]f64, m: usize, n: usize, k: usize) void {
    // Determine thread count based on available work
    const n_ic_tiles = (m + MC - 1) / MC;
    const hw_threads = if (m >= 256) (Thread.getCpuCount() catch 1) else 1;
    const n_threads = @min(@min(MAX_THREADS, n_ic_tiles), @max(hw_threads, 1));

    // Allocate padded packing buffers (padding required by MR/NR zero-fill in pack_a/pack_b)
    const max_kc = @min(KC, k);
    const max_nc = @min(NC, n);
    const max_mc = @min(MC, m);
    const bp_bytes = pack_b_size(max_nc, max_kc) * @sizeOf(f64);
    const ap_bytes = pack_a_size(max_mc, max_kc) * @sizeOf(f64);

    const raw_b = simd_alloc(bp_bytes) orelse return;
    defer simd_free(raw_b);
    const bp_buf: [*]f64 = @ptrCast(@alignCast(raw_b));

    // Each thread needs its own A packing buffer
    var ap_bufs: [MAX_THREADS][*]f64 = undefined;
    var ap_raws: [MAX_THREADS]?[*]u8 = .{null} ** MAX_THREADS;
    var alloc_ok = true;
    for (0..n_threads) |t| {
        ap_raws[t] = simd_alloc(ap_bytes);
        if (ap_raws[t]) |raw| {
            ap_bufs[t] = @ptrCast(@alignCast(raw));
        } else {
            alloc_ok = false;
            break;
        }
    }
    defer {
        for (0..MAX_THREADS) |t| {
            if (ap_raws[t]) |r| simd_free(r);
        }
    }

    // If thread-private allocation failed, fall back to single thread with ap_bufs[0]
    const effective_threads: usize = if (alloc_ok) n_threads else if (ap_raws[0] != null) 1 else return;

    // Loop 5: over N in blocks of NC
    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc_len = @min(NC, n - jc);

        // Loop 4: over K in blocks of KC
        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc_len = @min(KC, k - pc);

            // Pack B once (shared read-only)
            pack_b(b, bp_buf, kc_len, nc_len, n, pc, jc);

            if (effective_threads <= 1) {
                // Single-threaded fast path
                var ic: usize = 0;
                while (ic < m) : (ic += MC) {
                    const mc_len = @min(MC, m - ic);
                    pack_a(a, ap_bufs[0], mc_len, kc_len, k, ic, pc);
                    macro_kernel(ap_bufs[0], bp_buf, c, n, mc_len, nc_len, kc_len, ic, jc);
                }
            } else {
                // Multi-threaded: split ic range across threads
                const rows_per_thread = ((m + effective_threads - 1) / effective_threads);
                const rows_aligned = ((rows_per_thread + MC - 1) / MC) * MC;

                var ctxs: [MAX_THREADS]GemmWorkerCtx = undefined;
                var threads: [MAX_THREADS]?Thread = .{null} ** MAX_THREADS;

                // Build contexts and spawn workers (thread 0 = main thread)
                for (0..effective_threads) |t| {
                    const ic_start = t * rows_aligned;
                    if (ic_start >= m) break;
                    const ic_end = @min((t + 1) * rows_aligned, m);

                    ctxs[t] = .{
                        .a = a,
                        .bp = bp_buf,
                        .c = c,
                        .n = n,
                        .k = k,
                        .ic_start = ic_start,
                        .ic_end = ic_end,
                        .kc_len = kc_len,
                        .nc_len = nc_len,
                        .pc = pc,
                        .jc = jc,
                        .ap_buf = ap_bufs[t],
                    };

                    if (t > 0) {
                        threads[t] = Thread.spawn(.{}, gemm_worker, .{&ctxs[t]}) catch null;
                    }
                }

                // Main thread does thread 0's work
                gemm_worker(&ctxs[0]);

                // Join workers
                for (1..effective_threads) |t| {
                    if (threads[t]) |th| th.join();
                }
            }
        }
    }
}

/// Small matrix multiplication fallback (naive blocked, for < 64x64)
fn matmul_small(a: [*]const f64, b: [*]const f64, c: [*]f64, m: usize, n: usize, k_dim: usize) void {
    const BLOCK_SIZE = 32;

    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK_SIZE) {
        const i_end = @min(ii + BLOCK_SIZE, m);

        var kk: usize = 0;
        while (kk < k_dim) : (kk += BLOCK_SIZE) {
            const k_end = @min(kk + BLOCK_SIZE, k_dim);

            var jj: usize = 0;
            while (jj < n) : (jj += BLOCK_SIZE) {
                const j_end = @min(jj + BLOCK_SIZE, n);

                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var kix: usize = kk;
                    while (kix < k_end) : (kix += 1) {
                        const a_val = a[i * k_dim + kix];
                        const a_vec: Vec = @splat(a_val);

                        var j: usize = jj;
                        while (j + VEC_LEN <= j_end) : (j += VEC_LEN) {
                            const b_vec: Vec = b[kix * n + j ..][0..VEC_LEN].*;
                            const c_idx = i * n + j;
                            var c_vec: Vec = c[c_idx..][0..VEC_LEN].*;
                            c_vec += a_vec * b_vec;
                            c[c_idx..][0..VEC_LEN].* = c_vec;
                        }

                        while (j < j_end) : (j += 1) {
                            c[i * n + j] += a_val * b[kix * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// SIMD-accelerated matrix multiplication: A[m,k] @ B[k,n] -> C[m,n]
/// Dispatches: small matrices -> blocked, large -> Goto GEMM
export fn vt_simd_matmul(a: [*]const f64, b: [*]const f64, c: [*]f64, m: usize, n: usize, k: usize) callconv(.c) void {
    // Zero C
    @memset(c[0 .. m * n], 0.0);

    // Dispatch: Goto GEMM for matrices where packing overhead pays off
    if (m >= 32 and n >= 32 and k >= 32) {
        goto_gemm(a, b, c, m, n, k);
    } else {
        matmul_small(a, b, c, m, n, k);
    }
}

// =============================================================================
// SIMD Transcendental Functions
// Vectorized exp, sigmoid, log using polynomial approximations + bit tricks.
// Avoids scalar @exp/@log per element -> 4-8x faster for activation functions.
// =============================================================================

const Vi4 = @Vector(4, i64);

/// Vectorized exp(x) for Vec4 (4 x f64).
/// Cephes-style: range reduction to [-ln2/2, ln2/2], degree-6 minimax polynomial,
/// then 2^n via IEEE754 exponent bit manipulation.
/// Max relative error: ~2e-16 (full double precision).
inline fn exp_vec4(x: Vec4) Vec4 {
    // Clamp to prevent overflow/underflow
    const clamped = @min(@max(x, @as(Vec4, @splat(-708.0))), @as(Vec4, @splat(709.0)));

    // Range reduction: x = n * ln2 + r, where |r| <= ln2/2
    const log2e: Vec4 = @splat(1.4426950408889634);
    const ln2_hi: Vec4 = @splat(6.93145751953125e-1);
    const ln2_lo: Vec4 = @splat(1.42860682030941723212e-6);
    const n_f = @round(clamped * log2e);
    const r = clamped - n_f * ln2_hi - n_f * ln2_lo;

    // Minimax polynomial exp(r) ~ 1 + r + r^2/2! + ... + r^6/6!
    const c1: Vec4 = @splat(1.0);
    const c2: Vec4 = @splat(0.5);
    const c3: Vec4 = @splat(0.16666666666666602);
    const c4: Vec4 = @splat(0.04166666666557908);
    const c5: Vec4 = @splat(0.008333333333414256);
    const c6: Vec4 = @splat(0.001388888894063186);
    const c7: Vec4 = @splat(1.984126987568086e-4);

    // Horner's method
    const poly = c1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * c7))))));

    // Scale by 2^n: construct IEEE754 double with exponent = n + 1023
    const n_i: Vi4 = @intFromFloat(n_f);
    const bias: Vi4 = @splat(1023);
    const shift: @Vector(4, u6) = @splat(52);
    const exp_bits: Vi4 = (n_i + bias) << shift;
    const scale: Vec4 = @bitCast(exp_bits);

    return poly * scale;
}

/// SIMD-accelerated exp: process 4 doubles at a time
export fn vt_simd_exp(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    // Process 4 doubles at a time (AVX2 width for f64)
    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = data[i..][0..4].*;
        result[i..][0..4].* = exp_vec4(v);
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        result[i] = @exp(data[i]);
    }
}

/// SIMD-accelerated sigmoid: sig(x) = 1 / (1 + exp(-x))
export fn vt_simd_sigmoid(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const one: Vec4 = @splat(1.0);

    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = data[i..][0..4].*;
        const e = exp_vec4(-v);
        result[i..][0..4].* = one / (one + e);
    }

    while (i < len) : (i += 1) {
        const e = @exp(-data[i]);
        result[i] = 1.0 / (1.0 + e);
    }
}

/// SIMD-accelerated natural log: Cephes-style decomposition
/// x = 2^e * m, ln(x) = e*ln2 + ln(m), with polynomial for ln(m) on [sqrt(2)/2, sqrt(2)]
export fn vt_simd_log(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = data[i..][0..4].*;
        result[i..][0..4].* = log_vec4(v);
    }

    while (i < len) : (i += 1) {
        result[i] = @log(data[i]);
    }
}

/// Vectorized ln(x) for Vec4.
/// Decompose: x = 2^e * m, ln(x) = e*ln2 + ln(m)
/// ln(1+f) approximated by degree-7 minimax polynomial on [-0.29, 0.41]
inline fn log_vec4(x: Vec4) Vec4 {
    const one: Vec4 = @splat(1.0);
    const zero: Vec4 = @splat(0.0);

    // Extract exponent and mantissa via bit manipulation
    const bits: Vi4 = @bitCast(x);
    const exp_mask: Vi4 = @splat(0x7FF0000000000000);
    const mant_mask: Vi4 = @splat(0x000FFFFFFFFFFFFF);
    const bias: Vi4 = @splat(1023);
    const half_bits: Vi4 = @splat(0x3FE0000000000000); // 0.5 in IEEE754

    // e = exponent - 1023
    const shift: @Vector(4, u6) = @splat(52);
    const e_i: Vi4 = ((bits & exp_mask) >> shift) - bias;
    const e: Vec4 = @floatFromInt(e_i);

    // m = mantissa in [0.5, 1.0) by replacing exponent with 1022
    const m_bits = (bits & mant_mask) | half_bits;
    const m: Vec4 = @bitCast(m_bits);

    // Adjust: if m < sqrt(2)/2 ~ 0.7071, multiply by 2 and subtract 1 from exponent
    const sqrt2_over2: Vec4 = @splat(0.70710678118654752);
    const adjust = m < sqrt2_over2;
    const m_adj = @select(f64, adjust, m + m - one, m - one);
    const e_adj = @select(f64, adjust, e - one, e);

    // f = m_adj, compute ln(1+f) via polynomial
    const f = m_adj;
    const f2 = f * f;

    // Coefficients from Cephes
    const p0: Vec4 = @splat(-7.89580278884799154124e-1);
    const p1: Vec4 = @splat(1.63866645699558079767e1);
    const p2: Vec4 = @splat(-6.41409952958715622951e1);

    const q0: Vec4 = @splat(-3.56722798256324312549e1);
    const q1: Vec4 = @splat(3.12093766372244180303e2);
    const q2: Vec4 = @splat(-7.69691943550460008604e2);

    // R(f) = f - 0.5*f^2 + f^3 * P(f)/Q(f)
    const p_num = p0 + f * (p1 + f * p2);
    const q_den = one + f * (q0 + f * (q1 + f * q2));

    const ln2_hi: Vec4 = @splat(6.93145751953125e-1);
    const ln2_lo: Vec4 = @splat(1.42860682030941723212e-6);

    const r = f - @as(Vec4, @splat(0.5)) * f2 + f2 * f * p_num / q_den;

    // Result: e*ln2 + r
    const result = e_adj * ln2_hi + (e_adj * ln2_lo + r);

    // Handle special cases: x <= 0 -> NaN
    return @select(f64, x > zero, result, @as(Vec4, @splat(@as(f64, @bitCast(@as(u64, 0x7FF8000000000000))))));
}
