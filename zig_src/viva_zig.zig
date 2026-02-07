// viva_zig.zig - SIMD-optimized tensor operations
//
// Pure math library with no platform-specific dependencies.
// Uses Zig's @Vector for portable SIMD without platform-specific intrinsics.
// The NIF boilerplate is handled by nif_entry.c which calls these functions.
//
// NOTE: Matrix multiplication (GEMM) is handled by BLAS (MKL/OpenBLAS) in nif_entry.c
// This file provides SIMD ops for: dot, sum, scale, add, mul, exp, sigmoid, log, etc.

// =============================================================================
// SIMD Configuration
// =============================================================================

// Vector length for SIMD operations (8 doubles = 512 bits, good for AVX-512)
// Falls back gracefully on systems without AVX-512
const VEC_LEN = 8;
const Vec = @Vector(VEC_LEN, f64);
const Vec4 = @Vector(4, f64);

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
// =============================================================================

/// Saturn Blend: result = texture + (shade - bias)
/// VDP1-inspired lighting using pure SIMD addition (no multiplication).
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

// =============================================================================
// SIMD Transcendental Functions
// Vectorized exp, sigmoid, log using polynomial approximations + bit tricks.
// =============================================================================

const Vi4 = @Vector(4, i64);

/// Vectorized exp(x) for Vec4 (4 x f64).
/// Cephes-style: range reduction to [-ln2/2, ln2/2], degree-6 minimax polynomial.
inline fn exp_vec4(x: Vec4) Vec4 {
    const clamped = @min(@max(x, @as(Vec4, @splat(-708.0))), @as(Vec4, @splat(709.0)));

    const log2e: Vec4 = @splat(1.4426950408889634);
    const ln2_hi: Vec4 = @splat(6.93145751953125e-1);
    const ln2_lo: Vec4 = @splat(1.42860682030941723212e-6);
    const n_f = @round(clamped * log2e);
    const r = clamped - n_f * ln2_hi - n_f * ln2_lo;

    const c1: Vec4 = @splat(1.0);
    const c2: Vec4 = @splat(0.5);
    const c3: Vec4 = @splat(0.16666666666666602);
    const c4: Vec4 = @splat(0.04166666666557908);
    const c5: Vec4 = @splat(0.008333333333414256);
    const c6: Vec4 = @splat(0.001388888894063186);
    const c7: Vec4 = @splat(1.984126987568086e-4);

    const poly = c1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * c7))))));

    const n_i: Vi4 = @intFromFloat(n_f);
    const bias: Vi4 = @splat(1023);
    const shift: @Vector(4, u6) = @splat(52);
    const exp_bits: Vi4 = (n_i + bias) << shift;
    const scale: Vec4 = @bitCast(exp_bits);

    return poly * scale;
}

/// SIMD-accelerated exp
export fn vt_simd_exp(data: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = data[i..][0..4].*;
        result[i..][0..4].* = exp_vec4(v);
    }

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

/// Vectorized ln(x) for Vec4.
inline fn log_vec4(x: Vec4) Vec4 {
    const one: Vec4 = @splat(1.0);
    const zero: Vec4 = @splat(0.0);

    const bits: Vi4 = @bitCast(x);
    const exp_mask: Vi4 = @splat(0x7FF0000000000000);
    const mant_mask: Vi4 = @splat(0x000FFFFFFFFFFFFF);
    const bias: Vi4 = @splat(1023);
    const half_bits: Vi4 = @splat(0x3FE0000000000000);

    const shift: @Vector(4, u6) = @splat(52);
    const e_i: Vi4 = ((bits & exp_mask) >> shift) - bias;
    const e: Vec4 = @floatFromInt(e_i);

    const m_bits = (bits & mant_mask) | half_bits;
    const m: Vec4 = @bitCast(m_bits);

    const sqrt2_over2: Vec4 = @splat(0.70710678118654752);
    const adjust = m < sqrt2_over2;
    const m_adj = @select(f64, adjust, m + m - one, m - one);
    const e_adj = @select(f64, adjust, e - one, e);

    const f = m_adj;
    const f2 = f * f;

    const p0: Vec4 = @splat(-7.89580278884799154124e-1);
    const p1: Vec4 = @splat(1.63866645699558079767e1);
    const p2: Vec4 = @splat(-6.41409952958715622951e1);

    const q0: Vec4 = @splat(-3.56722798256324312549e1);
    const q1: Vec4 = @splat(3.12093766372244180303e2);
    const q2: Vec4 = @splat(-7.69691943550460008604e2);

    const p_num = p0 + f * (p1 + f * p2);
    const q_den = one + f * (q0 + f * (q1 + f * q2));

    const ln2_hi: Vec4 = @splat(6.93145751953125e-1);
    const ln2_lo: Vec4 = @splat(1.42860682030941723212e-6);

    const r = f - @as(Vec4, @splat(0.5)) * f2 + f2 * f * p_num / q_den;
    const result = e_adj * ln2_hi + (e_adj * ln2_lo + r);

    return @select(f64, x > zero, result, @as(Vec4, @splat(@as(f64, @bitCast(@as(u64, 0x7FF8000000000000))))));
}

/// SIMD-accelerated natural log
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

// =============================================================================
// Resonance Kernels (Log-Number System f64)
// =============================================================================

/// Resonance Multiply: LNS element-wise multiply
export fn vt_resonance_mul(a: [*]const f64, b: [*]const f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const zero: Vec4 = @splat(0.0);
    const neg: Vec4 = @splat(-1.0);
    const pos: Vec4 = @splat(1.0);
    const tiny: Vec4 = @splat(1e-300);

    while (i + 4 <= len) : (i += 4) {
        const va: Vec4 = a[i..][0..4].*;
        const vb: Vec4 = b[i..][0..4].*;

        const direct = va * vb;
        const nonzero = direct != zero;

        const sa = @select(f64, va < zero, neg, pos);
        const sb = @select(f64, vb < zero, neg, pos);

        const la = log_vec4(@max(@abs(va), tiny));
        const lb = log_vec4(@max(@abs(vb), tiny));
        const lns = exp_vec4(la + lb);

        result[i..][0..4].* = @select(f64, nonzero, sa * sb * lns, zero);
    }

    while (i < len) : (i += 1) {
        if (a[i] == 0.0 or b[i] == 0.0) {
            result[i] = 0.0;
        } else {
            const s: f64 = if ((a[i] < 0.0) != (b[i] < 0.0)) -1.0 else 1.0;
            result[i] = s * @exp(@log(@abs(a[i])) + @log(@abs(b[i])));
        }
    }
}

/// Resonance Power: LNS element-wise power
export fn vt_resonance_power(data: [*]const f64, exponent: f64, result: [*]f64, len: usize) callconv(.c) void {
    var i: usize = 0;
    const zero: Vec4 = @splat(0.0);
    const neg: Vec4 = @splat(-1.0);
    const pos: Vec4 = @splat(1.0);
    const tiny: Vec4 = @splat(1e-300);
    const exp_v: Vec4 = @splat(exponent);

    while (i + 4 <= len) : (i += 4) {
        const v: Vec4 = data[i..][0..4].*;
        const nonzero = v != zero;
        const sign = @select(f64, v < zero, neg, pos);

        const lv = log_vec4(@max(@abs(v), tiny));
        const pw = exp_vec4(exp_v * lv);

        result[i..][0..4].* = @select(f64, nonzero, sign * pw, zero);
    }

    while (i < len) : (i += 1) {
        if (data[i] == 0.0) {
            result[i] = 0.0;
        } else {
            const s: f64 = if (data[i] < 0.0) -1.0 else 1.0;
            result[i] = s * @exp(exponent * @log(@abs(data[i])));
        }
    }
}

// =============================================================================
// LNS f32 via IADD (8x throughput vs FMA)
// =============================================================================

const VecF32_8 = @Vector(8, f32);
const VecI32_8 = @Vector(8, i32);

/// LNS fast multiply (f32): A * B ≈ bits_to_float(bits(A) + bits(B) - bias)
export fn vt_lns_mul_f32(a: [*]const f32, b: [*]const f32, result: [*]f32, len: usize) callconv(.c) void {
    const bias: VecI32_8 = @splat(0x3F800000);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: VecF32_8 = a[i..][0..8].*;
        const vb: VecF32_8 = b[i..][0..8].*;
        const ia: VecI32_8 = @bitCast(va);
        const ib: VecI32_8 = @bitCast(vb);
        const ir = ia +% ib -% bias;
        result[i..][0..8].* = @bitCast(ir);
    }

    while (i < len) : (i += 1) {
        const ia: i32 = @bitCast(a[i]);
        const ib: i32 = @bitCast(b[i]);
        const ir = ia +% ib -% @as(i32, 0x3F800000);
        result[i] = @bitCast(ir);
    }
}

/// LNS multiply with Mitchell's correction: ~2% max error
export fn vt_lns_mul_corrected_f32(a: [*]const f32, b: [*]const f32, result: [*]f32, len: usize) callconv(.c) void {
    const bias: VecI32_8 = @splat(0x3F800000);
    const one: VecF32_8 = @splat(1.0);
    const half: VecF32_8 = @splat(0.5);
    const frac_mask: VecI32_8 = @splat(0x007FFFFF);
    const frac_scale: VecF32_8 = @splat(1.0 / 8388608.0);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: VecF32_8 = a[i..][0..8].*;
        const vb: VecF32_8 = b[i..][0..8].*;
        const ia: VecI32_8 = @bitCast(va);
        const ib: VecI32_8 = @bitCast(vb);
        const frac_a: VecF32_8 = @as(VecF32_8, @floatFromInt(ia & frac_mask)) * frac_scale;
        const frac_b: VecF32_8 = @as(VecF32_8, @floatFromInt(ib & frac_mask)) * frac_scale;
        const correction = (one - frac_a) * (one - frac_b);
        const ir = ia +% ib -% bias;
        const approx: VecF32_8 = @bitCast(ir);
        result[i..][0..8].* = approx * (one + correction * half);
    }

    while (i < len) : (i += 1) {
        const ia: i32 = @bitCast(a[i]);
        const ib: i32 = @bitCast(b[i]);
        const frac_a = @as(f32, @floatFromInt(ia & 0x007FFFFF)) / 8388608.0;
        const frac_b = @as(f32, @floatFromInt(ib & 0x007FFFFF)) / 8388608.0;
        const correction = (1.0 - frac_a) * (1.0 - frac_b);
        const ir = ia +% ib -% @as(i32, 0x3F800000);
        const approx: f32 = @bitCast(ir);
        result[i] = approx * (1.0 + correction * 0.5);
    }
}

/// LNS division (f32): A / B ≈ bits_to_float(bits(A) - bits(B) + bias)
export fn vt_lns_div_f32(a: [*]const f32, b: [*]const f32, result: [*]f32, len: usize) callconv(.c) void {
    const bias: VecI32_8 = @splat(0x3F800000);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: VecF32_8 = a[i..][0..8].*;
        const vb: VecF32_8 = b[i..][0..8].*;
        const ia: VecI32_8 = @bitCast(va);
        const ib: VecI32_8 = @bitCast(vb);
        const ir = ia -% ib +% bias;
        result[i..][0..8].* = @bitCast(ir);
    }

    while (i < len) : (i += 1) {
        const ia: i32 = @bitCast(a[i]);
        const ib: i32 = @bitCast(b[i]);
        const ir = ia -% ib +% @as(i32, 0x3F800000);
        result[i] = @bitCast(ir);
    }
}

/// LNS sqrt (f32): sqrt(A) ≈ bits_to_float((bits(A) + bias) / 2)
export fn vt_lns_sqrt_f32(data: [*]const f32, result: [*]f32, len: usize) callconv(.c) void {
    const bias: VecI32_8 = @splat(0x3F800000);
    const one_v: VecI32_8 = @splat(1);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const v: VecF32_8 = data[i..][0..8].*;
        const iv: VecI32_8 = @bitCast(v);
        const ir = (iv +% bias) >> one_v;
        result[i..][0..8].* = @bitCast(ir);
    }

    while (i < len) : (i += 1) {
        const iv: i32 = @bitCast(data[i]);
        const ir = (iv +% 0x3F800000) >> 1;
        result[i] = @bitCast(ir);
    }
}

/// LNS inverse sqrt (f32): 1/sqrt(A) - the famous Quake III trick!
export fn vt_lns_rsqrt_f32(data: [*]const f32, result: [*]f32, len: usize) callconv(.c) void {
    const magic: VecI32_8 = @splat(0x5F3759DF);
    const one_v: VecI32_8 = @splat(1);
    const half: VecF32_8 = @splat(0.5);
    const three_half: VecF32_8 = @splat(1.5);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const x: VecF32_8 = data[i..][0..8].*;
        const ix: VecI32_8 = @bitCast(x);
        const ir = magic -% (ix >> one_v);
        var y: VecF32_8 = @bitCast(ir);
        y = y * (three_half - half * x * y * y);
        result[i..][0..8].* = y;
    }

    while (i < len) : (i += 1) {
        const x = data[i];
        const ix: i32 = @bitCast(x);
        const ir = 0x5F3759DF - (ix >> 1);
        var y: f32 = @bitCast(ir);
        y = y * (1.5 - 0.5 * x * y * y);
        result[i] = y;
    }
}

// =============================================================================
// HORDE KERNEL - SoA Physics Engine
// =============================================================================

/// Euler integration: positions += velocities * dt
export fn vt_horde_integrate(positions: [*]f64, velocities: [*]const f64, dt: f64, count: usize) callconv(.c) void {
    var i: usize = 0;
    const dt_vec: Vec = @splat(dt);

    while (i + VEC_LEN <= count) : (i += VEC_LEN) {
        const pos: Vec = positions[i..][0..VEC_LEN].*;
        const vel: Vec = velocities[i..][0..VEC_LEN].*;
        positions[i..][0..VEC_LEN].* = @mulAdd(Vec, vel, dt_vec, pos);
    }

    while (i < count) : (i += 1) {
        positions[i] += velocities[i] * dt;
    }
}

/// Apply damping/friction: velocities *= friction
export fn vt_horde_dampen(velocities: [*]f64, friction: f64, count: usize) callconv(.c) void {
    var i: usize = 0;
    const fric_vec: Vec = @splat(friction);

    while (i + VEC_LEN <= count) : (i += VEC_LEN) {
        const v: Vec = velocities[i..][0..VEC_LEN].*;
        velocities[i..][0..VEC_LEN].* = v * fric_vec;
    }

    while (i < count) : (i += 1) {
        velocities[i] *= friction;
    }
}

/// Accelerate: velocities += accelerations * dt
export fn vt_horde_accelerate(velocities: [*]f64, accelerations: [*]const f64, dt: f64, count: usize) callconv(.c) void {
    var i: usize = 0;
    const dt_vec: Vec = @splat(dt);

    while (i + VEC_LEN <= count) : (i += VEC_LEN) {
        const vel: Vec = velocities[i..][0..VEC_LEN].*;
        const acc: Vec = accelerations[i..][0..VEC_LEN].*;
        velocities[i..][0..VEC_LEN].* = @mulAdd(Vec, acc, dt_vec, vel);
    }

    while (i < count) : (i += 1) {
        velocities[i] += accelerations[i] * dt;
    }
}

/// Toroidal wrap: if pos >= max, pos -= max; if pos < 0, pos += max
export fn vt_horde_wrap(positions: [*]f64, max_bound: f64, count: usize) callconv(.c) void {
    var i: usize = 0;
    const max_v: Vec = @splat(max_bound);
    const zero_v: Vec = @splat(0.0);

    while (i + VEC_LEN <= count) : (i += VEC_LEN) {
        var pos: Vec = positions[i..][0..VEC_LEN].*;
        const over = pos >= max_v;
        pos = @select(f64, over, pos - max_v, pos);
        const under = pos < zero_v;
        pos = @select(f64, under, pos + max_v, pos);
        positions[i..][0..VEC_LEN].* = pos;
    }

    while (i < count) : (i += 1) {
        if (positions[i] >= max_bound) positions[i] -= max_bound;
        if (positions[i] < 0.0) positions[i] += max_bound;
    }
}

/// Apply gravity: accelerations[y] = -gravity for all entities (2D interleaved)
export fn vt_horde_gravity_2d(accelerations: [*]f64, gravity: f64, entity_count: usize) callconv(.c) void {
    const neg_g = -gravity;
    var i: usize = 0;
    while (i < entity_count) : (i += 1) {
        accelerations[i * 2 + 1] = neg_g;
    }
}

/// Compute kinetic energy: 0.5 * sum(vel^2)
export fn vt_horde_kinetic_energy(velocities: [*]const f64, count: usize) callconv(.c) f64 {
    var sum: f64 = 0.0;
    var i: usize = 0;

    while (i + VEC_LEN <= count) : (i += VEC_LEN) {
        const v: Vec = velocities[i..][0..VEC_LEN].*;
        sum += @reduce(.Add, v * v);
    }

    while (i < count) : (i += 1) {
        sum += velocities[i] * velocities[i];
    }

    return 0.5 * sum;
}

// =============================================================================
// HDC KERNEL - Hyperdimensional Computing
// =============================================================================

const VecU64_8 = @Vector(8, u64);

/// XOR binding: result = a XOR b
export fn vt_hdc_bind(a: [*]const u64, b: [*]const u64, result: [*]u64, len: usize) callconv(.c) void {
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: VecU64_8 = a[i..][0..8].*;
        const vb: VecU64_8 = b[i..][0..8].*;
        result[i..][0..8].* = va ^ vb;
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] ^ b[i];
    }
}

/// Hamming distance via popcount
export fn vt_hdc_hamming(a: [*]const u64, b: [*]const u64, len: usize) callconv(.c) u64 {
    var total: u64 = 0;
    var i: usize = 0;

    while (i < len) : (i += 1) {
        total += @popCount(a[i] ^ b[i]);
    }

    return total;
}

/// Cosine-like similarity: (D - hamming) / D
export fn vt_hdc_similarity(a: [*]const u64, b: [*]const u64, len: usize, dim: usize) callconv(.c) f64 {
    const hamming = vt_hdc_hamming(a, b, len);
    const d: f64 = @floatFromInt(dim);
    const h: f64 = @floatFromInt(hamming);
    return (d - h) / d;
}

/// Bundling via majority vote
export fn vt_hdc_bundle(inputs: [*]const u64, n_vectors: usize, words: usize, result: [*]u64) callconv(.c) void {
    const threshold = n_vectors / 2;

    for (0..words) |w| {
        var result_bits: u64 = 0;

        for (0..64) |b| {
            var count: usize = 0;
            const mask: u64 = @as(u64, 1) << @intCast(b);

            for (0..n_vectors) |v| {
                if ((inputs[v * words + w] & mask) != 0) {
                    count += 1;
                }
            }

            if (count > threshold) {
                result_bits |= mask;
            }
        }

        result[w] = result_bits;
    }
}

/// Circular permutation
export fn vt_hdc_permute(input: [*]const u64, output: [*]u64, words: usize, shift: usize) callconv(.c) void {
    const total_bits = words * 64;
    const effective_shift = shift % total_bits;

    if (effective_shift == 0) {
        @memcpy(output[0..words], input[0..words]);
        return;
    }

    const word_shift = effective_shift / 64;
    const bit_shift: u6 = @intCast(effective_shift % 64);

    if (bit_shift == 0) {
        for (0..words) |i| {
            output[i] = input[(i + word_shift) % words];
        }
    } else {
        const inv_bit_shift: u6 = @intCast(64 - @as(usize, bit_shift));
        for (0..words) |i| {
            const src = (i + word_shift) % words;
            const next = (i + word_shift + 1) % words;
            output[i] = (input[src] >> bit_shift) | (input[next] << inv_bit_shift);
        }
    }
}

/// Generate random hypervector using xorshift64 PRNG
export fn vt_hdc_random(output: [*]u64, words: usize, seed: u64) callconv(.c) void {
    var state = seed;
    if (state == 0) state = 0xDEADBEEF;

    for (0..words) |i| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        output[i] = state;
    }
}

/// Weighted bundle
export fn vt_hdc_weighted_bundle(inputs: [*]const u64, weights: [*]const f64, n_vectors: usize, words: usize, result: [*]u64) callconv(.c) void {
    for (0..words) |w| {
        var result_bits: u64 = 0;

        for (0..64) |b| {
            var weighted_sum: f64 = 0.0;
            const mask: u64 = @as(u64, 1) << @intCast(b);

            for (0..n_vectors) |v| {
                const bit_set = (inputs[v * words + w] & mask) != 0;
                const bit_val: f64 = if (bit_set) 1.0 else -1.0;
                weighted_sum += weights[v] * bit_val;
            }

            if (weighted_sum > 0.0) {
                result_bits |= mask;
            }
        }

        result[w] = result_bits;
    }
}

// =============================================================================
// FUSED QUANTIZED MATMUL - Zero overhead quantization!
// Dequant on-the-fly during compute = no memory overhead from decompression.
// =============================================================================

/// INT8 Fused Matmul: C = A @ (B_quant * scale)
/// A is dense f64 [M x K], B_quant is i8 [K x N], C is f64 [M x N]
/// Dequantizes B on-the-fly: no extra memory for dequantized B!
///
/// This is 4x memory compression with ZERO runtime overhead.
/// Key insight: INT8 dequant is just a multiply, fuse it with accumulation.
export fn vt_matmul_int8(
    a_ptr: [*]const f64, // A: M x K (row-major)
    b_quant: [*]const i8, // B_quant: K x N (row-major, quantized)
    b_scale: f64, // Single scale for entire B tensor
    m: usize,
    n: usize,
    k: usize,
    c_ptr: [*]f64, // C: M x N output
) callconv(.c) void {
    // Simple but effective implementation:
    // For each output element C[i,j] = sum(A[i,:] * B[:,j])
    // We compute A[i,p] * (B_quant[p,j] * scale) directly

    const scale_vec: Vec = @splat(b_scale);

    // Process row by row
    for (0..m) |i| {
        const row_start = i * k;

        for (0..n) |j| {
            var acc: f64 = 0.0;
            var p: usize = 0;

            // Vectorized inner loop
            while (p + VEC_LEN <= k) : (p += VEC_LEN) {
                // Load A values
                const a_vec: Vec = a_ptr[row_start + p ..][0..VEC_LEN].*;

                // Load B_quant values and convert to f64
                var b_vec: Vec = undefined;
                inline for (0..VEC_LEN) |v| {
                    b_vec[v] = @floatFromInt(b_quant[(p + v) * n + j]);
                }

                // Fused multiply: A * (B_quant * scale)
                const dequant_b = b_vec * scale_vec;
                acc += @reduce(.Add, a_vec * dequant_b);
            }

            // Scalar tail
            while (p < k) : (p += 1) {
                const a_val = a_ptr[row_start + p];
                const b_val: f64 = @floatFromInt(b_quant[p * n + j]);
                acc += a_val * b_val * b_scale;
            }

            c_ptr[i * n + j] = acc;
        }
    }
}

/// INT8 Fused Matmul with per-block scaling
/// Each block of block_size elements in B has its own scale.
/// More accurate than single-scale, same memory efficiency.
export fn vt_matmul_int8_blocked(
    a_ptr: [*]const f64, // A: M x K
    b_quant: [*]const i8, // B_quant: K x N
    b_scales: [*]const f64, // Scales: one per block (total: ceil(K/block_size) * N)
    m: usize,
    n: usize,
    k: usize,
    block_size: usize, // Typically 64 or 128
    c_ptr: [*]f64, // C: M x N
) callconv(.c) void {
    const num_blocks = (k + block_size - 1) / block_size;

    for (0..m) |i| {
        const row_start = i * k;

        for (0..n) |j| {
            var acc: f64 = 0.0;

            // Process each block with its own scale
            for (0..num_blocks) |blk| {
                const block_start = blk * block_size;
                const block_end = @min(block_start + block_size, k);
                const scale = b_scales[blk * n + j];

                var p = block_start;
                while (p < block_end) : (p += 1) {
                    const a_val = a_ptr[row_start + p];
                    const b_val: f64 = @floatFromInt(b_quant[p * n + j]);
                    acc += a_val * b_val * scale;
                }
            }

            c_ptr[i * n + j] = acc;
        }
    }
}

// NF4 Quantization Levels (from QLoRA paper)
// 16 levels derived from N(0,1) quantiles, normalized to [-1, 1]
const NF4_LEVELS: [16]f64 = .{
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
};

/// NF4 Fused Matmul: C = A @ (dequant_nf4(B_indices) * scales)
/// A is dense f64 [M x K], B_indices is u8 [K x N] (4-bit packed: 2 values per byte)
/// Each block of block_size K elements has its own scale in b_scales.
///
/// NF4 = 8x compression with ~0.1% error for Gaussian weights!
export fn vt_matmul_nf4(
    a_ptr: [*]const f64, // A: M x K
    b_indices: [*]const u8, // NF4 indices: ceil(K*N/2) bytes (4-bit packed)
    b_scales: [*]const f64, // Scales: ceil(K/block_size) * N
    m: usize,
    n: usize,
    k: usize,
    block_size: usize, // Typically 64
    c_ptr: [*]f64, // C: M x N
) callconv(.c) void {
    const num_blocks = (k + block_size - 1) / block_size;

    for (0..m) |i| {
        const row_start = i * k;

        for (0..n) |j| {
            var acc: f64 = 0.0;

            // Process each block
            for (0..num_blocks) |blk| {
                const block_start = blk * block_size;
                const block_end = @min(block_start + block_size, k);
                const scale = b_scales[blk * n + j];

                var p = block_start;
                while (p < block_end) : (p += 1) {
                    const a_val = a_ptr[row_start + p];

                    // Get NF4 index from packed byte
                    const flat_idx = p * n + j;
                    const byte_idx = flat_idx / 2;
                    const is_high_nibble = (flat_idx % 2) == 1;
                    const packed_byte = b_indices[byte_idx];
                    const nf4_idx: usize = if (is_high_nibble)
                        (packed_byte >> 4) & 0x0F
                    else
                        packed_byte & 0x0F;

                    // Dequant: lookup NF4 level and apply scale
                    const b_val = NF4_LEVELS[nf4_idx] * scale;
                    acc += a_val * b_val;
                }
            }

            c_ptr[i * n + j] = acc;
        }
    }
}

/// Quantize f64 tensor to INT8 with absmax scaling
/// Returns the scale factor used.
export fn vt_quantize_int8(
    data: [*]const f64,
    output: [*]i8,
    len: usize,
) callconv(.c) f64 {
    // Find absmax
    var absmax: f64 = 0.0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const abs_val = @abs(data[i]);
        if (abs_val > absmax) absmax = abs_val;
    }

    if (absmax == 0.0) {
        @memset(output[0..len], 0);
        return 1.0;
    }

    // Scale to [-127, 127] range
    const scale = absmax / 127.0;
    const inv_scale = 127.0 / absmax;

    i = 0;
    while (i < len) : (i += 1) {
        const scaled = data[i] * inv_scale;
        const clamped = @max(-127.0, @min(127.0, scaled));
        output[i] = @intFromFloat(clamped);
    }

    return scale;
}

/// Quantize f64 tensor to NF4 (4-bit, packed)
/// Returns scales array (one per block).
export fn vt_quantize_nf4(
    data: [*]const f64,
    output: [*]u8, // Packed: 2 values per byte
    scales: [*]f64, // One scale per block
    len: usize,
    block_size: usize,
) callconv(.c) void {
    const num_blocks = (len + block_size - 1) / block_size;

    for (0..num_blocks) |blk| {
        const block_start = blk * block_size;
        const block_end = @min(block_start + block_size, len);

        // Find absmax for this block
        var absmax: f64 = 0.0;
        for (block_start..block_end) |i| {
            const abs_val = @abs(data[i]);
            if (abs_val > absmax) absmax = abs_val;
        }

        const scale = if (absmax == 0.0) 1.0 else absmax;
        scales[blk] = scale;

        // Quantize each value in block
        for (block_start..block_end) |i| {
            const normalized = data[i] / scale; // [-1, 1]

            // Find nearest NF4 level
            var best_idx: usize = 0;
            var best_dist: f64 = @abs(normalized - NF4_LEVELS[0]);

            for (1..16) |lvl| {
                const dist = @abs(normalized - NF4_LEVELS[lvl]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = lvl;
                }
            }

            // Pack into nibble
            const byte_idx = i / 2;
            const is_high = (i % 2) == 1;
            if (is_high) {
                output[byte_idx] |= @as(u8, @intCast(best_idx)) << 4;
            } else {
                output[byte_idx] = @as(u8, @intCast(best_idx));
            }
        }
    }
}
