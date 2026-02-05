// viva_zig.zig - SIMD-optimized tensor operations for BEAM
//
// This NIF provides cross-platform SIMD acceleration for tensor operations.
// Uses Zig's @Vector for portable SIMD without platform-specific intrinsics.
//
// Operations:
// - simd_dot: Vectorized dot product
// - simd_sum: Vectorized sum reduction
// - simd_scale: Vectorized scalar multiplication
// - simd_add: Vectorized element-wise addition
// - simd_matmul: Blocked matrix multiplication with SIMD

const std = @import("std");
const erl = @cImport({
    @cInclude("erl_nif.h");
});

// =============================================================================
// SIMD Configuration
// =============================================================================

// Vector length for SIMD operations (8 doubles = 512 bits, good for AVX-512)
// Falls back gracefully on systems without AVX-512
const VEC_LEN = 8;
const Vec = @Vector(VEC_LEN, f64);

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert Erlang list of floats to Zig slice
fn list_to_slice(env: ?*erl.ErlNifEnv, list: erl.ERL_NIF_TERM, allocator: std.mem.Allocator) ![]f64 {
    var length: c_uint = 0;
    if (erl.enif_get_list_length(env, list, &length) == 0) {
        return error.InvalidList;
    }

    const slice = try allocator.alloc(f64, length);
    errdefer allocator.free(slice);

    var current = list;
    var i: usize = 0;
    while (i < length) : (i += 1) {
        var head: erl.ERL_NIF_TERM = undefined;
        var tail: erl.ERL_NIF_TERM = undefined;

        if (erl.enif_get_list_cell(env, current, &head, &tail) == 0) {
            return error.InvalidList;
        }

        var value: f64 = undefined;
        if (erl.enif_get_double(env, head, &value) == 0) {
            // Try integer
            var int_val: c_int = undefined;
            if (erl.enif_get_int(env, head, &int_val) != 0) {
                value = @floatFromInt(int_val);
            } else {
                var long_val: c_long = undefined;
                if (erl.enif_get_long(env, head, &long_val) != 0) {
                    value = @floatFromInt(long_val);
                } else {
                    return error.InvalidNumber;
                }
            }
        }

        slice[i] = value;
        current = tail;
    }

    return slice;
}

/// Convert Zig slice to Erlang list
fn slice_to_list(env: ?*erl.ErlNifEnv, slice: []const f64) erl.ERL_NIF_TERM {
    var result = erl.enif_make_list(env, 0);
    var i: usize = slice.len;
    while (i > 0) {
        i -= 1;
        result = erl.enif_make_list_cell(env, erl.enif_make_double(env, slice[i]), result);
    }
    return result;
}

/// Create {ok, Value} tuple
fn make_ok(env: ?*erl.ErlNifEnv, value: erl.ERL_NIF_TERM) erl.ERL_NIF_TERM {
    return erl.enif_make_tuple2(
        env,
        erl.enif_make_atom(env, "ok"),
        value,
    );
}

/// Create {error, Reason} tuple
fn make_error(env: ?*erl.ErlNifEnv, reason: [*:0]const u8) erl.ERL_NIF_TERM {
    return erl.enif_make_tuple2(
        env,
        erl.enif_make_atom(env, "error"),
        erl.enif_make_atom(env, reason),
    );
}

// =============================================================================
// SIMD Operations
// =============================================================================

/// SIMD-accelerated dot product
fn simd_dot_impl(a: []const f64, b: []const f64) f64 {
    if (a.len != b.len) return 0.0;

    var sum: f64 = 0.0;
    var i: usize = 0;
    const len = a.len;

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
fn simd_sum_impl(data: []const f64) f64 {
    var sum: f64 = 0.0;
    var i: usize = 0;
    const len = data.len;

    // Process VEC_LEN elements at a time
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        sum += @reduce(.Add, v);
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        sum += data[i];
    }

    return sum;
}

/// SIMD-accelerated scale (multiply by scalar)
fn simd_scale_impl(data: []const f64, scalar: f64, result: []f64) void {
    var i: usize = 0;
    const len = data.len;
    const scalar_vec: Vec = @splat(scalar);

    // Process VEC_LEN elements at a time
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: Vec = data[i..][0..VEC_LEN].*;
        const scaled = v * scalar_vec;
        result[i..][0..VEC_LEN].* = scaled;
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        result[i] = data[i] * scalar;
    }
}

/// SIMD-accelerated element-wise addition
fn simd_add_impl(a: []const f64, b: []const f64, result: []f64) void {
    var i: usize = 0;
    const len = @min(a.len, b.len);

    // Process VEC_LEN elements at a time
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: Vec = a[i..][0..VEC_LEN].*;
        const vb: Vec = b[i..][0..VEC_LEN].*;
        result[i..][0..VEC_LEN].* = va + vb;
    }

    // Handle remaining elements
    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-accelerated matrix multiplication with blocking
/// A[m,k] @ B[k,n] -> C[m,n]
fn simd_matmul_impl(a: []const f64, b: []const f64, c: []f64, m: usize, n: usize, k: usize) void {
    // Initialize result to zero
    @memset(c, 0.0);

    // Blocked matrix multiplication for better cache utilization
    const BLOCK_SIZE = 32;

    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK_SIZE) {
        const i_end = @min(ii + BLOCK_SIZE, m);

        var kk: usize = 0;
        while (kk < k) : (kk += BLOCK_SIZE) {
            const k_end = @min(kk + BLOCK_SIZE, k);

            var jj: usize = 0;
            while (jj < n) : (jj += BLOCK_SIZE) {
                const j_end = @min(jj + BLOCK_SIZE, n);

                // Process block
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var kix: usize = kk;
                    while (kix < k_end) : (kix += 1) {
                        const a_val = a[i * k + kix];
                        const a_vec: Vec = @splat(a_val);

                        var j: usize = jj;
                        // SIMD inner loop
                        while (j + VEC_LEN <= j_end) : (j += VEC_LEN) {
                            const b_vec: Vec = b[kix * n + j ..][0..VEC_LEN].*;
                            const c_idx = i * n + j;
                            var c_vec: Vec = c[c_idx..][0..VEC_LEN].*;
                            c_vec += a_vec * b_vec;
                            c[c_idx..][0..VEC_LEN].* = c_vec;
                        }

                        // Handle remaining columns
                        while (j < j_end) : (j += 1) {
                            c[i * n + j] += a_val * b[kix * n + j];
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// NIF Functions
// =============================================================================

/// NIF: SIMD dot product
fn nif_simd_dot(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    if (argc != 2) return erl.enif_make_badarg(env);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a = list_to_slice(env, argv[0], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(a);

    const b = list_to_slice(env, argv[1], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(b);

    if (a.len != b.len) {
        return make_error(env, "length_mismatch");
    }

    const result = simd_dot_impl(a, b);
    return make_ok(env, erl.enif_make_double(env, result));
}

/// NIF: SIMD sum
fn nif_simd_sum(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    if (argc != 1) return erl.enif_make_badarg(env);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = list_to_slice(env, argv[0], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(data);

    const result = simd_sum_impl(data);
    return make_ok(env, erl.enif_make_double(env, result));
}

/// NIF: SIMD scale
fn nif_simd_scale(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    if (argc != 2) return erl.enif_make_badarg(env);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = list_to_slice(env, argv[0], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(data);

    var scalar: f64 = undefined;
    if (erl.enif_get_double(env, argv[1], &scalar) == 0) {
        var int_val: c_int = undefined;
        if (erl.enif_get_int(env, argv[1], &int_val) != 0) {
            scalar = @floatFromInt(int_val);
        } else {
            return make_error(env, "invalid_scalar");
        }
    }

    const result = allocator.alloc(f64, data.len) catch {
        return make_error(env, "out_of_memory");
    };
    defer allocator.free(result);

    simd_scale_impl(data, scalar, result);

    return make_ok(env, slice_to_list(env, result));
}

/// NIF: SIMD element-wise add
fn nif_simd_add(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    if (argc != 2) return erl.enif_make_badarg(env);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const a = list_to_slice(env, argv[0], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(a);

    const b = list_to_slice(env, argv[1], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(b);

    if (a.len != b.len) {
        return make_error(env, "length_mismatch");
    }

    const result = allocator.alloc(f64, a.len) catch {
        return make_error(env, "out_of_memory");
    };
    defer allocator.free(result);

    simd_add_impl(a, b, result);

    return make_ok(env, slice_to_list(env, result));
}

/// NIF: SIMD matrix multiplication
fn nif_simd_matmul(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    if (argc != 5) return erl.enif_make_badarg(env);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get dimensions
    var m_int: c_int = undefined;
    var n_int: c_int = undefined;
    var k_int: c_int = undefined;

    if (erl.enif_get_int(env, argv[2], &m_int) == 0 or
        erl.enif_get_int(env, argv[3], &n_int) == 0 or
        erl.enif_get_int(env, argv[4], &k_int) == 0)
    {
        return make_error(env, "invalid_dimensions");
    }

    const m: usize = @intCast(m_int);
    const n: usize = @intCast(n_int);
    const k: usize = @intCast(k_int);

    // Get input matrices
    const a = list_to_slice(env, argv[0], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(a);

    const b = list_to_slice(env, argv[1], allocator) catch {
        return make_error(env, "invalid_input");
    };
    defer allocator.free(b);

    // Validate sizes
    if (a.len != m * k or b.len != k * n) {
        return make_error(env, "size_mismatch");
    }

    // Allocate result
    const c = allocator.alloc(f64, m * n) catch {
        return make_error(env, "out_of_memory");
    };
    defer allocator.free(c);

    // Perform SIMD matrix multiplication
    simd_matmul_impl(a, b, c, m, n, k);

    return make_ok(env, slice_to_list(env, c));
}

/// NIF: Check if SIMD is available (always true for Zig)
fn nif_simd_available(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    _ = argc;
    _ = argv;
    return erl.enif_make_atom(env, "true");
}

/// NIF: Get backend info (returns binary)
fn nif_backend_info(env: ?*erl.ErlNifEnv, argc: c_int, argv: [*c]const erl.ERL_NIF_TERM) callconv(.c) erl.ERL_NIF_TERM {
    _ = argc;
    _ = argv;
    const info = "Zig SIMD (Vector length: 8, f64)";
    var bin: erl.ErlNifBinary = undefined;
    if (erl.enif_alloc_binary(info.len, &bin) == 0) {
        return erl.enif_make_atom(env, "error");
    }
    @memcpy(bin.data[0..info.len], info);
    return erl.enif_make_binary(env, &bin);
}

// =============================================================================
// NIF Initialization
// =============================================================================

const func_count = 7;
var nif_funcs = [func_count]erl.ErlNifFunc{
    .{ .name = "nif_simd_dot", .arity = 2, .fptr = nif_simd_dot, .flags = 0 },
    .{ .name = "nif_simd_sum", .arity = 1, .fptr = nif_simd_sum, .flags = 0 },
    .{ .name = "nif_simd_scale", .arity = 2, .fptr = nif_simd_scale, .flags = 0 },
    .{ .name = "nif_simd_add", .arity = 2, .fptr = nif_simd_add, .flags = 0 },
    .{ .name = "nif_simd_matmul", .arity = 5, .fptr = nif_simd_matmul, .flags = 0 },
    .{ .name = "simd_available", .arity = 0, .fptr = nif_simd_available, .flags = 0 },
    .{ .name = "backend_info", .arity = 0, .fptr = nif_backend_info, .flags = 0 },
};

var entry = erl.ErlNifEntry{
    .major = erl.ERL_NIF_MAJOR_VERSION,
    .minor = erl.ERL_NIF_MINOR_VERSION,
    .name = "viva_tensor_zig",
    .num_of_funcs = func_count,
    .funcs = &nif_funcs,
    .load = null,
    .reload = null,
    .upgrade = null,
    .unload = null,
    .vm_variant = "beam.vanilla",
    .options = 1,
    .sizeof_ErlNifResourceTypeInit = @sizeOf(erl.ErlNifResourceTypeInit),
    .min_erts = "erts-10.4",
};

export fn nif_init() *erl.ErlNifEntry {
    return &entry;
}
