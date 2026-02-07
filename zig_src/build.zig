const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get Erlang include path (needed for nif_entry.c, not for Zig code)
    const erl_include = b.option(
        []const u8,
        "erl_include",
        "Path to Erlang NIF headers",
    ) orelse "/usr/local/lib/erlang/usr/include";

    // Get MKL path for Windows (via winget install Intel.oneMKL)
    const mkl_root = b.option(
        []const u8,
        "mkl_root",
        "Path to Intel MKL installation",
    ) orelse "C:/PROGRA~2/Intel/oneAPI/mkl/latest";

    // Zig 0.15+ API: addLibrary with explicit linkage
    const lib = b.addLibrary(.{
        .name = "viva_tensor_zig",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("viva_zig.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // On Unix, allow undefined symbols (BEAM resolves enif_* at NIF load time)
    // On Windows, ERL_NIF_INIT uses TWinDynNifCallbacks - no undefined symbols
    if (target.result.os.tag != .windows) {
        lib.linker_allow_shlib_undefined = true;
    }

    // Compile the C NIF entry point (uses erl_nif.h for NIF boilerplate)
    lib.addCSourceFile(.{
        .file = b.path("nif_entry.c"),
        .flags = &.{},
    });

    // Add Erlang NIF headers (for nif_entry.c)
    lib.addIncludePath(.{ .cwd_relative = erl_include });

    // Link with libc (needed by nif_entry.c)
    lib.linkLibC();

    // Link with optimized BLAS for GEMM (800+ GFLOPS with MKL)
    // Platform-specific backends: MKL (Windows/Linux), Accelerate (macOS)
    if (target.result.os.tag == .windows) {
        // Windows: Intel MKL via winget install Intel.oneMKL
        // nif_entry.c uses cblas_dgemm directly when _WIN32 is defined
        // (no cuda_gemm.c needed - MKL is called directly from nif_entry.c)
        const mkl_inc = b.fmt("{s}/include", .{mkl_root});
        const mkl_lib = b.fmt("{s}/lib", .{mkl_root});
        lib.addIncludePath(.{ .cwd_relative = mkl_inc });
        lib.addLibraryPath(.{ .cwd_relative = mkl_lib });
        lib.linkSystemLibrary("mkl_rt");
    } else if (target.result.os.tag == .macos) {
        // macOS: Apple Accelerate framework (vDSP + BLAS)
        lib.addCSourceFile(.{
            .file = b.path("accelerate.c"),
            .flags = &.{},
        });
        lib.linkFramework("Accelerate");
    } else {
        // Linux: Intel MKL (apt install intel-mkl) - 800+ GFLOPS!
        lib.addCSourceFile(.{
            .file = b.path("cuda_gemm.c"),
            .flags = &.{"-DUSE_MKL_DIRECT"},
        });

        // cuSPARSELt for 2:4 structured sparsity (660/1320 TFLOPS!)
        lib.addCSourceFile(.{
            .file = b.path("cuda_sparselt.c"),
            .flags = &.{},
        });

        // SageAttention: INT8 QK^T + FP8 support (2-5x faster than FlashAttention!)
        lib.addCSourceFile(.{
            .file = b.path("cuda_sage.c"),
            .flags = &.{},
        });

        lib.root_module.addCMacro("USE_MKL_DIRECT", "1");

        // MKL headers and libs (Ubuntu: apt install intel-mkl)
        lib.addIncludePath(.{ .cwd_relative = "/usr/include/mkl" });
        lib.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        lib.linkSystemLibrary("mkl_rt");

        // OpenBLAS as fallback (kept for systems without MKL)
        lib.addLibraryPath(.{ .cwd_relative = "../deps/openblas-tuned/lib" });
        lib.addIncludePath(.{ .cwd_relative = "../deps/openblas-tuned/include" });

        // dlopen for CUDA and cuSPARSELt
        lib.linkSystemLibrary("dl");
        lib.linkSystemLibrary("pthread");
    }

    // Install the library
    b.installArtifact(lib);
}
