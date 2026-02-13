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

    // Platform detection: BLAS backend + CPU topology (extracted from nif_entry.c)
    lib.addCSourceFile(.{
        .file = b.path("nif_platform.c"),
        .flags = &.{},
    });

    // CudaTensor (FP32) NIFs: persistent GPU memory, SGEMM
    lib.addCSourceFile(.{
        .file = b.path("nif_cuda_fp32.c"),
        .flags = &.{},
    });

    // CudaTensor16 (FP16) NIFs: Tensor Core HGEMM, fused ops, benchmarks
    lib.addCSourceFile(.{
        .file = b.path("nif_cuda_fp16.c"),
        .flags = &.{},
    });

    // CudaInt8Tensor NIFs: INT8 IMMA Tensor Cores via cublasLt
    lib.addCSourceFile(.{
        .file = b.path("nif_cuda_int8.c"),
        .flags = &.{},
    });

    // CPU NIF ops: element-wise, reductions, matmul, activations, in-place, fused
    lib.addCSourceFile(.{
        .file = b.path("nif_cpu_ops.c"),
        .flags = &.{},
    });

    // Tensor resource types, lifecycle, helpers, constructors, accessors (extracted from nif_entry.c)
    lib.addCSourceFile(.{
        .file = b.path("nif_tensor_core.c"),
        .flags = &.{},
    });

    // Specialized backends: Resonance/LNS, Horde physics, HDC (extracted from nif_entry.c)
    lib.addCSourceFile(.{
        .file = b.path("nif_specialized.c"),
        .flags = &.{},
    });

    // SparseTensor NIFs (FP16 + INT8 via cuSPARSELt)
    lib.addCSourceFile(.{
        .file = b.path("nif_sparse.c"),
        .flags = &.{},
    });

    // Quantization NIFs (INT8 + NF4 fused matmul)
    lib.addCSourceFile(.{
        .file = b.path("nif_quant.c"),
        .flags = &.{},
    });

    // SageAttention NIFs (CPU + GPU paths)
    lib.addCSourceFile(.{
        .file = b.path("nif_sage_nif.c"),
        .flags = &.{},
    });

    // Legacy list-based CPU NIFs (backward compatibility)
    lib.addCSourceFile(.{
        .file = b.path("nif_legacy.c"),
        .flags = &.{},
    });

    // Add Erlang NIF headers (for nif_entry.c)
    lib.addIncludePath(.{ .cwd_relative = erl_include });

    // Link with libc (needed by nif_entry.c)
    lib.linkLibC();

    // BLAS backend: MKL (Windows/Linux), Accelerate (macOS)
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
        // Linux: Intel MKL (apt install intel-mkl)
        lib.addCSourceFile(.{
            .file = b.path("cuda_gemm.c"),
            .flags = &.{"-DUSE_MKL_DIRECT"},
        });

        // cuSPARSELt for 2:4 structured sparsity
        lib.addCSourceFile(.{
            .file = b.path("cuda_sparselt.c"),
            .flags = &.{},
        });

        // SageAttention CUDA kernels
        lib.addCSourceFile(.{
            .file = b.path("cuda_sage.c"),
            .flags = &.{},
        });

        lib.root_module.addCMacro("USE_MKL_DIRECT", "1");

        // MKL headers and libs (Intel oneAPI MKL)
        lib.addIncludePath(.{ .cwd_relative = "/opt/intel/oneapi/mkl/2025.3/include" });
        lib.addLibraryPath(.{ .cwd_relative = "/opt/intel/oneapi/mkl/2025.3/lib/intel64" });
        lib.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        lib.linkSystemLibrary("mkl_rt");

        // OpenBLAS as fallback (kept for systems without MKL)
        lib.addLibraryPath(.{ .cwd_relative = "../deps/openblas-tuned/lib" });
        lib.addIncludePath(.{ .cwd_relative = "../deps/openblas-tuned/include" });

        // CUTLASS FP8 sparse GEMM (pre-compiled with nvcc)
        lib.addObjectFile(.{ .cwd_relative = "libcutlass_fp8.a" });

        // cuSPARSELt INT8 2:4 sparse GEMM (needs libcusparseLt at runtime)
        lib.addObjectFile(.{ .cwd_relative = "libcusparselt_int8.a" });

        // CUTLASS INT4 2:4 sparse GEMM
        lib.addObjectFile(.{ .cwd_relative = "libcutlass_int4_sparse.a" });

        // CUDA libraries: use standard system paths
        // cuSPARSELt: install via `pip install nvidia-cusparselt` then symlink to CUDA lib dir
        // cuda_sparselt.c also uses dlopen for graceful fallback if unavailable
        const cuda_path = b.option(
            []const u8,
            "cuda_path",
            "Path to CUDA lib64 directory",
        ) orelse "/usr/local/cuda/lib64";
        lib.addLibraryPath(.{ .cwd_relative = cuda_path });
        lib.addRPath(.{ .cwd_relative = cuda_path });
        lib.linkSystemLibrary("cusparseLt");
        lib.linkSystemLibrary("cudart");
        lib.linkSystemLibrary("stdc++");

        // dlopen for CUDA and cuSPARSELt
        lib.linkSystemLibrary("dl");
        lib.linkSystemLibrary("pthread");
    }

    // Install the library
    b.installArtifact(lib);
}
