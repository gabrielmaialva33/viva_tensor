const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get Erlang include path from environment or use default
    const erl_include = b.option(
        []const u8,
        "erl_include",
        "Path to Erlang NIF headers",
    ) orelse "/usr/local/lib/erlang/usr/include";

    const lib = b.addSharedLibrary(.{
        .name = "viva_tensor_zig",
        .root_source_file = b.path("viva_zig.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add Erlang NIF headers
    lib.addIncludePath(.{ .cwd_relative = erl_include });

    // Link with libc for erl_nif
    lib.linkLibC();

    // Install the library
    b.installArtifact(lib);

    // Add a run step
    const run_step = b.step("run", "Build the NIF library");
    run_step.dependOn(b.getInstallStep());
}
