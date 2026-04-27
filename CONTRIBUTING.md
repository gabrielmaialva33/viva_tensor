# Contributing

Thanks for helping improve viva_tensor. This project is a Gleam tensor library with native BEAM/NIF acceleration through Zig, C, MKL, CUDA, and specialized kernels. Contributions should keep the public Gleam API small, documented, and predictable while allowing native backends to evolve behind internal modules.

## Project Layout

- `src/viva_tensor.gleam` is the stable public entry point.
- `src/viva_tensor/` contains domain modules and internal implementation modules.
- `src/*.erl` contains Erlang FFI wrappers used by Gleam externals.
- `zig_src/` contains Zig/C/CUDA NIF sources and native build configuration.
- `test/` contains Gleam tests.
- `bench/` contains external benchmark scripts and generated reports under ignored folders.
- `docs/` contains additional documentation pages included by `gleam.toml`.

## Development Setup

Install Gleam, Erlang/OTP, and Zig. Native acceleration also needs the relevant platform stack:

- Linux CPU: Intel oneMKL when testing MKL paths.
- NVIDIA GPU: CUDA libraries and driver access for CUDA paths.
- macOS: Apple Accelerate for native BLAS paths.

Download Gleam dependencies:

```sh
gleam deps download
```

## Quality Checks

Run the lightweight checks before opening a pull request:

```sh
gleam format --check src test
gleam check
gleam test
gleam docs build
```

For native work, also build the NIF:

```sh
make zig
gleam test
```

CUDA tests may require running outside restricted sandboxes so the BEAM process can access `/dev/nvidia*`.

## Benchmarks

Use benchmarks to support performance claims. Prefer comparing against a clear baseline and include hardware, driver, CUDA/MKL versions, matrix sizes, warmup, iteration count, and whether timings include upload/download.

Common commands:

```sh
make bench
gleam run -m viva_tensor/bench/rtx
python3 bench/benchmark.py
```

Generated benchmark data belongs in ignored output folders such as `bench/data/` and `bench/reports/`.

## API Guidelines

- Keep `viva_tensor` as the preferred public entry point.
- Mark implementation modules as internal in `gleam.toml`.
- Use `Result` for fallible operations.
- Prefer domain error types over `String` errors at public API boundaries.
- Do not use `panic` or `let assert` in library code.
- Use doc comments for every public type and function.
- Keep native resources opaque from user code unless there is a strong API reason.
- Avoid committing generated binaries, object files, archives, or local build outputs.

## Pull Requests

Keep pull requests focused. A good PR explains:

- What changed.
- Why it is needed.
- Which public APIs changed.
- Which checks and benchmarks were run.
- Any hardware-specific assumptions.

Bug fixes should include regression tests when practical. Performance changes should include enough benchmark detail for another contributor to reproduce or challenge the result.
