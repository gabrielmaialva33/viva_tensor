# Getting started

Quick path from zero to a running tensor program. The pure-Gleam path
needs nothing but `gleam`. The CUDA inference path needs a recent CUDA
toolkit and an Ada-or-better NVIDIA GPU.

## Pure-Gleam path

```bash
gleam new my_app
cd my_app
gleam add viva_tensor
```

```gleam
// src/my_app.gleam
import gleam/io
import viva_tensor as t

pub fn main() {
  let assert Ok(a) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(b) = t.matrix(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
  let assert Ok(c) = t.matmul(a, b)
  io.println(string.inspect(t.to_list(c)))
}
```

```bash
gleam run
```

That works on any platform Gleam supports. No NIF needed.

## CUDA inference path

Prerequisites:

- NVIDIA GPU, Ada SM89 (RTX 4090) or newer recommended
- CUDA 12.0+ toolkit (`nvcc`)
- Driver 555+
- `zig` 0.14+ (build system)
- `g++` 14 (GCC 16 has known nvcc breakage on `<functional>`; the
  Makefile auto-detects `g++-15` as the host compiler)

Build:

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor
cd viva_tensor
make cutlass-libs     # CUTLASS + cuSPARSELt static archives
make zig              # the NIF .so
gleam test            # 789 tests, all should pass with NIF loaded
```

If the NIF .so isn't present, the same `gleam test` still runs — it
just skips the native-only paths.

## Run TinyLlama-1.1B end-to-end

See [`inference.md`](inference.md) for the full walkthrough. Quick
version:

```bash
mkdir -p tmp/tinyllama && cd tmp/tinyllama
for f in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/$f
done
cd ../..

erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

Expected: the model prints a continuation of "Hello" generated at
~5.6 tok/sec.

## Verify your install

```bash
# Check NIF loaded
erl -pa build/dev/erlang/viva_tensor/ebin -noshell -eval \
    'io:format("~p~n", [viva_tensor_zig:cuda_available()]), halt(0).'
# -> true   (or false on CPU-only build)

# Run a quick CUDA matmul
gleam run -m viva_tensor/bench/peak
```

## Next steps

- [`inference.md`](inference.md) — full end-to-end Llama walkthrough.
- [`../api/tensor.md`](../api/tensor.md) — public tensor API reference.
- [`../api/inference.md`](../api/inference.md) — prepack / linear /
  sampling / tokenizer reference.
- [`ffi-architecture.md`](ffi-architecture.md) — how the Gleam ↔ NIF
  boundary works (maintainer-facing).
