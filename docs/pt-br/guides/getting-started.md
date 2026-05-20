# Começando

Caminho rápido do zero até um programa de tensor rodando. O caminho 100%
Gleam não precisa de nada além de `gleam`. O caminho de inferência CUDA
precisa de um toolkit CUDA recente e uma GPU NVIDIA Ada-ou-melhor.

## Rodar um modelo com a API pública

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor
cd viva_tensor
make cutlass-libs     # archives estáticos CUTLASS + cuSPARSELt
make zig              # a NIF .so
```

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")
  let opts = t.default_generate_opts()
  let assert Ok(result) = t.generate(model, "Hello", opts)
  result.text
}
```

Esse é o caminho preferido v2.2.102 pra checkpoints HF da família Llama. A
mesma API foi validada em TinyLlama-1.1B e Llama-3.2-1B-Instruct.

## Caminho 100% Gleam (sem CUDA)

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

Isso roda em qualquer plataforma que o Gleam suporta. Nada de NIF.

## Caminho de inferência CUDA

Pré-requisitos:

- GPU NVIDIA, Ada SM89 (RTX 4090) ou mais novo recomendado
- Toolkit CUDA 12.0+ (`nvcc`)
- Driver 555+
- `zig` 0.14+ (sistema de build)
- `g++` 14 (GCC 16 tem quebra conhecida do nvcc em `<functional>`; o
  Makefile auto-detecta `g++-15` como host compiler)

Build:

```bash
make cutlass-libs     # archives estáticos CUTLASS + cuSPARSELt
make zig              # a NIF .so
gleam test            # 792 tests, todos devem passar com NIF carregada
```

Se a NIF .so não estiver presente, o mesmo `gleam test` ainda roda — só
pula os caminhos só-nativos.

## Rodar TinyLlama-1.1B end-to-end

Veja [`inference.md`](inference.md) pro walkthrough completo. Versão
rápida:

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

Esperado: o modelo imprime uma continuação de "Hello" gerada a
~2.31 ms/token pelo caminho do handle público. O runner
`dev/llama_forward.erl` é mantido pra debug avançado e bisects de kernel.

## Verifica teu install

```bash
# Checa se a NIF carregou
erl -pa build/dev/erlang/viva_tensor/ebin -noshell -eval \
    'io:format("~p~n", [viva_tensor_zig:cuda_available()]), halt(0).'
# -> true   (ou false em build CPU-only)

# Roda um matmul CUDA rápido
gleam run -m viva_tensor/bench/peak
```

## Próximos passos

- [`inference.md`](inference.md) — walkthrough Llama end-to-end completo.
- [`../api/tensor.md`](../api/tensor.md) — referência da API pública de
  tensor.
- [`../api/inference.md`](../api/inference.md) — referência prepack /
  linear / sampling / tokenizer.
- [`ffi-architecture.md`](ffi-architecture.md) — como a fronteira Gleam ↔
  NIF funciona (voltado pra mantenedor).
