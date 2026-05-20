# Começando

Caminho rápido do zero até um programa rodando. O caminho pure-Gleam
precisa apenas de `gleam`. O caminho de inferência CUDA precisa de
toolkit CUDA recente e uma GPU NVIDIA Ada ou mais nova.

## Caminho pure-Gleam

```bash
gleam new my_app
cd my_app
gleam add viva_tensor
```

```gleam
// src/my_app.gleam
import gleam/io
import gleam/string
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

Funciona em qualquer plataforma que o Gleam suporta. Sem NIF.

## Caminho com CUDA

Pré-requisitos:

- GPU NVIDIA, Ada SM89 (RTX 4090) ou mais nova recomendado
- CUDA 12.0+ (`nvcc`)
- Driver 555+
- `zig` 0.14+ (sistema de build)
- `g++` 14 (GCC 16 tem problema conhecido com `nvcc` em `<functional>`;
  o Makefile detecta `g++-15` automaticamente como host compiler)

Build:

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor
cd viva_tensor
make cutlass-libs     # arquivos estáticos CUTLASS + cuSPARSELt
make zig              # o .so do NIF
gleam test            # 789 testes, todos devem passar com NIF carregado
```

Se o .so do NIF não estiver presente, `gleam test` ainda roda — apenas
pula os caminhos native-only.

## Rodar TinyLlama-1.1B end-to-end

Versão curta (veja [`../../en/guides/inference.md`](../../en/guides/inference.md)
para o walkthrough completo em inglês):

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

Esperado: o modelo imprime uma continuação de "Hello" gerada a ~5.6 tok/s.

## Verificar instalação

```bash
# Confere se o NIF carregou
erl -pa build/dev/erlang/viva_tensor/ebin -noshell -eval \
    'io:format("~p~n", [viva_tensor_zig:cuda_available()]), halt(0).'
# -> true   (ou false em build CPU-only)

# Roda um matmul CUDA rapidinho
gleam run -m viva_tensor/bench/peak
```

## Próximos passos

- [`api.md`](../api.md) — referência da API tensor (em português).
- [`../../en/api/inference.md`](../../en/api/inference.md) — prepack /
  linear / sampling / tokenizer (em inglês).
- [`../../en/guides/inference.md`](../../en/guides/inference.md) —
  walkthrough completo Llama end-to-end (em inglês).
