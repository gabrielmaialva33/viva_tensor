# Estrutura do projeto

`viva_tensor` é um pacote Gleam com um façade raiz estável, módulos de
implementação interna e aceleração nativa opcional. Mantém esse split
claro ao adicionar features: usuários do pacote devem depender do contrato
público Gleam, não dos internals atuais de nativo ou planner.

## Layout do pacote

| Path                                                               | Propósito                                                                                                                                                                 |
|:-------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/viva_tensor.gleam`                                            | Façade público estável. Exemplos de usuário devem preferir `import viva_tensor as t`.                                                                                     |
| `src/viva_tensor/axis.gleam`, `layout.gleam`, `named.gleam`        | Módulos companheiros públicos pra conceitos duráveis de tensor.                                                                                                           |
| `src/viva_tensor/tensor.gleam`                                     | Implementação interna do tensor usada pelo façade. Dono de operações puras, dispatch nativo, comportamento de shape e caminhos de fallback.                               |
| `src/viva_tensor/core/`                                            | Storage interno, shape, dtype, errors, layout math e wrappers FFI.                                                                                                        |
| `src/viva_tensor/backend/`                                         | Protocolo de backend e descrições de capability usadas por código de seleção estilo planner.                                                                              |
| `src/viva_tensor/native/`                                          | Helpers nativos voltados pra Gleam pra BLAS, CUDA, kernels esparsos e diagnósticos TFLOPS/backend.                                                                        |
| `src/viva_tensor/quant.gleam`                                      | Entrypoint interno de quantização que re-exporta os módulos de quantização suportados.                                                                                    |
| `src/viva_tensor/quant/`                                           | Implementações de quantização: compressão, NF4, AWQ, preprocessing Hadamard, helpers de layout tensor-core e código de referência TurboQuant.                             |
| `src/viva_tensor/nn/`, `optim/`, `observability/`, `experimental/` | Módulos internos de domínio até os contratos deles estabilizarem o suficiente pra API pública.                                                                            |
| `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl`                  | Módulos de bridge Erlang exigidos pelo target BEAM e caminho de loading da NIF.                                                                                           |
| `zig_src/`                                                         | Implementação nativa C, CUDA e Zig pra NIF opcional.                                                                                                                      |
| `priv/`                                                            | Artefatos nativos de runtime carregados pelo Erlang quando presentes.                                                                                                     |
| `test/`                                                            | Testes unitários, de comportamento, de contrato da API pública e de compatibilidade NIF/sem-NIF.                                                                          |
| `dev/`                                                             | Exemplos Gleam só-pra-dev e entrypoints de benchmark. Esses módulos são rodáveis com `gleam run -m ...` mas não fazem parte da API suportada do pacote.                   |
| `bench/`                                                           | Scripts externos de benchmark, agrupados por runtime ou ferramenta: `python/`, `r/`, `erlang/`, `cuda/`, `scripts/` e `windows/`. `data/` e `reports/` gerados ficam ignorados. |
| `docs/`                                                            | Guias e documentação extensa escritos pelo mantenedor.                                                                                                                    |

## Fronteira da API pública

A fronteira do pacote é definida pelo `gleam.toml`: o módulo raiz e um
pequeno conjunto de módulos companheiros são públicos, enquanto `backend`,
`core`, `native`, `quant`, `tensor`, `nn`, `optim`, `observability` e
`experimental` são internos.

Promove um módulo pra fora de `internal_modules` só quando ele tem:

- comportamento documentado de shape e dtype
- falhas recuperáveis representadas com `Result`
- testes via o façade raiz ou um módulo companheiro estável
- um comportamento puro Gleam definido quando aceleração nativa não tá
  disponível
- documentação gerada que ajuda usuários do pacote, não só mantenedores

Benchmarks, demos e probes de pesquisa pertencem a `dev/` ou `bench/` até
virarem features de runtime suportadas.

## Gleam puro e fallback de NIF

Aceleração nativa é opcional. Operações públicas de tensor devem continuar
funcionando quando a NIF tá faltando, falha em carregar ou retorna um
erro.

O fluxo usual é:

1. O façade público delega pra código interno de tensor.
2. Código interno valida comportamento de shape/broadcasting no Gleam.
3. Se os inputs são tensors nativos e uma operação NIF correspondente
   existe, o caminho nativo é tentado via `core/ffi.gleam` e os módulos
   de bridge Erlang.
4. Se o caminho nativo tá indisponível ou falha, a operação cai no
   fallback puro Gleam.

Esse contrato é importante pra usuários do Hex, portabilidade de CI e dev
em máquinas sem CUDA, MKL ou um artefato `priv/viva_tensor_zig.*`
compilado. Funções que são genuinamente só-nativas devem dizer isso
explicitamente nos docs e testes.

O contrato detalhado de ownership FFI e split vive em
[`Arquitetura FFI`](../guides/ffi-architecture.md). Mantém `core/ffi.gleam`
como façade de forwarding até qualquer módulo de split `core/ffi/*` ser
validado no Gleam e migrado uma família disjunta de recurso por vez.

## Planner de backend

A seleção de backend é dividida entre pequenas camadas internas:

- `backend/protocol.gleam` define tipos de backend, checks de
  disponibilidade, operações puras, auto-seleção local e hooks de matmul
  distribuído.
- `backend/capability.gleam` descreve sobre o que o planner consegue
  raciocinar, incluindo records de capability CPU, native, CUDA e
  tensor-core.
- `native/cuda.gleam` contém o planner de aceleração de alto nível pra
  CUDA, MKL/native CPU e fallback CPU. Tensors CUDA ficam no device até
  uma fronteira de API exigir conversão de volta pra tensors CPU.
- `native/blas.gleam`, `native/sparse.gleam` e `native/tflops.gleam`
  expõem detecção de backend e diagnósticos usados por testes, benchmarks
  e decisões do planner.

Mantém o código do planner descritivo em vez de mágico: registra por que
um backend foi escolhido, preserva fallback CPU e não deixa assumptions
só-pra-benchmark vazarem pro façade estável.

## Layout de quantização

Código de quantização é intencionalmente em camadas:

- `quant/compression.gleam`, `nf4.gleam` e `awq.gleam` guardam algoritmos
  concretos de quantização.
- `quant/hadamard.gleam` e `quant/turboquant.gleam` são caminhos de
  referência puro Gleam pra preprocessing estilo Hadamard e experimentos
  low-bit antes de kernels nativos existirem.
- `quant/layout.gleam` documenta assumptions de packing orientadas a
  tensor-core, como shapes de bloco e tile.
- `zig_src/nif_quant.c` e arquivos CUDA em `zig_src/` são a landing zone
  nativa depois de um contrato de quantização ser travado pela
  implementação Gleam pura e testes.

Prefere uma implementação de referência legível primeiro. Move loops
quentes pra NIF/CUDA só depois do contrato Gleam, comportamento de input
inválido e fallback sem-NIF estarem cobertos.

## Locais dos backends nativos

O build nativo é centrado em `zig_src/build.zig`.

- MKL é ligado a partir de `zig_src/build.zig` e código CPU/NIF como
  `zig_src/nif_entry.c` e `zig_src/nif_cpu_ops.c`.
- Suporte a macOS Accelerate vive em `zig_src/accelerate.c`.
- Trabalho de CUDA e sparse GPU vive em `zig_src/cuda_*.c`,
  `zig_src/cuda_*.cu`, `zig_src/nif_cuda_*.c`, `zig_src/nif_sparse.c` e
  `zig_src/sage/`.
- Registro de NIF e declarações compartilhadas vivem em
  `zig_src/nif_entry.c` e `zig_src/viva_nif.h`.

Módulos Gleam devem chamar código nativo só pelos wrappers FFI existentes
e módulos de bridge. Não faz APIs públicas dependerem de uma biblioteca
nativa específica estar instalada, a não ser que a API seja explicitamente
documentada como só-nativa.
