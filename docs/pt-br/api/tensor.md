# Guia da API

`viva_tensor` expõe uma superfície pequena e estável a partir do módulo raiz
`viva_tensor`. Módulos de implementação pra backends nativos, quantização,
kernels esparsos, telemetria, benchmarking e helpers experimentais de rede
neural ficam internos até os contratos deles estabilizarem.

Entrypoints de benchmark e exemplos só-pra-dev ficam em `dev/` pra rodarem
localmente sem virar parte da superfície empacotada da lib.

Veja [Política de Estabilidade](../reference/stability.md) pra fronteira
estável/experimental e as expectativas de compatibilidade pra adições no
módulo raiz. Veja [Estrutura do Projeto](../reference/project-structure.md)
pra layout do pacote e regras de fronteira entre módulos.

## Imports estáveis

```gleam
import gleam/result
import viva_tensor as t
```

Use o módulo raiz pra trabalho normal com tensor. Importe
`viva_tensor/layout` quando precisar inspecionar metadata de storage, e
`viva_tensor/axis` ou `viva_tensor/named` ao trabalhar com dimensões
semânticas.

## Criação de tensor

| Função                                   | Descrição                                              |
|:-----------------------------------------|:-------------------------------------------------------|
| `zeros(shape)`                           | Cria um tensor preenchido com zeros.                   |
| `ones(shape)`                            | Cria um tensor preenchido com uns.                     |
| `fill(shape, value)`                     | Cria um tensor preenchido com um único valor escalar.  |
| `from_list(data)`                        | Cria um tensor unidimensional.                         |
| `from_list2d(rows)`                      | Cria uma matriz a partir de linhas, validando tamanho. |
| `linspace(start, stop, steps)`           | Cria valores igualmente espaçados num intervalo fechado. |
| `try_linspace(start, stop, steps)`       | linspace falível rejeitando contagens de step inválidas. |
| `logspace(start, stop, steps, base)`     | Cria valores espaçados logaritmicamente.               |
| `try_logspace(start, stop, steps, base)` | logspace falível rejeitando steps/base inválidos.      |
| `zeros_like(tensor)`                     | Cria um tensor de zeros com a mesma shape.             |
| `ones_like(tensor)`                      | Cria um tensor de uns com a mesma shape.               |
| `full_like(tensor, value)`               | Cria um tensor preenchido com a mesma shape.           |
| `eye(n)` / `identity(n)`                 | Cria uma matriz identidade quadrada.                   |
| `try_eye(n)`                             | Matriz identidade falível rejeitando tamanho inválido. |
| `diag(tensor)`                           | Cria uma matriz diagonal a partir de um vetor.         |
| `try_diag(tensor)`                       | Criação falível de matriz diagonal.                    |
| `matrix(rows, cols, data)`               | Cria uma matriz com dimensões explícitas.              |

```gleam
let a = t.zeros([2, 3])
let b = t.fill([2, 3], 1.5)
```

## Operações falíveis

Operações de tensor que mudam shape ou são binárias retornam `Result` em
vez de panicar. Encadeia com `gleam/result.try`. Use `try_to_list()` no
lugar de `to_list()` em código falível quando falhas de materialização
nativa precisarem ser preservadas.

```gleam
pub fn example() {
  let a = t.ones([2, 3])
  let b = t.fill([2, 3], 2.0)

  use c <- result.try(t.add(a, b))
  use flat <- result.try(t.reshape(c, [6]))

  Ok(t.mean(flat))
}
```

## Math element-wise

| Função                                          | Descrição                                                                |
|:------------------------------------------------|:-------------------------------------------------------------------------|
| `add(a, b)`                                     | Adição element-wise pra shapes iguais.                                   |
| `sub(a, b)`                                     | Subtração element-wise pra shapes iguais.                                |
| `mul(a, b)`                                     | Multiplicação element-wise pra shapes iguais.                            |
| `div(a, b)`                                     | Divisão element-wise pra shapes iguais.                                  |
| `scale(tensor, scalar)`                         | Multiplica todo elemento por um escalar.                                 |
| `try_scale(tensor, scalar)`                     | Multiplicação escalar falível preservando erros de materialização nativa. |
| `add_scalar(tensor, scalar)`                    | Adiciona um escalar em todo elemento.                                    |
| `try_add_scalar(tensor, scalar)`                | Adição escalar falível preservando erros de materialização nativa.       |
| `negate(tensor)`                                | Nega todo elemento.                                                      |
| `try_negate(tensor)`                            | Negação falível preservando erros de materialização nativa.              |
| `clamp(tensor, min, max)`                       | Trava valores num intervalo fechado.                                     |
| `try_clamp(tensor, min, max)`                   | Clamp falível preservando erros de materialização nativa.                |
| `clip(tensor, min, max)`                        | Apelido pra clamp num intervalo fechado.                                 |
| `try_clip(tensor, min, max)`                    | Clip falível rejeitando intervalos inválidos.                            |
| `abs(tensor)` / `try_abs(tensor)`               | Valor absoluto pra todo elemento.                                        |
| `square(tensor)` / `try_square(tensor)`         | Eleva todo elemento ao quadrado.                                         |
| `sqrt(tensor)` / `try_sqrt(tensor)`             | Raiz quadrada, rejeitando valores negativos em `try_sqrt`.               |
| `exp(tensor)` / `try_exp(tensor)`               | Exponencial pra todo elemento.                                           |
| `log(tensor)` / `try_log(tensor)`               | Logaritmo natural, rejeitando valores não-positivos em `try_log`.        |
| `floor(tensor)` / `try_floor(tensor)`           | Floor em todo elemento.                                                  |
| `ceil(tensor)` / `try_ceil(tensor)`             | Ceiling em todo elemento.                                                |
| `round(tensor)` / `try_round(tensor)`           | Arredonda todo elemento pro inteiro mais próximo.                        |
| `sign(tensor)` / `try_sign(tensor)`             | Retorna -1, 0 ou 1 pra cada elemento.                                    |
| `reciprocal(tensor)` / `try_reciprocal(tensor)` | Recíproca, rejeitando valores zero em `try_reciprocal`.                  |
| `map(tensor, fun)`                              | Aplica uma função escalar em todo elemento.                              |
| `try_map(tensor, fun)`                          | Map escalar falível preservando erros de materialização nativa.          |
| `softmax_axis(tensor, axis)`                    | Normaliza cada slice ao longo de um eixo.                                |
| `try_softmax_axis(tensor, axis)`                | Softmax falível preservando erros de materialização nativa e indexação.  |

Use funções específicas de broadcasting quando shapes diferirem.

| Função                                          | Descrição                                          |
|:------------------------------------------------|:---------------------------------------------------|
| `can_broadcast(a, b)`                           | Checa se duas shapes são compatíveis.              |
| `broadcast_shape(a, b)`                         | Computa a shape comum pra duas shapes.             |
| `broadcast_shapes(shapes)`                      | Computa a shape comum pra várias shapes.           |
| `broadcast_to(tensor, shape)`                   | Cria uma view de broadcast quando possível.        |
| `broadcast_pair(a, b)`                          | Faz broadcast de dois tensors pra views comuns.    |
| `add_broadcast(a, b)`                           | Adição com broadcasting estilo NumPy.              |
| `sub_broadcast(a, b)`                           | Subtração com broadcasting estilo NumPy.           |
| `mul_broadcast(a, b)`                           | Multiplicação com broadcasting estilo NumPy.       |
| `div_broadcast(a, b)`                           | Divisão com broadcasting estilo NumPy.             |
| `maximum(a, b)`                                 | Máximo element-wise com broadcasting.              |
| `minimum(a, b)`                                 | Mínimo element-wise com broadcasting.              |
| `equal(a, b)` / `not_equal(a, b)`               | Máscaras de igualdade element-wise com broadcasting. |
| `greater(a, b)` / `greater_equal(a, b)`         | Máscaras de comparação element-wise com broadcasting. |
| `less(a, b)` / `less_equal(a, b)`               | Máscaras de comparação element-wise com broadcasting. |
| `where(condition, true, false)`                 | Seleciona valores usando uma máscara de condição não-zero. |
| `logical_not(mask)`                             | Inverte uma máscara numérica.                      |
| `logical_and(a, b)` / `logical_or(a, b)`        | Combina máscaras numéricas com broadcasting.       |
| `logical_xor(a, b)`                             | Ou-exclusivo sobre máscaras numéricas.             |
| `any(mask)` / `all(mask)`                       | Reduz uma máscara numérica pra booleano.           |
| `count_nonzero(tensor)`                         | Conta valores não-zero do tensor.                  |
| `any_axis(mask, axis)` / `all_axis(mask, axis)` | Reduz máscaras numéricas ao longo de um eixo.      |
| `count_nonzero_axis(tensor, axis)`              | Conta valores não-zero ao longo de um eixo.        |
| `take(tensor, indices)`                         | Pega valores achatados por índices explícitos.     |
| `nonzero(tensor)`                               | Retorna índices não-zero achatados como floats.    |
| `masked_select(tensor, mask)`                   | Seleciona valores achatados usando máscara de broadcast. |

## Reduções

| Função                                 | Descrição                                                                       |
|:---------------------------------------|:--------------------------------------------------------------------------------|
| `sum(tensor)`                          | Soma todos os elementos.                                                        |
| `try_sum(tensor)`                      | Soma falível preservando erros de materialização nativa.                        |
| `sum_axis(tensor, axis)`               | Soma ao longo de um eixo.                                                       |
| `try_sum_axis(tensor, axis)`           | Soma falível ao longo de um eixo.                                               |
| `sum_axis_keepdims(tensor, axis)`      | Soma ao longo de um eixo mantendo dimensão size-1.                              |
| `mean(tensor)`                         | Média sobre todos elementos.                                                    |
| `try_mean(tensor)`                     | Média falível preservando erros de materialização e tensor vazio.               |
| `product(tensor)`                      | Produto sobre todos elementos.                                                  |
| `try_product(tensor)`                  | Produto falível preservando erros de materialização.                            |
| `cumsum(tensor)`                       | Soma cumulativa sobre valores achatados, preservando shape.                     |
| `try_cumsum(tensor)`                   | Soma cumulativa falível preservando erros de materialização.                    |
| `cumsum_axis(tensor, axis)`            | Soma cumulativa ao longo de um eixo, preservando shape.                         |
| `try_cumsum_axis(tensor, axis)`        | Soma cumulativa falível ao longo de um eixo.                                    |
| `cumprod(tensor)`                      | Produto cumulativo sobre valores achatados, preservando shape.                  |
| `try_cumprod(tensor)`                  | Produto cumulativo falível preservando erros de materialização.                 |
| `cumprod_axis(tensor, axis)`           | Produto cumulativo ao longo de um eixo, preservando shape.                      |
| `try_cumprod_axis(tensor, axis)`       | Produto cumulativo falível ao longo de um eixo.                                 |
| `median(tensor)`                       | Mediana sobre todos elementos.                                                  |
| `try_median(tensor)`                   | Mediana falível preservando erros de materialização e tensor vazio.             |
| `percentile(tensor, percentile)`       | Percentil usando interpolação linear.                                           |
| `try_percentile(tensor, percentile)`   | Percentil falível com bordas explícitas e erros de tensor vazio.                |
| `mean_axis(tensor, axis)`              | Média ao longo de um eixo.                                                      |
| `try_mean_axis(tensor, axis)`          | Média falível ao longo de um eixo.                                              |
| `mean_axis_keepdims(tensor, axis)`     | Média ao longo de um eixo mantendo dimensão size-1.                             |
| `variance_axis(tensor, axis)`          | Variância ao longo de um eixo.                                                  |
| `try_variance_axis(tensor, axis)`      | Variância falível ao longo de um eixo.                                          |
| `variance_axis_keepdims(tensor, axis)` | Variância ao longo de um eixo mantendo dimensão size-1.                         |
| `std_axis(tensor, axis)`               | Desvio padrão ao longo de um eixo.                                              |
| `try_std_axis(tensor, axis)`           | Desvio padrão falível ao longo de um eixo.                                      |
| `std_axis_keepdims(tensor, axis)`      | Desvio padrão ao longo de um eixo mantendo dimensão size-1.                     |
| `max_axis(tensor, axis)`               | Máximo ao longo de um eixo.                                                     |
| `try_max_axis(tensor, axis)`           | Máximo falível ao longo de um eixo.                                             |
| `max_axis_keepdims(tensor, axis)`      | Máximo ao longo de um eixo mantendo dimensão size-1.                            |
| `min_axis(tensor, axis)`               | Mínimo ao longo de um eixo.                                                     |
| `try_min_axis(tensor, axis)`           | Mínimo falível ao longo de um eixo.                                             |
| `min_axis_keepdims(tensor, axis)`      | Mínimo ao longo de um eixo mantendo dimensão size-1.                            |
| `argmax_axis(tensor, axis)`            | Índice argmax ao longo de um eixo, como floats.                                 |
| `try_argmax_axis(tensor, axis)`        | Índice argmax falível ao longo de um eixo.                                      |
| `argmin_axis(tensor, axis)`            | Índice argmin ao longo de um eixo, como floats.                                 |
| `try_argmin_axis(tensor, axis)`        | Índice argmin falível ao longo de um eixo.                                      |
| `max(tensor)`                          | Valor máximo.                                                                   |
| `try_max(tensor)`                      | Máximo falível preservando erros de materialização e tensor vazio.              |
| `min(tensor)`                          | Valor mínimo.                                                                   |
| `try_min(tensor)`                      | Mínimo falível preservando erros de materialização e tensor vazio.              |
| `argmax(tensor)`                       | Índice flat do valor máximo.                                                    |
| `try_argmax(tensor)`                   | Índice flat falível do valor máximo.                                            |
| `argmin(tensor)`                       | Índice flat do valor mínimo.                                                    |
| `try_argmin(tensor)`                   | Índice flat falível do valor mínimo.                                            |
| `variance(tensor)`                     | Variância sobre todos elementos.                                                |
| `try_variance(tensor)`                 | Variância falível preservando erros de materialização e tensor vazio.           |
| `std(tensor)`                          | Desvio padrão sobre todos elementos.                                            |
| `try_std(tensor)`                      | Desvio padrão falível preservando erros de materialização e tensor vazio.       |

## Álgebra linear

| Função                       | Descrição                                                                |
|:-----------------------------|:-------------------------------------------------------------------------|
| `dot(a, b)`                  | Produto interno pra vetores.                                             |
| `matmul(a, b)`               | Multiplicação de matrizes.                                               |
| `matmul_planned(a, b)`       | Multiplicação usando o planner estável de backend com fallback.          |
| `matmul_vec(matrix, vector)` | Multiplicação matriz-vetor.                                              |
| `transpose(tensor)`          | Transpose de matriz.                                                     |
| `outer(a, b)`                | Produto externo.                                                         |

Variantes backed-by-NIF como `matmul_into`, `to_accelerated` e
`matmul_accelerated_into` ficam disponíveis no módulo raiz pra caminhos
quentes que podem reusar buffers ou memória GPU persistente.

Use `capabilities()` pra inspecionar se a VM atual carregou a NIF nativa, o
backend Zig SIMD, quais backends de TFLOPS tão visíveis e os records de
capability do backend estável. Use `backend_capabilities()` quando só
precisar da tabela de capability, ou `plan_backend(operation)` pra ver qual
backend o planner estável escolheria pra uma operação. Plans incluem entries
`rejected` com razões legíveis pra backends indisponíveis ou não-adequados.

Use `hardware_profiles()` ao planejar pra trabalho específico de
acelerador: hardware atual é marcado disponível só quando detectado,
enquanto perfis futuros como Blackwell, Rubin, Vera e Rubin CPX continuam
explícitos mas indisponíveis até um caminho de runtime poder provar
suporte.

```gleam
let plan = t.plan_backend(t.OperationMatmul(m: 1024, n: 1024, k: 1024))
```

## Prontidão pra quantização

| Função                                                 | Descrição                                                       |
|:-------------------------------------------------------|:----------------------------------------------------------------|
| `nvfp4_block_scaled_layout(shape)`                     | Descreve um layout NVFP4 micro-block estilo Rubin.              |
| `int2_progressive_layout(shape, block_size)`           | Descreve layout experimental INT2 progressive quantization.     |
| `int3_progressive_layout(shape, block_size)`           | Descreve layout experimental INT3 progressive quantization.     |
| `quant_layout_memory_bytes(layout)`                    | Estima bytes do payload pra um layout quantizado.               |
| `quant_layout_compression_ratio_against(layout, bits)` | Estima compressão vs uma largura de elemento baseline.          |
| `quant_layout_is_rubin_native_candidate(layout)`       | Checa se um layout bate com as assumptions Rubin micro-block.   |
| `try_hadamard_preprocess(tensor, seed)`                | Aplica preprocessing Hadamard randomizado reversível a um vetor. |
| `try_inverse_hadamard_preprocess(plan)`                | Restaura um vetor após preprocessing Hadamard.                  |
| `try_normalized_walsh_hadamard(values)`                | Transforma dados de vetor potência-de-dois com WHT normalizada. |

## Shape e layout

| Função                        | Descrição                                                                |
|:------------------------------|:-------------------------------------------------------------------------|
| `shape(tensor)`               | Dimensões do tensor.                                                     |
| `size(tensor)`                | Contagem total de elementos.                                             |
| `rank(tensor)`                | Número de dimensões.                                                     |
| `reshape(tensor, shape)`      | Muda shape preservando contagem de elementos.                            |
| `device(tensor)`              | Classe de device do payload.                                             |
| `dtype(tensor)`               | Tipo do elemento do tensor.                                              |
| `try_to_list(tensor)`         | Materializa dados do tensor preservando falhas nativas.                  |
| `flatten(tensor)`             | Converte pra uma dimensão.                                               |
| `try_flatten(tensor)`         | Flatten falível preservando falhas de materialização.                    |
| `squeeze(tensor)`             | Remove dimensões size-one.                                               |
| `unsqueeze(tensor, axis)`     | Insere uma dimensão size-one.                                            |
| `try_unsqueeze(tensor, axis)` | Unsqueeze falível preservando erros de axis inválido.                    |
| `to_strided(tensor)`          | Converte dados dense numa view strided zero-copy.                        |
| `try_to_strided(tensor)`      | Conversão strided falível preservando erros de materialização nativa.    |
| `to_contiguous(tensor)`       | Materializa uma view strided em storage dense contíguo.                  |
| `try_to_contiguous(tensor)`   | Conversão contígua falível preservando erros de materialização.          |
| `layout(tensor)`              | Inspeciona storage, device, dtype, strides, offset e contiguity.         |

```gleam
let info = t.layout(t.zeros([2, 3]))
```

Broadcasting, squeeze, unsqueeze e reshape contíguo preservam views
strided quando possível. Chama `to_contiguous()` antes de um caminho
nativo pesado se uma view fosse mais lenta que um buffer dense.

## Utilidades

| Função                           | Descrição                                                  |
|:---------------------------------|:-----------------------------------------------------------|
| `norm(tensor)`                   | Norma L2.                                                  |
| `try_norm(tensor)`               | Norma L2 falível preservando erros de materialização.      |
| `normalize(tensor)`              | Normaliza pra comprimento unitário.                        |
| `try_normalize(tensor)`          | Normalização falível preservando erros de materialização.  |
| `abs(tensor)`                    | Valor absoluto pra todo elemento.                          |
| `square(tensor)`                 | Eleva todo elemento ao quadrado.                           |
| `sqrt(tensor)`                   | Raiz quadrada de todo elemento.                            |
| `try_sqrt(tensor)`               | Raiz quadrada falível rejeitando valores negativos.        |
| `exp(tensor)`                    | Exponencial pra todo elemento.                             |
| `log(tensor)`                    | Logaritmo natural pra todo elemento.                       |
| `try_log(tensor)`                | Log natural falível rejeitando valores não-positivos.      |
| `is_close(a, b, rtol, atol)`     | Compara dois escalares com tolerâncias numéricas.          |
| `all_close(a, b, rtol, atol)`    | Compara dois tensors element-wise com tolerâncias.         |
| `euclidean_distance(a, b)`       | Distância euclidiana pra tensors com mesma shape.          |
| `try_euclidean_distance(a, b)`   | Distância euclidiana falível.                              |
| `manhattan_distance(a, b)`       | Distância de Manhattan pra tensors com mesma shape.        |
| `cosine_similarity(a, b)`        | Similaridade de cosseno pra tensors com mesma shape.       |
| `dot_similarity(a, b)`           | Similaridade por dot product pra tensors com mesma shape.  |
| `zscore(tensor)`                 | Padronização Z-score sobre todos elementos.                |
| `standardize(tensor)`            | Apelido pra `zscore`.                                      |
| `minmax_scale(tensor, min, max)` | Escala valores num intervalo alvo.                         |
| `clip_by_norm(tensor, max_norm)` | Trava a norma L2 num valor máximo.                         |

## Módulos companheiros públicos

| Módulo               | Propósito                                          |
|:---------------------|:---------------------------------------------------|
| `viva_tensor/layout` | Metadata canônica de layout de tensor.             |
| `viva_tensor/axis`   | Nomes semânticos de eixos e especificações.        |
| `viva_tensor/named`  | Wrapper de tensor com eixos nomeados.              |

## Política de estabilidade

Módulos públicos são documentados pelo `gleam docs build` e devem evitar
panics, preferir `Result` pra erros recuperáveis, preservar compatibilidade
semver e manter um fallback portátil quando possível. Módulos internos
podem mudar enquanto as APIs de aceleração nativa, quantização, sparse e
rede neural continuam amadurecendo. A política detalhada vive em
[Política de Estabilidade](../reference/stability.md).
