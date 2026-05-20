# Política de estabilidade

`viva_tensor` trata o módulo raiz `viva_tensor` como a superfície estável
de usuário. Novos exemplos voltados pra usuário devem começar com:

```gleam
import viva_tensor as t
```

Isso mantém o pacote usável como biblioteca mesmo enquanto kernels
nativos, quantização, formatos esparsos e helpers de rede neural continuam
evoluindo.

## Superfície estável

A superfície pública estável é:

| Módulo               | Status | Propósito                                                                                                            |
|:---------------------|:------:|:---------------------------------------------------------------------------------------------------------------------|
| `viva_tensor`        | Estável | Criação de tensor, math, reduções, inspeção de layout, broadcasting, planejamento de backend e execução fallback segura. |
| `viva_tensor/layout` | Estável | Metadata canônica de layout de tensor.                                                                              |
| `viva_tensor/axis`   | Estável | Nomes semânticos de eixos e especificações de eixo.                                                                 |
| `viva_tensor/named`  | Estável | Wrapper de tensor com eixos nomeados.                                                                               |

Funções estáveis devem preservar compatibilidade semver, retornar `Result`
pra falhas recuperáveis e manter um fallback BEAM puro a não ser que a
função seja explicitamente documentada como só-nativa.

Operações de tensor falíveis não devem converter silenciosamente falhas de
backend ou materialização em tensors vazios, zeros ou valores parcialmente
computados. Quando dados precisam ser materializados de storage nativo,
usa `try_to_list()` em caminhos que retornam `Result`.

Funções de conveniência legadas que retornam `Tensor` ou `Float` simples
continuam por compatibilidade, mas código novo sério deve preferir
variantes falíveis como `try_map()`, `try_scale()` e `try_sum()` quando
storage nativo possa estar envolvido.

## Superfície experimental

As seguintes áreas são intencionalmente experimentais até os contratos
delas serem documentados e cobertos por testes de compatibilidade:

- `viva_tensor/core/*`
- `viva_tensor/backend/*`
- módulos diretos de CUDA, BLAS, sparse, quantization, neural network,
  optimization, telemetry e benchmark
- exemplos só-pra-dev e entrypoints de benchmark embaixo de `dev/`
- entrypoints NIF nativos que expõem detalhes de recurso específicos do
  backend

Módulos experimentais podem mudar de forma conforme a implementação
amadurece. Prefere o módulo raiz a não ser que esteja trabalhando em
internals ou benchmarkando um backend específico.

As regras de ownership de diretório e módulo são documentadas em
[Estrutura do Projeto](project-structure.md).

## Guardrails da API pública

Cada adição estável de API deve incluir:

- um comentário de doc que diga o que a função faz
- restrições de argumento e shape quando relevante
- comportamento de return e erro
- seleção de backend ou comportamento de fallback quando aceleração nativa
  estiver envolvida
- pelo menos um teste de contrato de API pública quando a função pertence
  ao módulo raiz

A suite `test/public_api_contract_test.gleam` é o tripwire de
compatibilidade pro façade raiz. Ela verifica criação, metadata de
layout, broadcasting, softmax, álgebra linear e planejamento de backend
via `import viva_tensor as t`.

Broadcasting segue a convenção madura de biblioteca de tensor usada por
NumPy e PyTorch: shapes são alinhadas à direita, dimensões batem quando
iguais ou quando um lado é `1`, e tensors expandidos são representados
como views com strides zero quando possível.

## Maturidade dos backends

Execução puro Gleam é o baseline portátil. Zig SIMD, MKL, CUDA FP32, CUDA
FP16, CUDA INT8, sparse, FP8 e kernels fundidos são expostos via records
de capability e o planner de backend primeiro. Módulos diretos de backend
de baixo nível não devem ser tratados como estáveis até contratos de
operação, suporte de dtype, restrições de shape, comportamento de erro e
regras de fallback estarem documentados.
