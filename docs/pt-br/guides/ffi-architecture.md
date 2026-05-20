# Arquitetura FFI

Essa página define o contrato de ownership pro código FFI do `viva_tensor`.
É um contrato voltado pra mantenedor, não uma garantia de API pública.

## Fronteira atual

O caminho de chamada suportado é:

```text
API pública Gleam
  -> src/viva_tensor/tensor.gleam ou módulos de domínio
  -> src/viva_tensor/core/ffi.gleam
  -> src/viva_tensor_ffi.erl, src/viva_tensor_nif.erl ou src/viva_tensor_zig.erl
  -> zig_src/
```

`src/viva_tensor/core/ffi.gleam` continua sendo o único façade de
compatibilidade pra código Gleam. Callers existentes devem continuar
importando ele direto até um split ser completado e validado.

## Regras de ownership

| Área                    | Dono                                                | Regra                                                                       |
|:------------------------|:----------------------------------------------------|:----------------------------------------------------------------------------|
| API pública             | `src/viva_tensor.gleam` e companheiros documentados | Não pode expor requisitos só-nativos a não ser que explicitamente documentado. |
| Comportamento de tensor | `src/viva_tensor/tensor.gleam` e módulos de domínio | Dono da seleção de fallback e semântica de tensor.                          |
| Façade FFI              | `src/viva_tensor/core/ffi.gleam`                    | Dono dos nomes de wrapper interno estável usados pelos call sites Gleam.    |
| Módulos de FFI split    | `src/viva_tensor/core/ffi/*`                        | Pode ter ownership de wrappers internos agrupados depois da compatibilidade de import validada. |
| Bridge Erlang           | `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl`   | Dono dos exports do módulo BEAM e stubs da NIF.                             |
| Implementação nativa    | `zig_src/`                                          | Dono dos detalhes de implementação C, CUDA e Zig.                           |
| Documentação            | `docs/en/ffi-architecture.md`                       | Dono do contrato de split e regras de migração.                             |

## Contrato de split

Módulos FFI futuros embaixo de `src/viva_tensor/core/ffi/` devem ser
disjuntos por backend ou família de recurso. Não dividir por nomes
arbitrários de operação se isso criar ownership duplicada sobre o mesmo
tipo de recurso.

Grupos recomendados:

- `core/ffi/erlang_array.gleam`: `ErlangArray` e helpers puros de array
  Erlang.
- `core/ffi/math.gleam`: wrappers finos de `math` e `rand` do Erlang.
- `core/ffi/native_tensor.gleam`: construtores de recurso
  `NativeTensorRef`, operações element-wise, reduções, operações
  matriciais, mutação e kernels CPU fundidos.
- `core/ffi/cuda.gleam`: famílias de recurso de tensor CUDA.
- `core/ffi/research.gleam`: LNS, Horde, HDC, sparse, quantizado e
  recursos nativos experimentais até graduarem pra um dono específico de
  domínio.

Cada módulo de split deve ter ownership dos seus tipos de recurso e dos
seus bindings `@external` privados junto. Um wrapper não pode viver num
módulo enquanto seu tipo opaco ou declaração externa correspondente vive
em outro módulo a não ser que tenha um módulo de tipo compartilhado
deliberado.

## Regras de migração

1. Adiciona módulos de split primeiro sem mudar call sites existentes.
2. Valida que o Gleam aceita `src/viva_tensor/core/ffi.gleam` e
   `src/viva_tensor/core/ffi/*.gleam` juntos nesse pacote.
3. Move um grupo disjunto de cada vez e mantém `core/ffi.gleam` como
   façade de forwarding.
4. Roda formatação, type checking, testes sem-NIF e testes do caminho
   nativo depois de cada grupo.
5. Só atualiza `tensor.gleam` ou o façade público quando a mudança for
   puramente mecânica e o caminho de import antigo continuar disponível.

## Requisitos de fallback

Aceleração nativa é opcional. Novos wrappers FFI devem retornar valores
`Result` recuperáveis pra falhas nativas, exceto se envolverem funções
determinísticas da stdlib Erlang. Código a nível de tensor continua
responsável por escolher execução nativa e cair no comportamento Gleam
puro.

Nenhum módulo de split pode exigir uma NIF compilada no load time do
pacote. Se a NIF tá faltando, o pacote precisa continuar compilando e o
caminho sem-NIF precisa continuar testável.
