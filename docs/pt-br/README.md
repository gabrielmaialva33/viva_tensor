# viva_tensor

**[English](../en/README.md)** | **[中文](../zh-cn/README.md)**

Biblioteca de tensors para Gleam/BEAM com API pública pure-Gleam e NIF
opcional CUDA + CUTLASS para inferência de alto throughput. O mesmo
código `import viva_tensor as t` roda num notebook sem CUDA e numa
RTX 4090 com FP8 Tensor Cores.

> **Idioma canônico**: a documentação completa está em
> [`docs/en/`](../en/README.md). Esta pasta cobre o essencial em
> português — abra uma issue ou PR se quiser mais páginas traduzidas.

## O que está pronto hoje

| Subsistema                          | Status        | Destaques                                                                          |
| :---------------------------------- | :------------ | :--------------------------------------------------------------------------------- |
| API tensor pure-Gleam               | Estável       | Shape, broadcast, autograd básico, named axes, fallback puro.                      |
| FP8 dense (CUTLASS + cuBLASLt)      | Produção      | ~588 TFLOPS na RTX 4090. Buffer de saída FP32 (sem saturação FP16).                |
| FP8 W8A16 + escalas per-block-16    | Produção      | Fecha o gap numérico vs HF transformers; argmax bate fp32 reference.               |
| SwiGLU fundido (NIF)                | Produção      | Kernel único para `silu(gate)·up` com dequant per-channel dentro do kernel.        |
| INT8 2:4 sparse (cuSPARSELt)        | Produção      | ~1320 TOPS. Metadata byte-exata via reorder_meta shim.                             |
| INT4 2:4 sparse (CUTLASS Sm80)      | Produção      | ~1854 TOPS. Pipeline byte-exato (kernel + reorder + encoding self-tested).         |
| Loader SafeTensors                  | Funcional     | bf16 → fp32, transpose via NIF (3 min → 25 s para TinyLlama).                       |
| Tokenizer BPE                       | Funcional     | Encode/decode bit-exato vs HuggingFace transformers.                               |
| Llama-1.1B forward end-to-end       | Funcional     | 22 layers + RoPE + GQA + KV cache + LM head + argmax. Mesmo token que HF fp32.    |
| Sampling avançado (temp/top-k/p)    | Funcional     | Multinomial com seed reprodutível.                                                  |

## Início rápido

```bash
gleam add viva_tensor
```

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(a) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(b) = t.matrix(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
  let assert Ok(c) = t.matmul(a, b)
  c |> t.to_list
}
```

Para o caminho de inferência (requer CUDA):

```gleam
let assert Ok(packed) = t.prepack_fp8_weight(weight_tensor)
let assert Ok(logits) = t.linear_fp8(input, packed, None)
```

Veja [`guides/getting-started.md`](guides/getting-started.md) para o passo
completo de instalação + primeiro programa rodando.

## Recursos em português

| Página                                                  | O que é                                                              |
| :------------------------------------------------------ | :------------------------------------------------------------------- |
| [`guides/getting-started.md`](guides/getting-started.md) | Instalação, build, primeiro programa.                                |
| [`api.md`](api.md)                                      | Referência rápida da API tensor pure-Gleam.                          |
| [`paper.md`](paper.md)                                  | Paper técnico (versão em português).                                  |

Para tópicos não traduzidos — guia de inferência fim-a-fim, referência
de FP8/INT4, contrato FFI — abra o [`docs/en/`](../en/README.md).
