# API de Inferência

`viva_tensor` expõe uma superfície estável de inferência pra FP8 dense, INT8
2:4 sparse, INT4 4:8 em pares adjacentes e a FFN SwiGLU fundida. Os mesmos tipos
opacos de handle `PackedWeight*` são usados em todos os caminhos pra callers
poderem misturar dtypes no nível do modelo.

```gleam
import viva_tensor as t
```

> Nativo (CUDA + CUTLASS) é obrigatório. Com `VIVA_NO_CUDA=1` as chamadas de
> prepack retornam `Error(_)` e as chamadas linear degradam pra
> `nif_not_loaded` no BEAM. A API de tensor 100% Gleam
> ([`tensor.md`](tensor.md)) continua disponível.

## Handles de peso empacotado

Cada dtype tem um handle opaco retornado pela sua chamada `prepack_*`. Eles
carregam o peso quantizado residente no device mais o buffer de scale
per-channel (ou per-block) que a chamada `linear_*` correspondente espera.

| Handle                       | Backed by                       | Usado por                                                                |
| :--------------------------- | :------------------------------ | :----------------------------------------------------------------------- |
| `PackedWeightFp8`            | `nt_prepack_fp8` / `_blocked`   | `linear_fp8`, `linear_fp8_w8a16`, `linear_gelu_fp8`, `linear_swiglu_fp8` |
| `PackedWeightInt8Sparse`     | `nt_prepack_int8_sparse`        | `linear_int8_sparse`                                                     |
| `PackedWeightInt4Sparse`     | `nt_prepack_int4_sparse` / `_pair_4_8` | `linear_int4_sparse`                                              |

Handles são recursos Erlang reference-counted; o buffer do device é liberado
quando o GC do BEAM coleta o handle. Código chamador NÃO deve chamar
`cudaFree` diretamente — não tem API pública de release.

## FP8 dense (E4M3)

```gleam
let assert Ok(packed) = t.prepack_fp8_weight(weight)
let assert Ok(out)    = t.linear_fp8(input, packed, bias)
```

| Função                                           | Dtype de saída | Notas                                                                                                  |
| :----------------------------------------------- | :------------- | :----------------------------------------------------------------------------------------------------- |
| `prepack_fp8_weight(weight)`                     | `PackedWeightFp8` | Scale FP8 E4M3 per-channel; FP32 armazenado no device.                                              |
| `prepack_fp8_weight_blocked(w, blk)`             | `PackedWeightFp8` | Scale per-block-K (típico `blk=16` ou `128`). Fecha o gap numérico em pesos reais de LLM.           |
| `linear_fp8(input, packed, bias)`                | Tensor (FP16)  | GEMM FP8 dense CUTLASS, buffer de saída FP32 + dequantização no host.                                  |
| `linear_fp8_w8a16(input, packed, bias)`          | Tensor (FP16)  | Input FP16 × weight FP8 via kernel de dequant + GEMM FP16 cuBLAS. Elimina o passo de quantização do input FP8. |
| `linear_gelu_fp8(input, packed, bias)`           | Tensor (FP16)  | GEMM FP8 cuBLASLt com epilogue BIAS+GELU fundido.                                                      |
| `linear_swiglu_fp8(input, gate_pk, up_pk, bias)` | Tensor (FP16)  | Dois GEMMs FP8 + silu·mul fundido com dequantização per-channel dentro do kernel.                      |

### W8A16 vs W8A8

O `linear_fp8` default quantiza o input on-the-fly (absmax per-row / 448) e
roda um GEMM FP8×FP8 verdadeiro. Pra pesos reais de LLM com signs mistos
isso pode cancelar ~50% dos canais de saída via ruído do acumulador. A
variante `_w8a16` pula a quantização do input (input fica FP16) e é
recomendada pra inferência. Veja [`guides/inference.md`](../guides/inference.md)
pra história completa de diagnóstico.

### Scales por bloco

`prepack_fp8_weight_blocked(w, block_size)` emite um scale FP32 por
`block_size` pesos no eixo K em vez de um por canal de saída. Pra
TinyLlama-1.1B `block_size=16` alinha o token argmax com a referência fp32
do HF transformers.

### Caminho público de decode LLM

Código de aplicação deve usar `viva_tensor.load_model` e
`viva_tensor.generate`; veja [`llm.md`](llm.md) pro contrato do `ModelHandle`.
Internamente, `nt_embedding_table_new/3` carrega `embed_tokens.weight` uma
vez como uma tabela FP16 residente no device. `nt_forward_decode_step/8`
então pega um token id, esse recurso de embedding, os records de layer
blocados, weights do RMSNorm final, `lm_head` empacotado, recursos de KV
cache, posição e frequências do RoPE. Ele executa o embedding lookup, todos
os blocos transformer, RMSNorm final, `lm_head` e argmax dentro de uma
chamada NIF por token decodado.

O harness de dev histórico ainda expõe esse caminho pra debug de kernel:

```sh
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
  -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

## INT8 2:4 sparse (cuSPARSELt)

```gleam
let assert Ok(packed) = t.prepack_int8_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int8_sparse(input, packed, bias)
```

Peso 2:4 podado por magnitude armazenado no formato comprimido do
cuSPARSELt. Roda ~1320 TOPS em Ada SM89. Scale de peso per-channel + scale
de input per-row, dequantizado no host após o acumulador int32 do GEMM.

## INT4 4:8 em pares adjacentes (CUTLASS Sm80)

```gleam
let assert Ok(packed) = t.prepack_int4_sparse_24_weight(weight)
let assert Ok(out)    = t.linear_int4_sparse(input, packed, bias)
```

O prepack de conveniência acima escolhe por magnitude dois pares adjacentes
dentro de cada grupo de 8 pesos no eixo K. Pra um peso podado por SparseGPT,
passe o mask autoritativo:

```gleam
let assert Ok(packed) =
  t.prepack_int4_sparse_pair_4_8_weight(weight, pair_mask)
```

`pair_mask` tem shape `[out_features, in_features / 8]`, com um byte por grupo;
cada byte precisa manter exatamente dois pares adjacentes completos. O caminho
estrito rejeita masks 2:4 escalares e nunca repoda o peso. O comando
`dev/sparsegpt_2_4.py export-pair48` gera o checkpoint HuggingFace podado, o
safetensors de masks e o manifesto. O prepack no host escreve o metadata
ElementE no layout `ColumnMajorInterleaved<2>`. Roda ~1854 TOPS.

## Sampling

Um módulo Erlang puro separado expõe as primitivas padrão de sampling:

```erlang
%% dev/llama_sampling.erl — também usado direto do Gleam via FFI helpers
sample(Logits, #{temperature => 0.8, top_k => 40, top_p => 0.95, seed => 42}).
```

| Função                    | Notas                                                                  |
| :------------------------ | :--------------------------------------------------------------------- |
| `argmax/1`                | `{TokenId, Logit}` a partir dos logits crus.                           |
| `softmax/1`               | Softmax estável (subtração do max).                                    |
| `sample/2`                | Multinomial com `temperature`, `top_k`, `top_p`, `seed`. Reprodutível. |

## Tokenizer

```gleam
let assert Ok(tk) = viva_tensor_tokenizer_ffi.load("tmp/tinyllama/tokenizer.json")
let ids = viva_tensor_tokenizer_ffi.encode(tk, "Hello")
let text = viva_tensor_tokenizer_ffi.decode(tk, ids)
```

BPE estilo SentencePiece com byte-fallback. encode/decode é bit-exact vs
HuggingFace `transformers` em TinyLlama-1.1B.

## Loader SafeTensors

```gleam
let assert Ok(header) = viva_tensor_safetensors_ffi.open_header(path)
let assert Ok(bf16)   = viva_tensor_safetensors_ffi.read_tensor_bf16(header, name)
let fp32              = viva_tensor_safetensors_ffi.bf16_to_fp32_binary(bf16)
let assert Ok(trans)  = viva_tensor_safetensors_ffi.transpose_fp32(fp32, rows, cols)
```

Parseia o header JSON via módulo `json` do OTP 27, lê bytes do tensor, e
expõe um transpose backed-by-NIF rápido (tiled 32×32, ~110× mais rápido que
o fallback Erlang puro).

## Veja também

- [`guides/inference.md`](../guides/inference.md) — walkthrough end-to-end
  completo do TinyLlama-1.1B.
- [`guides/ffi-architecture.md`](../guides/ffi-architecture.md) — contrato
  da fronteira Gleam → Erlang → C/CUDA.
- [`api/tensor.md`](tensor.md) — API de tensor 100% Gleam que não precisa
  de CUDA.
