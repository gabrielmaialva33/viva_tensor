# API LLM ModelHandle

`viva_tensor` expõe uma API pública `ModelHandle` pra modelos da família Llama
do HuggingFace armazenados como SafeTensors. Ela empacota o caminho de decode
de produção do TinyLlama em duas chamadas: carrega o modelo uma vez e depois
gera a partir do handle cacheado.

A API foi desenhada pra checkpoints HF BF16 locais com os nomes de tensor
padrão do Llama:

- `model.embed_tokens.weight`
- `model.layers.N.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.N.mlp.{gate,up,down}_proj.weight`
- `model.layers.N.{input,post_attention}_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

Se tiver um `config.json` do lado do arquivo SafeTensors, o `viva_tensor` lê
hidden size, contagem de layers, contagem de heads, contagem de KV heads,
epsilon do RMSNorm, theta do RoPE, intermediate size e vocab size dali. Caso
contrário, infere o que dá das shapes dos tensores e usa os defaults
compatíveis com o TinyLlama.

## Gleam

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")

  let opts =
    t.GenerateOpts(
      max_new_tokens: 50,
      temperature: 0.0,
      top_k: t.TopKInfinity,
      top_p: 1.0,
      seed: 42,
      stop_on_eos: True,
    )

  let assert Ok(result) = t.generate(model, "Hello", opts)
  result.text
}
```

`temperature: 0.0` usa a NIF fundida de argmax do decode-step pra
reprodutibilidade byte-idêntica. `temperature > 0.0` usa logits top-k
fundidos mais temperature, top-k, top-p e sampling multinomial com seed no
host.

Pra sampling reprodutível:

```gleam
let opts =
  t.GenerateOpts(
    max_new_tokens: 20,
    temperature: 0.8,
    top_k: t.TopK(40),
    top_p: 0.95,
    seed: 42,
    stop_on_eos: True,
  )

let assert Ok(result) = t.generate(model, "Hello", opts)
```

## Erlang

```erlang
{ok, Model} = viva_tensor_llm:load(
    <<"tmp/tinyllama/model.safetensors">>,
    #{block_size => 16}
),

{ok, Result} = viva_tensor_llm:generate(
    Model,
    <<"Hello">>,
    #{max_new_tokens => 50, temperature => 0.0}
),

#{tokens := Tokens,
  text := Text,
  ms_per_token := MsPerToken,
  total_tokens := TotalTokens} = Result.
```

## Opções de Load

`viva_tensor_llm:load/2` aceita:

| Opção | Default | Notas |
| :-- | :-- | :-- |
| `num_layers` | detectado do SafeTensors / `config.json` | Quantidade de blocos do decoder pra carregar. |
| `block_size` | `16` | Tamanho do prepack blocado FP8 usado pelo caminho decode-step. |
| `tokenizer_path` | `<model>_tokenizer.json`, depois fallback pro `tokenizer.json` irmão | JSON do tokenizer HF. |

## Opções de Geração

`viva_tensor_llm:generate/3` aceita:

| Opção | Default | Notas |
| :-- | :-- | :-- |
| `max_new_tokens` | `50` | Máximo de tokens gerados. |
| `temperature` | `0.0` | `0.0` mantém o caminho argmax e reprodutibilidade absoluta; valores acima de zero ativam sampling. |
| `top_k` | `infinity` | Cap do candidato pra sampling. `infinity` usa até 256 logits top-k fundidos; valores explícitos são capados em 256. |
| `top_p` | `1.0` | Probabilidade do nucleus sampling aplicada sobre o conjunto fundido de candidatos. |
| `seed` | `42` | Seed determinístico; o mesmo prompt, modelo e opções reproduzem os mesmos tokens amostrados. |
| `stop_on_eos` | `true` | Para depois de emitir EOS. |

## Cached vs Per Call

O `ModelHandle` cacheia:

- estado do tokenizer
- tabela de embedding BF16 ou F16 como recurso nativo
- todos os pesos das layers prepackados com scales FP8 blocados
- pesos empacotados de QKV e gate-up fundidos
- bytes do RMSNorm final
- `lm_head` empacotado
- bytes das frequências do RoPE
- metadata do modelo de `config.json` e shapes dos tensores

Cada chamada `generate` aloca KV caches frescos antes do prefill. Recursos de
KV cache são mutáveis durante o decode, então são intencionalmente por
chamada pra manter um `ModelHandle` reutilizável entre prompts.

## Modelos testados

| Modelo | Status | Velocidade de decode | Notas |
| :-- | :-- | --: | :-- |
| TinyLlama-1.1B-Chat-v1.0 | validado | `2.31 ms/token` | `head_dim=64`, fast path GQA, tokenizer BPE byte-level. |
| Llama-3.2-1B-Instruct | validado | `2.47 ms/token` | SafeTensors sharded, embeddings tied / `lm_head`, caminho do tokenizer Llama-3. |
| NousResearch/Llama-2-7b-chat-hf | validado | `113.18 ms/token` | SafeTensors sharded F16, `head_dim=128`, sem GQA; exercita o caminho CUDA fallback dinâmico. |

A mesma API pública dirige ambos modelos:

```gleam
let assert Ok(model) = t.load_model("tmp/llama32_1b/model-00001-of-00002.safetensors")
let opts = t.default_generate_opts()
let assert Ok(result) = t.generate(model, "Hello", opts)
```

## Performance

Na RTX 4090, a API atual do handle público foi validada em
`2.31 ms/token` pra TinyLlama-1.1B, `2.47 ms/token` pra
Llama-3.2-1B-Instruct, e `113.18 ms/token` pra
NousResearch/Llama-2-7b-chat-hf. O run do Llama-2-7B é funcional e coerente,
mas bem mais lento porque exercita o caminho dinâmico `head_dim=128` atual.
A melhor run de decode FP8 W8A16 do TinyLlama chega a `448 tok/s`, à frente
do baseline local do Ollama em `352 tok/s`.

A geração ainda chama `nt_forward_decode_step/8` uma vez por token decodado.
O prefill também é token-por-token hoje; um caminho de prefill em batch é
trabalho futuro.

## Limitações

- **Phi-2 não é alvo drop-in.** A arquitetura e nomenclatura de tensor dele
  divergem do contrato de loader da família Llama usado pelo `ModelHandle`.
- **Llama-2-7B usa o caminho de atenção dinâmico lento hoje.**
  `NousResearch/Llama-2-7b-chat-hf` valida loading F16 sharded e correctness
  de `head_dim=128`, mas o throughput de decode ainda não foi otimizado.
- **Sem caminho de prefill em batch ainda.** O kernel de decode tá otimizado
  pra `batch=1`; processamento de prompt em batch ainda é expresso como
  chamadas decode-step repetidas.
- **FP8×FP8 verdadeiro não é usado pro decode do LLM.** O caminho validado
  numericamente é W8A16 com pesos FP8 blocados. Quantizar a ativação de
  token único economizaria só uns poucos KB por token enquanto arriscaria o
  comportamento argmax/EOS já validado no TinyLlama e Llama-3.2-1B.
