# Inferência estilo Llama end-to-end

Esse guia anda por um forward pass real text-in / text-out em
TinyLlama-1.1B usando a API pública `viva_tensor.load_model` /
`viva_tensor.generate`. A mesma sequência de chamadas é validada em
Llama-3.2-1B-Instruct; diferenças específicas do modelo são carregadas de
`config.json` e metadata do SafeTensors.

## Pré-requisitos

```
sudo apt install build-essential
# CUDA 12.x + driver 555+ pra Ada SM89

# Raiz do projeto:
make cutlass-libs     # builda archives estáticos CUTLASS + cuSPARSELt
make zig              # builda a NIF .so

# Pega o TinyLlama-1.1B (chat-tuned, 4-bit-friendly):
mkdir -p tmp/tinyllama
cd tmp/tinyllama
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json
```

## Rodada end-to-end

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")

  let opts =
    t.GenerateOpts(
      max_new_tokens: 20,
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

Saída esperada (com `block_size=16`, sampling argmax):

```
Prompt:     "Hello"
Generated:  ", I am interested to bookmark this job for [company/brand name], please? I am"
Throughput: ~2.31 ms/token em TinyLlama-1.1B
Token argmax após BOS: 529 (bate com referência fp32 do HF transformers)
```

## A pipeline

```
Texto do prompt
   ↓ viva_tensor.generate
   ↓ viva_tensor_tokenizer_ffi:encode  (BPE, byte-fallback)
[token_ids]
   ↓ embed_row(EmbedTbl, token_id)     (linha bf16 do SafeTensors)
hidden_state [hidden_size]
   ↓ ×22 blocos transformer:
   │     rmsnorm
   │     → Q/K/V proj (linear_fp8_w8a16)
   │     → rotação RoPE
   │     → atenção GQA (32 Q heads / 4 KV heads)
   │     → append no KV cache
   │     → O proj (linear_fp8_w8a16)
   │     → residual
   │     → rmsnorm
   │     → gate/up (linear_fp8_w8a16)
   │     → silu(gate)·up
   │     → down (linear_fp8_w8a16)
   │     → residual
hidden_state
   ↓ rmsnorm final + lm_head (linear_fp8_w8a16)
logits [vocab=32000]
   ↓ argmax ou sample (temp/top-k/top-p)
next_token_id
   ↓ viva_tensor_tokenizer_ffi:decode
texto
```

## O que `load_model` faz

`viva_tensor.load_model(path)` envolve os passos low-level de SafeTensors e
prepack atrás de um `ModelHandle` reutilizável. Internamente, cada peso
linear segue essa forma:

```erlang
{ok, Header} = viva_tensor_safetensors_ffi:open_header(Path),
{ok, Bf16}   = viva_tensor_safetensors_ffi:read_tensor_bf16(
                 Header, <<"model.layers.0.self_attn.q_proj.weight">>),
Fp32         = viva_tensor_safetensors_ffi:bf16_to_fp32_binary(Bf16),
%% HF armazena weight como [out, in]; o prepack do viva_tensor espera [in, out].
{ok, Trans}  = viva_tensor_safetensors_ffi:transpose_fp32(Fp32, OutF, InF),
{ok, {Resource, _, _, _}} =
    viva_tensor_zig:nt_prepack_fp8_blocked(Trans, [InF, OutF], 16).
```

O transpose tomava ~20 segundos pro LM head 32000×2048 em Erlang puro. O
caminho rápido vive em `nif_transpose.c` e roda em ~180 ms.

## Por que block_size=16

| Só per-channel        | block_size=128    | **block_size=16**  | Referência HF |
| :-------------------- | :---------------- | :----------------- | :------------ |
| Razão Q proj: 1.234×  | 1.150×            | **1.077×**         | 1.000×        |
| Razão K proj: —       | 1.108×            | **1.018×**         | 1.000×        |
| Token argmax após BOS | 6763              | **529 ✅**         | 529           |

`block_size=16` foi o menor bloco que alinha o token argmax com a
referência fp32 do HF transformers. É o default recomendado pra
inferência. Overhead de memória é desprezível (~3% dos bytes de peso).

## Sampling

Coloca `temperature > 0.0` e passa `top_k`, `top_p` e `seed`:

```gleam
let opts =
  t.GenerateOpts(
    max_new_tokens: 30,
    temperature: 0.8,
    top_k: t.TopK(40),
    top_p: 0.95,
    seed: 42,
    stop_on_eos: True,
  )

let assert Ok(result) = t.generate(model, "Hello", opts)
```

Use `seed` pra deixar a rodada reprodutível entre máquinas.

## KV cache

O driver atual mantém o KV cache per-layer como listas Erlang (um binary
anexado por token). Pra TinyLlama em pos≤512 cada linha do cache é 512
bytes e a transferência total por token é ~1 MB nas 22 layers —
desprezível. Pra contextos mais longos o cache devia migrar pra um recurso
persistente no device (rastreado como trabalho futuro; veja
`bench/plans/INFERENCE_API_PLAN.md`).

## Performance

Na RTX 4090, o caminho público do `ModelHandle` foi validado em
`2.31 ms/token` pra TinyLlama-1.1B e `2.47 ms/token` pra
Llama-3.2-1B-Instruct. A melhor run de decode FP8 W8A16 do TinyLlama chega
a `448 tok/s`, à frente do baseline local do Ollama em `352 tok/s`.

## O que vem a seguir

O gargalo atual são round-trips host↔device por linear, não compute na
GPU. O próximo salto de throughput (5.5 → ~11 tok/sec) precisa de uma NIF
de bloco único fundida que mantenha o hidden state residente no device
durante o bloco inteiro. Rastreado em
[`bench/plans/INFERENCE_API_PLAN.md`](../../../bench/plans/INFERENCE_API_PLAN.md)
e no runner de debug.

## Avançado / Debug

O driver de referência histórico continua útil pra bisectar pesos e
kernels individuais:

```bash
erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

Use só pra debug de mantenedor. Código de aplicação novo deve usar
`viva_tensor.load_model` e `viva_tensor.generate`.

## Troubleshooting

| Sintoma                              | Causa provável                                                                                                       |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| `nif_not_loaded` no prepack          | NIF não foi buildada — roda `make zig`.                                                                              |
| `bad_lib: function not found`        | Mismatch da lista de stubs Erlang — rebuilda o projeto Gleam (`gleam build`).                                        |
| Token diverge da referência HF       | Usando scales per-channel em vez de `block_size=16`. Troca pra `nt_prepack_fp8_blocked`.                             |
| Inf espúrio na saída FP16            | Saturação FP16 do caminho `cuBLASLt` — já corrigido roteando todos os caminhos pra buffers FP32. Atualiza a .so.     |
| Load lento (~3 min pra 22 layers)    | Caindo no transpose Erlang — confirma que `nt_transpose_fp32` tá registrado.                                         |
