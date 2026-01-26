# Início Rápido

## Instalação

```bash
gleam add viva_tensor
```

## Requisitos

| Ferramenta | Versão |
|:-----------|:-------|
| Gleam | >= 1.6 |
| Erlang/OTP | >= 27 |

## Primeiro Exemplo

```gleam
import viva_tensor/tensor
import viva_tensor/nf4

pub fn main() {
  // Criar tensor aleatório
  let t = tensor.random_uniform([1024, 512])

  // Quantizar para NF4 (8x compressão)
  let config = nf4.default_config()
  let compressed = nf4.quantize(t, config)

  // Verificar compressão
  io.println("Ratio: " <> float.to_string(compressed.compression_ratio))

  // Restaurar
  let restored = nf4.dequantize(compressed)
}
```

## Pipeline de Compressão

```mermaid
sequenceDiagram
    participant U as Usuário
    participant T as Tensor
    participant Q as Quantizador
    participant M as Memória

    U->>T: Criar tensor FP32
    T->>Q: Enviar para quantização
    Q->>Q: Calcular scales
    Q->>Q: Mapear para níveis
    Q->>M: Armazenar comprimido
    M-->>U: 8x menos memória
```

## Escolhendo o Algoritmo

```mermaid
flowchart TD
    Start[Tensor FP32] --> Q1{Precisa de velocidade?}

    Q1 -->|Sim| INT8[INT8 - 4x]
    Q1 -->|Não| Q2{Tem dados de calibração?}

    Q2 -->|Sim| AWQ[AWQ - 7.7x]
    Q2 -->|Não| NF4[NF4 - 7.5x]
```

| Algoritmo | Compressão | Quando usar |
|:----------|:----------:|:------------|
| INT8 | 4x | Velocidade, inferência simples |
| NF4 | 7.5x | Sem calibração, uso geral |
| AWQ | 7.7x | Máxima qualidade com calibração |
