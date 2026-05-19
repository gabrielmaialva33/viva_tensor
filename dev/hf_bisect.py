#!/usr/bin/env python3
"""
Reference forward of TinyLlama-1.1B layer 0 on BOS token, dumping
mean_abs of the hidden state at every meaningful stage.

Match list (against dev/llama_forward.erl's pipeline):
  - embed_lookup
  - rmsnorm(input_layernorm)
  - q_proj, k_proj, v_proj (raw, pre-RoPE)
  - RoPE(q), RoPE(k) at pos=0
  - attention output (concat heads)
  - o_proj
  - residual_1
  - rmsnorm(post_attention_layernorm)
  - gate_proj, up_proj
  - silu(gate) * up
  - down_proj
  - residual_2 (= block 0 hidden output)

Then final norm + lm_head + top-5 tokens.

Run:
  /home/gabrielmaia/Documents/projects/viva_tensor/tmp/hf_ref/bin/python \
    dev/hf_bisect.py
"""

import torch
from transformers import LlamaForCausalLM, LlamaConfig

MODEL_PATH = "tmp/tinyllama"
BOS = 1
EPS = 1e-5
HEAD_DIM = 64
NUM_HEADS = 32
NUM_KV_HEADS = 4

torch.set_grad_enabled(False)


def m(name, t):
    """Print mean_abs + first 5 values of a 1-D tensor."""
    x = t.detach().flatten().float()
    first5 = ", ".join(f"{v:.6f}" for v in x[:5].tolist())
    print(f"  {name:40s} mean_abs={x.abs().mean().item():.6f}  [{first5}, ...]")


def rmsnorm(x, weight, eps=EPS):
    var = x.float().pow(2).mean(-1, keepdim=True)
    h = x.float() * torch.rsqrt(var + eps)
    return (weight.float() * h).to(x.dtype)


def rope(x, pos, head_dim=HEAD_DIM, theta=10000.0):
    # x: [num_heads, head_dim]
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim)
    angle = pos * freqs                       # [half]
    cos = angle.cos()
    sin = angle.sin()
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)


def main():
    print(f"=== HF reference bisect — layer 0, BOS (pos=0) ===\n")
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    print(f"hidden={cfg.hidden_size}, kv={cfg.num_key_value_heads}*{HEAD_DIM}, ffn={cfg.intermediate_size}")
    sd = model.state_dict()

    # --- Stage 0: embedding ---
    embed = sd["model.embed_tokens.weight"]
    x = embed[BOS].clone()       # [hidden]
    m("embed[BOS]", x)

    # --- Layer 0 ---
    prefix = "model.layers.0."
    n1 = sd[prefix + "input_layernorm.weight"]
    qw = sd[prefix + "self_attn.q_proj.weight"]
    kw = sd[prefix + "self_attn.k_proj.weight"]
    vw = sd[prefix + "self_attn.v_proj.weight"]
    ow = sd[prefix + "self_attn.o_proj.weight"]
    n2 = sd[prefix + "post_attention_layernorm.weight"]
    gw = sd[prefix + "mlp.gate_proj.weight"]
    uw = sd[prefix + "mlp.up_proj.weight"]
    dw = sd[prefix + "mlp.down_proj.weight"]

    # --- Stage 1: input_layernorm ---
    x_norm1 = rmsnorm(x, n1)
    m("after input_layernorm", x_norm1)

    # --- Stage 2: Q/K/V projections (HF stores [out, in], y = x @ W^T) ---
    q = x_norm1 @ qw.T           # [hidden=2048]
    k = x_norm1 @ kw.T           # [kv_dim=256]
    v = x_norm1 @ vw.T           # [kv_dim=256]
    m("Q proj raw", q)
    m("K proj raw", k)
    m("V proj raw", v)

    # --- Stage 3: RoPE ---
    q_heads = q.view(NUM_HEADS, HEAD_DIM)
    k_heads = k.view(NUM_KV_HEADS, HEAD_DIM)
    q_rot = rope(q_heads, pos=0).flatten()
    k_rot = rope(k_heads, pos=0).flatten()
    m("Q after RoPE", q_rot)
    m("K after RoPE", k_rot)

    # --- Stage 4: single-token attention (softmax(1 scalar)=1, attn=v_head) ---
    v_heads = v.view(NUM_KV_HEADS, HEAD_DIM)
    q_per_kv = NUM_HEADS // NUM_KV_HEADS
    attn = torch.cat([
        v_heads[h // q_per_kv]
        for h in range(NUM_HEADS)
    ])
    m("attention output", attn)

    # --- Stage 5: O proj + residual 1 ---
    o = attn @ ow.T
    m("O proj", o)
    h1 = x + o
    m("residual 1", h1)

    # --- Stage 6: post_attention_layernorm ---
    x_norm2 = rmsnorm(h1, n2)
    m("after post_attention_layernorm", x_norm2)

    # --- Stage 7: SwiGLU FFN ---
    g = x_norm2 @ gw.T
    u = x_norm2 @ uw.T
    m("gate proj", g)
    m("up proj", u)
    inter = torch.nn.functional.silu(g) * u
    m("silu(gate) * up", inter)
    ffn = inter @ dw.T
    m("down proj (FFN out)", ffn)

    h2 = h1 + ffn
    m("residual 2 (block 0 hidden)", h2)

    # --- Final stages (only meaningful if going through all 22 layers, but we
    # show what would happen if h2 were the final hidden) ---
    print("\n--- (For reference) if block 0 hidden were final: ---")
    final_norm_w = sd["model.norm.weight"]
    lm = sd["lm_head.weight"]
    norm_final = rmsnorm(h2, final_norm_w)
    logits = norm_final @ lm.T
    top5_val, top5_idx = logits.topk(5)
    print(f"  norm_final mean_abs={norm_final.abs().mean().item():.6f}")
    print(f"  logits mean_abs={logits.abs().mean().item():.6f}")
    print(f"  top-5 ids:    {top5_idx.tolist()}")
    print(f"  top-5 logits: {[round(v, 3) for v in top5_val.tolist()]}")


if __name__ == "__main__":
    main()
