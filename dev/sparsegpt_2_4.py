"""SparseGPT 2:4 pruning + viability probe for Llama-family models.

This is the reference pipeline behind viva_tensor's INT4 2:4 sparse path. The
sparse *compute* (CUTLASS `nt_linear_int4_sparse`) and the throughput win are
already in the library; the missing piece for usable sparse *inference* is the
weight prep: 2:4 **magnitude** pruning (what `nt_prepack_int4_sparse` currently
does) destroys an unpruned model, so quality requires one-shot SparseGPT pruning
(Hessian-based, with compensating updates to the kept weights).

Measured on Llama-3.2-1B, wikitext-2 perplexity:
    dense           ~13.2
    magnitude 2:4   ~2739    (dead)
    SparseGPT 2:4   ~33.2    (coherent; "The capital of France is" -> "Paris")

SparseGPT recovers ~82x over magnitude in ~45s of calibration. On a 1B model the
2:4 hit is large (~2.5x ppl); larger models degrade far less.

Usage (needs torch + transformers + datasets, e.g. the tmp/hf_ref uv env):
    VIVA_MODEL=tmp/llama32_1b python dev/sparsegpt_2_4.py dense|mag|sparsegpt|gen|all

The SparseGPT core is inlined (sparsity-only) and robust to transformers 5.x:
the Catcher captures *all* decoder-layer kwargs (position_embeddings, mask, ...).

Next step for end-to-end sparse inference: export the pruned weights, run them
through `nt_prepack_int4_sparse` (int4 + 2:4 pack), and add a device-resident
sparse forward path. See the int4-2x4 sparse notes in the project memory.
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("VIVA_MODEL", "tmp/llama32_1b")
DEV = "cuda"
SEQLEN = 2048
NSAMPLES = 128
PROMPTS = ["The capital of France is", "Once upon a time"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


class SparseGPT:
    """SparseGPT (Frantar & Alistarh, 2023), sparsity-only core."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t().float()
        tmp = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def prune_2_4(self, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone().float()
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        prunen, prunem = 2, 4
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = torch.zeros_like(W1) == 1
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (
                        torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))
                    ) ** 2
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        self.layer.weight.data = W.to(self.layer.weight.data.dtype)
        self.H = None


def magnitude_2_4(weight):
    """Zero the 2 smallest of every 4 contiguous columns (along K), per row."""
    W = weight.data.float()
    r, c = W.shape
    Wg = W.reshape(r, c // 4, 4)
    idx = Wg.abs().argsort(dim=2)[:, :, :2]
    mask = torch.ones_like(Wg)
    mask.scatter_(2, idx, 0.0)
    weight.data = (Wg * mask).reshape(r, c).to(weight.data.dtype)


def get_loaders(tokenizer):
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    trainenc = tokenizer(" ".join(train["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    random.seed(0)
    loader = [
        trainenc.input_ids[:, i:i + SEQLEN]
        for i in (
            random.randint(0, trainenc.input_ids.shape[1] - SEQLEN - 1)
            for _ in range(NSAMPLES)
        )
    ]
    return loader, testenc


@torch.no_grad()
def eval_ppl(model, testenc):
    model.eval()
    ids = testenc.input_ids.to(DEV)
    nsteps = ids.shape[1] // SEQLEN
    total = 0.0
    for i in range(nsteps):
        batch = ids[:, i * SEQLEN:(i + 1) * SEQLEN]
        logits = model(batch).logits.float()
        loss = nn.CrossEntropyLoss()(
            logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        )
        total += loss.item() * (SEQLEN - 1)
    return math.exp(total / (nsteps * (SEQLEN - 1)))


def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").to(DEV)
    m.config.use_cache = False
    return m


@torch.no_grad()
def sparsegpt_prune(model, loader):
    layers = model.model.layers
    inps = torch.zeros(
        (NSAMPLES, SEQLEN, model.config.hidden_size),
        dtype=next(model.parameters()).dtype, device=DEV,
    )
    cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in loader:
        try:
            model(b.to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].m
    kwargs = cache["kwargs"]
    outs = torch.zeros_like(inps)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_linears(layer)
        gpts = {n: SparseGPT(subset[n]) for n in subset}
        handles = [
            subset[n].register_forward_hook(
                lambda _, inp, out, name=n: gpts[name].add_batch(inp[0].data)
            )
            for n in subset
        ]
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        for h in handles:
            h.remove()
        for n in subset:
            gpts[n].prune_2_4()
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        inps, outs = outs, inps
    return model


@torch.no_grad()
def gen_samples(model, tok, tag):
    model.eval()
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=28, do_sample=False)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  [{tag}] {p!r} -> {txt!r}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    loader, testenc = get_loaders(tok)

    if mode == "gen":
        m = load_model()
        m.config.use_cache = True
        gen_samples(m, tok, "DENSE")
        m.config.use_cache = False
        sparsegpt_prune(m, loader)
        m.config.use_cache = True
        gen_samples(m, tok, "SPARSEGPT 2:4")
        return

    if mode in ("all", "dense"):
        m = load_model()
        print(f"DENSE         ppl={eval_ppl(m, testenc):.3f}")
        del m
        torch.cuda.empty_cache()

    if mode in ("all", "mag"):
        m = load_model()
        for _, lin in find_linears(m.model.layers).items():
            magnitude_2_4(lin.weight)
        print(f"MAG 2:4       ppl={eval_ppl(m, testenc):.3f}")
        del m
        torch.cuda.empty_cache()

    if mode in ("all", "sparsegpt"):
        m = load_model()
        t = time.time()
        sparsegpt_prune(m, loader)
        ppl = eval_ppl(m, testenc)
        print(f"SPARSEGPT 2:4 ppl={ppl:.3f}  (prune {time.time()-t:.0f}s)")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
