"""SparseGPT structured pruning + viability probe for Llama-family models.

This is the reference pruning pipeline for viva_tensor's CUTLASS INT4 sparse
path. The sparse compute and throughput win are already in the library; this
probe measures whether one-shot SparseGPT weight prep can retain usable model
quality and exports the exact mask consumed by prepack.

Measured with scalar 2:4 pruning on Llama-3.2-1B, wikitext-2 perplexity:
    dense           ~13.2
    magnitude 2:4   ~2739    (dead)
    SparseGPT 2:4   ~33.2    (coherent; "The capital of France is" -> "Paris")

SparseGPT recovers ~82x over magnitude in ~45s of calibration. On a 1B model the
2:4 hit is large (~2.5x ppl); larger models degrade far less.

Measured with the hardware-compatible pair-4:8 constraint on the same model:
    SparseGPT pair-4:8   ~80.2    (experimental; quality gate not cleared)

The INT4 CUTLASS kernel has a stricter hardware layout than scalar 2:4: every
8-value K group must keep two *adjacent pairs* (4:8 pair-structured). Use the
``pair48`` modes for artifacts that can be consumed without silent re-pruning.

Usage (needs torch + transformers + datasets, e.g. the tmp/hf_ref uv env):
    VIVA_MODEL=tmp/llama32_1b tmp/hf_ref/bin/python dev/sparsegpt_2_4.py MODE

Modes:
    dense | mag | sparsegpt | gen | all
    sparsegpt-pair48 | gen-pair48 | export-pair48 | selftest

``export-pair48`` writes a HuggingFace checkpoint plus an explicit mask file:
    VIVA_SPARSE_EXPORT=tmp/llama32_1b_pair48 ... export-pair48

The SparseGPT core is inlined (sparsity-only) and robust to transformers 5.x:
the Catcher captures *all* decoder-layer kwargs (position_embeddings, mask, ...).

The exported mask uses one uint8 per 8-value group. Bit ``j`` marks logical
lane ``j`` as kept; valid bytes contain exactly two complete adjacent pairs.
The strict viva prepack consumes that mask instead of choosing a new one.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("VIVA_MODEL", "tmp/llama32_1b")
DEV = "cuda"
SEQLEN = 2048
NSAMPLES = 128
PROMPTS = ["The capital of France is", "Once upon a time"]
PAIR48_PATTERN = "pair48"
SCALAR24_PATTERN = "scalar24"

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

    def prune(self, pattern=SCALAR24_PATTERN, blocksize=128, percdamp=0.01):
        group_size = 4 if pattern == SCALAR24_PATTERN else 8
        if pattern not in (SCALAR24_PATTERN, PAIR48_PATTERN):
            raise ValueError(f"unsupported sparse pattern: {pattern}")
        if self.columns % group_size != 0:
            raise ValueError(f"{pattern} requires K divisible by {group_size}, got {self.columns}")
        if blocksize % group_size != 0:
            raise ValueError(f"blocksize must be divisible by {group_size}, got {blocksize}")

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
        pruned = torch.zeros_like(W, dtype=torch.bool)
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
                if i % group_size == 0:
                    scores = (
                        W1[:, i : i + group_size] ** 2
                        / (torch.diag(Hinv1)[i : i + group_size].reshape((1, -1))) ** 2
                    )
                    if pattern == SCALAR24_PATTERN:
                        prune_indices = torch.topk(scores, 2, dim=1, largest=False).indices
                    else:
                        pair_scores = scores.reshape(self.rows, 4, 2).sum(dim=2)
                        prune_pairs = torch.topk(pair_scores, 2, dim=1, largest=False).indices
                        pair_lanes = torch.arange(2, device=self.dev).reshape(1, 1, 2)
                        prune_indices = (prune_pairs.unsqueeze(2) * 2 + pair_lanes).reshape(
                            self.rows, 4
                        )
                    mask1.scatter_(1, i + prune_indices, True)
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            W[:, i1:i2] = Q1
            pruned[:, i1:i2] = mask1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        self.layer.weight.data = W.to(self.layer.weight.data.dtype)
        self.H = None
        return ~pruned


def encode_pair48_mask(keep_mask):
    """Encode a boolean [out, in] pair-4:8 mask as uint8 [out, in/8]."""
    if keep_mask.ndim != 2 or keep_mask.shape[1] % 8 != 0:
        raise ValueError(f"pair48 mask must be [out, in%8==0], got {tuple(keep_mask.shape)}")
    groups = keep_mask.bool().reshape(keep_mask.shape[0], -1, 8)
    if not torch.equal(groups[:, :, 0::2], groups[:, :, 1::2]):
        raise ValueError("pair48 mask contains a partially-kept adjacent pair")
    if not torch.all(groups[:, :, 0::2].sum(dim=2) == 2):
        raise ValueError("pair48 mask must keep exactly two adjacent pairs per 8 lanes")
    shifts = torch.arange(8, device=groups.device, dtype=torch.uint8).reshape(1, 1, 8)
    return torch.sum(groups.to(torch.uint8) << shifts, dim=2).to(torch.uint8)


def validate_pair48_mask_bytes(mask):
    """Validate the packed byte contract shared with the strict viva NIF."""
    if mask.dtype != torch.uint8 or mask.ndim != 2:
        raise ValueError(f"encoded pair48 mask must be 2-D uint8, got {mask.dtype}/{mask.ndim}D")
    lanes = torch.arange(8, device=mask.device, dtype=torch.uint8)
    decoded = ((mask.unsqueeze(2) >> lanes) & 1).bool()
    encode_pair48_mask(decoded.reshape(mask.shape[0], -1))


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
        trainenc.input_ids[:, i : i + SEQLEN]
        for i in (
            random.randint(0, trainenc.input_ids.shape[1] - SEQLEN - 1) for _ in range(NSAMPLES)
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
        batch = ids[:, i * SEQLEN : (i + 1) * SEQLEN]
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
def sparsegpt_prune(model, loader, pattern=SCALAR24_PATTERN, capture_masks=False):
    if capture_masks and pattern != PAIR48_PATTERN:
        raise ValueError("authoritative mask export is only defined for pair48")
    layers = model.model.layers
    inps = torch.zeros(
        (NSAMPLES, SEQLEN, model.config.hidden_size),
        dtype=next(model.parameters()).dtype,
        device=DEV,
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

    encoded_masks = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_linears(layer)
        gpts = {n: SparseGPT(subset[n]) for n in subset}
        handles = [
            subset[n].register_forward_hook(
                lambda _, inp, _out, name=n, gpts=gpts: gpts[name].add_batch(inp[0].data)
            )
            for n in subset
        ]
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        for h in handles:
            h.remove()
        for n in subset:
            keep_mask = gpts[n].prune(pattern=pattern)
            if capture_masks:
                key = f"model.layers.{i}.{n}.weight"
                encoded_masks[key] = encode_pair48_mask(keep_mask).cpu().contiguous()
        for j in range(NSAMPLES):
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
        inps, outs = outs, inps
    return model, encoded_masks


def save_pair48_export(model, tokenizer, masks, output_dir):
    """Save a pruned HF checkpoint and its authoritative pair-4:8 masks."""
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for _name, mask in masks.items():
        validate_pair48_mask_bytes(mask)

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    mask_path = output / "viva_pair48_masks.safetensors"
    save_file(
        dict(sorted(masks.items())),
        mask_path,
        metadata={
            "format": "viva_tensor_pair48_v1",
            "pattern": PAIR48_PATTERN,
            "layout": "out_by_k_group",
            "encoding": "one_lane_bit_per_k8_group",
        },
    )

    tensors = []
    for name, mask in sorted(masks.items()):
        weight = model.state_dict()[name]
        tensors.append(
            {
                "name": name,
                "out_features": weight.shape[0],
                "in_features": weight.shape[1],
                "mask_shape": list(mask.shape),
            }
        )
    manifest = {
        "format": "viva_tensor_pair48_v1",
        "source_model": MODEL,
        "pattern": PAIR48_PATTERN,
        "weight_layout": "huggingface_out_in",
        "mask_layout": "out_by_k_group",
        "mask_encoding": "one_lane_bit_per_k8_group",
        "mask_file": mask_path.name,
        "tensors": tensors,
    }
    with (output / "viva_pair48_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def selftest_pair48():
    keep = torch.tensor([[True, True, True, True, False, False, False, False]], dtype=torch.bool)
    encoded = encode_pair48_mask(keep)
    assert encoded.tolist() == [[0x0F]]
    validate_pair48_mask_bytes(encoded)
    invalid = torch.tensor([[0x55]], dtype=torch.uint8)
    try:
        validate_pair48_mask_bytes(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("scalar 2:4 mask 0x55 must be rejected by pair48 validation")


@torch.no_grad()
def gen_samples(model, tok, tag):
    model.eval()
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=28, do_sample=False)
        txt = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        print(f"  [{tag}] {p!r} -> {txt!r}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "selftest":
        selftest_pair48()
        print("PAIR48 SELFTEST ok")
        return

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    loader, testenc = get_loaders(tok)

    if mode in ("gen", "gen-pair48"):
        pattern = PAIR48_PATTERN if mode == "gen-pair48" else SCALAR24_PATTERN
        m = load_model()
        m.config.use_cache = True
        gen_samples(m, tok, "DENSE")
        m.config.use_cache = False
        m, _ = sparsegpt_prune(m, loader, pattern=pattern)
        m.config.use_cache = True
        gen_samples(m, tok, f"SPARSEGPT {pattern}")
        return

    if mode == "export-pair48":
        m = load_model()
        started = time.time()
        m, masks = sparsegpt_prune(m, loader, pattern=PAIR48_PATTERN, capture_masks=True)
        ppl = eval_ppl(m, testenc)
        m.config.use_cache = True
        gen_samples(m, tok, "SPARSEGPT pair48")
        output = os.environ.get("VIVA_SPARSE_EXPORT", "tmp/llama32_1b_pair48")
        save_pair48_export(m, tok, masks, output)
        print(
            f"EXPORTED pair48 tensors={len(masks)} ppl={ppl:.3f} "
            f"elapsed={time.time() - started:.0f}s path={output}"
        )
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
        m, _ = sparsegpt_prune(m, loader)
        ppl = eval_ppl(m, testenc)
        print(f"SPARSEGPT 2:4 ppl={ppl:.3f}  (prune {time.time() - t:.0f}s)")
        del m
        torch.cuda.empty_cache()

    if mode == "sparsegpt-pair48":
        m = load_model()
        t = time.time()
        m, _ = sparsegpt_prune(m, loader, pattern=PAIR48_PATTERN)
        ppl = eval_ppl(m, testenc)
        print(f"SPARSEGPT pair48 ppl={ppl:.3f}  (prune {time.time() - t:.0f}s)")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
