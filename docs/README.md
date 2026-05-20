# Documentation

Canonical documentation lives under [`en/`](en/) and is the source for
the HexDocs pages built by `gleam docs build`. The other language
folders are partial translations of the most-needed sections; raise an
issue or PR if you want more pages translated.

## Languages

| Language     | Coverage                       | Entry point                |
| :----------- | :----------------------------- | :------------------------- |
| **English**  | Canonical, complete            | [`en/README.md`](en/README.md) |
| **Português**| README + getting-started       | [`pt-br/README.md`](pt-br/README.md) |
| **中文**      | README + getting-started       | [`zh-cn/README.md`](zh-cn/README.md) |

## English structure

```
docs/en/
├── README.md              ← what ships today + doc map
├── paper.md               ← technical paper (closing the HF gap, kernel numbers)
├── api/
│   ├── tensor.md          ← pure-Gleam tensor API (creation, ops, broadcasting)
│   └── inference.md       ← FP8 / INT8 sparse / INT4 sparse / fused SwiGLU
├── guides/
│   ├── getting-started.md ← install, build, first run
│   ├── inference.md       ← TinyLlama-1.1B end-to-end
│   └── ffi-architecture.md ← maintainer-facing NIF contract
└── reference/
    ├── project-structure.md ← package layout
    └── stability.md         ← stable vs experimental boundary
```

## Building the HexDocs site

```
gleam docs build
```

The pages listed under `[[documentation.pages]]` in `gleam.toml` are
assembled into a single HexDocs site. See
[`../bench/README.md`](../bench/README.md) for benchmarks and
methodology, [`landing/README.md`](landing/README.md) for the
GitHub Pages landing.
