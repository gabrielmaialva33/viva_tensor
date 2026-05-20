# bench/plans/ — technical plans + future work

Design documents and roadmaps that informed (or still inform) work on
the native backends. Anything labelled **[DONE]** is preserved as
context — the implementation has shipped and is referenced in the file
header.

## Documents

| File                       | Status      | Summary                                                                          |
| :------------------------- | :---------- | :------------------------------------------------------------------------------- |
| `INFERENCE_API_PLAN.md`    | **[DONE]**  | FP8 prepack + linear + SwiGLU + INT4 sparse + TinyLlama end-to-end. All shipped. |
| `CUTLASS_DSL_NOTES.md`     | Active      | Migration notes for CUTLASS 4 CuTeDSL — guides future kernel work.               |
| `NVFP4_EVT_PLAN.md`        | Parked      | NVFP4 fused dequant + GEMM. Requires Blackwell-class hardware (not yet on hand). |

## Why keep [DONE] docs

The completed `INFERENCE_API_PLAN.md` documents the full evolution from
the proposed inference API to what actually shipped (W8A16 + per-block-16
scales reaching HF-equivalent tokens, fused SwiGLU, INT4 sparse with
byte-exact CUTLASS metadata). That story is more useful as context for
next-generation work than a deleted file would be.

The header on each [DONE] doc lists where the live code now lives.
