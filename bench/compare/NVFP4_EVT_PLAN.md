# CUTLASS EVT NVFP4 — Design plan for Tensor Core integration

## Status

PoC done in `zig_src/cuda_nvfp4_fused.cu`: 3.5 TFLOPS @ 1024² with a
hand-written CUDA kernel that fuses dequant into matmul. Confirms the
data path; doesn't use Tensor Cores so the throughput is 30× below the
dense FP16 path.

This doc plans the next jump: fuse dequant into a CUTLASS Tensor Core
GEMM via the **Epilogue Visitor Tree (EVT)** mechanism.

## Why EVT (not a hand-written Tensor Core kernel)

CUTLASS Tensor Core kernels are template-heavy and brittle. EVT lets
us **inject a custom callable** before the MMA tile is computed — same
fusion power as the cuBLASLt epilogue (BIAS/RELU/GELU) we already use
in `cuda_fp16_fused_bench.cu`, but generalised to the input side.

The "visitor" runs in the prologue (loading A) instead of the epilogue
(storing C). On loads from HBM, the visitor sees raw FP4 bytes;
dequantises to FP16 in registers; passes the FP16 tile to the standard
m16n8k16 MMA instruction. The dequantised intermediate **never touches
SMEM or HBM**.

## Reference shapes (NVFP4 on Ada SM89)

- Tile shape: 128 × 128 × 64 (M × N × K)
- Warp shape:  64 ×  64 × 64
- MMA shape:   16 ×   8 × 16 (`mma.sync.aligned.m16n8k16.f16.f16.f16.f16`)
- Stages: 3 (pipelined cp.async loads)

K dimension is doubled in the *A loader prologue* because each byte of
HBM carries 2 FP4 values. The MMA instruction itself stays unmodified;
EVT only changes the prologue.

## EVT prologue sketch (Python-ish pseudocode for clarity)

```
class NVFP4LoadVisitor(cutlass.PrologueVisitor):
    def __init__(self, dequant_table, scale_block):
        self.lut = dequant_table       # __constant__ uint16_t[16]
        self.scale = scale_block       # FP8 E4M3 per 16-element block

    @cute.kernel_op
    def visit(self, source_byte: uint8) -> Tuple[half, half]:
        lo = source_byte & 0xF
        hi = source_byte >> 4
        s = decode_fp8(self.scale)     # constant per block
        return (
            ushort_as_half(self.lut[lo]) * s,
            ushort_as_half(self.lut[hi]) * s,
        )
```

CUTLASS 4 CuTeDSL exposes this `PrologueVisitor` protocol directly.
The current C++ template path (CUTLASS 3.x) can express the same
fusion via `cutlass::epilogue::collective::CollectiveBuilder` with a
custom `LoadCallbacks` type, but the boilerplate is ~200 lines per
kernel instantiation.

## Roadmap

1. **Lock CUTLASS DSL toolchain** — requires `pip install cutlass`
   (~600MB). Already deferred in `CUTLASS_DSL_NOTES.md`; resolving
   that doc unblocks this one.

2. **Generate one specialization**: 4096² M=N=K, NVFP4 input,
   FP16 output, no fused activation, no sparsity. Just dequant + GEMM.
   Validate the data path against the naïve `nvfp4_fused_gemm_bench`
   from `cuda_nvfp4_fused.cu` — same Frobenius norm (within FP4 quant
   error).

3. **Target throughput**: matched dense FP16 (102-165 TFLOPS on
   4090 Ada). Anything less means the dequant became a bottleneck and
   we need shared-mem tiling.

4. **Add scale handling**: per-16-block FP8 scale + per-tensor FP32
   scale. Tested via per-tile dot product against a reference FP16
   GEMM with scaled inputs.

5. **Wire into NIF**: `nt_matmul_nvfp4_fused(M, N, K, A_packed,
   B_fp16, scale_block, scale_global) -> C_fp16`. Replace
   `nvfp4_fused_gemm_bench` (which uses the naïve kernel) with the
   EVT-backed version.

## Expected payoff

If the EVT prologue keeps up with the MMA pipeline (no bubble), we
should match dense FP16 throughput (~165 TFLOPS theoretical) on a
problem that **uses 4× less HBM bandwidth**. That's a free 4× speedup
on memory-bound inference layers (small batch, large weights).

On bandwidth-bound shapes — exactly the dominant case in LLM inference
with KV-cache pressure — this is the single biggest win we can extract
from Ada hardware before the Blackwell upgrade lands.

## Why this is parked (not implemented)

- CUTLASS DSL install is a ~600MB dep, not a 5-minute job.
- The C++ EVT path (~200 lines/kernel) is doable but maintenance cost
  outweighs the value for a benchmark — we want a *real* model
  workload to validate the win before paying that cost.
- Naïve PoC (cuda_nvfp4_fused.cu) already proves the data path; the
  speedup is gated on the Tensor Core integration, not the algorithm.

Reopen when: we have a concrete inference workload (LLM under 70B,
small batch, weight-bound) that would benefit from the 4× bandwidth
reduction, or when CUTLASS DSL ships an Ada SM89 NVFP4 example we can
adapt.
