# Prior art — fused MoE decode kernels

Research pass, 2026-11-20. What others have published on the problem this repo
attacks (single-token MoE decode on a bandwidth-rich, latency-bound GPU), and
what it means for the design here.

## Cursor — "warp decode"

Cursor's engineering blog post on their MoE decode kernels (Blackwell GPUs)
validates the structure used here almost 1:1:

- Output-major mapping: one warp (here: simdgroup) per output neuron.
- Two kernels: a fused `moe_gate_up` (gate + up + activation in one pass,
  activation read once) and a fused `moe_down` with the routing weight folded
  into a running FP32 accumulator.
- Result: 1.84× decode speedup, and outputs *closer* to the FP32 reference
  than the unfused chain (fewer intermediate roundings).

Checklist adopted from it: activation staged once for gate+up; routing weight
applied inside the down kernel, never materialised; FP32 accumulation
throughout.

## llama.cpp upstream — MoE fusion PRs

Upstream ggml has been landing MoE fusions (e.g. PR #28948: top-k router
fusion, MoE weighted-reduction fusion). Two lessons:

- The router fusion is softmax-shaped; MiniMax's router is sigmoid +
  correction bias, so a direct port does not apply — a sigmoid variant is the
  follow-up if router time ever matters.
- A MUL_MAT+UNARY fusion *regressed* Qwen3.6-35B-A3B decode by ~8% (lost
  overlap between dispatches). Fusion is not free: every merge removes a
  chance for the GPU to overlap independent work. Measure per merge.

## Hanzo / llama.cpp IQ-quant kernels

Field notes on IQ1_S-class formats: at batch 1 the dequant path can be
*compute*-bound, not bandwidth-bound — per-element FP32 LUT math is the wall,
and integer-dot (dp4a) restructures paid up to 2.27× where that held. On Apple
Silicon there is no dp4a equivalent in Metal shading language for this shape,
but the diagnostic matters: if a fused kernel lands far under the bandwidth
roofline, suspect dequant arithmetic before memory layout.

## zmlx — MLX fused MoE

`gather_qmm_swiglu` (fused gate+up+SwiGLU over gathered experts in MLX) gained
only +5–8% — because MLX already ran the block in ~1 kernel. Fusion headroom
is proportional to how many dispatches you start from; a ~20-op chain has
more to gain than a 2-op one.

## MonoMoE / FlashInfer — persistent megakernels

The documented endgame if a two-kernel split plateaus: one persistent kernel
for the whole MoE block (or layer), weights streamed weight-major, no
intermediate global-memory round-trips at all.

## gamozolabs llama.cpp fork — fused MoE on CPU

A fused MoE FFN CPU kernel took a small model from 3.26 → 18.73 tok/s,
mostly by removing per-op overhead and intermediate buffers. Same lesson at
the other end of the hardware spectrum: at batch 1, structure beats FLOPs.

## dmlx (zouyee) — expert streaming

Reference for the offload half of the roadmap: dense trunk resident, experts
streamed. Relevant when the working set outgrows the GPU, not at 93 GB on a
128 GB machine — noted, not adopted.

## What this means for maya-minimax-kernel

1. The two-kernel split (this repo) is the validated first step.
2. Per-merge measurement is mandatory — one fusion in the literature was a
   net regression.
3. If the split plateaus near the unfused chain, the bottleneck is either
   dequant compute (restructure the dot) or dispatch latency elsewhere in
   the layer (widen fusion beyond the MoE block) — profile before guessing.
