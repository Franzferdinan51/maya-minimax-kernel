# MiniMax kernel — profile findings + design (Juno, 2026-11-20)

## Measured (real model, engine --ids mode, logged in ~/minimax-bench/)

- decode, no DSpark: **19.65 tok/s** (128 tok in 6.51 s; GPU 5.79 s, host 0.57 s)
- decode, DSpark on: **19.82 tok/s** (synthetic prompt — acceptance unrepresentative)
- serve-path figure (Hermes, 2026-11-20): 13.17 tok/s — different conditions, not the engine's ceiling
- GPU share 89% → kernel-bound, not host/dispatch-bound
- Per token: ~45 ms GPU, 5 graph executions

## Why it's slow (code-verified, src/metal/mm3_metal.cpp)

Per MoE layer at decode (T=1), the graph runs ~20 ops:
router (sigmoid, add bias, argsort_top_k, get_rows, sum/clamp/div renorm, scale),
then THREE separate ggml_mul_mat_id passes over the experts (up, gate, down)
with a separate swiglu_oai elementwise pass between, a mul by the router
weights, and a 7-add chain to sum the 8 experts' outputs. ~60 MoE layers.

Active expert bytes/token ≈ 1.2 GB (8 of 256 experts × gate+up+down, IQ1_S).
M5 Max peak bandwidth ≈ 546 GB/s → bandwidth ceiling ≈ 440 tok/s.
At 19.65 tok/s the engine uses ~4.5% of peak bandwidth: it is latency- and
kernel-efficiency-bound (generic ggml-metal IQ1_S mul_mat_id, tiny matvecs,
intermediate tensors round-tripping between dispatches), NOT bandwidth-bound.
That is exactly the gap the Qwen Strata engine closed with custom kernels.

## Kernel plan (derived from the Qwen Strata kernel, Maya-compatible)

Integration follows the repo's own mechanism: ggml is built from a build-tree
copy with patches from third_party/ggml/ applied by cmake/metal_backend.cmake
(existing: metal-sparse-mla, metal-tensor-fa, metal-allocator-free-blocks).
New work ships as **third_party/ggml/metal-mm3-fused-moe.patch** + graph
changes in src/metal/mm3_metal.cpp on branch **minimax-metal**
(worktree /Volumes/Sabrent/project-maya-minimax, seeded with Hermes'
2026-11-20 mm3_metal.cpp so the baseline matches the benched binary).

- **K1 — fused MoE block kernel (Metal)**: one dispatch per MoE layer doing
  gate+up IQ1_S dequant-matvec for the token's 8 experts, swiglu_oai in
  registers/threadgroup memory, down dequant-matvec, router-weighted sum.
  Kills 2 of 3 weight passes' intermediates and ~15 dispatches per layer.
  Pattern follows Strata's dequant + mma row work (include/strata/artifact/
  dequant.hpp, core/exl3_mma_rows.hpp) and the CUDA mmq-fused-swiglu.patch
  already in third_party/ggml (fuse gate into the tile load).
- **Balance (Duckets' requirement)**: threadgroups take slices of the OUTPUT
  dimension across all 8 experts (not one expert per threadgroup), so routed
  load is spread evenly and no threadgroup waits on a hot expert; the
  n-gram/draft (DSpark) side stays a separate, bounded pass so drafting
  neither starves nor swamps verify.
- **K2 — router fusion** (if K1 leaves it visible): sigmoid+bias+topk+renorm
  as one small kernel.
- Verify at each step: metal_mm3 parity fixture (ctest) + real-model --ids
  bench vs the 19.65 baseline, numbers from logs only.
- Promotion gate: front-door serve with the new engine, real completion,
  then the Maya link goes to Duckets for his own test.
