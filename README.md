# maya-minimax-kernel

A fused Metal kernel for **MiniMax-M3.1-Flash** in [project-maya](https://github.com/Franzferdinan51/project-maya), derived from the Qwen Strata kernel work — so MiniMax runs on a purpose-built kernel instead of ggml-metal's generic path.

## Why

Measured on an M5 Max (128 GB), the stock engine decodes at **19.65 tok/s** while reading ~2.5 GB of expert weights per token — about 9% of peak memory bandwidth. The MoE block runs as ~20 separate ggml ops per layer (three `mul_mat_id` passes, a separate swiglu_oai pass, a 7-add expert sum). The bottleneck is kernel structure, not the model.

## What the kernel does

Two dispatches per MoE layer replace the whole block:

- `kernel_moe_mm3_gate_up_swiglu` — gate + up IQ1_S dequant-matvec fused, swiglu_oai applied in-register, writing the hidden activations once.
- `kernel_moe_mm3_down_sum` — down dequant-matvec with the router-weighted expert sum folded in.

**Expert balance:** threadgroups slice the output rows uniformly and every threadgroup loops all k routed experts itself, so routed load spreads evenly by construction — no threadgroup waits on a hot expert.

## Model geometry (from the GGUF)

60 layers (3 dense lead, 57 MoE), hidden 6144, expert FF 3072, 128 experts, top-4, swiglu_oai alpha 1.702 / limit 7.0. Experts are **not** one format: the pack mixes IQ1_S (50 B / 256 values) with per-layer IQ2_XXS bumps (66 B / 256) on the sensitive matrices — gate/up/down can differ within a single layer, and the kernels cover every pairing.

## Status (2026-11-20)

- **Parity: PASS.** Synthetic-fixture test vs the numpy oracle: prompt logits rel L2 4.2e-05, greedy continuation bitwise-identical. On the real model, fused vs stock continuations agree for 14 tokens then split at a single greedy flip — the signature of FP32 reduction-order drift, not a dequant error (a wrong LUT flips token 0).
- **Coverage: all 57 MoE layers fused.** v1 spoke only IQ1_S and crashed on the pack's IQ2_XXS layers; the kernels are now templated over both formats with mixed gate/up pairings (`moe_mm3.metal`), with matching CPU reference forwards.
- **Speed: at par, not yet faster.** Real-model decode: fused 19.28 tok/s vs stock 19.86 tok/s (same harness, same prompt). A threadgroup-memory staging variant (v3) regressed to 18.67 and was reverted. The two-kernel split alone does not beat ggml's tuned `mul_mat_id` chain — see `PRIOR-ART.md` for what the literature says about where the remaining time lives (dispatch latency across the whole layer; dequant compute).
- Full ggml integration patch: `ggml-mm3-fused.patch` (ops, Metal dispatch, CPU forwards, kernel). Developed on project-maya's `minimax-metal` branch.
- Next: shared-expert fold-in, batched (verify-window) variants for the DSpark serve path, per-section GPU profiling.

See `DESIGN.md` for the full profile findings and kernel plan.

## Credits

Built on [ggml](https://github.com/ggml-org/ggml) (MIT) and the project-maya Metal engines. MiniMax-M3.1-Flash is MiniMax's model; the Maya GGUF quants are peasantsmith's.
