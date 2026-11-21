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

## Status (2026-11-21)

Correct and fully general, at par on plain decode. The expert-major window
revision is built and measured (below): ahead of stock in the 2026-11-21
serve A/B, with an honest caveat about cross-day stock variance.

- Parity: F32 fixture PASS (prompt rel L2 4.24e-05, sliced prompt 4.234e-05,
  greedy continuation bitwise). The engine's divergence harness verdict:
  A-consistent - the batched verify path is token-exact under forced
  rejections; batched-vs-single logits differ only at the 6.2e-05 batching
  scale. All 57 MoE layers fuse (mixed IQ1_S/IQ2_XXS included).
- Batched verify windows: the fused ops now take T rows (decode T=1, DSpark
  verify windows, prompt slices up to T=32; larger prefill slices keep
  ggml's batched mul_mm path, which is faster there).
- Plain decode, same harness/prompt: fused 19.34 tok/s vs stock 19.86.
- Speculative serve (counting prompt, 256 tokens, DSpark on): fused 39.33
  tok/s vs stock 43.94. The fused window kernel LOSES ~11% here: it
  re-dequantizes each expert row once per token row, while ggml's
  mul_mat_id amortizes dequant across the window.
- Expert-major window kernel LANDED (2026-11-21) for the gate/up pass:
  grid over (expert, row), each weight row dequantized once into registers
  and reused across every window slot that selected the expert. Selected
  for 1 < T with T*k <= 128; T == 1 keeps the token-major kernel, and the
  per-lane accumulation order is unchanged, so outputs are bitwise
  identical (fixture parity numbers unchanged, divergence still
  A-consistent). The down pass is still token-major.
- Speculative serve A/B, 2026-11-21 (counting prompt, 256 tokens, DSpark
  on; 6 interleaved runs per side, medians): fused 39.71 tok/s vs stock
  36.73 - fused ahead by ~8%, every fused run ahead of every stock run.
  Plain decode at par (fused 19.66 vs stock 19.78, medians; the T == 1
  path is unchanged). Caveat: stock benched 43.94 on 2026-11-20 under a
  different machine state, while fused is cross-day consistent
  (39.33 -> 39.71), so the isolated expert-major gain over the v1 window
  kernel is ~1% at DSpark window sizes - expert collisions are rare at
  T ~ 5-8, so there is little repeated dequant to amortize. The remaining
  levers are an expert-major down pass and wider windows.
- Physical bound for this pack: ~7.2 GB of weights move per token
  (lm_head ~1.0 GB, shared experts ~2.2 GB, routed ~2.5 GB at top-4), so
  single-token decode caps near bandwidth; 50-100+ tok/s runs through
  speculation yield, not through the plain path.

## Credits

Built on [ggml](https://github.com/ggml-org/ggml) (MIT) and the project-maya Metal engines. MiniMax-M3.1-Flash is MiniMax's model; the Maya GGUF quants are peasantsmith's.
