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

60 layers (3 dense lead, 57 MoE), hidden 6144, expert FF 3072, 128 experts, top-4, IQ1_S experts (256-value blocks, 50-byte blocks), swiglu_oai alpha 1.702 / limit 7.0.

## Status

- v1 kernel written, compiles clean as a standalone Metal object against the pinned ggml headers (`dequantize_iq1_s` reused as-is).
- In progress: the ggml op plumbing (new op through ggml core + ggml-metal dispatch), parity verification against the synthetic fixture / numpy oracle, first real-model bench vs the 19.65 tok/s baseline.
- Integration route: a `third_party/ggml/metal-mm3-fused-moe.patch` in project-maya, applied by `cmake/metal_backend.cmake` like the existing Metal patches, developed on the `minimax-metal` branch.

See `DESIGN.md` for the full profile findings and kernel plan.

## Credits

Built on [ggml](https://github.com/ggml-org/ggml) (MIT) and the project-maya Metal engines. MiniMax-M3.1-Flash is MiniMax's model; the Maya GGUF quants are peasantsmith's.
