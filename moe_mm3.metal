// moe_mm3.metal — fused MiniMax-M3.1 MoE block kernels for ggml-metal (decode, T=1).
//
// Replaces the ~20-op MoE block in mm3_metal.cpp (3x mul_mat_id + swiglu_oai +
// mul + 7-add expert sum + router-side glue) with two dispatches per layer:
//
//   kernel_moe_mm3_gate_up_swiglu: hh[e][r] = swiglu_oai(gate_e[r].x, up_e[r].x)
//   kernel_moe_mm3_down_sum:       out[r]   = sum_e wts[e] * (down_e[r].hh[e])
//
// Expert balance: threadgroups slice the OUTPUT rows uniformly and every
// threadgroup loops all k routed experts itself, so routed load is spread
// evenly by construction — no threadgroup waits on a hot expert.
//
// Geometry is passed in the args struct (the real model: n_embd 6144,
// n_ff_exp 3072, k 4, IQ1_S blocks of 256). Accumulation is f32 throughout,
// matching the ggml mul_mat_id f32 path the parity fixture is calibrated to.

#include "common.h"
#include "dequantize.h"

struct moe_mm3_args {
    int n_embd;      // input dim of gate/up rows (6144)
    int n_ff;        // expert intermediate dim (3072)
    int k;           // routed experts per token (4)
    float alpha;     // swiglu_oai alpha (1.702)
    float limit;     // swiglu_oai limit (7.0)
};

// dot of one IQ1_S row (n_blocks * 256 values) with the f32 vector x,
// one simdgroup cooperating: lane l takes il-steps l, l+32, l+64, ...
static float moe_mm3_dot_iq1_s(device const block_iq1_s * row, int n_blocks,
                               device const float * x, uint lane) {
    float sum = 0.0f;
    const int steps = n_blocks * 16;  // 16 il-steps of 16 values per block
    for (int s = (int)lane; s < steps; s += 32) {
        const int b = s / 16;
        const short il = (short)(s % 16);
        float4x4 reg;
        dequantize_iq1_s(row + b, il, reg);
        device const float * xv = x + b * 256 + il * 16;
        sum += reg[0][0] * xv[0] + reg[0][1] * xv[1] + reg[0][2] * xv[2] + reg[0][3] * xv[3]
             + reg[1][0] * xv[4] + reg[1][1] * xv[5] + reg[1][2] * xv[6] + reg[1][3] * xv[7]
             + reg[2][0] * xv[8] + reg[2][1] * xv[9] + reg[2][2] * xv[10] + reg[2][3] * xv[11]
             + reg[3][0] * xv[12] + reg[3][1] * xv[13] + reg[3][2] * xv[14] + reg[3][3] * xv[15];
    }
    return simd_sum(sum);
}

// hh layout: [k, n_ff] f32. grid: (k * n_ff / 8) threadgroups of 256 threads;
// simdgroup s of a threadgroup computes row (tgid * 8 + s) of expert e = row / n_ff.
kernel void kernel_moe_mm3_gate_up_swiglu(
        constant moe_mm3_args & args,
        device const float * x,          // [n_embd]
        device const char * gate_w,      // IQ1_S [k experts selected from tensor], strides below
        device const char * up_w,
        device const int32_t * ids,      // [k] expert ids
        device float * hh,               // [k, n_ff]
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    const int row = (int)(tgpig.x * 8 + simd_gid);
    const int e = row / args.n_ff;
    const int r = row % args.n_ff;
    if (e >= args.k) return;
    const int32_t expert = ids[e];
    const int nb_row = (args.n_embd / 256) * (int)sizeof(block_iq1_s);         // bytes per weight row
    const int nb_exp = nb_row * args.n_ff;                                     // bytes per expert (gate/up)
    device const block_iq1_s * grow = (device const block_iq1_s *)(gate_w + (int64_t)expert * nb_exp + r * nb_row);
    device const block_iq1_s * urow = (device const block_iq1_s *)(up_w + (int64_t)expert * nb_exp + r * nb_row);
    const int n_blocks = args.n_embd / 256;
    const float g = moe_mm3_dot_iq1_s(grow, n_blocks, x, lane);
    const float u = moe_mm3_dot_iq1_s(urow, n_blocks, x, lane);
    if (lane == 0) {
        const float gc = min(g, args.limit);
        const float uc = clamp(u, -args.limit, args.limit);
        hh[e * args.n_ff + r] = gc / (1.0f + exp(-args.alpha * gc)) * (uc + 1.0f);
    }
}

// out layout: [n_embd] f32. grid: (n_embd / 8) threadgroups of 256 threads;
// simdgroup s computes output row (tgid * 8 + s), looping all k experts.
kernel void kernel_moe_mm3_down_sum(
        constant moe_mm3_args & args,
        device const float * hh,         // [k, n_ff]
        device const char * down_w,      // IQ1_S, rows of n_ff values
        device const int32_t * ids,      // [k]
        device const float * wts,        // [k] router weights (renormalised, scaled)
        device float * out,              // [n_embd]
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    const int r = (int)(tgpig.x * 8 + simd_gid);
    if (r >= args.n_embd) return;
    const int nb_row = (args.n_ff / 256) * (int)sizeof(block_iq1_s);
    const int nb_exp = nb_row * args.n_embd;
    const int n_blocks = args.n_ff / 256;
    float acc = 0.0f;
    for (int e = 0; e < args.k; ++e) {
        const int32_t expert = ids[e];
        device const block_iq1_s * row = (device const block_iq1_s *)(down_w + (int64_t)expert * nb_exp + r * nb_row);
        acc += wts[e] * moe_mm3_dot_iq1_s(row, n_blocks, hh + e * args.n_ff, lane);
    }
    if (lane == 0) out[r] = acc;
}
