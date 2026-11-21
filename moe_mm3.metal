// moe_mm3.metal — fused MiniMax-M3.1 MoE block kernels for ggml-metal.
// Covers decode (T=1) and DSpark verify windows (T>1) alike.
//
// Replaces the ~20-op MoE block in mm3_metal.cpp (3x mul_mat_id + swiglu_oai +
// mul + 7-add expert sum + router-side glue) with two dispatches per layer:
//
//   kernel_moe_mm3_gate_up_swiglu: hh[t][e][r] = swiglu_oai(gate_e[r].x_t, up_e[r].x_t)
//   kernel_moe_mm3_down_sum:       out[t][r]   = sum_e wts[t][e] * (down_e[r].hh[t][e])
//
// Expert balance: threadgroups slice the OUTPUT rows uniformly and every
// threadgroup loops all k routed experts itself, so routed load is spread
// evenly by construction — no threadgroup waits on a hot expert.
//
// Weight types: the Maya-XS pack is NOT uniform — Unsloth's recipe bumps the
// sensitive matrices to IQ2_XXS per layer, so gate/up/down can mix IQ1_S and
// IQ2_XXS within one layer (e.g. gate IQ2_XXS, up IQ1_S, down IQ2_XXS). The
// quant kernels are therefore instantiated for every gate/up pairing and both
// down types. Both formats share the same dequant shape (256-value blocks, 16
// il-steps of 16 values into a float4x4), so one templated dot serves all.
//
// Geometry is passed in the args struct (the real model: n_embd 6144,
// n_ff_exp 3072, k 4). Accumulation is f32 throughout, matching the ggml
// mul_mat_id f32 path the parity fixture is calibrated to.

#include "common.h"
#include "dequantize.h"

struct moe_mm3_args {
    int n_embd;      // input dim of gate/up rows (6144)
    int n_ff;        // expert intermediate dim (3072)
    int k;           // routed experts per token (4)
    int n_tokens;    // T: 1 for decode, >1 for verify windows
    float alpha;     // swiglu_oai alpha (1.702)
    float limit;     // swiglu_oai limit (7.0)
};

static inline float moe_mm3_swiglu_oai(float g, float u, float alpha, float limit) {
    g = min(g, limit);
    u = clamp(u, -limit, limit);
    return g / (1.0f + exp(-alpha * g)) * (u + 1.0f);
}

// ---- quant dot: one templated implementation for IQ1_S and IQ2_XXS ----

static inline void moe_mm3_dequant(device const block_iq1_s * xb, short il, thread float4x4 & reg) {
    dequantize_iq1_s(xb, il, reg);
}

static inline void moe_mm3_dequant(device const block_iq2_xxs * xb, short il, thread float4x4 & reg) {
    dequantize_iq2_xxs(xb, il, reg);
}

// dot of one quant row (n_blocks * 256 values) with the f32 vector x,
// one simdgroup cooperating: lane l takes il-steps l, l+32, l+64, ...
template <typename block_t>
static float moe_mm3_dot_q(device const block_t * row, int n_blocks,
                           device const float * x, uint lane) {
    float sum = 0.0f;
    const int steps = n_blocks * 16;  // 16 il-steps of 16 values per block
    for (int s = (int)lane; s < steps; s += 32) {
        const int b = s / 16;
        const short il = (short)(s % 16);
        float4x4 reg;
        moe_mm3_dequant(row + b, il, reg);
        device const float * xv = x + b * 256 + il * 16;
        sum += reg[0][0] * xv[0] + reg[0][1] * xv[1] + reg[0][2] * xv[2] + reg[0][3] * xv[3]
             + reg[1][0] * xv[4] + reg[1][1] * xv[5] + reg[1][2] * xv[6] + reg[1][3] * xv[7]
             + reg[2][0] * xv[8] + reg[2][1] * xv[9] + reg[2][2] * xv[10] + reg[2][3] * xv[11]
             + reg[3][0] * xv[12] + reg[3][1] * xv[13] + reg[3][2] * xv[14] + reg[3][3] * xv[15];
    }
    return simd_sum(sum);
}

// hh layout: [T, k, n_ff] f32 (token-major). grid: (T * k * n_ff / 8)
// threadgroups of 256 threads; simdgroup s computes flat row
// (tgid * 8 + s) = ((t * k) + e) * n_ff + r.
template <typename gblock_t, typename ublock_t>
static void moe_mm3_gu_impl(
        constant moe_mm3_args & args,
        device const float * x,          // [T, n_embd]
        device const char * gate_w,      // quant [n_exp experts], strides below
        device const char * up_w,
        device const int32_t * ids,      // [T, k] expert ids
        device float * hh,               // [T, k, n_ff]
        uint tgpig_x, uint simd_gid, uint lane) {
    const int row = (int)(tgpig_x * 8 + simd_gid);
    const int r = row % args.n_ff;
    const int ek = (row / args.n_ff);
    const int e = ek % args.k;
    const int t = ek / args.k;
    if (t >= args.n_tokens) return;
    const int32_t expert = ids[t * args.k + e];
    device const float * xt = x + (int64_t)t * args.n_embd;
    const int n_blocks = args.n_embd / 256;
    const int nb_grow = n_blocks * (int)sizeof(gblock_t);   // bytes per gate row
    const int nb_urow = n_blocks * (int)sizeof(ublock_t);   // bytes per up row
    device const gblock_t * grow = (device const gblock_t *)(gate_w + (int64_t)expert * nb_grow * args.n_ff + r * nb_grow);
    device const ublock_t * urow = (device const ublock_t *)(up_w   + (int64_t)expert * nb_urow * args.n_ff + r * nb_urow);
    const float g = moe_mm3_dot_q(grow, n_blocks, xt, lane);
    const float u = moe_mm3_dot_q(urow, n_blocks, xt, lane);
    if (lane == 0) {
        hh[(int64_t)ek * args.n_ff + r] = moe_mm3_swiglu_oai(g, u, args.alpha, args.limit);
    }
}

// out layout: [T, n_embd] f32. grid: (T * n_embd / 8) threadgroups of 256
// threads; simdgroup s computes flat row (tgid * 8 + s) = t * n_embd + r,
// looping all k experts for that token.
template <typename block_t>
static void moe_mm3_down_impl(
        constant moe_mm3_args & args,
        device const float * hh,         // [T, k, n_ff]
        device const char * down_w,      // quant, rows of n_ff values
        device const int32_t * ids,      // [T, k]
        device const float * wts,        // [T, k] router weights
        device float * out,              // [T, n_embd]
        uint tgpig_x, uint simd_gid, uint lane) {
    const int row = (int)(tgpig_x * 8 + simd_gid);
    const int r = row % args.n_embd;
    const int t = row / args.n_embd;
    if (t >= args.n_tokens) return;
    const int n_blocks = args.n_ff / 256;
    const int nb_row = n_blocks * (int)sizeof(block_t);
    const int64_t nb_exp = (int64_t)nb_row * args.n_embd;
    float acc = 0.0f;
    for (int e = 0; e < args.k; ++e) {
        const int32_t expert = ids[t * args.k + e];
        device const block_t * wrow = (device const block_t *)(down_w + (int64_t)expert * nb_exp + r * nb_row);
        acc += wts[t * args.k + e] * moe_mm3_dot_q(wrow, n_blocks, hh + ((int64_t)t * args.k + e) * args.n_ff, lane);
    }
    if (lane == 0) out[row] = acc;
}

#define MOE_MM3_GU_KERNEL(NAME, GT, UT) \
kernel void NAME( \
        constant moe_mm3_args & args, \
        device const float * x, \
        device const char * gate_w, \
        device const char * up_w, \
        device const int32_t * ids, \
        device float * hh, \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        uint simd_gid [[simdgroup_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    moe_mm3_gu_impl<GT, UT>(args, x, gate_w, up_w, ids, hh, tgpig.x, simd_gid, lane); \
}

#define MOE_MM3_DOWN_KERNEL(NAME, BT) \
kernel void NAME( \
        constant moe_mm3_args & args, \
        device const float * hh, \
        device const char * down_w, \
        device const int32_t * ids, \
        device const float * wts, \
        device float * out, \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        uint simd_gid [[simdgroup_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) { \
    moe_mm3_down_impl<BT>(args, hh, down_w, ids, wts, out, tgpig.x, simd_gid, lane); \
}

MOE_MM3_GU_KERNEL(kernel_moe_mm3_gate_up_swiglu,         block_iq1_s,   block_iq1_s)
MOE_MM3_GU_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq2,     block_iq2_xxs, block_iq2_xxs)
MOE_MM3_GU_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq2_iq1, block_iq2_xxs, block_iq1_s)
MOE_MM3_GU_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq1_iq2, block_iq1_s,   block_iq2_xxs)

MOE_MM3_DOWN_KERNEL(kernel_moe_mm3_down_sum,     block_iq1_s)
MOE_MM3_DOWN_KERNEL(kernel_moe_mm3_down_sum_iq2, block_iq2_xxs)

// ---- F32 variants (parity fixture; T=1 shapes, float weights) ----

static inline float moe_mm3_dot_f32(device const float * row,
                                    device const float * x,
                                    int n,
                                    uint lane) {
    float sum = 0.0f;
    for (int i = (int) lane; i < n; i += 32) {
        sum += row[i] * x[i];
    }
    return simd_sum(sum);
}

kernel void kernel_moe_mm3_gate_up_swiglu_f32(
        constant moe_mm3_args & args,
        device const float * x,
        device const float * gate_w,
        device const float * up_w,
        device const int   * ids,
        device float       * hh,
        uint3 tgpig    [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  lane     [[thread_index_in_simdgroup]]) {
    const int row = (int) (tgpig.x * 8 + simd_gid);
    if (row >= args.n_tokens * args.k * args.n_ff) {
        return;
    }
    const int r      = row % args.n_ff;
    const int ek     = row / args.n_ff;
    const int e_slot = ek % args.k;
    const int t      = ek / args.k;
    const int e      = ids[t * args.k + e_slot];

    device const float * grow = gate_w + ((size_t) e * args.n_ff + r) * args.n_embd;
    device const float * urow = up_w   + ((size_t) e * args.n_ff + r) * args.n_embd;
    device const float * xt   = x + (size_t) t * args.n_embd;

    const float g = moe_mm3_dot_f32(grow, xt, args.n_embd, lane);
    const float u = moe_mm3_dot_f32(urow, xt, args.n_embd, lane);

    if (lane == 0) {
        hh[row] = moe_mm3_swiglu_oai(g, u, args.alpha, args.limit);
    }
}

kernel void kernel_moe_mm3_down_sum_f32(
        constant moe_mm3_args & args,
        device const float * hh,
        device const float * down_w,
        device const int   * ids,
        device const float * wts,
        device float       * out,
        uint3 tgpig    [[threadgroup_position_in_grid]],
        uint  simd_gid [[simdgroup_index_in_threadgroup]],
        uint  lane     [[thread_index_in_simdgroup]]) {
    const int row = (int) (tgpig.x * 8 + simd_gid);
    if (row >= args.n_tokens * args.n_embd) {
        return;
    }
    const int r = row % args.n_embd;
    const int t = row / args.n_embd;

    float acc = 0.0f;
    for (int e_slot = 0; e_slot < args.k; ++e_slot) {
        const int e = ids[t * args.k + e_slot];
        device const float * drow = down_w + ((size_t) e * args.n_embd + r) * args.n_ff;
        device const float * hvec = hh + (size_t) (t * args.k + e_slot) * args.n_ff;
        acc += wts[t * args.k + e_slot] * moe_mm3_dot_f32(drow, hvec, args.n_ff, lane);
    }

    if (lane == 0) {
        out[row] = acc;
    }
}

// ---- expert-major window variant (T > 1), appended 2026-11-21 ----
//
// Same math and the same per-lane accumulation order as the token-major
// kernels above, so results are bitwise identical; only the grid changes.
// Token-major launches one simdgroup per (token, slot, row) and therefore
// re-dequantizes an expert's row once per token row that selected it.
// Expert-major launches one simdgroup per (expert, row): the row is
// dequantized ONCE into registers (in step chunks) and reused for every
// window slot that selected the expert — the amortization ggml's
// mul_mat_id gets from its grouped layout. Selected by the host op when
// 1 < T and T*k <= MOE_MM3_EM_MAXSLOTS (the engine gate already caps
// T <= 32); T == 1 keeps the token-major kernels, and n_ff % 8 == 0 is a
// dispatch precondition (threadgroups never straddle an expert boundary).

#define MOE_MM3_EM_MAXSLOTS 128
#define MOE_MM3_EM_CS 2

struct moe_mm3_block_f32 { float v[256]; };

static inline void moe_mm3_dequant(device const moe_mm3_block_f32 * xb, short il, thread float4x4 & reg) {
    device const float * v = xb->v + (int)il * 16;
    reg = float4x4(float4(v[0],  v[1],  v[2],  v[3]),
                   float4(v[4],  v[5],  v[6],  v[7]),
                   float4(v[8],  v[9],  v[10], v[11]),
                   float4(v[12], v[13], v[14], v[15]));
}

static inline float moe_mm3_dot16(thread const float4x4 & reg, device const float * xv) {
    return reg[0][0] * xv[0]  + reg[0][1] * xv[1]  + reg[0][2] * xv[2]  + reg[0][3] * xv[3]
         + reg[1][0] * xv[4]  + reg[1][1] * xv[5]  + reg[1][2] * xv[6]  + reg[1][3] * xv[7]
         + reg[2][0] * xv[8]  + reg[2][1] * xv[9]  + reg[2][2] * xv[10] + reg[2][3] * xv[11]
         + reg[3][0] * xv[12] + reg[3][1] * xv[13] + reg[3][2] * xv[14] + reg[3][3] * xv[15];
}

template <typename gblock_t, typename ublock_t>
static void moe_mm3_gu_em_impl(
        constant moe_mm3_args & args,
        device const float * x,          // [T, n_embd]
        device const char * gate_w,
        device const char * up_w,
        device const int32_t * ids,      // flat slots [T*k]
        device float * hh,               // [T, k, n_ff]
        uint tgpig_x, uint simd_gid, uint lane, uint tid_in_tg,
        threadgroup int32_t * tg_flag,
        threadgroup int32_t * tg_slot,
        threadgroup int32_t * tg_count) {
    const int row = (int)(tgpig_x * 8 + simd_gid);
    const int r = row % args.n_ff;
    const int expert = row / args.n_ff;
    const int slots = args.n_tokens * args.k;

    // match scan, once per threadgroup: all 8 simdgroups share this expert
    if (tid_in_tg == 0) tg_count[0] = 0;
    if ((int)tid_in_tg < slots) tg_flag[tid_in_tg] = (ids[tid_in_tg] == expert) ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid_in_tg == 0) {
        int c = 0;
        for (int j = 0; j < slots; ++j) {
            if (tg_flag[j]) tg_slot[c++] = j;
        }
        tg_count[0] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int count = tg_count[0];
    if (count == 0) return;

    const int n_blocks = args.n_embd / 256;
    const int steps = n_blocks * 16;
    const int nb_grow = n_blocks * (int)sizeof(gblock_t);
    const int nb_urow = n_blocks * (int)sizeof(ublock_t);
    device const gblock_t * grow = (device const gblock_t *)(gate_w + (int64_t)expert * nb_grow * args.n_ff + r * nb_grow);
    device const ublock_t * urow = (device const ublock_t *)(up_w   + (int64_t)expert * nb_urow * args.n_ff + r * nb_urow);

    float pacc_g[MOE_MM3_EM_MAXSLOTS];
    float pacc_u[MOE_MM3_EM_MAXSLOTS];
    for (int j = 0; j < count; ++j) { pacc_g[j] = 0.0f; pacc_u[j] = 0.0f; }

    // step chunks: dequantize MOE_MM3_EM_CS steps per lane into registers,
    // then run every matching slot's x segment through those registers
    // before dequantizing the next chunk. Per-lane step order stays
    // ascending, exactly as in moe_mm3_dot_q.
    for (int base = 0; base < steps; base += 32 * MOE_MM3_EM_CS) {
        float4x4 greg[MOE_MM3_EM_CS];
        float4x4 ureg[MOE_MM3_EM_CS];
        int ss[MOE_MM3_EM_CS];
        for (int i = 0; i < MOE_MM3_EM_CS; ++i) {
            const int s = base + (int)lane + 32 * i;
            ss[i] = s;
            if (s < steps) {
                moe_mm3_dequant(grow + s / 16, (short)(s % 16), greg[i]);
                moe_mm3_dequant(urow + s / 16, (short)(s % 16), ureg[i]);
            } else {
                greg[i] = float4x4(0.0f);
                ureg[i] = float4x4(0.0f);
            }
        }
        for (int j = 0; j < count; ++j) {
            const int t = tg_slot[j] / args.k;
            device const float * xt = x + (int64_t)t * args.n_embd;
            float sg = 0.0f, su = 0.0f;
            for (int i = 0; i < MOE_MM3_EM_CS; ++i) {
                const int s = ss[i];
                if (s >= steps) break;
                device const float * xv = xt + (s / 16) * 256 + (s % 16) * 16;
                sg += moe_mm3_dot16(greg[i], xv);
                su += moe_mm3_dot16(ureg[i], xv);
            }
            pacc_g[j] += sg;
            pacc_u[j] += su;
        }
    }

    for (int j = 0; j < count; ++j) {
        const float g = simd_sum(pacc_g[j]);
        const float u = simd_sum(pacc_u[j]);
        if (lane == 0) {
            hh[(int64_t)tg_slot[j] * args.n_ff + r] = moe_mm3_swiglu_oai(g, u, args.alpha, args.limit);
        }
    }
}

#define MOE_MM3_GU_EM_KERNEL(NAME, GT, UT) \
kernel void NAME( \
        constant moe_mm3_args & args, \
        device const float * x, \
        device const char * gate_w, \
        device const char * up_w, \
        device const int32_t * ids, \
        device float * hh, \
        uint3 tgpig [[threadgroup_position_in_grid]], \
        uint simd_gid [[simdgroup_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]], \
        uint tid_in_tg [[thread_index_in_threadgroup]]) { \
    threadgroup int32_t tg_flag[MOE_MM3_EM_MAXSLOTS]; \
    threadgroup int32_t tg_slot[MOE_MM3_EM_MAXSLOTS]; \
    threadgroup int32_t tg_count[1]; \
    moe_mm3_gu_em_impl<GT, UT>(args, x, gate_w, up_w, ids, hh, tgpig.x, simd_gid, lane, tid_in_tg, tg_flag, tg_slot, tg_count); \
}

MOE_MM3_GU_EM_KERNEL(kernel_moe_mm3_gate_up_swiglu_em,         block_iq1_s,        block_iq1_s)
MOE_MM3_GU_EM_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq2_em,     block_iq2_xxs,      block_iq2_xxs)
MOE_MM3_GU_EM_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq2_iq1_em, block_iq2_xxs,      block_iq1_s)
MOE_MM3_GU_EM_KERNEL(kernel_moe_mm3_gate_up_swiglu_iq1_iq2_em, block_iq1_s,        block_iq2_xxs)
MOE_MM3_GU_EM_KERNEL(kernel_moe_mm3_gate_up_swiglu_f32_em,     moe_mm3_block_f32,  moe_mm3_block_f32)
