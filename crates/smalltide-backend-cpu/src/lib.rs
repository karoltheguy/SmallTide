//! Pure-Rust CPU reference backend for kernel numerical verification (Phase 1)
//! and Inkling-2-Mini logit parity tests (Phase 2).

pub use smalltide_core::{DType, Result};

/// Exact FP4 E2M1 decode (1 sign, 2 exp bias=1, 1 mantissa).
#[inline]
pub fn decode_fp4_e2m1(nibble: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nibble & 7) as usize];
    if (nibble & 8) != 0 {
        -v
    } else {
        v
    }
}

/// Exact OCP FP8 E4M3FN decode (1 sign, 4 exp bias=7, 3 mantissa; 0x7F/0xFF -> 0.0 NaN guard).
#[inline]
pub fn decode_fp8_e4m3fn(b: u8) -> f32 {
    let sign = ((b as u32) & 0x80) << 24;
    let exp = ((b as u32) >> 3) & 0x0F;
    let mant = (b as u32) & 0x07;
    if exp == 0 {
        let val = (mant as f32) * (1.0 / 512.0);
        if sign != 0 { -val } else { val }
    } else if exp == 15 && mant == 7 {
        0.0
    } else {
        let bits = sign | ((exp + 120) << 23) | (mant << 20);
        f32::from_bits(bits)
    }
}

/// Exact UE8M0 microscaling exponent decode (`2^(e - 127)`).
#[inline]
pub fn decode_ue8m0(e: u8) -> f32 {
    if e == 0 || e == 255 {
        0.0
    } else {
        f32::from_bits((e as u32) << 23)
    }
}

pub fn rmsnorm_f32(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv_rms * w)
        .collect()
}

pub fn gemv_fp8_e4m3_f32(
    x: &[f32],
    w_fp8: &[u8],
    row_scale: &[f32],
    n_rows: usize,
    k_cols: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), k_cols);
    assert_eq!(w_fp8.len(), n_rows * k_cols);
    assert_eq!(row_scale.len(), n_rows);
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let row = &w_fp8[r * k_cols..(r + 1) * k_cols];
        let mut acc = 0.0f32;
        for c in 0..k_cols {
            acc += decode_fp8_e4m3fn(row[c]) * x[c];
        }
        out[r] = acc * row_scale[r];
    }
    out
}

pub fn gemv_nvfp4_f32(
    x: &[f32],
    w_packed: &[u8],
    w_scales: &[u8],
    outer_scale: f32,
    n_rows: usize,
    k_cols: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), k_cols);
    assert_eq!(k_cols % 16, 0);
    let n_blocks = k_cols / 16;
    assert_eq!(w_packed.len(), n_rows * (k_cols / 2));
    assert_eq!(w_scales.len(), n_rows * n_blocks);
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let w_row = &w_packed[r * (k_cols / 2)..(r + 1) * (k_cols / 2)];
        let s_row = &w_scales[r * n_blocks..(r + 1) * n_blocks];
        let mut acc = 0.0f32;
        for b in 0..n_blocks {
            let blk_scale = decode_fp8_e4m3fn(s_row[b]);
            let mut sub = 0.0f32;
            for i in 0..8 {
                let byte = w_row[b * 8 + i];
                let lo = decode_fp4_e2m1(byte & 0x0F);
                let hi = decode_fp4_e2m1(byte >> 4);
                sub += lo * x[b * 16 + i * 2];
                sub += hi * x[b * 16 + i * 2 + 1];
            }
            acc += sub * blk_scale;
        }
        out[r] = acc * outer_scale;
    }
    out
}

pub fn gemv_mxfp8_f32(
    x: &[f32],
    w_fp8: &[u8],
    w_scales: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), k_cols);
    assert_eq!(k_cols % 32, 0);
    let n_blocks_k = k_cols / 32;
    assert_eq!(w_fp8.len(), n_rows * k_cols);
    assert_eq!(w_scales.len(), n_rows * n_blocks_k);
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let w_row = &w_fp8[r * k_cols..(r + 1) * k_cols];
        let s_row = &w_scales[r * n_blocks_k..(r + 1) * n_blocks_k];
        let mut acc = 0.0f32;
        for b in 0..n_blocks_k {
            let blk_scale = decode_ue8m0(s_row[b]);
            let mut sub = 0.0f32;
            for i in 0..32 {
                sub += decode_fp8_e4m3fn(w_row[b * 32 + i]) * x[b * 32 + i];
            }
            acc += sub * blk_scale;
        }
        out[r] = acc;
    }
    out
}

#[inline]
pub fn softcap_tanh(x: f32, cap: f32) -> f32 {
    if cap > 0.0 {
        cap * (x / cap).tanh()
    } else {
        x
    }
}

#[inline]
pub fn situ(gate: f32, up: f32, gate_cap: f32, up_cap: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-gate).exp());
    softcap_tanh(gate, gate_cap) * sig * softcap_tanh(up, up_cap)
}

/// CPU reference matching `smalltide_moe_w13_swiglu_indirect` + `smalltide_moe_w2_accum_indirect`.
#[allow(clippy::too_many_arguments)]
pub fn moe_swiglu_indirect_ref(
    x: &[f32],
    expert_pool: &[u8],
    slot_table: &[u32],
    topk_ids: &[u32],
    topk_weights: &[f32],
    slab_stride_bytes: usize,
    hidden_size: usize,
    intermediate_size: usize,
    gate_softcap: f32,
    up_softcap: f32,
) -> Vec<f32> {
    let total_w: f32 = topk_weights.iter().sum();
    let exec_w: f32 = topk_ids
        .iter()
        .zip(topk_weights.iter())
        .filter(|(&id, _)| slot_table[id as usize] != u32::MAX)
        .map(|(_, &w)| w)
        .sum();
    let renorm = if exec_w > 1e-8 { total_w / exec_w } else { 0.0 };

    let mut out = vec![0.0f32; hidden_size];
    for (&exp_id, &w) in topk_ids.iter().zip(topk_weights.iter()) {
        let slot = slot_table[exp_id as usize];
        if slot == u32::MAX {
            continue;
        }
        let slab = &expert_pool
            [(slot as usize) * slab_stride_bytes..(slot as usize + 1) * slab_stride_bytes];
        let w13_outer = f32::from_le_bytes(slab[0..4].try_into().unwrap());
        let w2_outer = f32::from_le_bytes(slab[4..8].try_into().unwrap());
        let w13_w_off = u32::from_le_bytes(slab[8..12].try_into().unwrap()) as usize;
        let w13_s_off = u32::from_le_bytes(slab[12..16].try_into().unwrap()) as usize;
        let w2_w_off = u32::from_le_bytes(slab[16..20].try_into().unwrap()) as usize;
        let w2_s_off = u32::from_le_bytes(slab[20..24].try_into().unwrap()) as usize;
        let dtype_tag = u32::from_le_bytes(slab[24..28].try_into().unwrap());

        let (gate_up, down) = if dtype_tag == 0 {
            // NVFP4
            let w13_w_len = (2 * intermediate_size) * (hidden_size / 2);
            let w13_s_len = (2 * intermediate_size) * (hidden_size / 16);
            let w13 = gemv_nvfp4_f32(
                x,
                &slab[w13_w_off..w13_w_off + w13_w_len],
                &slab[w13_s_off..w13_s_off + w13_s_len],
                w13_outer,
                2 * intermediate_size,
                hidden_size,
            );
            let mut mid = vec![0.0f32; intermediate_size];
            for r in 0..intermediate_size {
                mid[r] = situ(w13[2 * r], w13[2 * r + 1], gate_softcap, up_softcap);
            }
            let w2_w_len = hidden_size * (intermediate_size / 2);
            let w2_s_len = hidden_size * (intermediate_size / 16);
            let y = gemv_nvfp4_f32(
                &mid,
                &slab[w2_w_off..w2_w_off + w2_w_len],
                &slab[w2_s_off..w2_s_off + w2_s_len],
                w2_outer,
                hidden_size,
                intermediate_size,
            );
            (mid, y)
        } else {
            // MXFP8: on-disk [N, K/32] UE8M0 scales
            let w13_w_len = (2 * intermediate_size) * hidden_size;
            let w13_s_len = (2 * intermediate_size) * (hidden_size / 32);
            let w13 = gemv_mxfp8_f32(
                x,
                &slab[w13_w_off..w13_w_off + w13_w_len],
                &slab[w13_s_off..w13_s_off + w13_s_len],
                2 * intermediate_size,
                hidden_size,
            );
            let mut mid = vec![0.0f32; intermediate_size];
            for r in 0..intermediate_size {
                mid[r] = situ(w13[2 * r], w13[2 * r + 1], gate_softcap, up_softcap);
            }
            let w2_w_len = hidden_size * intermediate_size;
            let w2_s_len = hidden_size * (intermediate_size / 32);
            let y = gemv_mxfp8_f32(
                &mid,
                &slab[w2_w_off..w2_w_off + w2_w_len],
                &slab[w2_s_off..w2_s_off + w2_s_len],
                hidden_size,
                intermediate_size,
            );
            (mid, y)
        };
        let _ = gate_up;
        for d in 0..hidden_size {
            out[d] += down[d] * w * renorm;
        }
    }
    out
}
