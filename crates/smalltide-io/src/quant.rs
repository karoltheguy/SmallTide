//! Fast load-time BF16 -> FP8 E4M3FN per-row quantizer for `--fp8-dense`
//! (Inkling-2-Mini attention projections, loop controllers, and `lm_head`).

/// Max finite representable magnitude in OCP FP8 E4M3FN (`(1 + 6/8) * 2^8 = 448.0`).
pub const FP8_E4M3FN_MAX: f32 = 448.0;

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = ((bits >> 16) & 1).wrapping_add(0x7FFF);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Exact OCP FP8 E4M3FN decode (matches `smalltide_backend_hip` & `smalltide_backend_cpu`).
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
        f32::from_bits(sign | ((exp + 120) << 23) | (mant << 20))
    }
}

/// Round-to-nearest-even conversion from `f32` to OCP FP8 E4M3FN (saturating at ±448.0, never NaN).
#[inline]
pub fn f32_to_fp8_e4m3fn(x: f32) -> u8 {
    if !x.is_finite() {
        return 0;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let ax = x.abs();
    if ax == 0.0 {
        return sign;
    }
    if ax >= FP8_E4M3FN_MAX {
        // Max finite E4M3FN: exp=15 (0x78), mant=6 (0x06) => 0x7E (0x7F is NaN!)
        return sign | 0x7E;
    }
    // Subnormal range: < 2^-6 = 1/64 = 0.015625; step is 2^-9 = 1/512
    if ax < (1.0 / 64.0) {
        let m = (ax * 512.0).round() as u32;
        if m >= 8 {
            return sign | 0x08; // smallest normal: exp=1, mant=0
        }
        return sign | (m as u8);
    }

    let bits = ax.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x007F_FFFF;

    // FP8 E4M3FN bias = 7 => exp_field = f32_exp + 7 (guaranteed 1..=15 here)
    let mut exp_field = (f32_exp + 7) as u32;
    // Top 3 mantissa bits + round-to-nearest on remaining 20 bits
    let top3 = f32_mant >> 20;
    let rem = f32_mant & 0x000F_FFFF;
    let round_up = rem > 0x0008_0000 || (rem == 0x0008_0000 && (top3 & 1) == 1);
    let mut mant_field = top3 + if round_up { 1 } else { 0 };
    if mant_field == 8 {
        mant_field = 0;
        exp_field += 1;
    }
    if exp_field >= 15 && mant_field >= 7 {
        return sign | 0x7E; // clamp before NaN
    }
    sign | (((exp_field as u8) << 3) | (mant_field as u8))
}

/// Quantizes a row-major `[n_rows, k_cols]` BF16 matrix (raw little-endian bytes)
/// into `(fp8_weights [n_rows * k_cols], row_scales [n_rows])` where `W_approx[r, c] = decode(fp8) * scale[r]`.
pub fn quantize_bf16_bytes_to_fp8_rows(
    bf16_bytes: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(bf16_bytes.len(), n_rows * k_cols * 2);
    let mut fp8 = vec![0u8; n_rows * k_cols];
    let mut scales = vec![0.0f32; n_rows];

    for r in 0..n_rows {
        let row_byte_base = r * k_cols * 2;
        let mut max_abs = 0.0f32;
        for c in 0..k_cols {
            let b0 = bf16_bytes[row_byte_base + c * 2];
            let b1 = bf16_bytes[row_byte_base + c * 2 + 1];
            let v = bf16_to_f32(u16::from_le_bytes([b0, b1])).abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let scale = if max_abs > 0.0 {
            max_abs / FP8_E4M3FN_MAX
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;
        scales[r] = scale;

        let out_row = &mut fp8[r * k_cols..(r + 1) * k_cols];
        for c in 0..k_cols {
            let b0 = bf16_bytes[row_byte_base + c * 2];
            let b1 = bf16_bytes[row_byte_base + c * 2 + 1];
            let v = bf16_to_f32(u16::from_le_bytes([b0, b1]));
            out_row[c] = f32_to_fp8_e4m3fn(v * inv_scale);
        }
    }

    (fp8, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3fn_roundtrip_and_no_nan() {
        for &val in &[0.0f32, -0.0, 0.002, 0.125, 1.0, -1.5, 12.5, 240.0, 448.0, 999.0, -999.0] {
            let q = f32_to_fp8_e4m3fn(val);
            assert_ne!(q & 0x7F, 0x7F, "Must never produce NaN encoding");
            let d = decode_fp8_e4m3fn(q);
            let clamped = val.clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX);
            let tol = (clamped.abs() * 0.07).max(0.002);
            assert!((d - clamped).abs() <= tol, "val={val} q=0x{q:02x} d={d}");
        }
    }

    #[test]
    fn test_bf16_matrix_to_fp8_rows() {
        let n_rows = 4;
        let k_cols = 64;
        let mut src = Vec::with_capacity(n_rows * k_cols);
        let mut bf16_bytes = Vec::with_capacity(n_rows * k_cols * 2);
        for r in 0..n_rows {
            for c in 0..k_cols {
                let v = ((r * 17 + c) as f32 * 0.13).sin() * ((r + 1) as f32 * 0.5);
                let b = f32_to_bf16(v);
                src.push(bf16_to_f32(b));
                bf16_bytes.extend_from_slice(&b.to_le_bytes());
            }
        }
        let (fp8, scales) = quantize_bf16_bytes_to_fp8_rows(&bf16_bytes, n_rows, k_cols);
        for r in 0..n_rows {
            for c in 0..k_cols {
                let orig = src[r * k_cols + c];
                let rec = decode_fp8_e4m3fn(fp8[r * k_cols + c]) * scales[r];
                let tol = (orig.abs() * 0.075).max(scales[r] * 0.01);
                assert!((orig - rec).abs() <= tol, "r={r} c={c} orig={orig} rec={rec}");
            }
        }
    }
}
