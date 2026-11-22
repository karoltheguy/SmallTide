//! Image inputs for Inkling 2's vision tower, as transformers' Inkling2ImageProcessor makes them
//! (ported from TideLM's tidelm-image): an image is used at its own resolution, one
//! language-model token a 40 x 40 pixel cell.
//!
//! - Decode PNG or JPEG to RGB, or RGBA when the file has transparency.
//! - Alpha is composited onto the processor's checkerboard (204 / 153 in 8-pixel squares).
//! - `rows = ceil(H / 40)`, `cols = ceil(W / 40)`; the image is padded at the bottom and right
//!   with -1 on the 0..255 scale and normalized `(x - 127.5) / 127.5`, then rounded to bf16.
//! - The patches are `[tokens * 4][1200]`: token-major (tokens row-major), a token's four
//!   20 x 20 patches in (row, column) order, a patch's values (y, x, channel).
//!
//! Ours, not the reference's: `max_tokens`. An image whose grid would hold more tokens is first
//! scaled down (aspect kept, an antialiased bicubic) to the largest size whose grid fits.

/// A language-model image token's cell, in pixels.
pub const CELL: usize = 40;
const PATCH: usize = 20;
const MERGE: usize = CELL / PATCH;
const PATCH_K: usize = 3 * PATCH * PATCH;
/// The pad on the 0..255 scale, before the normalization.
const PAD: f32 = -1.0;

/// An image ready for the tower.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// the token grid
    pub rows: usize,
    pub cols: usize,
    /// bf16 `[rows * cols * 4][1200]`
    pub patches: Vec<u16>,
    /// a 128-bit hash of the grid and the patches (prefix-cache keys)
    pub digest: u128,
}

impl Prepared {
    /// the image's placeholder tokens in the prompt: one a cell
    pub fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

struct Pixels {
    w: usize,
    h: usize,
    /// 3 (RGB) or 4 (RGBA, straight alpha)
    channels: usize,
    px: Vec<u8>,
}

fn decode(b: &[u8]) -> Result<Pixels, String> {
    if b.is_empty() {
        Err("an empty image (no bytes)".into())
    } else if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(b)
    } else if b.starts_with(&[0xff, 0xd8, 0xff]) {
        decode_jpeg(b)
    } else {
        Err("unsupported image format (PNG and JPEG are supported)".into())
    }
}

fn decode_png(b: &[u8]) -> Result<Pixels, String> {
    use png::{ColorType, Transformations};
    let mut dec = png::Decoder::new(std::io::Cursor::new(b));
    // palettes, low bit depths and tRNS (to an alpha channel) expanded; 16 bits to their high byte
    dec.set_transformations(Transformations::EXPAND | Transformations::STRIP_16);
    let mut rd = dec.read_info().map_err(|e| format!("PNG: {e}"))?;
    let mut buf = vec![0u8; rd.output_buffer_size().ok_or("PNG: image too large")?];
    let info = rd.next_frame(&mut buf).map_err(|e| format!("PNG: {e}"))?;
    let (w, h, stride) = (info.width as usize, info.height as usize, info.line_size);
    let (ch, alpha) = match info.color_type {
        ColorType::Grayscale => (1, false),
        ColorType::GrayscaleAlpha => (2, true),
        ColorType::Rgb => (3, false),
        ColorType::Rgba => (4, true),
        ColorType::Indexed => return Err("PNG: palette not expanded".into()),
    };
    let out_ch = if alpha { 4 } else { 3 };
    let mut px = Vec::with_capacity(w * h * out_ch);
    for y in 0..h {
        let row = &buf[y * stride..y * stride + w * ch];
        for p in row.chunks_exact(ch) {
            match ch {
                1 => px.extend_from_slice(&[p[0], p[0], p[0]]),
                2 => px.extend_from_slice(&[p[0], p[0], p[0], p[1]]),
                _ => px.extend_from_slice(p),
            }
        }
    }
    Ok(Pixels { w, h, channels: out_ch, px })
}

fn decode_jpeg(b: &[u8]) -> Result<Pixels, String> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(b, opts);
    let px = dec.decode().map_err(|e| format!("JPEG: {e}"))?;
    let info = dec.info().ok_or("JPEG: no header")?;
    let (w, h) = (info.width as usize, info.height as usize);
    if px.len() != w * h * 3 {
        return Err(format!("JPEG: {} bytes for {w} x {h} RGB pixels", px.len()));
    }
    Ok(Pixels { w, h, channels: 3, px })
}

/// An RGBA image over a checkerboard of 8-pixel squares, 204 where the square's row and column
/// sum is even and 153 elsewhere: `(rgb * a + background * (255 - a) + 127) / 255`.
fn composite(w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let p = &rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
            let a = p[3] as u32;
            let bg: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
            for &v in &p[..3] {
                px.push(((v as u32 * a + bg * (255 - a) + 127) / 255) as u8);
            }
        }
    }
    px
}

/// The token grid (rows, cols) of a `w` x `h` image.
pub fn grid(w: usize, h: usize) -> (usize, usize) {
    (h.div_ceil(CELL), w.div_ceil(CELL))
}

/// The size (w, h) an image is prepared at under a cap of `max_tokens` (0: none): its own when
/// its grid fits, else the largest scaled-down one whose grid does.
pub fn fit(w: usize, h: usize, max_tokens: usize) -> (usize, usize) {
    let fits = |(w, h): (usize, usize)| {
        let (r, c) = grid(w, h);
        r.saturating_mul(c) <= max_tokens
    };
    if max_tokens == 0 || w == 0 || h == 0 || fits((w, h)) {
        return (w, h);
    }
    let (long, short) = (w.max(h) as u128, w.min(h) as u128);
    let size = |l: usize| {
        let s = (((short * l as u128 + long / 2) / long) as usize).max(1);
        if w >= h { (l, s) } else { (s, l) }
    };
    let (mut lo, mut hi) = (1, w.max(h) - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(size(mid)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    size(lo)
}

/// One axis of an antialiased bicubic resize (a = -0.5, the kernel widened by the scale when
/// shrinking): for each output index, the first source index and the normalized weights.
fn axis_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let fs = scale.max(1.0);
    let support = 2.0 * fs;
    let cubic = |x: f64| -> f64 {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(src);
            let mut w: Vec<f64> = (lo..hi).map(|j| cubic((j as f64 + 0.5 - center) / fs)).collect();
            let sum: f64 = w.iter().sum();
            for v in &mut w {
                *v /= sum;
            }
            (lo, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

/// RGB `w` x `h` -> `nw` x `nh`.
fn resize(w: usize, h: usize, px: &[u8], nw: usize, nh: usize) -> Vec<u8> {
    let (wx, wy) = (axis_weights(w, nw), axis_weights(h, nh));
    // horizontal pass into f32, then vertical
    let mut mid = vec![0f32; nw * h * 3];
    for y in 0..h {
        for (x, (lo, ws)) in wx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (j, &wt) in ws.iter().enumerate() {
                let p = &px[(y * w + lo + j) * 3..];
                acc[0] += wt * p[0] as f32;
                acc[1] += wt * p[1] as f32;
                acc[2] += wt * p[2] as f32;
            }
            mid[(y * nw + x) * 3..(y * nw + x) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut out = vec![0u8; nw * nh * 3];
    for (y, (lo, ws)) in wy.iter().enumerate() {
        for x in 0..nw {
            let mut acc = [0f32; 3];
            for (j, &wt) in ws.iter().enumerate() {
                let p = &mid[((lo + j) * nw + x) * 3..];
                acc[0] += wt * p[0];
                acc[1] += wt * p[1];
                acc[2] += wt * p[2];
            }
            for c in 0..3 {
                out[(y * nw + x) * 3 + c] = (acc[c] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// fp32 to bf16, round to nearest even
fn bf16(f: f32) -> u16 {
    let b = f.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A value on the 0..255 scale normalized `(x - 127.5) / 127.5`, as bf16.
fn norm(v: f32) -> u16 {
    bf16((v - 127.5f32) / 127.5f32)
}

/// Two independent 64-bit hashes of the grid and the patches: equal digests are equal inputs
/// (to 2^-128; nothing here is adversarial - a collision would only reuse a cached prefix).
fn digest(rows: usize, cols: usize, patches: &[u16]) -> u128 {
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64 ^ rows as u64, 0x9E37_79B9_7F4A_7C15u64 ^ cols as u64);
    for &v in patches {
        a = (a ^ v as u64).wrapping_mul(0x0000_0100_0000_01B3);
        b = (b ^ v as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29);
    }
    a ^= (rows as u64) << 32 | cols as u64;
    ((a as u128) << 64) | (b ^ (b >> 31)) as u128
}

/// An RGB image of its final size into the padded grid's patches.
fn cut(w: usize, h: usize, px: &[u8]) -> Prepared {
    let (rows, cols) = grid(w, h);
    let mut lut = [0u16; 256];
    for (v, o) in lut.iter_mut().enumerate() {
        *o = norm(v as f32);
    }
    let mut patches = vec![norm(PAD); rows * cols * MERGE * MERGE * PATCH_K];
    for y in 0..h {
        let (r, ir, py) = (y / CELL, y % CELL / PATCH, y % PATCH);
        // a row's runs of 20 pixels: each one row of one patch
        for (run, src) in px[y * w * 3..(y + 1) * w * 3].chunks(PATCH * 3).enumerate() {
            let (c, ic) = (run / MERGE, run % MERGE);
            let o = (((r * cols + c) * MERGE + ir) * MERGE + ic) * PATCH_K + py * PATCH * 3;
            for (d, &v) in patches[o..o + src.len()].iter_mut().zip(src) {
                *d = lut[v as usize];
            }
        }
    }
    let digest = digest(rows, cols, &patches);
    Prepared { rows, cols, patches, digest }
}

/// Image bytes (PNG or JPEG) to the tower's input, of at most `max_tokens` tokens (0: no cap).
pub fn prepare(bytes: &[u8], max_tokens: usize) -> Result<Prepared, String> {
    let p = decode(bytes)?;
    if p.w == 0 || p.h == 0 {
        return Err("an empty image".into());
    }
    if p.w.saturating_mul(p.h) > 64_000_000 {
        return Err(format!("an image of {} x {} pixels is too large", p.w, p.h));
    }
    let rgb = if p.channels == 4 { composite(p.w, p.h, &p.px) } else { p.px };
    let (nw, nh) = fit(p.w, p.h, max_tokens);
    if (nw, nh) == (p.w, p.h) {
        Ok(cut(p.w, p.h, &rgb))
    } else {
        Ok(cut(nw, nh, &resize(p.w, p.h, &rgb, nw, nh)))
    }
}

/// base64 as Python's b64decode reads it: characters outside the alphabet are skipped, missing
/// padding is tolerated.
pub fn b64(s: &str) -> Result<Vec<u8>, String> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let body = s.split('=').next().unwrap_or("");
    let digits: Vec<u32> = body.bytes().filter_map(val).collect();
    if digits.len() % 4 == 1 {
        return Err("base64: incorrect padding".into());
    }
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for chunk in digits.chunks(4) {
        let mut v = 0u32;
        for (i, &d) in chunk.iter().enumerate() {
            v |= d << (18 - 6 * i);
        }
        let bytes = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
        out.extend_from_slice(&bytes[..chunk.len() - 1]);
    }
    Ok(out)
}

/// The bytes of an image given as a `data:` URL or bare base64. The server does not fetch
/// http(s) URLs or read local files for a client.
pub fn load_url(u: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = u.strip_prefix("data:") {
        let (header, payload) = rest.split_once(',').unwrap_or((rest, ""));
        if !header.contains(";base64") {
            return Err(format!("unsupported data URL encoding: data:{header}"));
        }
        return b64(payload);
    }
    if u.starts_with("http://") || u.starts_with("https://") {
        return Err("image URLs are not fetched by this server: send the image as a data: URL (base64)".into());
    }
    Err("an image must be a data: URL (base64)".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_as_python_reads_it() {
        assert_eq!(b64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(b64("aGVs\nbG8").unwrap(), b"hello");
        assert_eq!(b64("aGVsbA==").unwrap(), b"hell");
        assert!(b64("aGVsb").is_err());
        assert_eq!(load_url("data:image/png;base64,aGk=").unwrap(), b"hi");
        assert!(load_url("/etc/passwd").is_err());
        assert!(load_url("https://example.com/a.png").is_err());
    }

    #[test]
    fn grid_and_fit() {
        assert_eq!(grid(1024, 1024), (26, 26));
        assert_eq!(grid(40, 41), (2, 1));
        assert_eq!(fit(1024, 1024, 0), (1024, 1024));
        assert_eq!(fit(1024, 1024, 676), (1024, 1024));
        let (w, h) = fit(4000, 3000, 1024);
        let (r, c) = grid(w, h);
        assert!(r * c <= 1024 && w < 4000);
        let (r2, c2) = grid(w + 40, h + 30);
        assert!(r2 * c2 > 1024, "{w} x {h} is not the largest that fits");
        assert!((w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn cut_lays_patches_token_major_and_pads() {
        // 60 x 40: 1 row of 2 tokens; the second token's right half is padding
        let (w, h) = (60usize, 40usize);
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                px[(y * w + x) * 3] = x as u8;
                px[(y * w + x) * 3 + 1] = y as u8;
                px[(y * w + x) * 3 + 2] = 7;
            }
        }
        let p = cut(w, h, &px);
        assert_eq!((p.rows, p.cols, p.patches.len()), (1, 2, 2 * 4 * PATCH_K));
        let at = |tok: usize, ir: usize, ic: usize, py: usize, pxl: usize, c: usize| p.patches[((tok * 2 + ir) * 2 + ic) * PATCH_K + (py * PATCH + pxl) * 3 + c];
        // token 0, patch (1, 1): pixel (y 20 + 3, x 20 + 5)
        assert_eq!(at(0, 1, 1, 3, 5, 0), norm(25.0));
        assert_eq!(at(0, 1, 1, 3, 5, 1), norm(23.0));
        assert_eq!(at(0, 1, 1, 3, 5, 2), norm(7.0));
        // token 1, patch (0, 0): x 40..60 is image, patch (0, 1): x 60..80 is padding
        assert_eq!(at(1, 0, 0, 0, 19, 0), norm(59.0));
        assert_eq!(at(1, 0, 1, 0, 0, 0), norm(PAD));
        assert_ne!(p.digest, cut(w, h, &vec![0u8; w * h * 3]).digest);
    }

    #[test]
    fn resize_keeps_a_flat_image_flat() {
        let px = vec![200u8; 100 * 80 * 3];
        let out = resize(100, 80, &px, 37, 29);
        assert_eq!(out.len(), 37 * 29 * 3);
        assert!(out.iter().all(|&v| v == 200));
    }

    #[test]
    fn composite_uses_the_checkerboard() {
        let rgba = vec![0u8; 16 * 8 * 4];
        let px = composite(16, 8, &rgba);
        assert_eq!((px[0], px[8 * 3]), (204, 153));
        let opaque = [10u8, 20, 30, 255];
        assert_eq!(composite(1, 1, &opaque), vec![10, 20, 30]);
    }
}
