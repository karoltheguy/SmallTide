//! Inkling 2's vision tower (transformers' Inkling2VisionModel; ported from TideLM's
//! inkling_vision.rs): an image's patches (bf16 `[tokens x 4][20 x 20 x 3]`, token-major, a token's
//! 2 x 2 patches in (in_row, in_col) order, each (y, x, channel)) in, one `text_hidden`-wide f32 row
//! an image token out - the rows the image placeholders take in place of their embeddings.
//!
//! Nothing of the tower is resident: its weights (636 MB BF16) stay in the checkpoint file and are
//! streamed through a pinned staging buffer into one device buffer a layer at a time, the scratch
//! is allocated for the image and freed after it. The text engine's pools are not touched.
//!
//! The sequence is [class token; patches]. Embeddings: patch_embedding (K padded 1200 -> 1216)
//! plus the learned 32 x 32 position table resampled bilinearly (half-pixel centres, taps clamped)
//! to the image's patch grid; the class token is class_embedding + cls_position_embedding; ln_pre.
//! Per layer: LayerNorm, q | k | v with biases, axial 2D RoPE (theta 1e4; patches at (row + 1,
//! col + 1), the class token at (0, 0)), full attention, out_proj, LayerScale, residual; LayerNorm,
//! fc + GELU (erf), proj, LayerScale, residual. No final norm. The merger: a token's 4 patch rows
//! are one row of 4096, proj_in, GELU, proj_out. Activations are f32 (the reference's are bf16:
//! the rows agree to rounding).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use smalltide_core::{ComputeBackend, DeviceBuf, Event, PinnedBuf, Result, SmallTideError, Stream};
use smalltide_io::safetensors::inspect_safetensors_header;

const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const INTER: usize = 4096;
const DEPTH: usize = 23;
/// a patch's values: 20 x 20 x 3
pub const PATCH_K: usize = 1200;
/// padded to the GEMM's multiple of 32
const PATCH_KP: usize = 1216;
/// patches a token (2 x 2)
pub const PER_TOKEN: usize = 4;
/// side of the learned position table
const SIDE: usize = 32;
const PROJ: usize = 4096;
const EPS: f32 = 1e-5;
const THETA: f32 = 10000.0;
/// rows of the MLP's 4096-wide activation held at once
const MLP_ROWS: usize = 1024;
const P: &str = "model.visual.";

fn err(m: String) -> SmallTideError {
    SmallTideError::InvalidModel(m)
}

fn bf(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// fp32 -> bf16, round to nearest even
fn to_bf(f: f32) -> u16 {
    let b = f.to_bits();
    if f.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// One axis' two taps into the position table and their weights for index `i` of `size`
/// (bilinear, align_corners False, border padding).
fn taps(i: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let src = (i as f32 + 0.5) * SIDE as f32 / size as f32 - 0.5;
    let fl = src.floor();
    let tap = |o: i64| (fl as i64 + o).clamp(0, SIDE as i64 - 1) as usize;
    let w = |o: f32| (1.0 - (src - fl - o).abs()).max(0.0);
    ([tap(0), tap(1)], [w(0.0), w(1.0)])
}

/// Where the tower's tensors are in the checkpoint file.
pub struct VisionTower {
    path: PathBuf,
    /// name (without `model.visual.`) -> (file offset, bytes)
    at: BTreeMap<String, (u64, usize)>,
    /// width of a feature row (the language model's hidden size)
    pub out: usize,
}

/// A layer's pieces in the device weight buffer, byte offsets (each a multiple of 256).
struct Bundle {
    names: Vec<(String, usize, usize)>, // (tensor, offset, bytes)
    bytes: usize,
}

fn bundle(pieces: &[(String, usize)]) -> Bundle {
    let mut off = 0usize;
    let mut names = Vec::new();
    for (n, b) in pieces {
        names.push((n.clone(), off, *b));
        off += b.div_ceil(256) * 256;
    }
    Bundle { names, bytes: off }
}

fn layer_pieces(i: usize) -> Vec<(String, usize)> {
    let l = format!("backbone.layers.{i}.");
    let a = format!("{l}self_attn.");
    vec![
        (format!("{a}q_proj.weight"), 2 * D * D),
        (format!("{a}k_proj.weight"), 2 * D * D),
        (format!("{a}v_proj.weight"), 2 * D * D),
        (format!("{a}q_proj.bias"), 2 * D),
        (format!("{a}k_proj.bias"), 2 * D),
        (format!("{a}v_proj.bias"), 2 * D),
        (format!("{a}out_proj.weight"), 2 * D * D),
        (format!("{a}out_proj.bias"), 2 * D),
        (format!("{l}ln_1.weight"), 2 * D),
        (format!("{l}ln_1.bias"), 2 * D),
        (format!("{l}ls_1.gamma"), 2 * D),
        (format!("{l}ln_2.weight"), 2 * D),
        (format!("{l}ln_2.bias"), 2 * D),
        (format!("{l}ls_2.gamma"), 2 * D),
        (format!("{l}mlp.c_fc.weight"), 2 * INTER * D),
        (format!("{l}mlp.c_fc.bias"), 2 * INTER),
        (format!("{l}mlp.c_proj.weight"), 2 * D * INTER),
        (format!("{l}mlp.c_proj.bias"), 2 * D),
    ]
}

/// What an encode took, for the log.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeStats {
    pub tokens: usize,
    pub ms: f64,
    pub device_bytes: usize,
}

impl VisionTower {
    /// The tower of the checkpoint at `safetensors` (only its header is read).
    pub fn open(safetensors: &Path, text_hidden: usize) -> Result<Self> {
        let hdr = inspect_safetensors_header(safetensors)?;
        let base = 8 + hdr.header_len;
        let mut at = BTreeMap::new();
        for (name, e) in &hdr.tensors {
            if let Some(n) = name.strip_prefix(P) {
                if e.dtype != "BF16" {
                    return Err(err(format!("{name}: {} (BF16 expected)", e.dtype)));
                }
                at.insert(n.to_string(), (base + e.data_offsets[0] as u64, e.data_offsets[1] - e.data_offsets[0]));
            }
        }
        let t = VisionTower { path: safetensors.to_path_buf(), at, out: text_hidden };
        // every tensor the tower reads, with the size this file is written for
        let mut want: Vec<(String, usize)> = vec![
            ("backbone.embeddings.patch_embedding.weight".into(), 2 * D * PATCH_K),
            ("backbone.embeddings.position_embedding".into(), 2 * SIDE * SIDE * D),
            ("backbone.embeddings.class_embedding".into(), 2 * D),
            ("backbone.embeddings.cls_position_embedding".into(), 2 * D),
            ("backbone.ln_pre.weight".into(), 2 * D),
            ("backbone.ln_pre.bias".into(), 2 * D),
            ("proj_in.weight".into(), 2 * PROJ * PER_TOKEN * D),
            ("proj_out.weight".into(), 2 * text_hidden * PROJ),
        ];
        for i in 0..DEPTH {
            want.extend(layer_pieces(i));
        }
        for (n, b) in &want {
            match t.at.get(n) {
                Some(&(_, len)) if len == *b => {}
                Some(&(_, len)) => return Err(err(format!("{P}{n}: {len} bytes ({b} expected)"))),
                None => return Err(err(format!("{P}{n}: not in the checkpoint"))),
            }
        }
        if t.at.contains_key(&format!("backbone.layers.{DEPTH}.ln_1.weight")) {
            return Err(err(format!("the vision tower has more than {DEPTH} layers")));
        }
        Ok(t)
    }

    fn read_into(&self, f: &mut File, name: &str, dst: &mut [u8]) -> Result<()> {
        let &(off, len) = self.at.get(name).ok_or_else(|| err(format!("{P}{name}: not in the checkpoint")))?;
        if len != dst.len() {
            return Err(err(format!("{P}{name}: {len} bytes ({} expected)", dst.len())));
        }
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(dst)?;
        Ok(())
    }

    fn read_u16(&self, f: &mut File, name: &str, elems: usize) -> Result<Vec<u16>> {
        let mut b = vec![0u8; elems * 2];
        self.read_into(f, name, &mut b)?;
        Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    /// Device bytes an encode of `tokens` image tokens allocates (weights buffer and scratch).
    pub fn device_bytes(&self, tokens: usize) -> usize {
        let s = PER_TOKEN * tokens + 1;
        let w = self.weight_buf_bytes();
        w + 4 * (PER_TOKEN * tokens * PATCH_KP + 3 * s * D + s * 3 * D + 2 * s * 32 + s.min(MLP_ROWS) * INTER + tokens * PROJ + tokens * self.out)
    }

    fn weight_buf_bytes(&self) -> usize {
        let layer = bundle(&layer_pieces(0)).bytes;
        layer.max(2 * PROJ * PER_TOKEN * D).max(2 * self.out * PROJ).max(2 * D * PATCH_KP + 1024)
    }

    /// One image of `rows` x `cols` tokens: `patches` bf16 `[rows * cols * 4][1200]` (token-major,
    /// (in_row, in_col) within a token, each patch (y, x, channel), normalized) -> the feature rows
    /// f32 `[rows * cols][out]`. Every device and pinned buffer it takes is freed before it returns.
    pub fn encode<B: ComputeBackend>(&self, b: &B, st: Stream, patches: &[u16], rows: usize, cols: usize) -> Result<(Vec<f32>, EncodeStats)> {
        let t = rows.checked_mul(cols).unwrap_or(0);
        if t == 0 || patches.len() != t * PER_TOKEN * PATCH_K {
            return Err(err(format!("vision: an image of {rows} x {cols} tokens, {} patch values", patches.len())));
        }
        let t0 = std::time::Instant::now();
        let mut dev: Vec<DeviceBuf> = Vec::new();
        let mut pin: Vec<PinnedBuf> = Vec::new();
        let mut evs: Vec<Event> = Vec::new();
        let r = self.encode_inner(b, st, patches, rows, cols, &mut dev, &mut pin, &mut evs);
        // nothing may still be reading the buffers when they go
        let _ = b.sync_stream(st);
        let bytes: usize = dev.iter().map(|d| d.size_bytes).sum();
        for d in dev {
            let _ = b.free_device(d);
        }
        for p in pin {
            let _ = b.free_pinned(p);
        }
        for e in evs {
            let _ = b.destroy_event(e);
        }
        r.map(|v| (v, EncodeStats { tokens: t, ms: t0.elapsed().as_secs_f64() * 1e3, device_bytes: bytes }))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_inner<B: ComputeBackend>(&self, b: &B, st: Stream, patches: &[u16], rows: usize, cols: usize,
        dev: &mut Vec<DeviceBuf>, pin: &mut Vec<PinnedBuf>, evs: &mut Vec<Event>) -> Result<Vec<f32>> {
        let t = rows * cols;
        let (n, gh, gw) = (PER_TOKEN * t, 2 * rows, 2 * cols);
        let s = n + 1;
        let out_w = self.out;
        let mut f = File::open(&self.path)?;
        let mut dalloc = |bytes: usize| -> Result<DeviceBuf> {
            let d = b.alloc_device(bytes)?;
            dev.push(d);
            Ok(d)
        };
        let wbytes = self.weight_buf_bytes();
        let wbuf = dalloc(wbytes)?;
        let d_patch = dalloc(4 * n * PATCH_KP)?;
        let x = dalloc(4 * s * D)?;
        let h = dalloc(4 * s * D)?;
        let att = dalloc(4 * s * D)?;
        let qkv = dalloc(4 * s * 3 * D)?;
        let cs = dalloc(4 * s * 32)?;
        let sn = dalloc(4 * s * 32)?;
        let mo = dalloc(4 * s.min(MLP_ROWS) * INTER)?;
        let y = dalloc(4 * t * PROJ)?;
        let outb = dalloc(4 * t * out_w)?;
        // two pinned staging buffers: a layer's weights are read from the file into one while the
        // GPU still computes with the previous layer's
        let stage_bytes = wbytes.max(4 * n * PATCH_KP).max(4 * s * D);
        let mut stage = Vec::new();
        for _ in 0..2 {
            let pb = b.alloc_pinned(stage_bytes)?;
            pin.push(pb);
            let ev = b.create_event()?;
            evs.push(ev);
            stage.push((pb, ev, false));
        }
        // host f32 -> device through staging buffer 0, synchronously (setup only)
        let put = |dst: DeviceBuf, v: &[f32]| -> Result<()> {
            let bytes = v.len() * 4;
            unsafe {
                let p = b.pinned_as_mut_slice(stage[0].0)?;
                std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, p.as_mut_ptr(), bytes);
            }
            b.memcpy_h2d_async(dst, 0, stage[0].0, 0, bytes, st)?;
            b.sync_stream(st)
        };
        let dump = |name: &str, buf: DeviceBuf, rows: usize, cols: usize| -> Result<()> {
            let Ok(prefix) = std::env::var("SMALLTIDE_VISION_DUMP") else { return Ok(()) };
            let mut v = vec![0u8; rows * cols * 4];
            let len = v.len();
            b.memcpy_d2h_sync(&mut v, buf.slice(0, len), 0, st)?;
            std::fs::write(format!("{prefix}.{name}.bin"), v)?;
            Ok(())
        };
        // patch p's place on the grid (the patches are block-major)
        let place = |p: usize| -> (usize, usize) {
            let (tok, inb) = (p / PER_TOKEN, p % PER_TOKEN);
            (2 * (tok / cols) + inb / 2, 2 * (tok % cols) + inb % 2)
        };
        // the patches as f32, padded to 1216 columns
        {
            let mut pp = vec![0f32; n * PATCH_KP];
            for r in 0..n {
                for (o, &v) in pp[r * PATCH_KP..r * PATCH_KP + PATCH_K].iter_mut().zip(&patches[r * PATCH_K..(r + 1) * PATCH_K]) {
                    *o = bf(v);
                }
            }
            put(d_patch, &pp)?;
        }
        // row 0: the class token; row 1 + p: patch p's position embedding (the table's four taps
        // weighted and summed in fp32, then bf16 as the reference)
        {
            let pos: Vec<f32> = self.read_u16(&mut f, "backbone.embeddings.position_embedding", SIDE * SIDE * D)?.iter().map(|&v| bf(v)).collect();
            let ce = self.read_u16(&mut f, "backbone.embeddings.class_embedding", D)?;
            let cp = self.read_u16(&mut f, "backbone.embeddings.cls_position_embedding", D)?;
            let (th, tw): (Vec<_>, Vec<_>) = ((0..gh).map(|i| taps(i, gh)).collect(), (0..gw).map(|i| taps(i, gw)).collect());
            let mut e = vec![0f32; s * D];
            for d in 0..D {
                e[d] = bf(to_bf(bf(ce[d]) + bf(cp[d])));
            }
            for p in 0..n {
                let (r, c) = place(p);
                let ((hi, hw), (wi, ww)) = (th[r], tw[c]);
                let row = |a: usize, bb: usize| &pos[(hi[a] * SIDE + wi[bb]) * D..][..D];
                let (t00, t01, t10, t11) = (row(0, 0), row(0, 1), row(1, 0), row(1, 1));
                let (w00, w01, w10, w11) = (hw[0] * ww[0], hw[0] * ww[1], hw[1] * ww[0], hw[1] * ww[1]);
                for (d, o) in e[(1 + p) * D..(2 + p) * D].iter_mut().enumerate() {
                    *o = bf(to_bf(t00[d] * w00 + t01[d] * w01 + t10[d] * w10 + t11[d] * w11));
                }
            }
            put(h, &e)?;
        }
        // the RoPE table [s][32]: pair j < 16 turns by the column's angle j, pair 16 + j by the row's
        {
            let inv: Vec<f32> = (0..HD / 4).map(|j| 1.0f32 / THETA.powf((2 * j) as f32 / (HD / 2) as f32)).collect();
            let (mut c, mut si) = (Vec::with_capacity(s * 32), Vec::with_capacity(s * 32));
            for q in 0..s {
                let (ph, pw) = if q == 0 { (0.0f32, 0.0f32) } else { let (r, cc) = place(q - 1); ((r + 1) as f32, (cc + 1) as f32) };
                for j in 0..HD / 2 {
                    let a = if j < HD / 4 { pw * inv[j] } else { ph * inv[j - HD / 4] };
                    c.push(a.cos());
                    si.push(a.sin());
                }
            }
            put(cs, &c)?;
            put(sn, &si)?;
        }
        // a bundle of tensors from the file -> a staging buffer -> wbuf, stream-ordered after the
        // kernels already queued (they read wbuf's previous contents)
        let mut turn = 0usize;
        let mut upload = |f: &mut File, bd: &Bundle, pad_patch: bool| -> Result<()> {
            let k = turn % 2;
            turn += 1;
            let (pb, ev, used) = stage[k];
            if used {
                b.sync_event(ev)?;
            }
            unsafe {
                let p = b.pinned_as_mut_slice(pb)?;
                if pad_patch {
                    // patch_embedding [1024][1200] -> [1024][1216], K padded with zeros
                    let w = self.read_u16(f, "backbone.embeddings.patch_embedding.weight", D * PATCH_K)?;
                    p[..2 * D * PATCH_KP].fill(0);
                    for o in 0..D {
                        for (i, v) in w[o * PATCH_K..(o + 1) * PATCH_K].iter().enumerate() {
                            p[2 * (o * PATCH_KP + i)..2 * (o * PATCH_KP + i) + 2].copy_from_slice(&v.to_le_bytes());
                        }
                    }
                } else {
                    for (name, off, bytes) in &bd.names {
                        self.read_into(f, name, &mut p[*off..*off + *bytes])?;
                    }
                }
            }
            b.memcpy_h2d_async(wbuf, 0, pb, 0, bd.bytes, st)?;
            b.record_event(ev, st)?;
            stage[k].2 = true;
            Ok(())
        };
        let piece = |bd: &Bundle, name: &str| -> DeviceBuf {
            let (_, off, bytes) = bd.names.iter().find(|(n, _, _)| n.ends_with(name)).expect("bundle piece");
            wbuf.slice(*off, *bytes)
        };

        // embeddings: h[1..] += patches . W^T, then ln_pre over the sequence
        let pre = bundle(&[
            ("backbone.embeddings.patch_embedding.weight".to_string(), 2 * D * PATCH_KP),
            ("backbone.ln_pre.weight".to_string(), 2 * D),
            ("backbone.ln_pre.bias".to_string(), 2 * D),
        ]);
        upload(&mut f, &Bundle { names: vec![], bytes: 2 * D * PATCH_KP }, true)?;
        let x1 = x.slice(4 * D, 4 * n * D);
        let h1 = h.slice(4 * D, 4 * n * D);
        b.vit_gemm_bf16(x1, d_patch, wbuf.slice(0, 2 * D * PATCH_KP), n, D, PATCH_KP, st)?;
        b.vit_epilogue(h1, x1, None, None, Some(h1), n, D, false, st)?;
        dump("embed", h, s, D)?;
        let lnb = bundle(&pre.names[1..].iter().map(|(nm, _, by)| (nm.clone(), *by)).collect::<Vec<_>>());
        upload(&mut f, &lnb, false)?;
        b.vit_ln(x, h, piece(&lnb, "ln_pre.weight"), piece(&lnb, "ln_pre.bias"), s, D, EPS, st)?;
        dump("pre", x, s, D)?;

        for li in 0..DEPTH {
            let bd = bundle(&layer_pieces(li));
            upload(&mut f, &bd, false)?;
            let w = |nm: &str| piece(&bd, nm);
            b.vit_ln(h, x, w("ln_1.weight"), w("ln_1.bias"), s, D, EPS, st)?;
            // q | k | v: three [1024][1024] projections into the row's three thirds
            // (q_proj.weight, k_proj.weight and v_proj.weight are adjacent in the bundle: one GEMM)
            let qkv_w = wbuf.slice(bd.names[0].1, 2 * 3 * D * D);
            let qkv_b = wbuf.slice(bd.names[3].1, 2 * 3 * D);
            b.vit_gemm_bf16(qkv, h, qkv_w, s, 3 * D, D, st)?;
            b.vit_epilogue(qkv, qkv, Some(qkv_b), None, None, s, 3 * D, false, st)?;
            b.vit_rope(qkv, s, HEADS, cs, sn, st)?;
            b.vit_attn(att, qkv, s, HEADS, st)?;
            b.vit_gemm_bf16(h, att, w("out_proj.weight"), s, D, D, st)?;
            b.vit_epilogue(x, h, Some(w("out_proj.bias")), Some(w("ls_1.gamma")), Some(x), s, D, false, st)?;
            b.vit_ln(h, x, w("ln_2.weight"), w("ln_2.bias"), s, D, EPS, st)?;
            for r0 in (0..s).step_by(MLP_ROWS) {
                let m = MLP_ROWS.min(s - r0);
                let (hr, xr, ar) = (h.slice(4 * r0 * D, 4 * m * D), x.slice(4 * r0 * D, 4 * m * D), att.slice(4 * r0 * D, 4 * m * D));
                let mob = mo.slice(0, 4 * m * INTER);
                b.vit_gemm_bf16(mob, hr, w("mlp.c_fc.weight"), m, INTER, D, st)?;
                b.vit_epilogue(mob, mob, Some(w("mlp.c_fc.bias")), None, None, m, INTER, true, st)?;
                b.vit_gemm_bf16(ar, mob, w("mlp.c_proj.weight"), m, D, INTER, st)?;
                b.vit_epilogue(xr, ar, Some(w("mlp.c_proj.bias")), Some(w("ls_2.gamma")), Some(xr), m, D, false, st)?;
            }
            dump(&format!("b{li}"), x, s, D)?;
        }
        // the merger: past the class token, a token's 4 patch rows are one row of 4096
        let pin_b = bundle(&[("proj_in.weight".to_string(), 2 * PROJ * PER_TOKEN * D)]);
        upload(&mut f, &pin_b, false)?;
        b.vit_gemm_bf16(y, x1, wbuf.slice(0, 2 * PROJ * PER_TOKEN * D), t, PROJ, PER_TOKEN * D, st)?;
        b.vit_epilogue(y, y, None, None, None, t, PROJ, true, st)?;
        dump("merge", y, t, PROJ)?;
        let pout_b = bundle(&[("proj_out.weight".to_string(), 2 * out_w * PROJ)]);
        upload(&mut f, &pout_b, false)?;
        b.vit_gemm_bf16(outb, y, wbuf.slice(0, 2 * out_w * PROJ), t, out_w, PROJ, st)?;
        let mut bytes = vec![0u8; 4 * t * out_w];
        b.memcpy_d2h_sync(&mut bytes, outb, 0, st)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(to_bf(1.0), 0x3f80);
        assert_eq!(bf(to_bf(-2.5)), -2.5);
        assert_eq!(to_bf(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(to_bf(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn taps_match_half_pixel_bilinear() {
        for i in 0..SIDE {
            let (t, w) = taps(i, SIDE);
            assert_eq!((t[0], w[0], w[1]), (i, 1.0, 0.0));
        }
        let (t, w) = taps(0, 2 * SIDE);
        assert_eq!((t, w), ([0, 0], [0.25, 0.75]));
        let (t, w) = taps(1, 2 * SIDE);
        assert_eq!((t, w), ([0, 1], [0.75, 0.25]));
        let (t, w) = taps(2 * SIDE - 1, 2 * SIDE);
        assert_eq!((t, w), ([SIDE - 1, SIDE - 1], [0.75, 0.25]));
    }

    #[test]
    fn a_layer_bundle_keeps_q_k_v_adjacent() {
        let bd = bundle(&layer_pieces(3));
        assert_eq!(bd.names[1].1, 2 * D * D);
        assert_eq!(bd.names[2].1, 4 * D * D);
        assert_eq!(bd.names[4].1 - bd.names[3].1, 2 * D);
        assert!(bd.bytes < 26 << 20);
    }
}
