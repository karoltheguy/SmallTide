//! DSpark drafter for Inkling-2-Mini (`mtp.safetensors`): a 3-layer block-diffusion drafter over
//! the target's taps (outputs of logical layers 4 / 22 / 40 + the final normed hidden) with a
//! rank-256 Markov head folded into the target's unembedding.
//!
//! Per anchor position `P` (the target has committed positions `< P` and sampled the anchor
//! token at `P`): a block of 8 rows (anchor embedding + 7 mask embeddings) at positions
//! `P..P+8` attends to the context K/V of positions `< P` (local layers: the last 1024) and to
//! itself (non-causal); rows 1..8 then feed the Markov head, which emits the drafts for
//! positions `P+1..` greedily, one step per draft, entirely on the GPU (one readback).
//!
//! Reference: `tools/inkling2_draft_ref.py`.

use crate::weights::{Fp8MatrixGpu, Inkling2DenseGpu, Uploader};
use smalltide_core::backend::GemvSegment;
use smalltide_core::{ComputeBackend, DeviceBuf, PinnedBuf, QueueKind, Result, SmallTideError, Stream};
use smalltide_io::dense_loader::{read_f32_vec, read_fp8_matrix, read_raw_bytes};
use smalltide_io::quant::bf16_to_f32;
use smalltide_io::safetensors::inspect_safetensors_header;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub const BLOCK: usize = 8;
const D: usize = 2304;
const HQ: usize = 24;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 9216;
const R: usize = 256;
const WINDOW: usize = 1024;
/// Local layers' context ring: the window plus a block of not-yet-committed positions.
const RING: usize = WINDOW + 16;
const THETA: f32 = 1.0e6;
const EPS: f32 = 1e-6;
const SOFTCAP: f32 = 30.0;

struct DrafterLayer {
    local: bool,
    in_ln: DeviceBuf,
    post_ln: DeviceBuf,
    dflare: DeviceBuf,
    q: Fp8MatrixGpu,
    k: Fp8MatrixGpu,
    v: Fp8MatrixGpu,
    k_ctx: Fp8MatrixGpu,
    v_ctx: Fp8MatrixGpu,
    o: Fp8MatrixGpu,
    gate: Fp8MatrixGpu,
    up: Fp8MatrixGpu,
    down: Fp8MatrixGpu,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    k_norm_ctx: DeviceBuf,
    /// Context K / V, f16 `[slots][HK * HD]` (local: ring of `RING`, else `max_seq`).
    kc: DeviceBuf,
    vc: DeviceBuf,
}

struct Scratch {
    stage: DeviceBuf, // [8][D] uploaded block input
    res: DeviceBuf,   // [8][D]
    xn: DeviceBuf,    // [8][D]
    tmp: DeviceBuf,   // [8][D]
    q: DeviceBuf,     // [8][HQ*HD]
    kb: DeviceBuf,    // [8][HK*HD]
    vb: DeviceBuf,    // [8][HK*HD]
    att: DeviceBuf,   // [8][HQ*HD]
    g: DeviceBuf,     // [8][FF]
    u: DeviceBuf,     // [8][FF]
    ctx_c: DeviceBuf, // [CHUNK][D]
    ctx_k: DeviceBuf, // [CHUNK][HK*HD]
    ctx_v: DeviceBuf, // [CHUNK][HK*HD]
    base: DeviceBuf,  // [7][V]
    frnn: DeviceBuf,  // [D]
    z: DeviceBuf,     // [2R + 2D]
    raw: DeviceBuf,   // [3R]
    state: DeviceBuf, // [R]
    latent: DeviceBuf,
    drafts: DeviceBuf, // [8] u32
    pmax: DeviceBuf,
    pidx: DeviceBuf,
    pinned: PinnedBuf, // [8][D] f32
}

pub struct Drafter<B: ComputeBackend> {
    backend: Arc<B>,
    layers: Vec<DrafterLayer>,
    norm_w: DeviceBuf,
    mask_embed: Vec<f32>,
    /// softmax(fuse_weight): per drafter layer + the Markov head, over the 4 taps.
    alpha: [[f32; 4]; 4],
    mk_norm: DeviceBuf,
    w1: DeviceBuf, // bf16 [V_pad][R]
    rnn_w: DeviceBuf,
    rnn_b: DeviceBuf,
    fold16: DeviceBuf, // f16 [R][V]
    fold_b: DeviceBuf, // [V]
    vocab: usize,
    max_seq: usize,
    chunk: usize,
    sc: Scratch,
    pub vram_bytes: usize,
}

fn bf16r(x: f32) -> f32 {
    // round-to-nearest-even to bf16
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    f32::from_bits(r & 0xFFFF_0000)
}

impl<B: ComputeBackend> Drafter<B> {
    /// Loads `mtp.safetensors` (dense projections quantized to FP8 per row) and folds the Markov
    /// head's output projection into the target's unembedding (`fold = U @ hidden_out`, f16).
    /// `chunk`: most rows one `ingest` carries (the prefill chunk).
    pub fn load(backend: Arc<B>, path: &Path, dense: &Inkling2DenseGpu<B>, vocab: usize, max_seq: usize, chunk: usize) -> Result<Self> {
        let hdr = inspect_safetensors_header(path)?;
        let base = 8u64 + hdr.header_len;
        let mut f = File::open(path)?;
        let staging = backend.alloc_pinned(64 * 1024 * 1024)?;
        let stream = backend.create_stream(QueueKind::Compute)?;
        let up = Uploader { backend: backend.as_ref(), staging, stream };
        let mut bytes = 0usize;
        let res = (|| -> Result<Self> {
            let fp8 = |f: &mut File, name: &str, bytes: &mut usize| -> Result<Fp8MatrixGpu> {
                let m = read_fp8_matrix(f, &hdr, base, name)?;
                *bytes += m.byte_size();
                up.upload_fp8(&m)
            };
            let mut layers = Vec::with_capacity(3);
            for l in 0..3 {
                let p = format!("layers.{l}");
                let q = fp8(&mut f, &format!("{p}.self_attn.q_proj.weight"), &mut bytes)?;
                let k = fp8(&mut f, &format!("{p}.self_attn.k_proj.weight"), &mut bytes)?;
                let v = fp8(&mut f, &format!("{p}.self_attn.v_proj.weight"), &mut bytes)?;
                let k_ctx = fp8(&mut f, &format!("{p}.self_attn.k_proj_ctx.weight"), &mut bytes)?;
                let v_ctx = fp8(&mut f, &format!("{p}.self_attn.v_proj_ctx.weight"), &mut bytes)?;
                let o = fp8(&mut f, &format!("{p}.self_attn.o_proj.weight"), &mut bytes)?;
                let gate = fp8(&mut f, &format!("{p}.mlp.gate_proj.weight"), &mut bytes)?;
                let upm = fp8(&mut f, &format!("{p}.mlp.up_proj.weight"), &mut bytes)?;
                let down = fp8(&mut f, &format!("{p}.mlp.down_proj.weight"), &mut bytes)?;
                let vec = |f: &mut File, name: &str| -> Result<DeviceBuf> { up.upload_f32(&read_f32_vec(f, &hdr, base, &format!("{p}.{name}"))?) };
                let in_ln = vec(&mut f, "input_layernorm.weight")?;
                let post_ln = vec(&mut f, "post_attention_layernorm.weight")?;
                let dflare = vec(&mut f, "dflare_norm.weight")?;
                let q_norm = vec(&mut f, "self_attn.q_norm.weight")?;
                let k_norm = vec(&mut f, "self_attn.k_norm.weight")?;
                let k_norm_ctx = vec(&mut f, "self_attn.k_norm_ctx.weight")?;
                let local = l != 1;
                let slots = if local { RING } else { max_seq };
                let kc = backend.alloc_device(slots * HK * HD * 2)?;
                let vc = backend.alloc_device(slots * HK * HD * 2)?;
                bytes += 2 * slots * HK * HD * 2;
                layers.push(DrafterLayer { local, in_ln, post_ln, dflare, q, k, v, k_ctx, v_ctx, o, gate, up: upm, down, q_norm, k_norm, k_norm_ctx, kc, vc });
            }
            let rdv = |f: &mut File, name: &str| read_f32_vec(f, &hdr, base, name);
            let norm_w = up.upload_f32(&rdv(&mut f, "norm.weight")?)?;
            let mask_embed: Vec<f32> = rdv(&mut f, "mask_embed")?.into_iter().map(bf16r).collect();
            let fw = rdv(&mut f, "fuse_weight")?;
            let mut alpha = [[0f32; 4]; 4];
            for r in 0..4 {
                let row = &fw[r * 4..r * 4 + 4];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = row.iter().map(|x| (x - mx).exp()).collect();
                let s: f32 = e.iter().sum();
                for t in 0..4 {
                    alpha[r][t] = bf16r(e[t] / s);
                }
            }
            let mk_norm = up.upload_f32(&rdv(&mut f, "markov_head.dflare_norm.weight")?)?;
            let w1 = up.upload_bytes(&read_raw_bytes(&mut f, &hdr, base, "markov_head.markov_w1.weight")?)?;
            let rnn_w = up.upload_f32(&rdv(&mut f, "markov_head.rnn_proj.weight")?)?;
            let rnn_b = up.upload_f32(&rdv(&mut f, "markov_head.rnn_proj.bias")?)?;
            bytes += w1.size_bytes + rnn_w.size_bytes;
            // fold: x[r] = hidden_out[:, r] (r < R), x[R] = hidden_out bias; y = U x -> [R + 1][V].
            let ho_w = rdv(&mut f, "markov_head.hidden_out.weight")?; // [D][R]
            let ho_b = rdv(&mut f, "markov_head.hidden_out.bias")?;
            let mut x = vec![0f32; (R + 1) * D];
            for d in 0..D {
                for r in 0..R {
                    x[r * D + d] = ho_w[d * R + r];
                }
                x[R * D + d] = ho_b[d];
            }
            let xd = up.upload_f32(&x)?;
            let y = backend.alloc_device((R + 1) * vocab * 4)?;
            let u = &dense.unembed_fp8;
            backend.gemm_fp8_rows(y, xd, u.data, u.row_scales, R + 1, vocab, D, stream)?;
            let fold16 = backend.alloc_device(R * vocab * 2)?;
            backend.f32_to_f16(fold16, y, R * vocab, stream)?;
            let fold_b = backend.alloc_device(vocab * 4)?;
            backend.vec_copy(fold_b, y.slice(R * vocab * 4, vocab * 4), vocab, stream)?;
            backend.sync_stream(stream)?;
            backend.free_device(y)?;
            backend.free_device(xd)?;
            bytes += fold16.size_bytes + fold_b.size_bytes;

            let a = |n: usize| backend.alloc_device(n * 4);
            let sc = Scratch {
                stage: a(BLOCK * D)?,
                res: a(BLOCK * D)?,
                xn: a(BLOCK * D)?,
                tmp: a(BLOCK * D)?,
                q: a(BLOCK * HQ * HD)?,
                kb: a(BLOCK * HK * HD)?,
                vb: a(BLOCK * HK * HD)?,
                att: a(BLOCK * HQ * HD)?,
                g: a(BLOCK * FF)?,
                u: a(BLOCK * FF)?,
                ctx_c: a(chunk * D)?,
                ctx_k: a(chunk * HK * HD)?,
                ctx_v: a(chunk * HK * HD)?,
                base: a((BLOCK - 1) * vocab)?,
                frnn: a(D)?,
                z: a(2 * R + 2 * D)?,
                raw: a(3 * R)?,
                state: a(R)?,
                latent: a(R)?,
                drafts: a(BLOCK)?,
                pmax: a(vocab.div_ceil(256))?,
                pidx: a(vocab.div_ceil(256))?,
                pinned: backend.alloc_pinned(BLOCK * D * 4)?,
            };
            bytes += ((BLOCK - 1) * vocab + chunk * (D + 2 * HK * HD)) * 4;
            Ok(Self {
                backend: backend.clone(),
                layers,
                norm_w,
                mask_embed,
                alpha,
                mk_norm,
                w1,
                rnn_w,
                rnn_b,
                fold16,
                fold_b,
                vocab,
                max_seq,
                chunk,
                sc,
                vram_bytes: bytes,
            })
        })();
        let _ = backend.sync_stream(stream);
        let _ = backend.free_pinned(staging);
        let _ = backend.destroy_stream(stream);
        res
    }

    /// Context K/V of `m` committed rows at positions `pos0..` from their taps
    /// (`taps` = `[4][tap_stride][D]`, rows from 0).
    pub fn ingest(&self, taps: DeviceBuf, tap_stride: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if m > self.chunk || pos0 + m > self.max_seq {
            return Err(SmallTideError::Backend { backend: "drafter", message: format!("ingest {m} rows at {pos0}") });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;
        let kvd = HK * HD;
        for (l, ly) in self.layers.iter().enumerate() {
            b.dspark_mix_rms(sc.ctx_c, taps, tap_stride, self.alpha[l], ly.dflare, m, D, EPS, s)?;
            if m <= 8 {
                let segs = [
                    GemvSegment { w: ly.k_ctx.data, row_scales: Some(ly.k_ctx.row_scales), x: sc.ctx_c, y: sc.ctx_k, n_rows: kvd, k_cols: D },
                    GemvSegment { w: ly.v_ctx.data, row_scales: Some(ly.v_ctx.row_scales), x: sc.ctx_c, y: sc.ctx_v, n_rows: kvd, k_cols: D },
                ];
                b.gemv_multi_rows(&segs, m, s)?;
            } else {
                b.gemm_fp8_rows(sc.ctx_k, sc.ctx_c, ly.k_ctx.data, ly.k_ctx.row_scales, m, kvd, D, s)?;
                b.gemm_fp8_rows(sc.ctx_v, sc.ctx_c, ly.v_ctx.data, ly.v_ctx.row_scales, m, kvd, D, s)?;
            }
            b.dspark_head_norm_rope(sc.ctx_k, ly.k_norm_ctx, HK, m, pos0, THETA, EPS, s)?;
            b.dspark_ctx_store(ly.kc, ly.vc, sc.ctx_k, sc.ctx_v, pos0, m, if ly.local { RING } else { 0 }, kvd, s)?;
        }
        Ok(())
    }

    /// Drafts `n` (<= 7) tokens after `anchor` at position `p`; `tap_row` is the row of position
    /// `p - 1` in `taps`. Context of positions `< p` must have been ingested.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(&mut self, dense: &Inkling2DenseGpu<B>, anchor: u32, p: usize, taps: DeviceBuf, tap_stride: usize, tap_row: usize, n: usize, s: Stream) -> Result<Vec<u32>> {
        let n = n.clamp(1, BLOCK - 1);
        if p + BLOCK > self.max_seq {
            return Err(SmallTideError::Backend { backend: "drafter", message: format!("draft at {p} past max_seq") });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;
        // Block input: anchor embedding (normed below), then bf16 mask embeddings.
        unsafe {
            let dst = b.pinned_as_mut_slice(sc.pinned)?;
            let row = &dense.embed_bf16_host[anchor as usize * D * 2..(anchor as usize + 1) * D * 2];
            for i in 0..D {
                let v = bf16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
                dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            for r in 1..BLOCK {
                for i in 0..D {
                    let o = (r * D + i) * 4;
                    dst[o..o + 4].copy_from_slice(&self.mask_embed[i].to_le_bytes());
                }
            }
        }
        b.memcpy_h2d_async(sc.stage, 0, sc.pinned, 0, BLOCK * D * 4, s)?;
        b.rms_norm(sc.res, sc.stage, dense.embed_norm, 1, D, EPS, s)?;
        b.vec_copy(sc.res.slice(D * 4, (BLOCK - 1) * D * 4), sc.stage.slice(D * 4, (BLOCK - 1) * D * 4), (BLOCK - 1) * D, s)?;
        b.rms_norm(sc.xn, sc.res, self.layers[0].in_ln, BLOCK, D, EPS, s)?;
        let seg = |m: &Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| GemvSegment { w: m.data, row_scales: Some(m.row_scales), x, y, n_rows: m.n_rows, k_cols: m.k_cols };
        let scale = 1.0 / (HD as f32).sqrt();
        for (l, ly) in self.layers.iter().enumerate() {
            b.gemv_multi_rows(&[seg(&ly.q, sc.xn, sc.q), seg(&ly.k, sc.xn, sc.kb), seg(&ly.v, sc.xn, sc.vb)], BLOCK, s)?;
            b.dspark_head_norm_rope(sc.q, ly.q_norm, HQ, BLOCK, p, THETA, EPS, s)?;
            b.dspark_head_norm_rope(sc.kb, ly.k_norm, HK, BLOCK, p, THETA, EPS, s)?;
            let (lo, ring) = if ly.local { (p.saturating_sub(WINDOW), RING) } else { (0, 0) };
            b.dspark_attn(sc.att, sc.q, sc.kb, sc.vb, ly.kc, ly.vc, BLOCK, p, lo, ring, HQ, HK, BLOCK, scale, s)?;
            b.gemv_multi_rows(&[seg(&ly.o, sc.att, sc.tmp)], BLOCK, s)?;
            b.vec_add(sc.res, sc.res, sc.tmp, BLOCK * D, s)?;
            b.rms_norm(sc.xn, sc.res, ly.post_ln, BLOCK, D, EPS, s)?;
            b.gemv_multi_rows(&[seg(&ly.gate, sc.xn, sc.g), seg(&ly.up, sc.xn, sc.u)], BLOCK, s)?;
            b.silu_mul(sc.g, sc.g, sc.u, BLOCK * FF, s)?;
            b.gemv_multi_rows(&[seg(&ly.down, sc.g, sc.tmp)], BLOCK, s)?;
            b.vec_add(sc.res, sc.res, sc.tmp, BLOCK * D, s)?;
            let next = if l + 1 < self.layers.len() { self.layers[l + 1].in_ln } else { self.norm_w };
            b.rms_norm(sc.xn, sc.res, next, BLOCK, D, EPS, s)?;
        }
        // Markov head: base logits of block rows 1.., the context feature of position p - 1.
        let hb = sc.xn.slice(D * 4, (BLOCK - 1) * D * 4);
        let u = &dense.unembed_fp8;
        b.gemm_fp8_rows_decode(sc.base, hb, u.data, u.row_scales, n, self.vocab, D, s)?;
        let toff = tap_row * D * 4;
        b.dspark_mix_rms(sc.frnn, taps.slice(toff, taps.size_bytes - toff), tap_stride, self.alpha[3], self.mk_norm, 1, D, EPS, s)?;
        b.zero_f32(sc.state, R, s)?;
        for i in 0..n {
            b.dspark_markov_step(
                sc.z, sc.raw, sc.state, sc.latent, self.w1, sc.drafts, i, anchor,
                hb.slice(i * D * 4, D * 4), sc.frnn, self.rnn_w, self.rnn_b,
                sc.base.slice(i * self.vocab * 4, self.vocab * 4), self.fold_b, self.fold16, sc.pmax, sc.pidx,
                R, D, self.vocab, SOFTCAP, s,
            )?;
        }
        let mut out = vec![0u8; n * 4];
        b.memcpy_d2h_sync(&mut out, sc.drafts.slice(0, n * 4), 0, s)?;
        Ok(out.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
