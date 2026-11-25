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
use smalltide_io::dense_loader::{read_f32_vec, read_fp8_matrix, read_raw_bytes, Fp8MatrixHost};
use smalltide_io::quant::{bf16_to_f32, f32_to_fp8_e4m3fn, FP8_E4M3FN_MAX};
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
const MAX_ATTN_SPLITS: usize = 16;
const THETA: f32 = 1.0e6;
const EPS: f32 = 1e-6;

struct DrafterLayer {
    local: bool,
    v_q4: bool,
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
    /// Context K / V in FP8 E4M3FN `[slots][HK * HD]` + per-head-pos f32 scales `[slots][HK]`.
    /// A local layer's are the current lane's ring (see `lane_ctx`); the global layer's hold
    /// every lane's context, a lane's positions mapped through its page table when paged.
    kc: DeviceBuf,
    ks: DeviceBuf,
    vc: DeviceBuf,
    vs: DeviceBuf,
    /// Local layers: each lane's ring (kc, ks, vc, vs). Empty for the global layer.
    lane_ctx: Vec<[DeviceBuf; 4]>,
}

struct Scratch {
    anchor_in: DeviceBuf, // [D] f32 uploaded anchor embedding
    res: DeviceBuf,       // [8][D]
    xn: DeviceBuf,        // [8][D]
    tmp: DeviceBuf,       // [8][D]
    q: DeviceBuf,         // [8][HQ*HD]
    kb: DeviceBuf,        // [8][HK*HD]
    vb: DeviceBuf,        // [8][HK*HD]
    att: DeviceBuf,       // [8][HQ*HD]
    part_o: DeviceBuf,    // [8][HQ][MAX_ATTN_SPLITS][HD]
    part_m: DeviceBuf,    // [8][HQ][MAX_ATTN_SPLITS]
    part_l: DeviceBuf,    // [8][HQ][MAX_ATTN_SPLITS]
    g: DeviceBuf,         // [8][FF]
    u: DeviceBuf,         // [8][FF]
    ctx_c: DeviceBuf,     // [CHUNK][D]
    ctx_k: DeviceBuf,     // [CHUNK][HK*HD]
    ctx_v: DeviceBuf,     // [CHUNK][HK*HD]
    base: DeviceBuf,      // [7][V]
    frnn: DeviceBuf,      // [D]
    frnn_proj: DeviceBuf, // [3R]
    hb_proj: DeviceBuf,   // [7][3R]
    state: DeviceBuf,     // [R]
    latent: DeviceBuf,    // [R]
    drafts: DeviceBuf,    // [8] u32 (followed by [8] f32 margins in `drafts_and_margins`)
    margins: DeviceBuf,   // [8] f32
    pmax: DeviceBuf,      // [ceil(V/1024)]
    pidx: DeviceBuf,      // [ceil(V/1024)]
    pmax2: DeviceBuf,     // [ceil(V/1024)]
    pinned: PinnedBuf,    // [D] f32 (single anchor row)
}

pub struct DSparkDrafter<B: ComputeBackend> {
    backend: Arc<B>,
    layers: Vec<DrafterLayer>,
    norm_w: DeviceBuf,
    mask_embed_gpu: DeviceBuf, // [D] f32 on GPU
    /// softmax(fuse_weight): per drafter layer + the Markov head, over the 4 taps.
    alpha: [[f32; 4]; 4],
    mk_norm: DeviceBuf,
    /// `markov_w1` (`[V_pad][R]` bf16, ~103 MiB) kept in host-mapped pinned memory: each draft
    /// step reads only 1 row (512 bytes), saving ~103 MiB of resident VRAM for experts.
    w1: DeviceBuf,
    w1_host: PinnedBuf,
    /// Split Markov RNN weights: `[3R][2R]` f16 for dynamic `[state | w1[prev]]` (786 KB in L2),
    /// plus per-row FP8 `[3R][D]` for `hb[0..n]` and `f_rnn` (precomputed once before the loop).
    rnn_w_dyn: DeviceBuf,
    rnn_w_h: Fp8MatrixGpu,
    rnn_w_frnn: Fp8MatrixGpu,
    rnn_b: DeviceBuf,
    /// Folded output projection `U @ hidden_out` quantized per-r to FP8 E4M3FN `[R][V_pad]`.
    fold_fp8: DeviceBuf,
    fold_scales: DeviceBuf, // [R] f32
    fold_b: DeviceBuf,      // [V] f32
    vocab: usize,
    vocab_pad: usize,
    max_seq: usize,
    chunk: usize,
    sc: Scratch,
    pub vram_bytes: usize,
    /// The lane whose context `ingest` writes and `draft` reads, and its page table into the
    /// shared KV pool (`None`: the global layer's cache is one sequence's, position = slot).
    lane: usize,
    page_table: Option<DeviceBuf>,
}

fn bf16r(x: f32) -> f32 {
    // round-to-nearest-even to bf16
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    f32::from_bits(r & 0xFFFF_0000)
}

pub(crate) fn quantize_f32_rows_to_fp8(src: &[f32], n_rows: usize, k_cols: usize) -> Fp8MatrixHost {
    let mut data = vec![0u8; n_rows * k_cols];
    let mut row_scales = vec![1.0f32; n_rows];
    for r in 0..n_rows {
        let row = &src[r * k_cols..(r + 1) * k_cols];
        let mx = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let (sc, inv) = if mx > 1e-12 { (mx / FP8_E4M3FN_MAX, FP8_E4M3FN_MAX / mx) } else { (1.0, 0.0) };
        row_scales[r] = sc;
        let dst = &mut data[r * k_cols..(r + 1) * k_cols];
        for (d, &v) in dst.iter_mut().zip(row) {
            *d = f32_to_fp8_e4m3fn(v * inv);
        }
    }
    Fp8MatrixHost { n_rows, k_cols, data, row_scales }
}

impl<B: ComputeBackend> DSparkDrafter<B> {
    /// Loads `mtp.safetensors` (dense projections quantized to FP8 per row), keeps `w1` in
    /// host-mapped pinned memory, splits `rnn_proj` into dynamic/static blocks, uses FP8 context
    /// KV, and folds `U @ hidden_out` into padded FP8 E4M3FN.
    /// `lanes`: sequences that each keep their own context (local rings a lane, about 4 MiB).
    /// `global_slots`: slots of the global layer's cache - the positions of the shared KV pool
    /// when the lanes page into it (pass each lane's table with `set_lane`), else `max_seq`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(backend: Arc<B>, path: &Path, dense: &Inkling2DenseGpu<B>, vocab: usize, max_seq: usize, chunk: usize, global_v_q4: bool,
                lanes: usize, global_slots: usize) -> Result<Self> {
        let hdr = inspect_safetensors_header(path)?;
        let base = 8u64 + hdr.header_len;
        let mut f = File::open(path)?;
        let staging = backend.alloc_pinned(64 * 1024 * 1024)?;
        let stream = backend.create_stream(QueueKind::Compute)?;
        let up = Uploader { backend: backend.as_ref(), staging, stream };
        let mut bytes = 0usize;
        let vocab_pad = (vocab + 15) & !15;
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
                let v_q4 = !local && global_v_q4;
                let slots = if local { RING } else if global_slots > 0 { global_slots } else { max_seq };
                let vc_bytes = slots * HK * if v_q4 { HD / 2 } else { HD };
                let vs_bytes = slots * HK * if v_q4 { 16 } else { 4 };
                let mut lane_ctx: Vec<[DeviceBuf; 4]> = Vec::new();
                for _ in 0..if local { lanes.max(1) } else { 1 } {
                    lane_ctx.push([
                        backend.alloc_device(slots * HK * HD)?,
                        backend.alloc_device(slots * HK * 4)?,
                        backend.alloc_device(vc_bytes)?,
                        backend.alloc_device(vs_bytes)?,
                    ]);
                    bytes += slots * HK * (HD + 4) + vc_bytes + vs_bytes;
                }
                let [kc, ks, vc, vs] = lane_ctx[0];
                if !local {
                    lane_ctx.clear();
                }
                layers.push(DrafterLayer {
                    local, v_q4, in_ln, post_ln, dflare, q, k, v, k_ctx, v_ctx, o, gate, up: upm, down,
                    q_norm, k_norm, k_norm_ctx, kc, ks, vc, vs, lane_ctx,
                });
            }
            let rdv = |f: &mut File, name: &str| read_f32_vec(f, &hdr, base, name);
            let norm_w = up.upload_f32(&rdv(&mut f, "norm.weight")?)?;
            let mask_embed_vec: Vec<f32> = rdv(&mut f, "mask_embed")?.into_iter().map(bf16r).collect();
            let mask_embed_gpu = up.upload_f32(&mask_embed_vec)?;
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

            // Keep markov_w1 (~103 MiB BF16) in host-mapped pinned memory: 0 VRAM!
            let w1_raw = read_raw_bytes(&mut f, &hdr, base, "markov_head.markov_w1.weight")?;
            let (w1, w1_host) = backend.alloc_host_mapped(w1_raw.len())?;
            unsafe {
                backend.pinned_as_mut_slice(w1_host)?[..w1_raw.len()].copy_from_slice(&w1_raw);
            }

            // Split rnn_proj.weight [3R][2R + 2D] into W_dyn [3R][2R] (f16), W_h [3R][D] (FP8), W_frnn [3R][D] (FP8).
            let rnn_w_full = rdv(&mut f, "markov_head.rnn_proj.weight")?;
            let rnn_cols = 2 * R + 2 * D;
            let mut w_dyn_f32 = vec![0.0f32; 3 * R * 2 * R];
            let mut w_h_f32 = vec![0.0f32; 3 * R * D];
            let mut w_frnn_f32 = vec![0.0f32; 3 * R * D];
            for row in 0..(3 * R) {
                let src = &rnn_w_full[row * rnn_cols..(row + 1) * rnn_cols];
                w_dyn_f32[row * 2 * R..(row + 1) * 2 * R].copy_from_slice(&src[0..2 * R]);
                w_h_f32[row * D..(row + 1) * D].copy_from_slice(&src[2 * R..2 * R + D]);
                w_frnn_f32[row * D..(row + 1) * D].copy_from_slice(&src[2 * R + D..2 * R + 2 * D]);
            }
            let w_dyn_tmp = up.upload_f32(&w_dyn_f32)?;
            let rnn_w_dyn = backend.alloc_device(3 * R * 2 * R * 2)?;
            backend.f32_to_f16(rnn_w_dyn, w_dyn_tmp, 3 * R * 2 * R, stream)?;
            backend.sync_stream(stream)?;
            backend.free_device(w_dyn_tmp)?;
            let rnn_w_h_host = quantize_f32_rows_to_fp8(&w_h_f32, 3 * R, D);
            let rnn_w_frnn_host = quantize_f32_rows_to_fp8(&w_frnn_f32, 3 * R, D);
            let rnn_w_h = up.upload_fp8(&rnn_w_h_host)?;
            let rnn_w_frnn = up.upload_fp8(&rnn_w_frnn_host)?;
            let rnn_b = up.upload_f32(&rdv(&mut f, "markov_head.rnn_proj.bias")?)?;
            bytes += rnn_w_dyn.size_bytes + rnn_w_h_host.byte_size() + rnn_w_frnn_host.byte_size();

            // fold: x[r] = hidden_out[:, r] (r < R), x[R] = hidden_out bias; y = U x -> [R + 1][V].
            // Quantize y[0..R] per-r to padded FP8 E4M3FN [R][V_pad] + [R] f32 scales (-51 MiB VRAM, 2x less BW).
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
            let fold_fp8 = backend.alloc_device(R * vocab_pad)?;
            let fold_scales = backend.alloc_device(R * 4)?;
            backend.dspark_quant_fold_fp8(fold_fp8, fold_scales, y, R, vocab, vocab_pad, stream)?;
            let fold_b = backend.alloc_device(vocab * 4)?;
            backend.vec_copy(fold_b, y.slice(R * vocab * 4, vocab * 4), vocab, stream)?;
            backend.sync_stream(stream)?;
            backend.free_device(y)?;
            backend.free_device(xd)?;
            bytes += fold_fp8.size_bytes + fold_scales.size_bytes + fold_b.size_bytes;

            let a = |n: usize| backend.alloc_device(n * 4);
            let dm = a(2 * BLOCK)?;
            let sc = Scratch {
                anchor_in: a(D)?,
                res: a(BLOCK * D)?,
                xn: a(BLOCK * D)?,
                tmp: a(BLOCK * D)?,
                q: a(BLOCK * HQ * HD)?,
                kb: a(BLOCK * HK * HD)?,
                vb: a(BLOCK * HK * HD)?,
                att: a(BLOCK * HQ * HD)?,
                part_o: a(BLOCK * HQ * MAX_ATTN_SPLITS * HD)?,
                part_m: a(BLOCK * HQ * MAX_ATTN_SPLITS)?,
                part_l: a(BLOCK * HQ * MAX_ATTN_SPLITS)?,
                g: a(BLOCK * FF)?,
                u: a(BLOCK * FF)?,
                ctx_c: a(chunk * D)?,
                ctx_k: a(chunk * HK * HD)?,
                ctx_v: a(chunk * HK * HD)?,
                base: a((BLOCK - 1) * vocab)?,
                frnn: a(D)?,
                frnn_proj: a(3 * R)?,
                hb_proj: a((BLOCK - 1) * 3 * R)?,
                state: a(R)?,
                latent: a(R)?,
                drafts: dm.slice(0, BLOCK * 4),
                margins: dm.slice(BLOCK * 4, BLOCK * 4),
                pmax: a(vocab.div_ceil(1024))?,
                pidx: a(vocab.div_ceil(1024))?,
                pmax2: a(vocab.div_ceil(1024))?,
                pinned: backend.alloc_pinned(D * 4)?,
            };
            bytes += ((BLOCK - 1) * vocab + chunk * (D + 2 * HK * HD) + BLOCK * HQ * MAX_ATTN_SPLITS * (HD + 2)) * 4;
            Ok(Self {
                backend: backend.clone(),
                layers,
                norm_w,
                mask_embed_gpu,
                alpha,
                mk_norm,
                w1,
                w1_host,
                rnn_w_dyn,
                rnn_w_h,
                rnn_w_frnn,
                rnn_b,
                fold_fp8,
                fold_scales,
                fold_b,
                vocab,
                vocab_pad,
                max_seq,
                chunk,
                sc,
                vram_bytes: bytes,
                lane: 0,
                page_table: None,
            })
        })();
        let _ = backend.sync_stream(stream);
        let _ = backend.free_pinned(staging);
        let _ = backend.destroy_stream(stream);
        res
    }

    /// Makes `lane` the sequence `ingest` writes and `draft` reads: its local rings and, when the
    /// global layer's cache is the shared KV pool, its page table (`None`: position = slot).
    pub fn set_lane(&mut self, lane: usize, page_table: Option<DeviceBuf>) {
        self.page_table = page_table;
        self.lane = lane;
        for ly in self.layers.iter_mut() {
            if let Some(&[kc, ks, vc, vs]) = ly.lane_ctx.get(lane) {
                (ly.kc, ly.ks, ly.vc, ly.vs) = (kc, ks, vc, vs);
            }
        }
    }

    /// Lanes that keep their own context.
    pub fn lanes(&self) -> usize {
        self.layers.iter().map(|ly| ly.lane_ctx.len()).max().unwrap_or(1).max(1)
    }

    /// Bytes per position in the drafter's global KV cache (for elastic KV sizing).
    pub fn global_kv_bytes_per_pos(&self) -> usize {
        self.layers
            .iter()
            .filter(|ly| !ly.local)
            .map(|ly| HK * (HD + 4 + if ly.v_q4 { HD / 2 + 16 } else { HD + 4 }))
            .sum()
    }

    /// Resizes the drafter's global-attention KV cache to `new_slots` positions, preserving
    /// existing pages `0..min(old, new)` via D2D copy on `s`.
    pub fn resize_global_slots(&mut self, new_slots: usize, s: Stream) -> Result<()> {
        let b = &self.backend;
        for ly in self.layers.iter_mut() {
            if ly.local {
                continue;
            }
            let kc_bytes = new_slots * HK * HD;
            let ks_bytes = new_slots * HK * 4;
            let vc_bytes = new_slots * HK * if ly.v_q4 { HD / 2 } else { HD };
            let vs_bytes = new_slots * HK * if ly.v_q4 { 16 } else { 4 };
            if ly.kc.size_bytes == kc_bytes {
                continue;
            }
            let old = [ly.kc, ly.ks, ly.vc, ly.vs];
            let old_bytes: usize = old.iter().map(|buf| buf.size_bytes).sum();
            let new = [
                b.alloc_device(kc_bytes)?,
                b.alloc_device(ks_bytes)?,
                b.alloc_device(vc_bytes)?,
                b.alloc_device(vs_bytes)?,
            ];
            for (&o, &n) in old.iter().zip(new.iter()) {
                let cp = o.size_bytes.min(n.size_bytes);
                if cp > 0 {
                    b.memcpy_d2d_async(n, 0, o, 0, cp, s)?;
                }
            }
            b.sync_stream(s)?;
            for &o in &old {
                b.free_device(o)?;
            }
            let new_bytes: usize = new.iter().map(|buf| buf.size_bytes).sum();
            self.vram_bytes = self.vram_bytes.saturating_sub(old_bytes) + new_bytes;
            (ly.kc, ly.ks, ly.vc, ly.vs) = (new[0], new[1], new[2], new[3]);
        }
        Ok(())
    }

    /// Context K/V of `m` committed rows at positions `pos0..` from their taps
    /// (`taps` = `[4][tap_stride][D]`, starting at row `row_off`).
    pub fn ingest(&self, taps: DeviceBuf, tap_stride: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        self.ingest_at_row(taps, tap_stride, 0, pos0, m, s)
    }

    pub fn ingest_at_row(&self, taps: DeviceBuf, tap_stride: usize, row_off: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if m > self.chunk || pos0 + m > self.max_seq {
            return Err(SmallTideError::Backend { backend: "drafter", message: format!("ingest {m} rows at {pos0}") });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;
        let kvd = HK * HD;
        let toff = row_off * D * 4;
        let taps_shifted = if toff > 0 { taps.slice(toff, taps.size_bytes - toff) } else { taps };
        for (l, ly) in self.layers.iter().enumerate() {
            b.dspark_mix_rms(sc.ctx_c, taps_shifted, tap_stride, self.alpha[l], ly.dflare, m, D, EPS, s)?;
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
            let pages = if ly.local { None } else { self.page_table };
            b.dspark_ctx_store_fp8(ly.kc, ly.ks, ly.vc, ly.vs, sc.ctx_k, sc.ctx_v, pos0, m, if ly.local { RING } else { 0 }, HK, ly.v_q4, pages, s)?;
        }
        Ok(())
    }

    /// Drafts `n` (<= 7) tokens after `anchor` at position `p`; `tap_row` is the row of position
    /// `p - 1` in `taps`. Context of positions `< p` must have been ingested.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(&mut self, dense: &Inkling2DenseGpu<B>, anchor: u32, p: usize, taps: DeviceBuf, tap_stride: usize, tap_row: usize, n: usize, s: Stream) -> Result<Vec<u32>> {
        Ok(self.draft_with_margins(dense, anchor, p, taps, tap_stride, tap_row, n, s)?.into_iter().map(|(t, _)| t).collect())
    }

    /// Like `draft`, returning `(token_id, top1_minus_top2_logit_margin)` per drafted token in one
    /// combined 64-byte readback so the speculative loop can trim low-confidence tail drafts.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_with_margins(&mut self, dense: &Inkling2DenseGpu<B>, anchor: u32, p: usize, taps: DeviceBuf, tap_stride: usize, tap_row: usize, n: usize, s: Stream) -> Result<Vec<(u32, f32)>> {
        let n = n.clamp(1, BLOCK - 1);
        if p + 1 + n > self.max_seq {
            return Err(SmallTideError::Backend { backend: "drafter", message: format!("draft {n} at {p} past max_seq {}", self.max_seq) });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;
        // Upload only the single anchor embedding row (9 KB); mask_embed lives on GPU.
        unsafe {
            let dst = b.pinned_as_mut_slice(sc.pinned)?;
            let row = &dense.embed_bf16_host[anchor as usize * D * 2..(anchor as usize + 1) * D * 2];
            for i in 0..D {
                let v = bf16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
                dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        b.memcpy_h2d_async(sc.anchor_in, 0, sc.pinned, 0, D * 4, s)?;
        b.dspark_embed_prep(sc.res, sc.xn, sc.anchor_in, dense.embed_norm, self.mask_embed_gpu, self.layers[0].in_ln, D, EPS, s)?;

        let seg = |m: &Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| GemvSegment { w: m.data, row_scales: Some(m.row_scales), x, y, n_rows: m.n_rows, k_cols: m.k_cols };
        let scale = 1.0 / (HD as f32).sqrt();
        for (l, ly) in self.layers.iter().enumerate() {
            b.gemv_multi_rows(&[seg(&ly.q, sc.xn, sc.q), seg(&ly.k, sc.xn, sc.kb), seg(&ly.v, sc.xn, sc.vb)], BLOCK, s)?;
            b.dspark_qk_norm_rope(sc.q, sc.kb, ly.q_norm, ly.k_norm, HQ, HK, BLOCK, p, THETA, EPS, s)?;
            let (lo, ring) = if ly.local { (p.saturating_sub(WINDOW), RING) } else { (0, 0) };
            b.dspark_attn_fp8(
                sc.att, sc.part_o, sc.part_m, sc.part_l,
                sc.q, sc.kb, sc.vb, ly.kc, ly.ks, ly.vc, ly.vs,
                BLOCK, p, lo, ring, HQ, HK, BLOCK, scale, ly.v_q4, if ly.local { None } else { self.page_table }, s,
            )?;
            b.gemv_multi_rows(&[seg(&ly.o, sc.att, sc.tmp)], BLOCK, s)?;
            b.add_rmsnorm_rows(sc.res, sc.xn, sc.tmp, ly.post_ln, BLOCK, D, EPS, s)?;
            b.gemv_multi_rows(&[seg(&ly.gate, sc.xn, sc.g), seg(&ly.up, sc.xn, sc.u)], BLOCK, s)?;
            b.silu_mul(sc.g, sc.g, sc.u, BLOCK * FF, s)?;
            b.gemv_multi_rows(&[seg(&ly.down, sc.g, sc.tmp)], BLOCK, s)?;
            let next = if l + 1 < self.layers.len() { self.layers[l + 1].in_ln } else { self.norm_w };
            b.add_rmsnorm_rows(sc.res, sc.xn, sc.tmp, next, BLOCK, D, EPS, s)?;
        }

        // Markov head: base logits of block rows 1..=n, plus precomputed loop-invariant RNN projections.
        let hb = sc.xn.slice(D * 4, (BLOCK - 1) * D * 4);
        let u = &dense.unembed_fp8;
        b.gemm_fp8_rows_decode(sc.base, hb, u.data, u.row_scales, n, self.vocab, D, s)?;
        let toff = tap_row * D * 4;
        b.dspark_mix_rms(sc.frnn, taps.slice(toff, taps.size_bytes - toff), tap_stride, self.alpha[3], self.mk_norm, 1, D, EPS, s)?;
        b.gemv_fp8(sc.frnn_proj, sc.frnn, self.rnn_w_frnn.data, self.rnn_w_frnn.row_scales, 3 * R, D, s)?;
        b.gemm_fp8_rows_decode(sc.hb_proj, hb, self.rnn_w_h.data, self.rnn_w_h.row_scales, n, 3 * R, D, s)?;
        b.zero_f32(sc.state, R, s)?;
        for i in 0..n {
            b.dspark_markov_step_fast(
                sc.state, sc.latent, self.w1, sc.drafts, sc.margins, i, anchor,
                self.rnn_w_dyn, sc.frnn_proj, sc.hb_proj.slice(i * 3 * R * 4, 3 * R * 4), self.rnn_b,
                sc.base.slice(i * self.vocab * 4, self.vocab * 4), self.fold_b,
                self.fold_fp8, self.fold_scales, sc.pmax, sc.pidx, sc.pmax2,
                R, self.vocab, self.vocab_pad, s,
            )?;
        }
        // drafts[0..BLOCK] and margins[0..BLOCK] are contiguous in one 64-byte allocation.
        let mut out = [0u8; 2 * BLOCK * 4];
        let full = DeviceBuf { id: sc.drafts.id, offset_bytes: 0, size_bytes: 2 * BLOCK * 4 };
        b.memcpy_d2h_sync(&mut out, full, 0, s)?;
        Ok((0..n)
            .map(|i| {
                let id = u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
                let mg = f32::from_le_bytes(out[(BLOCK + i) * 4..(BLOCK + i) * 4 + 4].try_into().unwrap());
                (id, mg)
            })
            .collect())
    }

    /// Fixed-size local context KV rings + scales (layers 0 & 2, ~4.2 MiB total) for prefix snapshots.
    pub fn local_kv_regions(&self) -> Vec<DeviceBuf> {
        let mut v = Vec::new();
        for ly in &self.layers {
            if ly.local {
                v.extend_from_slice(&[ly.kc, ly.ks, ly.vc, ly.vs]);
            }
        }
        v
    }

    /// Global context KV + scales (layer 1): `(buf, bytes_per_pos)` for prefix snapshots (~2.1 KB/pos).
    pub fn global_kv_regions(&self) -> Vec<(DeviceBuf, usize)> {
        let mut v = Vec::new();
        for ly in &self.layers {
            if !ly.local {
                v.push((ly.kc, HK * HD));
                v.push((ly.ks, HK * 4));
                v.push((ly.vc, HK * if ly.v_q4 { HD / 2 } else { HD }));
                v.push((ly.vs, HK * if ly.v_q4 { 16 } else { 4 }));
            }
        }
        v
    }
}

impl<B: ComputeBackend> Drop for DSparkDrafter<B> {
    fn drop(&mut self) {
        let b = self.backend.as_ref();
        for ly in &self.layers {
            for m in [&ly.q, &ly.k, &ly.v, &ly.k_ctx, &ly.v_ctx, &ly.o, &ly.gate, &ly.up, &ly.down] {
                let _ = b.free_device(m.data);
                let _ = b.free_device(m.row_scales);
            }
            for buf in [ly.in_ln, ly.post_ln, ly.dflare, ly.q_norm, ly.k_norm, ly.k_norm_ctx] {
                let _ = b.free_device(buf);
            }
            if ly.lane_ctx.is_empty() {
                for buf in [ly.kc, ly.ks, ly.vc, ly.vs] {
                    let _ = b.free_device(buf);
                }
            }
            for bufs in &ly.lane_ctx {
                for &buf in bufs {
                    let _ = b.free_device(buf);
                }
            }
        }
        for m in [&self.rnn_w_h, &self.rnn_w_frnn] {
            let _ = b.free_device(m.data);
            let _ = b.free_device(m.row_scales);
        }
        for buf in [
            self.norm_w, self.mask_embed_gpu, self.mk_norm, self.w1, self.rnn_w_dyn, self.rnn_b,
            self.fold_fp8, self.fold_scales, self.fold_b,
        ] {
            let _ = b.free_device(buf);
        }
        let _ = b.free_pinned(self.w1_host);
        let sc = &self.sc;
        for buf in [
            sc.anchor_in, sc.res, sc.xn, sc.tmp, sc.q, sc.kb, sc.vb, sc.att,
            sc.part_o, sc.part_m, sc.part_l, sc.g, sc.u, sc.ctx_c, sc.ctx_k, sc.ctx_v,
            sc.base, sc.frnn, sc.frnn_proj, sc.hb_proj, sc.state, sc.latent, sc.drafts, sc.pmax, sc.pidx, sc.pmax2,
        ] {
            let _ = b.free_device(buf);
        }
        let _ = b.free_pinned(sc.pinned);
    }
}

/// One speculative draft token proposal with confidence margin and optional sparse draft
/// probability distribution `q` (for lossless stochastic rejection sampling at `temperature > 0`).
#[derive(Clone, Debug)]
pub struct DraftProposal {
    pub token: u32,
    pub margin: f32,
    /// When `Some`, `(candidate_id, q_prob)` over the top-K candidates drawn by the drafter;
    /// when `None`, deterministic proposal `q(token) = 1`.
    pub q_dist: Option<Vec<(u32, f32)>>,
}

/// Unified speculative drafter: either Inkling-2-Mini's native DSpark (`mtp.safetensors`) or
/// Inco AI's DFlash 2 (`incoai/Inkling-2-Mini-DFlash2` `model.safetensors`).
pub enum Drafter<B: ComputeBackend> {
    DSpark(DSparkDrafter<B>),
    DFlash2(crate::dflash2::DFlash2Drafter<B>),
}

impl<B: ComputeBackend> Drafter<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        backend: Arc<B>,
        path: &Path,
        dense: &Inkling2DenseGpu<B>,
        vocab: usize,
        max_seq: usize,
        chunk: usize,
        global_v_q4: bool,
        lanes: usize,
        global_slots: usize,
    ) -> Result<Self> {
        Ok(Self::DSpark(DSparkDrafter::load(
            backend, path, dense, vocab, max_seq, chunk, global_v_q4, lanes, global_slots,
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_dflash2(
        backend: Arc<B>,
        path: &Path,
        dense: &Inkling2DenseGpu<B>,
        vocab: usize,
        max_seq: usize,
        chunk: usize,
        lanes: usize,
        paged: bool,
    ) -> Result<Self> {
        Ok(Self::DFlash2(crate::dflash2::DFlash2Drafter::load(
            backend, path, dense, vocab, max_seq, chunk, lanes, paged,
        )?))
    }

    pub fn is_dflash2_paged(&self) -> bool {
        matches!(self, Self::DFlash2(d) if d.is_paged())
    }

    pub fn bind_dflash2_kv_pool(
        &mut self,
        k: DeviceBuf,
        ks: DeviceBuf,
        v: DeviceBuf,
        vs: DeviceBuf,
        v_q4: bool,
        ring_slots: usize,
    ) {
        if let Self::DFlash2(d) = self {
            d.bind_kv_pool(k, ks, v, vs, v_q4, ring_slots);
        }
    }

    pub fn set_lane_unified(
        &mut self,
        lane: usize,
        drafter_tables: Option<&[DeviceBuf]>,
        dspark_global_table: Option<DeviceBuf>,
    ) {
        match self {
            Self::DSpark(d) => d.set_lane(lane, dspark_global_table),
            Self::DFlash2(d) => d.set_lane(lane, drafter_tables),
        }
    }

    pub fn vram_bytes(&self) -> usize {
        match self {
            Self::DSpark(d) => d.vram_bytes,
            Self::DFlash2(d) => d.vram_bytes,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::DSpark(_) => "dspark",
            Self::DFlash2(_) => "dflash2",
        }
    }

    pub fn set_lane(&mut self, lane: usize, page_table: Option<DeviceBuf>) {
        match self {
            Self::DSpark(d) => d.set_lane(lane, page_table),
            Self::DFlash2(d) => d.set_lane(lane, None),
        }
    }

    pub fn lanes(&self) -> usize {
        match self {
            Self::DSpark(d) => d.lanes(),
            Self::DFlash2(d) => d.lanes(),
        }
    }

    pub fn global_kv_bytes_per_pos(&self) -> usize {
        match self {
            Self::DSpark(d) => d.global_kv_bytes_per_pos(),
            Self::DFlash2(_) => 0,
        }
    }

    pub fn resize_global_slots(&mut self, new_slots: usize, s: Stream) -> Result<()> {
        match self {
            Self::DSpark(d) => d.resize_global_slots(new_slots, s),
            Self::DFlash2(_) => Ok(()),
        }
    }

    pub fn ingest(&self, taps: DeviceBuf, tap_stride: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        self.ingest_at_row(taps, tap_stride, 0, pos0, m, s)
    }

    pub fn ingest_at_row(&self, taps: DeviceBuf, tap_stride: usize, row_off: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        match self {
            Self::DSpark(d) => d.ingest_at_row(taps, tap_stride, row_off, pos0, m, s),
            Self::DFlash2(d) => d.ingest_at_row(taps, tap_stride, row_off, pos0, m, s),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        anchor: u32,
        p: usize,
        taps: DeviceBuf,
        tap_stride: usize,
        tap_row: usize,
        n: usize,
        s: Stream,
    ) -> Result<Vec<u32>> {
        Ok(self.draft_with_margins(dense, anchor, p, taps, tap_stride, tap_row, n, s)?.into_iter().map(|(t, _)| t).collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draft_with_margins(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        anchor: u32,
        p: usize,
        taps: DeviceBuf,
        tap_stride: usize,
        tap_row: usize,
        n: usize,
        s: Stream,
    ) -> Result<Vec<(u32, f32)>> {
        match self {
            Self::DSpark(d) => d.draft_with_margins(dense, anchor, p, taps, tap_stride, tap_row, n, s),
            Self::DFlash2(d) => d.draft_with_margins(dense, anchor, p, taps, tap_stride, tap_row, n, s),
        }
    }

    /// Drafts `n` tokens with optional stochastic sampling (`temperature > 0` for DFlash 2) and
    /// sparse proposal distributions `q` for lossless speculative rejection sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_proposals(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        anchor: u32,
        p: usize,
        taps: DeviceBuf,
        tap_stride: usize,
        tap_row: usize,
        n: usize,
        temperature: f32,
        rng: &mut u64,
        s: Stream,
    ) -> Result<Vec<DraftProposal>> {
        match self {
            Self::DSpark(d) => Ok(d
                .draft_with_margins(dense, anchor, p, taps, tap_stride, tap_row, n, s)?
                .into_iter()
                .map(|(token, margin)| DraftProposal { token, margin, q_dist: None })
                .collect()),
            Self::DFlash2(d) => d.draft_proposals(dense, anchor, p, n, temperature, rng, s),
        }
    }

    /// Batched multi-lane drafting (DFlash 2): reads DFlash 2 weights & unembed ONCE across all
    /// concurrent lanes using caller-provided scratch (`pf.blocks` + `batch_logits`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_proposals_batch(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        reqs: &[crate::dflash2::DraftLaneReq],
        lane_tables: &[Option<Vec<DeviceBuf>>],
        rngs: &mut [u64],
        scratch_pool: DeviceBuf,
        logits_buf: DeviceBuf,
        s: Stream,
    ) -> Result<Vec<Vec<DraftProposal>>> {
        match self {
            Self::DFlash2(d) => {
                d.draft_proposals_batch(dense, reqs, lane_tables, rngs, scratch_pool, logits_buf, s)
            }
            Self::DSpark(_) => Err(SmallTideError::Backend {
                backend: "drafter",
                message: "draft_proposals_batch is only supported for DFlash 2".into(),
            }),
        }
    }

    pub fn local_kv_regions(&self) -> Vec<DeviceBuf> {
        match self {
            Self::DSpark(d) => d.local_kv_regions(),
            Self::DFlash2(d) => d.local_kv_regions(),
        }
    }

    pub fn global_kv_regions(&self) -> Vec<(DeviceBuf, usize)> {
        match self {
            Self::DSpark(d) => d.global_kv_regions(),
            Self::DFlash2(_) => Vec::new(),
        }
    }
}
