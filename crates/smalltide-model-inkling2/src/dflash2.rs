//! DFlash 2 block-diffusion drafter for Inkling-2-Mini (`incoai/Inkling-2-Mini-DFlash2`):
//! a 5-layer sliding-window (2048) non-causal block-diffusion drafter over the target's 4 taps
//! (logical layers 4 / 22 / 40 post-residual + final normed hidden, matching `target_layer_ids = [5, 23, 41, 48]`),
//! with 2-tap grouped dynamic causal convolutions around attention and MLP and a rank-256
//! pairwise candidate path selector over the target `lm_head`'s top-16 candidates.
//!
//! Memory layout optimized for 16 GB GPUs:
//! - Backbone (`fc` + 5 layers) quantized per-row to FP8 E4M3FN (~374 MiB VRAM)
//! - All 5 layers use fixed 2064-slot FP8 sliding-window rings (~21.3 MiB/lane; 0 global KV)
//! - Both 103 MiB BF16 selector codebooks (`predecessor_codebook`, `successor_codebook`) stay in
//!   ordinary host RAM (0 VRAM, 0 pinned RAM): the GPU extracts exact Top-16 `(val, id)` and
//!   `hidden_projection(hb)` in one ~8 KB D2H copy, and the CPU runs the 7x16x256 path selector
//!   in < 0.05 ms.

use crate::drafter::{quantize_f32_rows_to_fp8, DraftProposal};
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
const NUM_LAYERS: usize = 5;
const D: usize = 2304;
const HQ: usize = 32;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 6912;
const CONV_G: usize = 16;
const CONV_GROUPS: usize = D / CONV_G; // 144
const CONV_PROJ_OUT: usize = 2 * 2 * CONV_GROUPS; // 576
const WINDOW: usize = 2048;
/// Sliding-window context ring: window plus a block of slack.
const RING: usize = WINDOW + 16;
const MAX_ATTN_SPLITS: usize = 16;
const THETA: f32 = 10_000.0;
const EPS: f32 = 1e-6;
const MASK_TOKEN_ID: usize = 200_056;
const R: usize = 256;
const TOP_K: usize = 16;
const SOFTCAP: f32 = 30.0;

fn next_rand_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 40) as f32) * (1.0 / ((1u64 << 24) as f32))
}

struct DFlash2Layer {
    in_ln: DeviceBuf,
    post_ln: DeviceBuf,
    attn_conv_base: DeviceBuf, // [2, 2, D] f32
    attn_conv_proj: Fp8MatrixGpu, // [576, D]
    mlp_conv_base: DeviceBuf,  // [2, 2, D] f32
    mlp_conv_proj: Fp8MatrixGpu,  // [576, D]
    q: Fp8MatrixGpu,
    k: Fp8MatrixGpu,
    v: Fp8MatrixGpu,
    o: Fp8MatrixGpu,
    gate: Fp8MatrixGpu,
    up: Fp8MatrixGpu,
    down: Fp8MatrixGpu,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    /// Active lane's sliding-window FP8 context K/V ring `[RING][HK * HD]` + scales `[RING][HK]`.
    kc: DeviceBuf,
    ks: DeviceBuf,
    vc: DeviceBuf,
    vs: DeviceBuf,
    /// Per-lane sliding-window rings.
    lane_ctx: Vec<[DeviceBuf; 4]>,
}

struct Scratch {
    anchor_in: DeviceBuf,   // [D]
    res: DeviceBuf,         // [8][D]
    xn: DeviceBuf,          // [8][D]
    xn_conv: DeviceBuf,     // [8][D]
    tmp: DeviceBuf,         // [8][D]
    conv_coeff: DeviceBuf,  // [8][576]
    q: DeviceBuf,           // [8][HQ*HD]
    kb: DeviceBuf,          // [8][HK*HD]
    vb: DeviceBuf,          // [8][HK*HD]
    att: DeviceBuf,         // [8][HQ*HD]
    part_o: DeviceBuf,      // [8][HQ][MAX_ATTN_SPLITS][HD]
    part_m: DeviceBuf,      // [8][HQ][MAX_ATTN_SPLITS]
    part_l: DeviceBuf,      // [8][HQ][MAX_ATTN_SPLITS]
    g: DeviceBuf,           // [8][FF]
    u: DeviceBuf,           // [8][FF]
    ctx_c: DeviceBuf,       // [chunk][D]
    ctx_tmp: DeviceBuf,     // [chunk][D]
    ctx_k: DeviceBuf,       // [chunk][HK*HD]
    ctx_v: DeviceBuf,       // [chunk][HK*HD]
    base: DeviceBuf,        // [7][vocab]
    part_val: DeviceBuf,    // [7][num_blocks * 16]
    part_idx: DeviceBuf,    // [7][num_blocks * 16]
    /// Contiguous D2H output block: `[topk_val (7*16) | topk_idx (7*16) | h_proj (7*256)]`.
    sel_out: DeviceBuf,
    topk_val: DeviceBuf,
    topk_idx: DeviceBuf,
    h_proj: DeviceBuf,
    pinned: PinnedBuf,      // [D] f32 (single anchor row)
}

/// One lane's request inside a batched multi-lane DFlash 2 draft pass (`draft_proposals_batch`).
#[derive(Clone, Copy, Debug)]
pub struct DraftLaneReq {
    pub lane: usize,
    pub anchor: u32,
    pub pos: usize,
    pub n_draft: usize,
    pub temperature: f32,
}

pub struct DFlash2Drafter<B: ComputeBackend> {
    backend: Arc<B>,
    /// `fc.weight [D, 4*D]` split into 4 FP8 `[D, D]` projections (one per target tap).
    fc: [Fp8MatrixGpu; 4],
    hidden_norm: DeviceBuf,
    layers: Vec<DFlash2Layer>,
    norm_w: DeviceBuf,
    mask_embed_gpu: DeviceBuf,
    sel_hidden_proj: Fp8MatrixGpu, // [R, D]
    /// Host RAM BF16 codebooks (`[201024, R]` raw little-endian BF16 bytes, 0 VRAM).
    pred_cb: Vec<u8>,
    succ_cb: Vec<u8>,
    vocab: usize,
    max_seq: usize,
    chunk: usize,
    sc: Scratch,
    pub vram_bytes: usize,
    lane: usize,
    /// When true, KV lives in the engine's unified `KvPoolBufs` (0 private KV VRAM).
    paged: bool,
    v_q4: bool,
    ring_slots: usize,
    page_tables: Option<Vec<DeviceBuf>>,
    lanes_count: usize,
}

impl<B: ComputeBackend> DFlash2Drafter<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        backend: Arc<B>,
        path: &Path,
        dense: &Inkling2DenseGpu<B>,
        vocab: usize,
        max_seq: usize,
        chunk: usize,
        lanes: usize,
        paged: bool,
    ) -> Result<Self> {
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

            // Split fc.weight [D, 4*D] into 4 [D, D] per-tap FP8 matrices.
            let fc_full = read_f32_vec(&mut f, &hdr, base, "fc.weight")?;
            if fc_full.len() != D * 4 * D {
                return Err(SmallTideError::Backend {
                    backend: "dflash2",
                    message: format!("unexpected fc.weight len {} (expected {})", fc_full.len(), D * 4 * D),
                });
            }
            let mut fc_vec = Vec::with_capacity(4);
            for t in 0..4 {
                let mut block = vec![0.0f32; D * D];
                for r in 0..D {
                    let src = &fc_full[r * 4 * D + t * D..r * 4 * D + (t + 1) * D];
                    block[r * D..(r + 1) * D].copy_from_slice(src);
                }
                let q = quantize_f32_rows_to_fp8(&block, D, D);
                bytes += q.byte_size();
                fc_vec.push(up.upload_fp8(&q)?);
            }
            let fc: [Fp8MatrixGpu; 4] = fc_vec.try_into().ok().unwrap();
            let hidden_norm = up.upload_f32(&read_f32_vec(&mut f, &hdr, base, "hidden_norm.weight")?)?;
            let norm_w = up.upload_f32(&read_f32_vec(&mut f, &hdr, base, "norm.weight")?)?;

            let mut layers = Vec::with_capacity(NUM_LAYERS);
            for l in 0..NUM_LAYERS {
                let p = format!("layers.{l}");
                let q = fp8(&mut f, &format!("{p}.self_attn.q_proj.weight"), &mut bytes)?;
                let k = fp8(&mut f, &format!("{p}.self_attn.k_proj.weight"), &mut bytes)?;
                let v = fp8(&mut f, &format!("{p}.self_attn.v_proj.weight"), &mut bytes)?;
                let o = fp8(&mut f, &format!("{p}.self_attn.o_proj.weight"), &mut bytes)?;
                let gate = fp8(&mut f, &format!("{p}.mlp.gate_proj.weight"), &mut bytes)?;
                let upm = fp8(&mut f, &format!("{p}.mlp.up_proj.weight"), &mut bytes)?;
                let down = fp8(&mut f, &format!("{p}.mlp.down_proj.weight"), &mut bytes)?;
                let attn_conv_proj = fp8(&mut f, &format!("{p}.attention_conv.kernel_projection.weight"), &mut bytes)?;
                let mlp_conv_proj = fp8(&mut f, &format!("{p}.mlp_conv.kernel_projection.weight"), &mut bytes)?;

                let vec = |f: &mut File, name: &str| -> Result<DeviceBuf> {
                    up.upload_f32(&read_f32_vec(f, &hdr, base, &format!("{p}.{name}"))?)
                };
                let in_ln = vec(&mut f, "input_layernorm.weight")?;
                let post_ln = vec(&mut f, "post_attention_layernorm.weight")?;
                let q_norm = vec(&mut f, "self_attn.q_norm.weight")?;
                let k_norm = vec(&mut f, "self_attn.k_norm.weight")?;
                let attn_conv_base = vec(&mut f, "attention_conv.base_kernel")?;
                let mlp_conv_base = vec(&mut f, "mlp_conv.base_kernel")?;

                let mut lane_ctx: Vec<[DeviceBuf; 4]> = Vec::new();
                let z = DeviceBuf { id: 0, offset_bytes: 0, size_bytes: 0 };
                let [kc, ks, vc, vs] = if paged {
                    [z, z, z, z]
                } else {
                    for _ in 0..lanes.max(1) {
                        lane_ctx.push([
                            backend.alloc_device(RING * HK * HD)?,
                            backend.alloc_device(RING * HK * 4)?,
                            backend.alloc_device(RING * HK * HD)?,
                            backend.alloc_device(RING * HK * 4)?,
                        ]);
                        bytes += RING * HK * (2 * HD + 8);
                    }
                    lane_ctx[0]
                };
                layers.push(DFlash2Layer {
                    in_ln, post_ln, attn_conv_base, attn_conv_proj, mlp_conv_base, mlp_conv_proj,
                    q, k, v, o, gate, up: upm, down, q_norm, k_norm, kc, ks, vc, vs, lane_ctx,
                });
            }

            let sel_hidden_proj = fp8(&mut f, "candidate_selector.hidden_projection.weight", &mut bytes)?;
            let pred_cb = read_raw_bytes(&mut f, &hdr, base, "candidate_selector.predecessor_codebook")?;
            let succ_cb = read_raw_bytes(&mut f, &hdr, base, "candidate_selector.successor_codebook")?;

            // Precompute mask_embed_gpu = RMSNorm(target.embed_tokens[MASK_TOKEN_ID], dense.embed_norm, EPS).
            let mask_off = MASK_TOKEN_ID * D * 2;
            if mask_off + D * 2 > dense.embed_bf16_host.len() {
                return Err(SmallTideError::Backend {
                    backend: "dflash2",
                    message: format!("mask_token_id {MASK_TOKEN_ID} out of range for embed table"),
                });
            }
            let mut mask_f32 = vec![0.0f32; D];
            let mask_raw = &dense.embed_bf16_host[mask_off..mask_off + D * 2];
            for i in 0..D {
                mask_f32[i] = bf16_to_f32(u16::from_le_bytes([mask_raw[2 * i], mask_raw[2 * i + 1]]));
            }
            let mask_raw_gpu = up.upload_f32(&mask_f32)?;
            let mask_embed_gpu = backend.alloc_device(D * 4)?;
            backend.rms_norm(mask_embed_gpu, mask_raw_gpu, dense.embed_norm, 1, D, EPS, stream)?;
            backend.sync_stream(stream)?;
            backend.free_device(mask_raw_gpu)?;

            let a = |n: usize| backend.alloc_device(n * 4);
            let num_topk_blocks = vocab.div_ceil(4096).max(1);
            let topk_val_bytes = (BLOCK - 1) * TOP_K * 4;
            let topk_idx_bytes = (BLOCK - 1) * TOP_K * 4;
            let h_proj_bytes = (BLOCK - 1) * R * 4;
            let sel_out = backend.alloc_device(topk_val_bytes + topk_idx_bytes + h_proj_bytes)?;
            let topk_val = sel_out.slice(0, topk_val_bytes);
            let topk_idx = sel_out.slice(topk_val_bytes, topk_idx_bytes);
            let h_proj = sel_out.slice(topk_val_bytes + topk_idx_bytes, h_proj_bytes);

            let sc = Scratch {
                anchor_in: a(D)?,
                res: a(BLOCK * D)?,
                xn: a(BLOCK * D)?,
                xn_conv: a(BLOCK * D)?,
                tmp: a(BLOCK * D)?,
                conv_coeff: a(BLOCK * CONV_PROJ_OUT)?,
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
                ctx_tmp: a(chunk * D)?,
                ctx_k: a(chunk * HK * HD)?,
                ctx_v: a(chunk * HK * HD)?,
                base: a((BLOCK - 1) * vocab)?,
                part_val: a((BLOCK - 1) * num_topk_blocks * TOP_K)?,
                part_idx: a((BLOCK - 1) * num_topk_blocks * TOP_K)?,
                sel_out,
                topk_val,
                topk_idx,
                h_proj,
                pinned: backend.alloc_pinned(32 * D * 4)?,
            };
            bytes += ((BLOCK - 1) * vocab
                + chunk * (2 * D + 2 * HK * HD)
                + BLOCK * HQ * MAX_ATTN_SPLITS * (HD + 2))
                * 4;
            Ok(Self {
                backend: backend.clone(),
                fc,
                hidden_norm,
                layers,
                norm_w,
                mask_embed_gpu,
                sel_hidden_proj,
                pred_cb,
                succ_cb,
                vocab,
                max_seq,
                chunk,
                sc,
                vram_bytes: bytes,
                lane: 0,
                paged,
                v_q4: false,
                ring_slots: RING,
                page_tables: None,
                lanes_count: lanes.max(1),
            })
        })();
        let _ = backend.sync_stream(stream);
        let _ = backend.free_pinned(staging);
        let _ = backend.destroy_stream(stream);
        res
    }

    pub fn is_paged(&self) -> bool {
        self.paged
    }

    /// Binds all 5 DFlash 2 layers to the engine's unified `KvPoolBufs`.
    pub fn bind_kv_pool(
        &mut self,
        k: DeviceBuf,
        ks: DeviceBuf,
        v: DeviceBuf,
        vs: DeviceBuf,
        v_q4: bool,
        ring_slots: usize,
    ) {
        self.paged = true;
        self.v_q4 = v_q4;
        self.ring_slots = ring_slots;
        for ly in &mut self.layers {
            ly.kc = k;
            ly.ks = ks;
            ly.vc = v;
            ly.vs = vs;
        }
    }

    pub fn set_lane(&mut self, lane: usize, drafter_tables: Option<&[DeviceBuf]>) {
        self.lane = lane;
        if self.paged {
            self.page_tables = drafter_tables.map(|t| t.to_vec());
            return;
        }
        for ly in self.layers.iter_mut() {
            if let Some(&[kc, ks, vc, vs]) = ly.lane_ctx.get(lane) {
                (ly.kc, ly.ks, ly.vc, ly.vs) = (kc, ks, vc, vs);
            }
        }
    }

    pub fn lanes(&self) -> usize {
        if self.paged {
            self.lanes_count
        } else {
            self.layers.iter().map(|ly| ly.lane_ctx.len()).max().unwrap_or(1).max(1)
        }
    }

    pub fn ingest(&self, taps: DeviceBuf, tap_stride: usize, pos0: usize, m: usize, s: Stream) -> Result<()> {
        self.ingest_at_row(taps, tap_stride, 0, pos0, m, s)
    }

    /// Ingests `m` committed rows starting at `row_off` inside `taps` (`[4][tap_stride][D]`)
    /// into positions `pos0..pos0+m` of the active lane.
    pub fn ingest_at_row(
        &self,
        taps: DeviceBuf,
        tap_stride: usize,
        row_off: usize,
        pos0: usize,
        m: usize,
        s: Stream,
    ) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if m > self.chunk || pos0 + m > self.max_seq {
            return Err(SmallTideError::Backend {
                backend: "dflash2",
                message: format!("ingest {m} rows at {pos0}"),
            });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;
        let kvd = HK * HD;
        let tap_row_bytes = m * D * 4;
        // Project concatenated 4 taps through fc [D, 4*D] -> ctx_c [m, D], then hidden_norm.
        for t in 0..4 {
            let src = taps.slice((t * tap_stride + row_off) * D * 4, tap_row_bytes);
            let dst = if t == 0 { sc.ctx_c } else { sc.ctx_tmp };
            if m <= 8 {
                let seg = [GemvSegment {
                    w: self.fc[t].data,
                    row_scales: Some(self.fc[t].row_scales),
                    x: src,
                    y: dst,
                    n_rows: D,
                    k_cols: D,
                }];
                b.gemv_multi_rows(&seg, m, s)?;
            } else {
                b.gemm_fp8_rows(dst, src, self.fc[t].data, self.fc[t].row_scales, m, D, D, s)?;
            }
            if t > 0 {
                b.vec_add(sc.ctx_c, sc.ctx_c, sc.ctx_tmp, m * D, s)?;
            }
        }
        b.rms_norm(sc.ctx_c, sc.ctx_c, self.hidden_norm, m, D, EPS, s)?;

        for (l, ly) in self.layers.iter().enumerate() {
            if m <= 8 {
                let segs = [
                    GemvSegment { w: ly.k.data, row_scales: Some(ly.k.row_scales), x: sc.ctx_c, y: sc.ctx_k, n_rows: kvd, k_cols: D },
                    GemvSegment { w: ly.v.data, row_scales: Some(ly.v.row_scales), x: sc.ctx_c, y: sc.ctx_v, n_rows: kvd, k_cols: D },
                ];
                b.gemv_multi_rows(&segs, m, s)?;
            } else {
                b.gemm_fp8_rows(sc.ctx_k, sc.ctx_c, ly.k.data, ly.k.row_scales, m, kvd, D, s)?;
                b.gemm_fp8_rows(sc.ctx_v, sc.ctx_c, ly.v.data, ly.v.row_scales, m, kvd, D, s)?;
            }
            b.dspark_head_norm_rope(sc.ctx_k, ly.k_norm, HK, m, pos0, THETA, EPS, s)?;
            let pt = self.page_tables.as_ref().and_then(|v| v.get(l).copied());
            b.dspark_ctx_store_fp8(ly.kc, ly.ks, ly.vc, ly.vs, sc.ctx_k, sc.ctx_v, pos0, m, self.ring_slots, HK, self.v_q4, pt, s)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draft_with_margins(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        anchor: u32,
        p: usize,
        _taps: DeviceBuf,
        _tap_stride: usize,
        _tap_row: usize,
        n: usize,
        s: Stream,
    ) -> Result<Vec<(u32, f32)>> {
        let mut dummy_rng = 0u64;
        Ok(self
            .draft_proposals(dense, anchor, p, n, 0.0, &mut dummy_rng, s)?
            .into_iter()
            .map(|p| (p.token, p.margin))
            .collect())
    }

    /// Drafts `n` tokens after `anchor` at position `p`.
    /// When `temperature > 0.0`, applies `selector_softcap_mode = "after_correction"` (`30 * tanh(x/30)`)
    /// and samples from `q = softmax(scores / temperature)` over the top-16 candidates, returning
    /// the sparse `q_dist` for exact lossless speculative rejection sampling.
    pub fn draft_proposals(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        anchor: u32,
        p: usize,
        n: usize,
        temperature: f32,
        rng: &mut u64,
        s: Stream,
    ) -> Result<Vec<DraftProposal>> {
        let n = n.clamp(1, BLOCK - 1);
        if p + 1 + n > self.max_seq {
            return Err(SmallTideError::Backend {
                backend: "dflash2",
                message: format!("draft {n} at {p} past max_seq {}", self.max_seq),
            });
        }
        let b = self.backend.as_ref();
        let sc = &self.sc;

        // Upload single anchor embedding row (9 KB); mask_embed_gpu is pre-normed on GPU.
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

        let seg = |m: &Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| GemvSegment {
            w: m.data,
            row_scales: Some(m.row_scales),
            x,
            y,
            n_rows: m.n_rows,
            k_cols: m.k_cols,
        };
        let scale = 1.0 / (HD as f32).sqrt();

        for (l, ly) in self.layers.iter().enumerate() {
            // Attention sublayer wrapped by 2-tap grouped dynamic conv.
            b.gemv_multi_rows(&[seg(&ly.attn_conv_proj, sc.xn, sc.conv_coeff)], BLOCK, s)?;
            b.dflash2_conv_prepare(sc.xn_conv, sc.xn, sc.conv_coeff, ly.attn_conv_base, BLOCK, D, CONV_G, s)?;
            b.gemv_multi_rows(
                &[seg(&ly.q, sc.xn_conv, sc.q), seg(&ly.k, sc.xn_conv, sc.kb), seg(&ly.v, sc.xn_conv, sc.vb)],
                BLOCK,
                s,
            )?;
            b.dspark_qk_norm_rope(sc.q, sc.kb, ly.q_norm, ly.k_norm, HQ, HK, BLOCK, p, THETA, EPS, s)?;
            let pt = self.page_tables.as_ref().and_then(|v| v.get(l).copied());
            b.dflash2_attn_fp8(
                sc.att, sc.part_o, sc.part_m, sc.part_l,
                sc.q, sc.kb, sc.vb, ly.kc, ly.ks, ly.vc, ly.vs,
                BLOCK, p, WINDOW, self.ring_slots, HQ, HK, BLOCK, scale, self.v_q4, pt, s,
            )?;
            b.gemv_multi_rows(&[seg(&ly.o, sc.att, sc.tmp)], BLOCK, s)?;
            b.dflash2_conv_finish_add_rmsnorm(
                sc.res, sc.xn, sc.tmp, sc.conv_coeff, ly.attn_conv_base, ly.post_ln, BLOCK, D, CONV_G, EPS, s,
            )?;

            // MLP sublayer wrapped by 2-tap grouped dynamic conv.
            b.gemv_multi_rows(&[seg(&ly.mlp_conv_proj, sc.xn, sc.conv_coeff)], BLOCK, s)?;
            b.dflash2_conv_prepare(sc.xn_conv, sc.xn, sc.conv_coeff, ly.mlp_conv_base, BLOCK, D, CONV_G, s)?;
            b.gemv_multi_rows(&[seg(&ly.gate, sc.xn_conv, sc.g), seg(&ly.up, sc.xn_conv, sc.u)], BLOCK, s)?;
            b.silu_mul(sc.g, sc.g, sc.u, BLOCK * FF, s)?;
            b.gemv_multi_rows(&[seg(&ly.down, sc.g, sc.tmp)], BLOCK, s)?;
            let next = if l + 1 < self.layers.len() { self.layers[l + 1].in_ln } else { self.norm_w };
            b.dflash2_conv_finish_add_rmsnorm(
                sc.res, sc.xn, sc.tmp, sc.conv_coeff, ly.mlp_conv_base, next, BLOCK, D, CONV_G, EPS, s,
            )?;
        }

        // Unary logits on rows 1..=n via target unembed_fp8 (restricted to unpadded valid_vocab),
        // exact Top-16 extraction, and selector hidden projection -> 1 tiny ~8 KB D2H copy.
        let hb = sc.xn.slice(D * 4, n * D * 4);
        let u = &dense.unembed_fp8;
        b.gemm_fp8_rows_decode(sc.base, hb, u.data, u.row_scales, n, self.vocab, D, s)?;
        b.topk16_rows(sc.topk_val, sc.topk_idx, sc.part_val, sc.part_idx, sc.base, n, self.vocab, s)?;
        b.gemv_multi_rows(&[seg(&self.sel_hidden_proj, hb, sc.h_proj)], n, s)?;

        let mut raw = [0u8; ((BLOCK - 1) * TOP_K * 2 + (BLOCK - 1) * R) * 4];
        b.memcpy_d2h_sync(&mut raw, sc.sel_out, 0, s)?;

        let val_off = 0usize;
        let idx_off = (BLOCK - 1) * TOP_K * 4;
        let hp_off = idx_off + (BLOCK - 1) * TOP_K * 4;

        Ok(self.select_proposals(&raw, val_off, idx_off, hp_off, anchor, n, temperature, rng))
    }

    #[allow(clippy::too_many_arguments)]
    fn select_proposals(
        &self,
        raw: &[u8],
        val_off: usize,
        idx_off: usize,
        hp_off: usize,
        anchor: u32,
        n: usize,
        temperature: f32,
        rng: &mut u64,
    ) -> Vec<DraftProposal> {
        let rd_f32 = |byte_off: usize| -> f32 {
            f32::from_le_bytes(raw[byte_off..byte_off + 4].try_into().unwrap())
        };
        let rd_u32 = |byte_off: usize| -> u32 {
            u32::from_le_bytes(raw[byte_off..byte_off + 4].try_into().unwrap())
        };

        // Pairwise path selection on CPU (< 0.05 ms for 7 x 16 x 256).
        let mut out = Vec::with_capacity(n);
        let mut pred = anchor as usize;
        let mut q_vec = [0.0f32; R];
        let mut cids = [0u32; TOP_K];
        let mut scores = [0.0f32; TOP_K];
        for pos in 0..n {
            let pred_base = pred * R * 2;
            let hp_row = hp_off + pos * R * 4;
            for r in 0..R {
                let p_bits = u16::from_le_bytes([self.pred_cb[pred_base + 2 * r], self.pred_cb[pred_base + 2 * r + 1]]);
                q_vec[r] = bf16_to_f32(p_bits) * rd_f32(hp_row + r * 4);
            }
            let mut best_c = 0usize;
            let mut best1 = f32::NEG_INFINITY;
            let mut best2 = f32::NEG_INFINITY;
            for c in 0..TOP_K {
                let slot = pos * TOP_K + c;
                let cid = rd_u32(idx_off + slot * 4);
                let unary = rd_f32(val_off + slot * 4);
                let succ_base = cid as usize * R * 2;
                let mut acc0 = 0.0f32;
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut acc3 = 0.0f32;
                let mut r = 0usize;
                while r < R {
                    let s0 = bf16_to_f32(u16::from_le_bytes([self.succ_cb[succ_base + 2 * r], self.succ_cb[succ_base + 2 * r + 1]]));
                    let s1 = bf16_to_f32(u16::from_le_bytes([self.succ_cb[succ_base + 2 * (r + 1)], self.succ_cb[succ_base + 2 * (r + 1) + 1]]));
                    let s2 = bf16_to_f32(u16::from_le_bytes([self.succ_cb[succ_base + 2 * (r + 2)], self.succ_cb[succ_base + 2 * (r + 2) + 1]]));
                    let s3 = bf16_to_f32(u16::from_le_bytes([self.succ_cb[succ_base + 2 * (r + 3)], self.succ_cb[succ_base + 2 * (r + 3) + 1]]));
                    acc0 += q_vec[r] * s0;
                    acc1 += q_vec[r + 1] * s1;
                    acc2 += q_vec[r + 2] * s2;
                    acc3 += q_vec[r + 3] * s3;
                    r += 4;
                }
                let raw_score = unary + (acc0 + acc1) + (acc2 + acc3);
                cids[c] = cid;
                scores[c] = raw_score;
                if raw_score > best1 {
                    best2 = best1;
                    best1 = raw_score;
                    best_c = c;
                } else if raw_score > best2 {
                    best2 = raw_score;
                }
            }
            let margin = (best1 - best2).max(0.0);
            if temperature > 0.0 {
                // Apply selector_softcap_mode = "after_correction" (30 * tanh(x / 30)), then softmax / T.
                let inv_t = 1.0 / temperature;
                let mut capped = [0.0f32; TOP_K];
                let mut max_s = f32::NEG_INFINITY;
                for c in 0..TOP_K {
                    let cs = SOFTCAP * (scores[c] / SOFTCAP).tanh();
                    capped[c] = cs;
                    if cs > max_s {
                        max_s = cs;
                    }
                }
                let mut probs = [0.0f32; TOP_K];
                let mut sum = 0.0f32;
                for c in 0..TOP_K {
                    let e = ((capped[c] - max_s) * inv_t).exp();
                    probs[c] = e;
                    sum += e;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 / TOP_K as f32 };
                let mut q_dist = Vec::with_capacity(TOP_K);
                let target_u = next_rand_f32(rng);
                let mut acc = 0.0f32;
                let mut chosen_c = TOP_K - 1;
                let mut picked = false;
                for c in 0..TOP_K {
                    let p = probs[c] * inv_sum;
                    q_dist.push((cids[c], p));
                    acc += p;
                    if !picked && acc >= target_u {
                        chosen_c = c;
                        picked = true;
                    }
                }
                let chosen_id = cids[chosen_c];
                pred = chosen_id as usize;
                out.push(DraftProposal { token: chosen_id, margin, q_dist: Some(q_dist) });
            } else {
                let best_id = cids[best_c];
                pred = best_id as usize;
                out.push(DraftProposal { token: best_id, margin, q_dist: None });
            }
        }
        out
    }

    /// Batched multi-lane DFlash 2 drafting: runs `reqs` (`1..=8` lanes, `sum(n_draft) <= 16`)
    /// through all 5 DFlash 2 layers and the target `unembed_fp8` with ONE shared weight read
    /// across all concurrent lanes, using caller-provided scratch (`pf.blocks` + `batch_logits`)
    /// for 0 extra VRAM.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_proposals_batch(
        &mut self,
        dense: &Inkling2DenseGpu<B>,
        reqs: &[DraftLaneReq],
        lane_tables: &[Option<Vec<DeviceBuf>>],
        rngs: &mut [u64],
        scratch_pool: DeviceBuf,
        logits_buf: DeviceBuf,
        s: Stream,
    ) -> Result<Vec<Vec<DraftProposal>>> {
        let k = reqs.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if k == 1 {
            self.set_lane(reqs[0].lane, lane_tables[0].as_deref());
            let mut r = rngs[0];
            let res = self.draft_proposals(dense, reqs[0].anchor, reqs[0].pos, reqs[0].n_draft, reqs[0].temperature, &mut r, s)?;
            rngs[0] = r;
            return Ok(vec![res]);
        }
        if k > 32 {
            return Err(SmallTideError::Backend {
                backend: "dflash2",
                message: format!("draft_proposals_batch: {k} lanes > 32"),
            });
        }
        let mut total_n = 0usize;
        for req in reqs {
            let n = req.n_draft.clamp(1, BLOCK - 1);
            if req.pos + 1 + n > self.max_seq {
                return Err(SmallTideError::Backend {
                    backend: "dflash2",
                    message: format!("draft {n} at {} past max_seq {}", req.pos, self.max_seq),
                });
            }
            total_n += n;
        }
        let total_rows = k * BLOCK;
        let num_topk_blocks = self.vocab.div_ceil(4096).max(1);
        // Carve 256-byte-aligned scratch buffers out of `scratch_pool` (e.g. `pf.blocks`, 72 MiB).
        let mut cur_off = 0usize;
        let mut alloc = |bytes: usize| -> Result<DeviceBuf> {
            let aligned = bytes.next_multiple_of(256);
            if cur_off + aligned > scratch_pool.size_bytes {
                return Err(SmallTideError::Backend {
                    backend: "dflash2",
                    message: format!(
                        "draft_proposals_batch scratch overflow: need {} > {}",
                        cur_off + aligned,
                        scratch_pool.size_bytes
                    ),
                });
            }
            let buf = scratch_pool.slice(cur_off, bytes);
            cur_off += aligned;
            Ok(buf)
        };
        let anchor_in = alloc(k * D * 4)?;
        let res = alloc(total_rows * D * 4)?;
        let xn = alloc(total_rows * D * 4)?;
        let xn_conv = alloc(total_rows * D * 4)?;
        let tmp = alloc(total_rows * D * 4)?;
        let conv_coeff = alloc(total_rows * CONV_PROJ_OUT * 4)?;
        let q = alloc(total_rows * HQ * HD * 4)?;
        let kb = alloc(total_rows * HK * HD * 4)?;
        let vb = alloc(total_rows * HK * HD * 4)?;
        let att = alloc(total_rows * HQ * HD * 4)?;
        let g = alloc(total_rows * FF * 4)?;
        let u_buf = alloc(total_rows * FF * 4)?;
        let hb_pack = alloc(total_n * D * 4)?;
        let topk_val_bytes = total_n * TOP_K * 4;
        let topk_idx_bytes = total_n * TOP_K * 4;
        let h_proj_bytes = total_n * R * 4;
        let sel_out = alloc(topk_val_bytes + topk_idx_bytes + h_proj_bytes)?;
        let topk_val = sel_out.slice(0, topk_val_bytes);
        let topk_idx = sel_out.slice(topk_val_bytes, topk_idx_bytes);
        let h_proj = sel_out.slice(topk_val_bytes + topk_idx_bytes, h_proj_bytes);
        let part_val = alloc(total_n * num_topk_blocks * TOP_K * 4)?;
        let part_idx = alloc(total_n * num_topk_blocks * TOP_K * 4)?;
        let base = logits_buf.slice(0, total_n * self.vocab * 4);

        let b = self.backend.as_ref();
        // Upload all `k` anchor embeddings in ONE single H2D transfer.
        unsafe {
            let dst = b.pinned_as_mut_slice(self.sc.pinned)?;
            for (i, req) in reqs.iter().enumerate() {
                let row = &dense.embed_bf16_host[req.anchor as usize * D * 2..(req.anchor as usize + 1) * D * 2];
                let out_off = i * D * 4;
                for j in 0..D {
                    let v = bf16_to_f32(u16::from_le_bytes([row[2 * j], row[2 * j + 1]]));
                    dst[out_off + j * 4..out_off + j * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        b.memcpy_h2d_async(anchor_in, 0, self.sc.pinned, 0, k * D * 4, s)?;
        for i in 0..k {
            b.dspark_embed_prep(
                res.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                xn.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                anchor_in.slice(i * D * 4, D * 4),
                dense.embed_norm,
                self.mask_embed_gpu,
                self.layers[0].in_ln,
                D,
                EPS,
                s,
            )?;
        }

        let seg = |m: &Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| GemvSegment {
            w: m.data,
            row_scales: Some(m.row_scales),
            x,
            y,
            n_rows: m.n_rows,
            k_cols: m.k_cols,
        };
        let mm = |y: DeviceBuf, x: DeviceBuf, w: &Fp8MatrixGpu| -> Result<()> {
            if total_rows <= 64 {
                b.gemv_multi_rows(&[seg(w, x, y)], total_rows, s)
            } else {
                b.gemm_fp8_rows(y, x, w.data, w.row_scales, total_rows, w.n_rows, w.k_cols, s)
            }
        };
        let scale = 1.0 / (HD as f32).sqrt();

        for (l, ly) in self.layers.iter().enumerate() {
            // 1. Attention dynamic conv prepare (projection shared across all k*8 rows!)
            mm(conv_coeff, xn, &ly.attn_conv_proj)?;
            for i in 0..k {
                b.dflash2_conv_prepare(
                    xn_conv.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    xn.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    conv_coeff.slice(i * BLOCK * CONV_PROJ_OUT * 4, BLOCK * CONV_PROJ_OUT * 4),
                    ly.attn_conv_base,
                    BLOCK,
                    D,
                    CONV_G,
                    s,
                )?;
            }
            // 2. Q/K/V projections shared across all k*8 rows (1 weight read!)
            if total_rows <= 32 {
                b.gemv_multi_rows(&[seg(&ly.q, xn_conv, q), seg(&ly.k, xn_conv, kb), seg(&ly.v, xn_conv, vb)], total_rows, s)?;
            } else {
                b.gemm_fp8_rows(q, xn_conv, ly.q.data, ly.q.row_scales, total_rows, ly.q.n_rows, ly.q.k_cols, s)?;
                b.gemm_fp8_rows(kb, xn_conv, ly.k.data, ly.k.row_scales, total_rows, ly.k.n_rows, ly.k.k_cols, s)?;
                b.gemm_fp8_rows(vb, xn_conv, ly.v.data, ly.v.row_scales, total_rows, ly.v.n_rows, ly.v.k_cols, s)?;
            }
            // 3. Per-lane QK norm + RoPE + context attention (each lane has its own pos & page table!)
            for (i, req) in reqs.iter().enumerate() {
                let qi = q.slice(i * BLOCK * HQ * HD * 4, BLOCK * HQ * HD * 4);
                let ki = kb.slice(i * BLOCK * HK * HD * 4, BLOCK * HK * HD * 4);
                let vi = vb.slice(i * BLOCK * HK * HD * 4, BLOCK * HK * HD * 4);
                let atti = att.slice(i * BLOCK * HQ * HD * 4, BLOCK * HQ * HD * 4);
                b.dspark_qk_norm_rope(qi, ki, ly.q_norm, ly.k_norm, HQ, HK, BLOCK, req.pos, THETA, EPS, s)?;
                let (kc, ks, vc, vs, pt) = if self.paged {
                    let p = lane_tables[i].as_ref().and_then(|v| v.get(l).copied());
                    (ly.kc, ly.ks, ly.vc, ly.vs, p)
                } else {
                    let c = ly.lane_ctx[req.lane.min(ly.lane_ctx.len() - 1)];
                    (c[0], c[1], c[2], c[3], None)
                };
                b.dflash2_attn_fp8(
                    atti, self.sc.part_o, self.sc.part_m, self.sc.part_l,
                    qi, ki, vi, kc, ks, vc, vs,
                    BLOCK, req.pos, WINDOW, self.ring_slots, HQ, HK, BLOCK, scale, self.v_q4, pt, s,
                )?;
            }
            // 4. O projection shared across all k*8 rows + per-lane conv finish
            mm(tmp, att, &ly.o)?;
            for i in 0..k {
                b.dflash2_conv_finish_add_rmsnorm(
                    res.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    xn.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    tmp.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    conv_coeff.slice(i * BLOCK * CONV_PROJ_OUT * 4, BLOCK * CONV_PROJ_OUT * 4),
                    ly.attn_conv_base,
                    ly.post_ln,
                    BLOCK,
                    D,
                    CONV_G,
                    EPS,
                    s,
                )?;
            }
            // 5. MLP dynamic conv + gate/up/down shared across all k*8 rows (1 weight read!)
            mm(conv_coeff, xn, &ly.mlp_conv_proj)?;
            for i in 0..k {
                b.dflash2_conv_prepare(
                    xn_conv.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    xn.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    conv_coeff.slice(i * BLOCK * CONV_PROJ_OUT * 4, BLOCK * CONV_PROJ_OUT * 4),
                    ly.mlp_conv_base,
                    BLOCK,
                    D,
                    CONV_G,
                    s,
                )?;
            }
            if total_rows <= 32 {
                b.gemv_multi_rows(&[seg(&ly.gate, xn_conv, g), seg(&ly.up, xn_conv, u_buf)], total_rows, s)?;
            } else {
                b.gemm_fp8_rows(g, xn_conv, ly.gate.data, ly.gate.row_scales, total_rows, ly.gate.n_rows, ly.gate.k_cols, s)?;
                b.gemm_fp8_rows(u_buf, xn_conv, ly.up.data, ly.up.row_scales, total_rows, ly.up.n_rows, ly.up.k_cols, s)?;
            }
            b.silu_mul(g, g, u_buf, total_rows * FF, s)?;
            mm(tmp, g, &ly.down)?;
            let next = if l + 1 < self.layers.len() { self.layers[l + 1].in_ln } else { self.norm_w };
            for i in 0..k {
                b.dflash2_conv_finish_add_rmsnorm(
                    res.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    xn.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    tmp.slice(i * BLOCK * D * 4, BLOCK * D * 4),
                    conv_coeff.slice(i * BLOCK * CONV_PROJ_OUT * 4, BLOCK * CONV_PROJ_OUT * 4),
                    ly.mlp_conv_base,
                    next,
                    BLOCK,
                    D,
                    CONV_G,
                    EPS,
                    s,
                )?;
            }
        }

        // Pack each lane's requested `n_i` draft rows (`1..=n_i`) contiguously so `unembed_fp8`
        // (460 MiB!), `topk16_rows`, and `sel_hidden_proj` each run ONCE across all `total_n <= 16` rows!
        let mut pack_row = 0usize;
        let mut lane_offsets = Vec::with_capacity(k);
        for (i, req) in reqs.iter().enumerate() {
            let n = req.n_draft.clamp(1, BLOCK - 1);
            let src = xn.slice((i * BLOCK + 1) * D * 4, n * D * 4);
            let dst = hb_pack.slice(pack_row * D * 4, n * D * 4);
            b.vec_copy(dst, src, n * D, s)?;
            lane_offsets.push((pack_row, n));
            pack_row += n;
        }
        let u = &dense.unembed_fp8;
        b.gemm_fp8_rows_decode(base, hb_pack, u.data, u.row_scales, total_n, self.vocab, D, s)?;
        b.topk16_rows(topk_val, topk_idx, part_val, part_idx, base, total_n, self.vocab, s)?;
        b.gemv_multi_rows(&[seg(&self.sel_hidden_proj, hb_pack, h_proj)], total_n, s)?;

        let mut raw = vec![0u8; topk_val_bytes + topk_idx_bytes + h_proj_bytes];
        b.memcpy_d2h_sync(&mut raw, sel_out, 0, s)?;

        let mut out = Vec::with_capacity(k);
        for (i, req) in reqs.iter().enumerate() {
            let (r0, n) = lane_offsets[i];
            let v_off = r0 * TOP_K * 4;
            let i_off = topk_val_bytes + r0 * TOP_K * 4;
            let h_off = topk_val_bytes + topk_idx_bytes + r0 * R * 4;
            out.push(self.select_proposals(&raw, v_off, i_off, h_off, req.anchor, n, req.temperature, &mut rngs[i]));
        }
        Ok(out)
    }

    pub fn local_kv_regions(&self) -> Vec<DeviceBuf> {
        if self.paged {
            return Vec::new();
        }
        let mut v = Vec::with_capacity(self.layers.len() * 4);
        for ly in &self.layers {
            v.extend_from_slice(&[ly.kc, ly.ks, ly.vc, ly.vs]);
        }
        v
    }
}

impl<B: ComputeBackend> Drop for DFlash2Drafter<B> {
    fn drop(&mut self) {
        let b = self.backend.as_ref();
        for m in &self.fc {
            let _ = b.free_device(m.data);
            let _ = b.free_device(m.row_scales);
        }
        for ly in &self.layers {
            for m in [&ly.attn_conv_proj, &ly.mlp_conv_proj, &ly.q, &ly.k, &ly.v, &ly.o, &ly.gate, &ly.up, &ly.down] {
                let _ = b.free_device(m.data);
                let _ = b.free_device(m.row_scales);
            }
            for buf in [ly.in_ln, ly.post_ln, ly.attn_conv_base, ly.mlp_conv_base, ly.q_norm, ly.k_norm] {
                let _ = b.free_device(buf);
            }
            for bufs in &ly.lane_ctx {
                for &buf in bufs {
                    let _ = b.free_device(buf);
                }
            }
        }
        let _ = b.free_device(self.sel_hidden_proj.data);
        let _ = b.free_device(self.sel_hidden_proj.row_scales);
        for buf in [self.hidden_norm, self.norm_w, self.mask_embed_gpu] {
            let _ = b.free_device(buf);
        }
        let sc = &self.sc;
        for buf in [
            sc.anchor_in, sc.res, sc.xn, sc.xn_conv, sc.tmp, sc.conv_coeff,
            sc.q, sc.kb, sc.vb, sc.att, sc.part_o, sc.part_m, sc.part_l,
            sc.g, sc.u, sc.ctx_c, sc.ctx_tmp, sc.ctx_k, sc.ctx_v,
            sc.base, sc.part_val, sc.part_idx, sc.sel_out,
        ] {
            let _ = b.free_device(buf);
        }
        let _ = b.free_pinned(sc.pinned);
    }
}
