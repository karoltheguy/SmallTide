//! Rows-batched chunked prefill for `Inkling2Engine` (up to `prefill_chunk()` tokens per chunk).
//!
//! Same math as `forward_token`, but every projection is a rows GEMM (weights read once per chunk
//! instead of once per token), attention is a multi-row causal kernel over the FP8 KV cache, short
//! convolutions scan the chunk sequentially per channel, and the MoE is expert-grouped: the
//! chunk's (row, k) assignments are grouped by expert on the host and each expert's weights are
//! streamed once for all of its rows. Experts that are not resident in VRAM are staged into the
//! Scratch Rescue slots in waves, so prefill is always exact (no skipped experts).

use crate::engine::{prefill_chunk, Inkling2Engine};
use smalltide_core::backend::GemvSegment;
use crate::weights::{Fp8MatrixGpu, LoopControllerGpu};
use smalltide_core::backend::{GroupedMoeParams, PrefillAttnParams};
use smalltide_core::{
    ComputeBackend, DeviceBuf, Event, ExpertId, PhysicalLayerId, PinnedBuf, Result, SlotIndex,
    Stream,
};
use smalltide_io::quant::bf16_to_f32;
use smalltide_tide::cache::ExpertClass;

const MAX_BLOCKS: usize = 8;
const MOE_PAIRS_PER_ITEM: usize = 16;
const MOE_PAIRS_PER_ITEM_M64: usize = 64;
const DESC_BYTES: usize = 160 * 1024;
const DESC_RING: usize = 4;

/// Rows-batched activation buffers (sized for `prefill_chunk()` rows).
pub(crate) struct PrefillScratch {
    /// Pre-decoded f16 V tiles for prefill attention (`None`: off): a local layer's ring, and a
    /// global layer's whole context (256 bytes per position and KV head: 268 MiB at 128K).
    v_tiles_local: Option<DeviceBuf>,
    v_tiles_global: Option<DeviceBuf>,
    cur: DeviceBuf,
    embed: DeviceBuf,
    inp_sa: DeviceBuf,
    ffn_inp: DeviceBuf,
    prelude_cur: DeviceBuf,
    bank_out: DeviceBuf,
    pub(crate) blocks: DeviceBuf,   // [8][M][H]
    pub(crate) bank: DeviceBuf,     // [8][M][H]
    pub(crate) messages: DeviceBuf, // [4][M][H]
    pub(crate) in_qkv: DeviceBuf,   // [3][M][H]
    pub(crate) ctrl_qkv: DeviceBuf, // [3][M][H]
    norm_qkv: DeviceBuf, // [3][M][H]
    ctrl_prep: DeviceBuf,
    ctrl_gate: DeviceBuf,
    ctrl_gain: DeviceBuf,
    ctrl_forget: DeviceBuf,
    q: DeviceBuf,
    k_raw: DeviceBuf,
    k: DeviceBuf,
    v_raw: DeviceBuf,
    v: DeviceBuf,
    k_dyn: DeviceBuf,
    v_dyn: DeviceBuf,
    r: DeviceBuf,
    sinks: DeviceBuf,
    rel_table: DeviceBuf,
    attn_out: DeviceBuf,
    gate_out: DeviceBuf,
    o_raw: DeviceBuf,
    sconv_tmp: DeviceBuf,
    moe_in: DeviceBuf,
    moe_ctrl: DeviceBuf,
    moe_norm: DeviceBuf,
    router_logits: DeviceBuf, // [M][128]
    route: DeviceBuf,         // [M][16] u32
    moe_mid: DeviceBuf,       // [M*8][I]
    pub(crate) moe_tmp: DeviceBuf, // [M*8][H]
    moe_out: DeviceBuf,
    desc: DeviceBuf,
    attn_part_o: DeviceBuf, // [MAX_BATCH_ROWS][n_q_heads][16][head_dim] for Split-K verify attn
    attn_part_m: DeviceBuf, // [MAX_BATCH_ROWS][n_q_heads][16]
    attn_part_l: DeviceBuf, // [MAX_BATCH_ROWS][n_q_heads][16]
    attn_part_owned: bool,
    embed_host: PinnedBuf,
    /// image rows of a chunk on their way to `cur` (same layout as `embed_host`)
    img_host: PinnedBuf,
    embed_done: Event,
    embed_used: bool,
    desc_host: Vec<(PinnedBuf, Event, bool)>,
    desc_cursor: usize,
}

const VERIFY_ATTN_SPLITS: usize = 16;

impl PrefillScratch {
    /// `global_ctx`: the longest global context of any lane (sizes the global V-tile scratch).
    pub(crate) fn new<B: ComputeBackend>(b: &B, cfg: &smalltide_core::Inkling2Config, global_ctx: usize) -> Result<Self> {
        let m = prefill_chunk();
        let h = cfg.hidden_size;
        let qd = cfg.num_attention_heads * cfg.head_dim;
        let kvd = cfg.num_key_value_heads * cfg.head_dim;
        let f = |n: usize| b.alloc_device(n * 4);
        let mut desc_host = Vec::with_capacity(DESC_RING);
        for _ in 0..DESC_RING {
            desc_host.push((b.alloc_pinned(DESC_BYTES)?, b.create_event()?, false));
        }
        // 256 bytes per ring position and KV head; the ring must be whole 16-key tiles.
        // Sized to match the unified paged KV local ring (`(window + m).div_ceil(PAGE_POS) + 1`).
        let ring = ((cfg.local_window_size + m).div_ceil(crate::kv_pager::PAGE_POS) + 1)
            * crate::kv_pager::PAGE_POS;
        // SMALLTIDE_ATTN_VTILES: all (default) | local (no global scratch) | 0 (off).
        // Cap global V-tile scratch at 128K positions (256 MiB) so 256K elastic KV does not
        // permanently burn 512 MiB of expert VRAM; prompts past 128K fall back to on-the-fly V decode.
        let v_tiles = std::env::var("SMALLTIDE_ATTN_VTILES").map(|v| v.trim().to_ascii_lowercase()).unwrap_or_else(|_| "all".into());
        let v_tiles_local = if v_tiles != "0" && ring % 16 == 0 {
            Some(b.alloc_device(ring * cfg.num_key_value_heads * 256)?)
        } else {
            None
        };
        let v_tiles_global_cap = global_ctx.min(131_072);
        let v_tiles_global = if v_tiles != "0" && v_tiles != "local" && v_tiles_global_cap > 0 {
            Some(b.alloc_device((v_tiles_global_cap + 16) * cfg.num_key_value_heads * 256)?)
        } else {
            None
        };
        let moe_tmp = f(m * 8 * h)?;
        let po_bytes = crate::engine::MAX_BATCH_ROWS * cfg.num_attention_heads * VERIFY_ATTN_SPLITS * cfg.head_dim * 4;
        let pm_bytes = crate::engine::MAX_BATCH_ROWS * cfg.num_attention_heads * VERIFY_ATTN_SPLITS * 4;
        // During m <= MAX_BATCH_ROWS (64), moe_tmp only uses 64*8*H*4 = 4.5 MiB,
        // so alias attn_part_o/m/l at offset 5 MiB (12.2 MiB, zero extra VRAM!).
        let alias_off = 5usize << 20;
        let pm_off = (alias_off + po_bytes).next_multiple_of(256);
        let pl_off = (pm_off + pm_bytes).next_multiple_of(256);
        let (attn_part_o, attn_part_m, attn_part_l, attn_part_owned) = if pl_off + pm_bytes <= moe_tmp.size_bytes {
            (
                moe_tmp.slice(alias_off, po_bytes),
                moe_tmp.slice(pm_off, pm_bytes),
                moe_tmp.slice(pl_off, pm_bytes),
                false,
            )
        } else {
            (
                b.alloc_device(po_bytes)?,
                b.alloc_device(pm_bytes)?,
                b.alloc_device(pm_bytes)?,
                true,
            )
        };
        Ok(Self {
            v_tiles_local,
            v_tiles_global,
            cur: f(m * h)?,
            embed: f(m * h)?,
            inp_sa: f(m * h)?,
            ffn_inp: f(m * h)?,
            prelude_cur: f(m * h)?,
            bank_out: f(m * h)?,
            blocks: f(MAX_BLOCKS * m * h)?,
            bank: f(MAX_BLOCKS * m * h)?,
            messages: f(4 * m * h)?,
            in_qkv: f(3 * m * h)?,
            ctrl_qkv: f(3 * m * h)?,
            norm_qkv: f(3 * m * h)?,
            ctrl_prep: f(m * 2 * h)?,
            ctrl_gate: f(m * cfg.loop_gate_rank)?,
            ctrl_gain: f(m * h)?,
            ctrl_forget: f(m * h)?,
            q: f(m * qd)?,
            k_raw: f(m * kvd)?,
            k: f(m * kvd)?,
            v_raw: f(m * kvd)?,
            v: f(m * kvd)?,
            k_dyn: f(m * cfg.num_key_value_heads * cfg.sconv_kernel_size)?,
            v_dyn: f(m * cfg.num_key_value_heads * cfg.sconv_kernel_size)?,
            r: f(m * cfg.num_attention_heads * cfg.d_rel)?,
            sinks: f(m * cfg.num_attention_heads)?,
            rel_table: f(m * cfg.num_attention_heads * cfg.rel_extent)?,
            attn_out: f(m * qd)?,
            gate_out: f(m * qd)?,
            o_raw: f(m * h)?,
            sconv_tmp: f(m * h)?,
            moe_in: f(m * h)?,
            moe_ctrl: f(m * h)?,
            moe_norm: f(m * h)?,
            router_logits: f(m * cfg.n_routed_experts)?,
            route: b.alloc_device(m * 16 * 4)?,
            moe_mid: f(m * 8 * cfg.intermediate_size)?,
            moe_tmp,
            moe_out: f(m * h)?,
            desc: b.alloc_device(DESC_BYTES)?,
            attn_part_o,
            attn_part_m,
            attn_part_l,
            attn_part_owned,
            embed_host: b.alloc_pinned(m * h * 4)?,
            img_host: b.alloc_pinned(m * h * 4)?,
            embed_done: b.create_event()?,
            embed_used: false,
            desc_host,
            desc_cursor: 0,
        })
    }

    pub(crate) fn free<B: ComputeBackend>(self, b: &B) {
        for d in [
            self.cur, self.embed, self.inp_sa, self.ffn_inp, self.prelude_cur, self.bank_out, self.blocks,
            self.bank, self.messages, self.in_qkv, self.ctrl_qkv, self.norm_qkv, self.ctrl_prep,
            self.ctrl_gate, self.ctrl_gain, self.ctrl_forget, self.q, self.k_raw, self.k, self.v_raw,
            self.v, self.k_dyn, self.v_dyn, self.r, self.sinks, self.rel_table, self.attn_out,
            self.gate_out, self.o_raw, self.sconv_tmp, self.moe_in, self.moe_ctrl, self.moe_norm,
            self.router_logits, self.route, self.moe_mid, self.moe_tmp, self.moe_out, self.desc,
        ] {
            let _ = b.free_device(d);
        }
        if self.attn_part_owned {
            let _ = b.free_device(self.attn_part_o);
            let _ = b.free_device(self.attn_part_m);
            let _ = b.free_device(self.attn_part_l);
        }
        let _ = b.free_pinned(self.embed_host);
        let _ = b.free_pinned(self.img_host);
        let _ = b.destroy_event(self.embed_done);
        for (p, e, _) in self.desc_host {
            let _ = b.free_pinned(p);
            let _ = b.destroy_event(e);
        }
    }
}

/// One lane's contiguous token slice inside a multi-lane `VerifyBatch` pass (`sum(width) <= 8`).
#[derive(Clone, Copy, Debug)]
pub struct VerifySlice {
    pub lane: usize,
    pub pos0: usize,
    pub row_off: usize,
    pub width: usize,
    /// Starting row offset in `verify_hist_slice(il, kind)` (`Some` when `width >= 2` so short-conv
    /// histories can be rewound independently per slice; `None` when `width == 1`).
    pub hist_off: Option<usize>,
}

/// What the rows of a rows-batched forward are.
#[derive(Clone, Copy)]
pub(crate) enum RowsMode<'a> {
    /// Consecutive tokens of the active lane at positions `pos0..pos0 + m`.
    Prefill { pos0: usize },
    /// One decode token per row: `(lane, pos)`. Lane-specific state (KV caches, short-conv
    /// histories) is per row; everything weight-heavy runs once over all rows.
    Decode { rows: &'a [(usize, usize)] },
    /// Speculative verify: up to 8 consecutive tokens of the active lane at `pos0..` (prefill
    /// semantics: causal attention, sequential short convs) with decode-style device-side MoE
    /// (no host sync), logits for every row, and short-conv histories for the rewind.
    Verify { pos0: usize, greedy: bool },
    /// Batched multi-lane speculative verify: up to `MAX_BATCH_ROWS = 8` total rows partitioned
    /// across multiple concurrent lanes (`slices`). All weight-bound ops (projections, MoE,
    /// rescued experts, lm_head) run ONCE shared across all rows; KV caches and short-convs run
    /// per slice.
    VerifyBatch { slices: &'a [VerifySlice], greedy: bool },
}

/// Row `r` of a `[m][n]` f32 buffer.
#[inline]
fn row(buf: DeviceBuf, r: usize, n: usize) -> DeviceBuf {
    buf.slice(r * n * 4, n * 4)
}

/// Contiguous rows `[r0 .. r0 + count)` of a `[m][n]` f32 buffer.
#[inline]
fn row_range(buf: DeviceBuf, r0: usize, count: usize, n: usize) -> DeviceBuf {
    buf.slice(r0 * n * 4, count * n * 4)
}

/// `[nq][m][H]` slice helper.
#[inline]
fn part(buf: DeviceBuf, idx: usize, m: usize, h: usize) -> DeviceBuf {
    buf.slice(idx * m * h * 4, m * h * 4)
}

impl<B: ComputeBackend> Inkling2Engine<B> {
    /// Prefills `tokens` at positions `pos0..` in chunks of up to `prefill_chunk()` rows. When
    /// `need_logits`, the logits of the last token are left in the logits buffer for sampling.
    pub fn prefill(&mut self, tokens: &[u32], pos0: usize, need_logits: bool) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        // Fast path for 1-token prefill cuts (e.g. the final `end - 1 .. end` token after the
        // server's end-of-prompt prefix snapshot): run the fused zero-sync verify path instead of
        // 48 host-synced `prefill_moe` passes (~15 ms vs ~80-120 ms on TTFT).
        if tokens.len() == 1
            && self.image_rows.is_empty()
            && self.dump.is_none()
            && self.verify_hist.is_some()
            && !matches!(self.tide.scheduler.tide_cfg.rescue_policy, smalltide_core::RescuePolicy::Exact)
        {
            self.verify_for_sampling(tokens, pos0, false)?;
            if need_logits {
                let vocab = self.cfg.unpadded_vocab_size;
                let s = self.tide.compute_stream;
                self.backend.vec_copy(
                    self.scratch.logits,
                    self.batch_logits.expect("batch logits").slice(0, vocab * 4),
                    vocab,
                    s,
                )?;
            }
            self.commit_batch_rows(1);
            self.post_batch_tide_step()?;
            self.verify_rewind(1, 1)?;
            return Ok(());
        }
        self.kv_ensure(self.active_lane(), pos0 + tokens.len())?;
        if self.pf.is_none() {
            self.pf = Some(PrefillScratch::new(&*self.backend, &self.cfg, self.lane_max_seq.iter().copied().max().unwrap_or(0))?);
        }
        // Token boundary bookkeeping (same as forward_token); keep any still-live scratch-ring
        // experts resident so Wave 0 of prefill_moe can reuse them BEFORE unloading them.
        self.run_deferred_tide_step()?;
        let published = self.tide.poll_pending()?;
        self.tide.scheduler.telemetry.stage1_async_published += published as u64;
        self.route_log_inflight = false;
        let mut off = 0usize;
        while off < tokens.len() {
            let m = (tokens.len() - off).min(prefill_chunk());
            let last = off + m == tokens.len();
            if off > 0 {
                // Prompt-aware loading: swaps queued after the previous chunk that have landed
                // are published now (non-blocking), the next ones are planned from this prompt's
                // routing so far and overlap the next chunk.
                // Background disk reads that finished are collected too: until then their experts
                // count as not in RAM and a rescue of one reads the disk.
                let _ = self.tide.poll_stage2_completions()?;
                let published = self.tide.poll_pending()?;
                self.tide.scheduler.telemetry.stage1_async_published += published as u64;
                if self.prefill_chunk_swaps > 0 {
                    let ops = self.tide.scheduler.plan_prefill_progress_swaps(self.prefill_chunk_swaps);
                    if !ops.is_empty() {
                        self.tide.execute_transfers(&ops, false, true)?;
                    }
                }
            }
            self.rows_forward(&tokens[off..off + m], RowsMode::Prefill { pos0: pos0 + off }, need_logits && last)?;
            if self.capture_taps {
                self.drafter_ingest(pos0 + off, m)?;
            }
            off += m;
        }
        Ok(())
    }

    /// Rows-batched forward of `tokens` (one per row) per `mode`. Prefill: logits of the last row
    /// into the logits buffer when `need_logits`. Decode: logits of every row into the batch
    /// logits buffer (`need_logits` ignored).
    pub(crate) fn rows_forward(&mut self, tokens: &[u32], mode: RowsMode, need_logits: bool) -> Result<()> {
        let pos0 = match mode {
            RowsMode::Prefill { pos0 } | RowsMode::Verify { pos0, .. } => pos0,
            RowsMode::VerifyBatch { slices, .. } => slices.first().map_or(0, |sl| sl.pos0),
            RowsMode::Decode { .. } => 0,
        };
        let verify = matches!(mode, RowsMode::Verify { .. } | RowsMode::VerifyBatch { .. });
        let decode = matches!(mode, RowsMode::Decode { .. }) || verify;
        // Verify and batched decode (<= 8 rows) use the GEMV-style kernels (faster than WMMA at
        // this size, and f32 activations like the single-row path).
        self.rows_decode = decode;
        let capture = self.capture_taps && !matches!(mode, RowsMode::Decode { .. });
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let m = tokens.len();
        let (cur, embed, embed_host, embed_done, embed_used) = {
            let pf = self.pf.as_ref().unwrap();
            (pf.cur, pf.embed, pf.embed_host, pf.embed_done, pf.embed_used)
        };
        if embed_used {
            // Wait only if the previous chunk's H2D copy from `embed_host` hasn't finished yet
            // (already complete on verify/decode steps, avoiding a mid-step GPU pipeline bubble).
            self.backend.sync_event(embed_done)?;
        }
        // Embedding rows (host BF16) -> pinned -> GPU -> embed RMSNorm.
        unsafe {
            let dst = self.backend.pinned_as_mut_slice(embed_host)?;
            let dst = std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, m * h);
            for (r, &t) in tokens.iter().enumerate() {
                let tok = (t as usize).min(self.cfg.vocab_size - 1);
                let row = &self.dense.embed_bf16_host[tok * h * 2..(tok + 1) * h * 2];
                for i in 0..h {
                    dst[r * h + i] = bf16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
                }
            }
        }
        self.backend.memcpy_h2d_async(embed, 0, embed_host, 0, m * h * 4, s)?;
        self.backend.record_event(embed_done, s)?;
        self.pf.as_mut().unwrap().embed_used = true;
        self.backend.rms_norm(cur, embed, self.dense.embed_norm, m, h, eps, s)?;
        // Image placeholders: the vision tower's rows replace the normed embeddings (runs of
        // consecutive rows, one copy each, stream-ordered after the norm).
        if matches!(mode, RowsMode::Prefill { .. }) && !self.image_rows.is_empty() {
            let img_host = self.pf.as_ref().unwrap().img_host;
            let mut runs: Vec<(usize, usize)> = Vec::new();
            unsafe {
                let dst = self.backend.pinned_as_mut_slice(img_host)?;
                let dst = std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, m * h);
                for r in 0..m {
                    if let Some(row) = self.image_row(pos0 + r) {
                        dst[r * h..(r + 1) * h].copy_from_slice(row);
                        match runs.last_mut() {
                            Some((r0, n)) if *r0 + *n == r => *n += 1,
                            _ => runs.push((r, 1)),
                        }
                    }
                }
            }
            for (r0, n) in runs {
                self.backend.memcpy_h2d_async(cur, r0 * h * 4, img_host, r0 * h * 4, n * h * 4, s)?;
            }
            // the next chunk waits for these copies too before it overwrites the staging rows
            self.backend.record_event(embed_done, s)?;
        }

        let n_log = self.layer_states.len();
        let dumping = self.dump.is_some() && !matches!(mode, RowsMode::Decode { .. });
        let mut dump_hidden = dumping.then(|| vec![0f32; n_log * m * h]);
        let mut dump_routes = dumping.then(|| vec![0u32; n_log * m * 8]);

        let start = self.cfg.loop_start_layer;
        let end = self.cfg.loop_end_exclusive;
        let core_len = end - start;
        let mut nb = 0usize;
        let mut run = |eng: &mut Self, il: usize, ip: usize, nb: &mut usize, msgs: Option<DeviceBuf>| -> Result<()> {
            let routes = eng.prefill_layer(il, ip, mode, m, nb, msgs)?;
            if capture {
                // Drafter taps: entries of logical layers 5, 23, 41 = outputs of 4, 22, 40.
                let slot = match il { 4 => Some(0usize), 22 => Some(1), 40 => Some(2), _ => None };
                if let Some(t) = slot {
                    let taps = eng.taps_buf.expect("taps buffer");
                    let cur = eng.pf.as_ref().unwrap().cur;
                    eng.backend.vec_copy(taps.slice(t * prefill_chunk() * h * 4, m * h * 4), cur.slice(0, m * h * 4), m * h, s)?;
                }
            }
            if let (Some(dh), Some(dr)) = (dump_hidden.as_mut(), dump_routes.as_mut()) {
                let mut hb = vec![0u8; m * h * 4];
                let cur = eng.pf.as_ref().unwrap().cur;
                eng.backend.memcpy_d2h_sync(&mut hb, cur.slice(0, m * h * 4), 0, s)?;
                for r in 0..m {
                    for i in 0..h {
                        let o = (r * h + i) * 4;
                        dh[(il * m + r) * h + i] = f32::from_le_bytes([hb[o], hb[o + 1], hb[o + 2], hb[o + 3]]);
                    }
                    for k in 0..8 {
                        dr[(il * m + r) * 8 + k] = routes.get(r * 8 + k).copied().unwrap_or(0);
                    }
                }
            }
            Ok(())
        };

        for ip in 0..start {
            run(self, ip, ip, &mut nb, None)?;
        }
        let pf = self.pf.as_ref().unwrap();
        let (prelude_cur, blocks, bank, bank_out, messages) = (pf.prelude_cur, pf.blocks, pf.bank, pf.bank_out, pf.messages);
        self.backend.vec_copy(prelude_cur, cur, m * h, s)?;
        let prelude_nb = nb;
        for pass in 0..self.cfg.loop_passes {
            let mut bank_count = 0usize;
            if pass > 0 {
                let bank_start = prelude_nb + 1;
                bank_count = nb.saturating_sub(bank_start);
                if bank_count > 0 {
                    self.backend.vec_copy(
                        bank.slice(0, bank_count * m * h * 4),
                        blocks.slice(bank_start * m * h * 4, bank_count * m * h * 4),
                        bank_count * m * h,
                        s,
                    )?;
                }
                self.backend.vec_copy(bank_out, cur, m * h, s)?;
                self.backend.vec_copy(cur, prelude_cur, m * h, s)?;
                nb = prelude_nb;
            }
            for ip in start..end {
                let il = start + pass * core_len + (ip - start);
                if pass == 0 {
                    run(self, il, ip, &mut nb, None)?;
                } else {
                    let msg_q = self.dense.layers[ip].loop_msg_queries.expect("loop msg queries");
                    let bv = (bank_count > 0).then(|| bank.slice(0, bank_count * m * h * 4));
                    self.backend.attn_res_mix_rows(messages, bv, bank_out, msg_q, bank_count, 4, h, eps, m, s)?;
                    run(self, il, ip, &mut nb, Some(messages))?;
                }
            }
        }
        let coda_base = start + self.cfg.loop_passes * core_len;
        for ip in end..self.cfg.num_physical_layers {
            run(self, coda_base + (ip - end), ip, &mut nb, None)?;
        }
        drop(run);

        if decode {
            // Final mix + norm and lm_head for every row.
            let pf = self.pf.as_ref().unwrap();
            let (moe_in, moe_norm) = (pf.moe_in, pf.moe_norm);
            let bv = Some(blocks.slice(0, nb * m * h * 4));
            self.backend.attn_res_mix_rows(moe_in, bv, cur, self.dense.output_res_query, nb, 1, h, eps, m, s)?;
            self.backend.rms_norm(moe_norm, moe_in, self.dense.final_norm, m, h, eps, s)?;
            if capture {
                let taps = self.taps_buf.expect("taps buffer");
                self.backend.vec_copy(taps.slice(3 * prefill_chunk() * h * 4, m * h * 4), moe_norm.slice(0, m * h * 4), m * h, s)?;
            }
            let vocab = self.cfg.unpadded_vocab_size;
            let logits = self.batch_logits.expect("batch logits");
            self.dense.unembed_fp8.gemm_rows_decode(self.backend.as_ref(), logits, moe_norm, m, vocab, s)?;
            let skip_softcap = matches!(
                mode,
                RowsMode::Verify { greedy: true, .. } | RowsMode::VerifyBatch { greedy: true, .. }
            ) && self.dump.is_none();
            if self.cfg.final_logit_softcapping > 0.0 && !skip_softcap {
                self.backend.final_logit_softcap(logits, m * vocab, self.cfg.final_logit_softcapping, s)?;
            }
            if !self.pending_checks.is_empty() {
                self.service_verify_rescue_checks(m, pos0)?;
                self.run_deferred_tide_step()?;
            } else if self.verify_rescue_active {
                self.run_deferred_tide_step()?;
            }
            if let (Some(dh), Some(dr)) = (dump_hidden, dump_routes) {
                for r in 0..m {
                    self.write_dump_final(pos0 + r, moe_norm.slice(r * h * 4, h * 4))?;
                    let d = self.dump.as_mut().unwrap();
                    for il in 0..n_log {
                        d.hidden[il * h..(il + 1) * h].copy_from_slice(&dh[(il * m + r) * h..(il * m + r + 1) * h]);
                        d.routes[il * 8..(il + 1) * 8].copy_from_slice(&dr[(il * m + r) * 8..(il * m + r + 1) * 8]);
                    }
                    self.dump_token = (tokens[r], pos0 + r);
                    self.write_dump(false)?;
                }
            }
            return Ok(());
        }
        let logits_needed = need_logits || self.dump.is_some() || capture;
        if logits_needed {
            // Final mix + norm for all rows (cheap), lm_head for the last row only.
            let pf = self.pf.as_ref().unwrap();
            let (moe_in, moe_norm) = (pf.moe_in, pf.moe_norm);
            let bv = Some(blocks.slice(0, nb * m * h * 4));
            self.backend.attn_res_mix_rows(moe_in, bv, cur, self.dense.output_res_query, nb, 1, h, eps, m, s)?;
            self.backend.rms_norm(moe_norm, moe_in, self.dense.final_norm, m, h, eps, s)?;
            if capture {
                let taps = self.taps_buf.expect("taps buffer");
                self.backend.vec_copy(taps.slice(3 * prefill_chunk() * h * 4, m * h * 4), moe_norm.slice(0, m * h * 4), m * h, s)?;
            }
            if self.dump.is_some() {
                for r in 0..m {
                    self.write_dump_final(pos0 + r, moe_norm.slice(r * h * 4, h * 4))?;
                }
            }
            let last = moe_norm.slice((m - 1) * h * 4, h * 4);
            self.dense.unembed_fp8.gemv(
                self.backend.as_ref(),
                self.scratch.logits,
                last,
                self.cfg.unpadded_vocab_size,
                s,
            )?;
            if self.cfg.final_logit_softcapping > 0.0 {
                self.backend.final_logit_softcap(
                    self.scratch.logits,
                    self.cfg.unpadded_vocab_size,
                    self.cfg.final_logit_softcapping,
                    s,
                )?;
            }
        }
        if let (Some(dh), Some(dr)) = (dump_hidden, dump_routes) {
            for r in 0..m {
                let d = self.dump.as_mut().unwrap();
                for il in 0..n_log {
                    d.hidden[il * h..(il + 1) * h].copy_from_slice(&dh[(il * m + r) * h..(il * m + r + 1) * h]);
                    d.routes[il * 8..(il + 1) * 8].copy_from_slice(&dr[(il * m + r) * 8..(il * m + r + 1) * 8]);
                }
                self.dump_token = (tokens[r], pos0 + r);
                // Only the last row's logits exist; earlier rows are dumped without logits.
                self.write_dump(r + 1 == m && logits_needed)?;
            }
        }
        Ok(())
    }

    fn mm_rows(&self, y: DeviceBuf, x: DeviceBuf, w: &Fp8MatrixGpu, m: usize, s: Stream) -> Result<()> {
        if self.rows_decode {
            w.gemm_rows_decode(self.backend.as_ref(), y, x, m, w.n_rows, s)
        } else {
            w.gemm_rows(self.backend.as_ref(), y, x, m, w.n_rows, s)
        }
    }

    /// Up to 4 controllers over the same <= MAX_BATCH_ROWS rows: down and gain / forget
    /// projections each in one launch (scratch slices of MAX_BATCH_ROWS per controller).
    fn control_rows_multi(
        &self,
        items: &[(DeviceBuf, DeviceBuf, DeviceBuf, &LoopControllerGpu)],
        norms: Option<&[(DeviceBuf, DeviceBuf)]>,
        m: usize,
        s: Stream,
    ) -> Result<()> {
        let pf = self.pf.as_ref().unwrap();
        let h = self.cfg.hidden_size;
        let r = self.cfg.loop_gate_rank;
        let r_stride = m;
        let sl = |buf: DeviceBuf, j: usize, n: usize| buf.slice(j * r_stride * n * 4, m * n * 4);
        let fused = norms.is_some();
        if fused {
            let prep: Vec<_> = items.iter().enumerate().map(|(j, it)| (sl(pf.ctrl_prep, j, 2 * h), it.1, it.2)).collect();
            self.backend.ctrl_prep_multi(&prep, h, self.cfg.rms_norm_eps, m, s)?;
        } else {
            for (j, (_, cur, msg, _)) in items.iter().enumerate() {
                self.backend.loop_control_prep_rows(sl(pf.ctrl_prep, j, 2 * h), *cur, *msg, h, self.cfg.rms_norm_eps, m, s)?;
            }
        }
        let seg = |mm: &crate::weights::Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| mm.seg(x, y);
        let downs: Vec<GemvSegment> = items.iter().enumerate().map(|(j, it)| seg(&it.3.down, sl(pf.ctrl_prep, j, 2 * h), sl(pf.ctrl_gate, j, r))).collect();
        let mut gf = Vec::with_capacity(2 * items.len());
        for (j, it) in items.iter().enumerate() {
            gf.push(seg(&it.3.gain, sl(pf.ctrl_gate, j, r), sl(pf.ctrl_gain, j, h)));
            gf.push(seg(&it.3.forget, sl(pf.ctrl_gate, j, r), sl(pf.ctrl_forget, j, h)));
        }
        self.backend.gemv_multi_rows(&downs, m, s)?;
        self.backend.gemv_multi_rows(&gf, m, s)?;
        if let Some(nw) = norms {
            let ap: Vec<_> = items.iter().enumerate().map(|(j, it)| (it.0, it.1, it.2, sl(pf.ctrl_gain, j, h), sl(pf.ctrl_forget, j, h), nw.get(j).copied())).collect();
            return self.backend.ctrl_apply_norm_multi(&ap, h, self.cfg.rms_norm_eps, m, s);
        }
        for (j, (out, cur, msg, _)) in items.iter().enumerate() {
            self.backend.loop_control_apply(*out, *cur, *msg, sl(pf.ctrl_gain, j, h), sl(pf.ctrl_forget, j, h), m * h, s)?;
        }
        Ok(())
    }

    fn control_rows(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        ctrl: &LoopControllerGpu,
        m: usize,
        s: Stream,
    ) -> Result<()> {
        let pf = self.pf.as_ref().unwrap();
        let h = self.cfg.hidden_size;
        self.backend.loop_control_prep_rows(pf.ctrl_prep, cur, msg, h, self.cfg.rms_norm_eps, m, s)?;
        self.mm_rows(pf.ctrl_gate, pf.ctrl_prep, &ctrl.down, m, s)?;
        self.mm_rows(pf.ctrl_gain, pf.ctrl_gate, &ctrl.gain, m, s)?;
        self.mm_rows(pf.ctrl_forget, pf.ctrl_gate, &ctrl.forget, m, s)?;
        self.backend.loop_control_apply(out, cur, msg, pf.ctrl_gain, pf.ctrl_forget, m * h, s)
    }

    /// One logical layer for `m` rows; returns the routed expert ids `[m][8]`.
    fn prefill_layer(
        &mut self,
        il: usize,
        ip: usize,
        mode: RowsMode,
        m: usize,
        nb: &mut usize,
        messages: Option<DeviceBuf>,
    ) -> Result<Vec<u32>> {
        let (pos0, lanes, verify, vbatch) = match mode {
            RowsMode::Prefill { pos0 } => (pos0, None, false, None),
            RowsMode::Verify { pos0, .. } => (pos0, None, true, None),
            RowsMode::VerifyBatch { slices, .. } => (0, None, true, Some(slices)),
            RowsMode::Decode { rows } => (0, Some(rows), false, None),
        };
        // Verify: every short conv of this layer records its history (rewind after acceptance).
        let hist_s: [Option<DeviceBuf>; 4] = if verify && vbatch.is_none() {
            [0, 1, 2, 3].map(|k| Some(self.verify_hist_slice(il, k)))
        } else {
            [None; 4]
        };
        let hist = move |kind: usize| -> Option<DeviceBuf> { hist_s[kind] };
        let hist_vb = |eng: &Self, kind: usize, sl: &VerifySlice| -> Option<DeviceBuf> {
            sl.hist_off.map(|hoff| eng.verify_hist_slice_at(il, kind, hoff, 3 + sl.width))
        };
        let s = self.tide.compute_stream;
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hb = m * h * 4;
        let eps = cfg.rms_norm_eps;
        let nqh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nqh * hd;
        let kv_dim = nkv * hd;
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        let is_local = layer.is_local;
        let backend = self.backend.clone();
        let b = &*backend;

        b.seg_mark("pf: mix,ctrl,norm", s);
        // Speculative verify: copy + mixes (+ norms) in one launch per side of the attention.
        let fuse_mix = verify && self.rows_decode && std::env::var_os("SMALLTIDE_VERIFY_UNFUSED").is_none();
        if !fuse_mix {
            b.vec_copy(pf.inp_sa, pf.cur, m * h, s)?;
        }
        let blocks_view = (*nb > 0).then(|| pf.blocks.slice(0, *nb * hb));
        let nrm = [part(pf.norm_qkv, 0, m, h), part(pf.norm_qkv, 1, m, h), part(pf.norm_qkv, 2, m, h)];
        let mut pre_normed: Option<(DeviceBuf, DeviceBuf, DeviceBuf)> = None;

        // 1. attention-residual mix
        let (in0, in1, in2) = if fuse_mix {
            let nqm = if is_local { 1 } else { 3 };
            let norms = match (messages.is_none(), is_local) {
                (true, true) => [layer.attn_norm, None, None],
                (true, false) => [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm],
                (false, _) => [None, None, None],
            };
            let mix_out = (messages.is_some() && layer.attn_res_query.is_some()).then_some(pf.in_qkv);
            b.mix_rows(
                smalltide_core::backend::MixRowsParams {
                    cur: pf.cur,
                    add_src: None,
                    residual: None,
                    blocks: blocks_view,
                    copy_out: Some(pf.inp_sa),
                    queries: layer.attn_res_query,
                    mix_out,
                    norm_w: norms,
                    norm_out: Some(pf.norm_qkv),
                    nb: *nb,
                    boundary: false,
                    nq: nqm,
                    hidden: h,
                    rows: m,
                    eps,
                },
                s,
            )?;
            if messages.is_none() {
                pre_normed = Some(if is_local { (nrm[0], nrm[0], nrm[0]) } else { (nrm[0], nrm[1], nrm[2]) });
            }
            match (layer.attn_res_query.is_some(), is_local) {
                (true, true) => (part(pf.in_qkv, 0, m, h), part(pf.in_qkv, 0, m, h), part(pf.in_qkv, 0, m, h)),
                (true, false) => (part(pf.in_qkv, 0, m, h), part(pf.in_qkv, 1, m, h), part(pf.in_qkv, 2, m, h)),
                (false, _) => (pf.cur, pf.cur, pf.cur),
            }
        } else if let Some(attn_q) = layer.attn_res_query {
            if is_local {
                let d0 = part(pf.in_qkv, 0, m, h);
                b.attn_res_mix_rows(d0, blocks_view, pf.cur, attn_q, *nb, 1, h, eps, m, s)?;
                (d0, d0, d0)
            } else {
                b.attn_res_mix_rows(pf.in_qkv, blocks_view, pf.cur, attn_q, *nb, 3, h, eps, m, s)?;
                (part(pf.in_qkv, 0, m, h), part(pf.in_qkv, 1, m, h), part(pf.in_qkv, 2, m, h))
            }
        } else {
            (pf.cur, pf.cur, pf.cur)
        };
        // 2. cross-pass controllers
        let (in0, in1, in2) = if let Some(msgs) = messages {
            let ctrls = layer.controllers.as_ref().expect("controllers");
            let c = [part(pf.ctrl_qkv, 0, m, h), part(pf.ctrl_qkv, 1, m, h), part(pf.ctrl_qkv, 2, m, h)];
            let ins = [in0, in1, in2];
            if self.rows_decode {
                let items: Vec<_> = (0..3).map(|j| (c[j], ins[j], part(msgs, j, m, h), &ctrls[j])).collect();
                if fuse_mix {
                    let ws = if is_local {
                        let w = layer.attn_norm.expect("attn_norm");
                        [w, w, w]
                    } else {
                        [layer.q_input_norm.unwrap(), layer.k_input_norm.unwrap(), layer.v_input_norm.unwrap()]
                    };
                    let nr: Vec<_> = (0..3).map(|j| (ws[j], nrm[j])).collect();
                    self.control_rows_multi(&items, Some(&nr), m, s)?;
                    pre_normed = Some((nrm[0], nrm[1], nrm[2]));
                } else {
                    self.control_rows_multi(&items, None, m, s)?;
                }
            } else {
                for j in 0..3 {
                    self.control_rows(c[j], ins[j], part(msgs, j, m, h), &ctrls[j], m, s)?;
                }
            }
            (c[0], c[1], c[2])
        } else {
            (in0, in1, in2)
        };
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        let has_msg = messages.is_some();
        // 3. input norms
        let n = [part(pf.norm_qkv, 0, m, h), part(pf.norm_qkv, 1, m, h), part(pf.norm_qkv, 2, m, h)];
        let (nq, nk, nv) = if let Some(x) = pre_normed {
            x
        } else if is_local {
            let w = layer.attn_norm.expect("attn_norm");
            if has_msg {
                b.rms_norm(n[0], in0, w, m, h, eps, s)?;
                b.rms_norm(n[1], in1, w, m, h, eps, s)?;
                b.rms_norm(n[2], in2, w, m, h, eps, s)?;
                (n[0], n[1], n[2])
            } else {
                b.rms_norm(n[0], in0, w, m, h, eps, s)?;
                (n[0], n[0], n[0])
            }
        } else {
            b.rms_norm(n[0], in0, layer.q_input_norm.unwrap(), m, h, eps, s)?;
            b.rms_norm(n[1], in1, layer.k_input_norm.unwrap(), m, h, eps, s)?;
            b.rms_norm(n[2], in2, layer.v_input_norm.unwrap(), m, h, eps, s)?;
            (n[0], n[1], n[2])
        };
        // 4. projections
        b.seg_mark("pf: qkvr proj,conv,clip", s);
        let fused_proj = self.rows_decode;
        let taps = nkv * cfg.sconv_kernel_size;
        if fused_proj {
            let seg = |mm: &crate::weights::Fp8MatrixGpu, x: DeviceBuf, y: DeviceBuf| mm.seg(x, y);
            let fp8_segs = [
                seg(&layer.wq, nq, pf.q),
                seg(&layer.wk, nk, pf.k_raw),
                seg(&layer.wv, nv, pf.v_raw),
                seg(&layer.wr, nq, pf.r),
                seg(&layer.gate_attn, nq, pf.gate_out),
            ];
            let mut f32_segs = vec![
                GemvSegment { w: layer.attn_sink, row_scales: None, x: nq, y: pf.sinks, n_rows: nqh, k_cols: h, is_bf16: false },
            ];
            if let Some(w) = layer.k_sconv_dyn_proj {
                f32_segs.push(GemvSegment { w, row_scales: None, x: nk, y: pf.k_dyn, n_rows: taps, k_cols: h, is_bf16: false });
            }
            if let Some(w) = layer.v_sconv_dyn_proj {
                f32_segs.push(GemvSegment { w, row_scales: None, x: nv, y: pf.v_dyn, n_rows: taps, k_cols: h, is_bf16: false });
            }
            if m >= 4 {
                // Separate FP8 segments so gemv_multi_wmma fires on all 5 large projections in ONE launch (up to 64 rows!).
                b.gemv_multi_rows(&fp8_segs, m, s)?;
                if m > 16 {
                    for sg in &f32_segs {
                        b.gemm_f32_rows(sg.y, sg.x, sg.w, m, sg.n_rows, sg.k_cols, s)?;
                    }
                } else {
                    b.gemv_multi_rows(&f32_segs, m, s)?;
                }
            } else {
                let mut segs = Vec::with_capacity(fp8_segs.len() + f32_segs.len());
                segs.extend_from_slice(&fp8_segs);
                segs.extend_from_slice(&f32_segs);
                b.gemv_multi_rows(&segs, m, s)?;
            }
        } else {
            self.mm_rows(pf.q, nq, &layer.wq, m, s)?;
            self.mm_rows(pf.k_raw, nk, &layer.wk, m, s)?;
            self.mm_rows(pf.v_raw, nv, &layer.wv, m, s)?;
            self.mm_rows(pf.r, nq, &layer.wr, m, s)?;
        }
        // During prefill, local sliding-window layers allocate pages layer-by-layer and evict
        // expired pages immediately after attention, keeping peak temporary overhead to ~4 pages.
        let prefill_local_li = if matches!(mode, RowsMode::Prefill { .. }) && is_local {
            let li = self.layer_states[il].local_idx;
            if let Some(li) = li {
                let win = self.cfg.local_window_size;
                self.kv_ensure_local(self.active_lane(), Some(li), pos0.saturating_sub(win), pos0 + m)?;
            }
            li
        } else {
            None
        };
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        let cfg = &self.cfg;
        let st = &self.layer_states[il];
        // Speculative verify: one fused launch for the K/V convs (+histories), clips, value norm,
        // relative tables and the KV store.
        let prepped = verify && fused_proj && std::env::var_os("SMALLTIDE_VERIFY_UNFUSED").is_none();
        if prepped {
            let tau = (!is_local).then_some((cfg.log_scaling_alpha, cfg.log_scaling_n_floor as f32));
            if let Some(slices) = vbatch {
                for sl in slices {
                    let ls = self.lane_layer_state(sl.lane, il);
                    let (off, w) = (sl.row_off, sl.width);
                    let hk = hist_vb(self, 0, sl);
                    let hv = hist_vb(self, 1, sl);
                    b.attn_prep_rows(
                        smalltide_core::backend::AttnPrepParams {
                            page_table: ls.page_table,
                            q: row_range(pf.q, off, w, q_dim),
                            r: row_range(pf.r, off, w, nqh * cfg.d_rel),
                            k_raw: row_range(pf.k_raw, off, w, kv_dim),
                            v_raw: row_range(pf.v_raw, off, w, kv_dim),
                            k_dyn: layer.k_sconv_dyn_proj.map(|_| row_range(pf.k_dyn, off, w, taps)),
                            v_dyn: layer.v_sconv_dyn_proj.map(|_| row_range(pf.v_dyn, off, w, taps)),
                            sconv_k: ls.sconv_k,
                            sconv_v: ls.sconv_v,
                            k_w: layer.k_sconv_static,
                            v_w: layer.v_sconv_static,
                            q_scale: layer.q_l2_clip_scale,
                            r_scale: layer.rel_l2_clip_scale,
                            rel_proj: layer.rel_logits_proj,
                            rel_table: row_range(pf.rel_table, off, w, nqh * cfg.rel_extent),
                            k_cache: ls.k_cache,
                            k_scales: ls.k_scales,
                            v_cache: ls.v_cache,
                            v_scales: ls.v_scales,
                            n_q_heads: nqh,
                            n_kv_heads: nkv,
                            max_seq: ls.max_seq,
                            slot: 0,
                            rel_extent: cfg.rel_extent,
                            qk_threshold: cfg.qk_l2_clip_threshold,
                            rel_threshold: cfg.rel_l2_clip_threshold,
                            tau: 1.0,
                            eps,
                            v_q4: ls.v_q4,
                        },
                        hk.zip(hv),
                        sl.pos0,
                        is_local,
                        w,
                        tau,
                        s,
                    )?;
                }
            } else {
                b.attn_prep_rows(
                    smalltide_core::backend::AttnPrepParams {
                        page_table: st.page_table,
                        q: pf.q,
                        r: pf.r,
                        k_raw: pf.k_raw,
                        v_raw: pf.v_raw,
                        k_dyn: layer.k_sconv_dyn_proj.map(|_| pf.k_dyn),
                        v_dyn: layer.v_sconv_dyn_proj.map(|_| pf.v_dyn),
                        sconv_k: st.sconv_k,
                        sconv_v: st.sconv_v,
                        k_w: layer.k_sconv_static,
                        v_w: layer.v_sconv_static,
                        q_scale: layer.q_l2_clip_scale,
                        r_scale: layer.rel_l2_clip_scale,
                        rel_proj: layer.rel_logits_proj,
                        rel_table: pf.rel_table,
                        k_cache: st.k_cache,
                        k_scales: st.k_scales,
                        v_cache: st.v_cache,
                        v_scales: st.v_scales,
                        n_q_heads: nqh,
                        n_kv_heads: nkv,
                        max_seq: st.max_seq,
                        slot: 0,
                        rel_extent: cfg.rel_extent,
                        qk_threshold: cfg.qk_l2_clip_threshold,
                        rel_threshold: cfg.rel_l2_clip_threshold,
                        tau: 1.0,
                        eps,
                        v_q4: st.v_q4,
                    },
                    hist(0).zip(hist(1)),
                    pos0,
                    is_local,
                    m,
                    tau,
                    s,
                )?;
            }
        }
        let k_dyn = match layer.k_sconv_dyn_proj {
            Some(w) => {
                if !fused_proj {
                    b.gemm_f32_rows(pf.k_dyn, nk, w, m, taps, h, s)?;
                }
                Some(pf.k_dyn)
            }
            None => None,
        };
        match (lanes, vbatch) {
            (_, _) if prepped => {}
            (None, Some(slices)) => {
                for sl in slices {
                    let ls = self.lane_layer_state(sl.lane, il);
                    let (off, w) = (sl.row_off, sl.width);
                    b.sconv_k4_rows_hist(
                        row_range(pf.k, off, w, kv_dim), ls.sconv_k, row_range(pf.k_raw, off, w, kv_dim),
                        layer.k_sconv_static, k_dyn.map(|d| row_range(d, off, w, taps)), kv_dim, hd, w, taps,
                        hist_vb(self, 0, sl), s,
                    )?;
                }
            }
            (None, None) => b.sconv_k4_rows_hist(pf.k, st.sconv_k, pf.k_raw, layer.k_sconv_static, k_dyn, kv_dim, hd, m, taps, hist(0), s)?,
            (Some(rows), _) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.k, r, kv_dim), ls.sconv_k, row(pf.k_raw, r, kv_dim), layer.k_sconv_static,
                        k_dyn.map(|d| row(d, r, taps)), kv_dim, hd, 1, taps, s)?;
                }
            }
        }
        let v_dyn = match layer.v_sconv_dyn_proj {
            Some(w) => {
                if !fused_proj {
                    b.gemm_f32_rows(pf.v_dyn, nv, w, m, taps, h, s)?;
                }
                Some(pf.v_dyn)
            }
            None => None,
        };
        match (lanes, vbatch) {
            (_, _) if prepped => {}
            (None, Some(slices)) => {
                for sl in slices {
                    let ls = self.lane_layer_state(sl.lane, il);
                    let (off, w) = (sl.row_off, sl.width);
                    b.sconv_k4_rows_hist(
                        row_range(pf.v, off, w, kv_dim), ls.sconv_v, row_range(pf.v_raw, off, w, kv_dim),
                        layer.v_sconv_static, v_dyn.map(|d| row_range(d, off, w, taps)), kv_dim, hd, w, taps,
                        hist_vb(self, 1, sl), s,
                    )?;
                }
            }
            (None, None) => b.sconv_k4_rows_hist(pf.v, st.sconv_v, pf.v_raw, layer.v_sconv_static, v_dyn, kv_dim, hd, m, taps, hist(1), s)?,
            (Some(rows), _) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.v, r, kv_dim), ls.sconv_v, row(pf.v_raw, r, kv_dim), layer.v_sconv_static,
                        v_dyn.map(|d| row(d, r, taps)), kv_dim, hd, 1, taps, s)?;
                }
            }
        }
        // 5. clips, value norm, tau (global layers)
        let tau_at = |p: usize| (!is_local).then_some((p, cfg.log_scaling_alpha, cfg.log_scaling_n_floor as f32));
        match (lanes, vbatch) {
            _ if prepped => {}
            (Some(rows), _) if !is_local => {
                // tau depends on each row's own position.
                for (r, &(_, p)) in rows.iter().enumerate() {
                    b.head_l2_clip_rows(row(pf.q, r, q_dim), Some(layer.q_l2_clip_scale), nqh, hd, nqh, cfg.qk_l2_clip_threshold, tau_at(p), s)?;
                    b.head_l2_clip_rows(row(pf.r, r, nqh * cfg.d_rel), Some(layer.rel_l2_clip_scale), nqh, cfg.d_rel, nqh,
                        cfg.rel_l2_clip_threshold, tau_at(p), s)?;
                }
            }
            (None, Some(slices)) if !is_local => {
                for sl in slices {
                    let (off, w) = (sl.row_off, sl.width);
                    b.head_l2_clip_rows(row_range(pf.q, off, w, q_dim), Some(layer.q_l2_clip_scale), w * nqh, hd, nqh, cfg.qk_l2_clip_threshold, tau_at(sl.pos0), s)?;
                    b.head_l2_clip_rows(row_range(pf.r, off, w, nqh * cfg.d_rel), Some(layer.rel_l2_clip_scale), w * nqh, cfg.d_rel, nqh, cfg.rel_l2_clip_threshold, tau_at(sl.pos0), s)?;
                }
            }
            _ => {
                b.head_l2_clip_rows(pf.q, Some(layer.q_l2_clip_scale), m * nqh, hd, nqh, cfg.qk_l2_clip_threshold, tau_at(pos0), s)?;
                b.head_l2_clip_rows(pf.r, Some(layer.rel_l2_clip_scale), m * nqh, cfg.d_rel, nqh, cfg.rel_l2_clip_threshold, tau_at(pos0), s)?;
            }
        }
        if !prepped {
            b.head_l2_clip_rows(pf.k, None, m * nkv, hd, nkv, cfg.qk_l2_clip_threshold, None, s)?;
            b.value_norm_no_gamma(pf.v, m * nkv, hd, eps, s)?;
        }
        // 6. sinks, relative bias, KV store, attention
        b.seg_mark("pf: sinks,rel,kv,attention", s);
        if !fused_proj {
            b.gemm_f32_rows(pf.sinks, nq, layer.attn_sink, m, nqh, h, s)?;
        }
        if !prepped {
            b.rel_table(pf.rel_table, pf.r, layer.rel_logits_proj, m * nqh, cfg.rel_extent, s)?;
        }
        match (lanes, vbatch) {
            (None, Some(slices)) => {
                for sl in slices {
                    let ls = self.lane_layer_state(sl.lane, il);
                    let (off, w) = (sl.row_off, sl.width);
                    if !prepped {
                        b.kv_store_fp8_rows(
                            ls.k_cache, ls.k_scales, ls.v_cache, ls.v_scales,
                            row_range(pf.k, off, w, kv_dim), row_range(pf.v, off, w, kv_dim),
                            nkv, ls.max_seq, hd, sl.pos0, w, is_local, ls.page_table, ls.v_q4, s,
                        )?;
                    }
                    let attn_params = PrefillAttnParams {
                        page_table: ls.page_table,
                        n_q_heads: nqh,
                        n_kv_heads: nkv,
                        max_seq: ls.max_seq,
                        window: if is_local { ls.window } else { 0 },
                        pos0: sl.pos0,
                        rows: w,
                        rel_extent: cfg.rel_extent,
                        inv_sqrt_d: 1.0 / (hd as f32).sqrt(),
                        v_q4: ls.v_q4,
                        v_tiles: None,
                    };
                    b.attn_prefill_fp8_split(
                        row_range(pf.attn_out, off, w, q_dim),
                        pf.attn_part_o, pf.attn_part_m, pf.attn_part_l,
                        row_range(pf.q, off, w, q_dim),
                        ls.k_cache, ls.k_scales, ls.v_cache, ls.v_scales,
                        Some(row_range(pf.rel_table, off, w, nqh * cfg.rel_extent)),
                        Some(row_range(pf.sinks, off, w, nqh)),
                        attn_params,
                        s,
                    )?;
                }
            }
            (None, None) => {
                if !prepped {
                    b.kv_store_fp8_rows(st.k_cache, st.k_scales, st.v_cache, st.v_scales, pf.k, pf.v, nkv, st.max_seq, hd, pos0, m, is_local, st.page_table, st.v_q4, s)?;
                }
                let attn_params = PrefillAttnParams {
                    page_table: st.page_table,
                    n_q_heads: nqh,
                    n_kv_heads: nkv,
                    max_seq: st.max_seq,
                    window: if is_local { st.window } else { 0 },
                    pos0,
                    rows: m,
                    rel_extent: cfg.rel_extent,
                    inv_sqrt_d: 1.0 / (hd as f32).sqrt(),
                    v_q4: st.v_q4,
                    // Many rows: the V tiles are decoded once for the pass (a local ring, or the
                    // whole context of a global layer) instead of once per query block.
                    v_tiles: if m <= 64 {
                        None
                    } else if is_local {
                        pf.v_tiles_local.filter(|b| st.max_seq % 16 == 0 && st.max_seq * nkv * 256 <= b.size_bytes)
                    } else {
                        pf.v_tiles_global.filter(|b| (pos0 + m + 16) * nkv * 256 <= b.size_bytes)
                    },
                };
                if m <= crate::engine::MAX_BATCH_ROWS {
                    b.attn_prefill_fp8_split(
                        pf.attn_out, pf.attn_part_o, pf.attn_part_m, pf.attn_part_l,
                        pf.q, st.k_cache, st.k_scales, st.v_cache, st.v_scales,
                        Some(pf.rel_table), Some(pf.sinks), attn_params, s,
                    )?;
                } else {
                    b.attn_prefill_fp8(
                        pf.attn_out, pf.q, st.k_cache, st.k_scales, st.v_cache, st.v_scales,
                        Some(pf.rel_table), Some(pf.sinks), attn_params, s,
                    )?;
                }
                if let Some(li) = prefill_local_li {
                    let win = self.cfg.local_window_size;
                    self.kv_evict_local(self.active_lane(), Some(li), (pos0 + m).saturating_sub(win), pos0 + m);
                }
            }
            (Some(rows), _) => {
                // Each row attends over its own lane's cache (decode flash attention).
                for (r, &(lane, p)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.kv_store_fp8_rows(ls.k_cache, ls.k_scales, ls.v_cache, ls.v_scales, row(pf.k, r, kv_dim), row(pf.v, r, kv_dim),
                        nkv, ls.max_seq, hd, p, 1, is_local, ls.page_table, ls.v_q4, s)?;
                    let n_valid = if is_local { (p + 1).min(ls.window) } else { p + 1 };
                    let num_splits = n_valid.div_ceil(128).clamp(1, crate::engine::MAX_FLASH_SPLITS);
                    b.flash_decode_fp8(
                        row(pf.attn_out, r, q_dim),
                        self.scratch.partial_out,
                        self.scratch.partial_m,
                        self.scratch.partial_l,
                        row(pf.q, r, q_dim),
                        ls.k_cache,
                        ls.k_scales,
                        ls.v_cache,
                        ls.v_scales,
                        Some(row(pf.rel_table, r, nqh * cfg.rel_extent)),
                        Some(row(pf.sinks, r, nqh)),
                        smalltide_core::backend::FlashDecodeParams {
                            page_table: ls.page_table,
                            n_q_heads: nqh,
                            n_kv_heads: nkv,
                            max_seq: ls.max_seq,
                            q_pos: p,
                            rel_extent: cfg.rel_extent,
                            is_local,
                            num_splits,
                            inv_sqrt_d: 1.0 / (hd as f32).sqrt(),
                            window: if is_local { ls.window } else { 0 },
                            v_q4: ls.v_q4,
                        },
                        s,
                    )?;
                }
            }
        }
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        let cfg = &self.cfg;
        let st = &self.layer_states[il];
        // 7. output gate, wo, post-attention conv
        b.seg_mark("pf: gate,wo,conv,mix,router", s);
        if !fused_proj {
            self.mm_rows(pf.gate_out, nq, &layer.gate_attn, m, s)?;
        }
        b.gate_attn_sigmoid_mul(pf.attn_out, pf.attn_out, pf.gate_out, m * q_dim, s)?;
        self.mm_rows(pf.o_raw, pf.attn_out, &layer.wo, m, s)?;
        match (lanes, vbatch) {
            (None, Some(slices)) => {
                for sl in slices {
                    let ls = self.lane_layer_state(sl.lane, il);
                    let (off, w) = (sl.row_off, sl.width);
                    b.sconv_k4_rows_hist(
                        row_range(pf.sconv_tmp, off, w, h), ls.sconv_attn, row_range(pf.o_raw, off, w, h),
                        layer.attn_sconv, None, h, 0, w, 0, hist_vb(self, 2, sl), s,
                    )?;
                }
            }
            (None, None) => b.sconv_k4_rows_hist(pf.sconv_tmp, st.sconv_attn, pf.o_raw, layer.attn_sconv, None, h, 0, m, 0, hist(2), s)?,
            (Some(rows), _) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.sconv_tmp, r, h), ls.sconv_attn, row(pf.o_raw, r, h), layer.attn_sconv, None, h, 0, 1, 0, s)?;
                }
            }
        }
        // 8. block boundary, 9. MLP input mix -> controller -> norm -> router
        let boundary = ip % cfg.n_layers_per_attn_res_block == 0;
        let mut post_normed = false;
        if fuse_mix {
            b.mix_rows(
                smalltide_core::backend::MixRowsParams {
                    cur: pf.cur,
                    add_src: Some(pf.sconv_tmp),
                    residual: Some(pf.inp_sa),
                    blocks: Some(pf.blocks.slice(0, (*nb + 1) * hb)),
                    copy_out: Some(pf.ffn_inp),
                    queries: Some(layer.mlp_res_query),
                    mix_out: messages.is_some().then_some(pf.moe_in),
                    norm_w: [messages.is_none().then_some(layer.mlp_norm), None, None],
                    norm_out: Some(pf.moe_norm),
                    nb: *nb,
                    boundary,
                    nq: 1,
                    hidden: h,
                    rows: m,
                    eps,
                },
                s,
            )?;
            if boundary {
                *nb += 1;
            }
            post_normed = messages.is_none();
        } else {
            if boundary {
                b.vec_copy(pf.blocks.slice(*nb * hb, hb), pf.inp_sa, m * h, s)?;
                *nb += 1;
                b.vec_copy(pf.cur, pf.sconv_tmp, m * h, s)?;
            } else {
                b.vec_add(pf.cur, pf.inp_sa, pf.sconv_tmp, m * h, s)?;
            }
            b.vec_copy(pf.ffn_inp, pf.cur, m * h, s)?;
            let bv = Some(pf.blocks.slice(0, *nb * hb));
            b.attn_res_mix_rows(pf.moe_in, bv, pf.cur, layer.mlp_res_query, *nb, 1, h, eps, m, s)?;
        }
        let moe_x = if let Some(msgs) = messages {
            let ctrls = layer.controllers.as_ref().unwrap();
            if self.rows_decode {
                let nr = [(layer.mlp_norm, pf.moe_norm)];
                self.control_rows_multi(&[(pf.moe_ctrl, pf.moe_in, part(msgs, 3, m, h), &ctrls[3])], fuse_mix.then_some(&nr[..]), m, s)?;
                post_normed = fuse_mix;
            } else {
                self.control_rows(pf.moe_ctrl, pf.moe_in, part(msgs, 3, m, h), &ctrls[3], m, s)?;
            }
            pf.moe_ctrl
        } else {
            pf.moe_in
        };
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        if !post_normed {
            b.rms_norm(pf.moe_norm, moe_x, layer.mlp_norm, m, h, eps, s)?;
        }
        b.gemm_f32_rows(pf.router_logits, pf.moe_norm, layer.router_weight, m, cfg.n_routed_experts, h, s)?;
        // Batched decode / verify: the router writes straight into this layer's routing-log rows.
        let route = if self.rows_decode {
            self.batch_route_log.expect("batch route log").slice(il * crate::engine::MAX_BATCH_ROWS * 64, m * 64)
        } else {
            pf.route
        };
        b.router_topk8_rows(route, pf.router_logits, Some(layer.router_bias), cfg.route_scale, m, 16, s)?;

        // 10. expert-grouped MoE (host plan from the routed ids)
        b.seg_mark("pf: moe route sync + host plan", s);
        if let Some(rows) = lanes {
            // The rows of a batched round are rescued like a verify block's (a no-op unless the
            // round armed it: `begin_rows_rescue`).
            self.queue_verify_rescue_check(il, ip, route, m)?;
            // Batched decode MoE: device-side routing, experts deduplicated across rows;
            // per-row renormalized sum + MLP short conv + residual.
            let cfg = &self.cfg;
            let pf = self.pf.as_ref().unwrap();
            let is_mx = cfg.is_mxfp8_layer(ip);
            let (pool, stride, dtype) = if is_mx {
                (self.tide.vram_pool_mxfp8, self.tide.mxfp8_layout.aligned_slab_bytes, smalltide_core::DType::MXFP8)
            } else {
                (self.tide.vram_pool_nvfp4, self.tide.nvfp4_layout.aligned_slab_bytes, smalltide_core::DType::NVFP4)
            };
            let n_exp = cfg.n_routed_experts;
            let slot_table = self.tide.gpu_slot_table.slice(ip * n_exp * 4, n_exp * 4);
            let params = smalltide_core::backend::MoeDispatchParams {
                batch_size: m,
                hidden_size: h,
                intermediate_size: cfg.intermediate_size,
                top_k: cfg.num_experts_per_tok,
                weight_dtype: dtype,
                expert_stride_bytes: stride,
            };
            if std::env::var_os("SMALLTIDE_BATCH_MOE_DEDUP").is_some() {
                b.moe_decode_rows(pf.moe_mid, pf.moe_tmp, pf.moe_norm, pool, slot_table, route, m, params, s)?;
            } else {
                b.moe_rows_grouped(pf.moe_mid, pf.moe_tmp, pf.moe_norm, pf.moe_out, route, slot_table, pool, pf.desc, m, params, s)?;
            }
            let layer = &self.dense.layers[ip];
            for (r, &(lane, _)) in rows.iter().enumerate() {
                let ls = self.lane_layer_state(lane, il);
                b.moe_decode_post(
                    row(pf.cur, r, h),
                    pf.moe_tmp.slice(r * 8 * h * 4, 8 * h * 4),
                    slot_table,
                    route.slice(r * 64, 32),
                    route.slice(r * 64 + 32, 32),
                    cfg.num_experts_per_tok,
                    Some((ls.sconv_mlp, layer.mlp_sconv, row(pf.ffn_inp, r, h))),
                    h,
                    s,
                )?;
            }
            return Ok(Vec::new());
        }
        let ids = if verify {
            // Inline GPU-side rescue check (zero-sync when layer is 100% resident or rescue disabled).
            self.queue_verify_rescue_check(il, ip, route, m)?;
            // Device-side multi-row MoE (as in batched decode), weighted sum per row into moe_out;
            // the MLP short conv below then runs over the rows with history.
            let pf = self.pf.as_ref().unwrap();
            let is_mx = self.cfg.is_mxfp8_layer(ip);
            let (pool, stride, dtype) = if is_mx {
                (self.tide.vram_pool_mxfp8, self.tide.mxfp8_layout.aligned_slab_bytes, smalltide_core::DType::MXFP8)
            } else {
                (self.tide.vram_pool_nvfp4, self.tide.nvfp4_layout.aligned_slab_bytes, smalltide_core::DType::NVFP4)
            };
            let n_exp = self.cfg.n_routed_experts;
            let slot_table = self.tide.gpu_slot_table.slice(ip * n_exp * 4, n_exp * 4);
            let params = smalltide_core::backend::MoeDispatchParams {
                batch_size: m,
                hidden_size: h,
                intermediate_size: self.cfg.intermediate_size,
                top_k: self.cfg.num_experts_per_tok,
                weight_dtype: dtype,
                expert_stride_bytes: stride,
            };
            let b = self.backend.clone();
            b.moe_rows_grouped(pf.moe_mid, pf.moe_tmp, pf.moe_norm, pf.moe_out, route, slot_table, pool, pf.desc, m, params, s)?;
            if std::env::var_os("SMALLTIDE_VERIFY_UNFUSED").is_none() {
                // Fused epilogue: weighted sum + MLP conv (+history) + residual; done here.
                let layer = &self.dense.layers[ip];
                if let Some(slices) = vbatch {
                    for sl in slices {
                        let ls = self.lane_layer_state(sl.lane, il);
                        let (off, w) = (sl.row_off, sl.width);
                        b.moe_post_sconv_rows(
                            row_range(pf.cur, off, w, h),
                            row_range(pf.ffn_inp, off, w, h),
                            pf.moe_tmp.slice(off * 8 * h * 4, w * 8 * h * 4),
                            slot_table,
                            route.slice(off * 64, w * 64),
                            ls.sconv_mlp,
                            layer.mlp_sconv,
                            hist_vb(self, 3, sl),
                            h,
                            w,
                            s,
                        )?;
                    }
                } else {
                    let st = &self.layer_states[il];
                    b.moe_post_sconv_rows(pf.cur, pf.ffn_inp, pf.moe_tmp, slot_table, route, st.sconv_mlp, layer.mlp_sconv, hist(3), h, m, s)?;
                }
                b.seg_mark("pf: moe post", s);
                return Ok(Vec::new());
            }
            b.moe_post_rows(pf.moe_out, pf.moe_tmp, slot_table, route, m, h, s)?;
            Vec::new()
        } else {
            self.prefill_moe(ip, m)?
        };
        b.seg_mark("pf: moe post", s);

        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        if let Some(slices) = vbatch {
            for sl in slices {
                let ls = self.lane_layer_state(sl.lane, il);
                let (off, w) = (sl.row_off, sl.width);
                b.sconv_k4_rows_hist(
                    row_range(pf.sconv_tmp, off, w, h), ls.sconv_mlp, row_range(pf.moe_out, off, w, h),
                    layer.mlp_sconv, None, h, 0, w, 0, hist_vb(self, 3, sl), s,
                )?;
            }
        } else {
            let st = &self.layer_states[il];
            b.sconv_k4_rows_hist(pf.sconv_tmp, st.sconv_mlp, pf.moe_out, layer.mlp_sconv, None, h, 0, m, 0, hist(3), s)?;
        }
        b.vec_add(pf.cur, pf.ffn_inp, pf.sconv_tmp, m * h, s)?;
        Ok(ids)
    }

    fn upload_desc(&mut self, words: &[u32]) -> Result<()> {
        let s = self.tide.compute_stream;
        let pf = self.pf.as_mut().unwrap();
        let i = pf.desc_cursor;
        pf.desc_cursor = (i + 1) % pf.desc_host.len();
        let (host, ev, used) = pf.desc_host[i];
        if used {
            self.backend.sync_event(ev)?;
        }
        let bytes = words.len() * 4;
        assert!(bytes <= DESC_BYTES, "MoE descriptor overflow");
        unsafe {
            let dst = self.backend.pinned_as_mut_slice(host)?;
            let src = std::slice::from_raw_parts(words.as_ptr() as *const u8, bytes);
            dst[..bytes].copy_from_slice(src);
        }
        self.backend.memcpy_h2d_async(pf.desc, 0, host, 0, bytes, s)?;
        self.backend.record_event(ev, s)?;
        pf.desc_host[i].2 = true;
        Ok(())
    }

    /// Plans and runs the expert-grouped MoE for one physical layer over `m` rows.
    fn prefill_moe(&mut self, ip: usize, m: usize) -> Result<Vec<u32>> {
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let n_exp = self.cfg.n_routed_experts;
        let route = self.pf.as_ref().unwrap().route;
        let mut rb = vec![0u8; m * 16 * 4];
        self.backend.memcpy_d2h_sync(&mut rb, route.slice(0, m * 64), 0, s)?;
        let word = |i: usize| u32::from_le_bytes([rb[4 * i], rb[4 * i + 1], rb[4 * i + 2], rb[4 * i + 3]]);
        let phys = PhysicalLayerId(ip as u16);
        let mut ids = vec![0u32; m * 8];
        let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); n_exp];
        for r in 0..m {
            let mut ri = [0u16; 8];
            let mut rw = [0f32; 8];
            for k in 0..8 {
                let e = word(r * 16 + k) as usize;
                ids[r * 8 + k] = e as u32;
                ri[k] = e as u16;
                rw[k] = f32::from_bits(word(r * 16 + 8 + k));
                by_expert[e.min(n_exp - 1)].push((r * 8 + k) as u32);
            }
            self.tide.scheduler.record_prefill_routing(phys, &ri, &rw);
        }

        // Per-pair normalized routing share (a row's 8 shares sum to 1).
        let mut row_sum = vec![0f32; m];
        for r in 0..m {
            row_sum[r] = (0..8).map(|k| f32::from_bits(word(r * 16 + 8 + k))).sum::<f32>().max(1e-20);
        }
        let share = |pr: u32| f32::from_bits(word((pr as usize >> 3) * 16 + 8 + (pr as usize & 7))) / row_sum[pr as usize >> 3];

        // Resident experts address their published slot; missing ones are staged through scratch
        // waves, or skipped per the rescue policy (renormalizing each affected row).
        //   exact     - stage every missing expert (prefill is exact)
        //   threshold - stage a missing expert only if its summed share over this chunk is at
        //               least `prefill_min_share` tokens' worth (one rare pair costs a full slab copy)
        //   disabled  - stage nothing
        let min_share: Option<f32> = match self.tide.scheduler.tide_cfg.rescue_policy {
            smalltide_core::RescuePolicy::Exact => None,
            smalltide_core::RescuePolicy::Threshold(_) => Some(self.prefill_min_share),
            smalltide_core::RescuePolicy::Disabled => Some(f32::INFINITY),
        };
        let mut resident: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut missing: Vec<(ExpertId, Vec<u32>)> = Vec::new();
        let mut skipped_pairs: Vec<u32> = Vec::new();
        let mut used_scratch = false;
        for (e, prs) in by_expert.into_iter().enumerate() {
            if prs.is_empty() {
                continue;
            }
            let id = ExpertId { layer: phys, expert: e as u16 };
            let entry = self.tide.scheduler.cache.entry(id);
            let slot = self.tide.scheduler.cache.gpu_slot_table()[self.tide.scheduler.cache.flat_index(id)];
            // Use BOTH resident experts AND still-live scratch-ring experts in Wave 0 BEFORE
            // Wave 1 unloads/overwrites the scratch ring!
            if slot != SlotIndex::MISSING.0 && !entry.vram_pending {
                if entry.is_scratch_slot {
                    used_scratch = true;
                }
                resident.push((slot, prs));
            } else if min_share.is_some_and(|t| prs.iter().map(|&pr| share(pr)).sum::<f32>() < t) {
                skipped_pairs.extend_from_slice(&prs);
            } else {
                missing.push((id, prs));
            }
        }
        let tel = &mut self.tide.scheduler.telemetry;
        tel.prefill_weight_total += m as f64;
        tel.prefill_weight_skipped += skipped_pairs.iter().map(|&pr| share(pr) as f64).sum::<f64>();
        // Skipped pairs: weight 0, the rest of the row rescaled to the original total. The
        // modified route rides along in every wave's descriptor upload.
        let route_override: Option<Vec<u32>> = if skipped_pairs.is_empty() {
            None
        } else {
            let mut rw: Vec<u32> = (0..m * 16).map(word).collect();
            let mut dropped = vec![0f32; m];
            for &pr in &skipped_pairs {
                let (r, k) = (pr as usize >> 3, pr as usize & 7);
                dropped[r] += f32::from_bits(rw[r * 16 + 8 + k]);
                rw[r * 16 + 8 + k] = 0f32.to_bits();
            }
            for r in 0..m {
                if dropped[r] > 0.0 {
                    let kept = row_sum[r] - dropped[r];
                    let scale = if kept > 1e-20 { row_sum[r] / kept } else { 0.0 };
                    for k in 0..8 {
                        let w = f32::from_bits(rw[r * 16 + 8 + k]);
                        rw[r * 16 + 8 + k] = (w * scale).to_bits();
                    }
                }
            }
            Some(rw)
        };
        let is_mx = self.cfg.is_mxfp8_layer(ip);
        let (pool, stride, cls) = if is_mx {
            (self.tide.vram_pool_mxfp8, self.tide.mxfp8_layout.aligned_slab_bytes, ExpertClass::Mxfp8)
        } else {
            (self.tide.vram_pool_nvfp4, self.tide.nvfp4_layout.aligned_slab_bytes, ExpertClass::Nvfp4)
        };
        // Prefill stages a wave's missing experts in the first slots of the rescue ring, 8 at a
        // time whatever the ring's size (`SMALLTIDE_RESCUE_RING_*` is for decode).
        let ring = (self.tide.scheduler.cache.scratch_ring(cls).1 as usize).min(smalltide_tide::runtime::DEFAULT_SCRATCH_RING_SLOTS);

        let (out, mid, tmp, x) = {
            let pf = self.pf.as_ref().unwrap();
            (pf.moe_out, pf.moe_mid, pf.moe_tmp, pf.moe_norm)
        };
        if route_override.is_some() {
            // Skipped pairs are never written by the expert kernels: their tmp rows must be 0.
            self.backend.zero_f32(tmp, m * 8 * h, s)?;
        }
        // Wave 0: resident experts (plus any still-live scratch-ring experts we use BEFORE
        // unloading the scratch ring!). When Wave 0 does not touch the scratch ring, fence before
        // Wave 0 so Wave 1 DMA overlaps Wave 0; when Wave 0 reads unloading scratch experts, run
        // Wave 0 first and release/fence the scratch ring right before Wave 1 overwrites it.
        self.backend.seg_mark("pf: moe kernels (+staging)", s);
        let split = !resident.is_empty() && !missing.is_empty();
        let mut scratch_released = false;
        if split && !used_scratch {
            self.tide.release_scratch_ring_for_class(cls)?;
            scratch_released = true;
            self.tide.fence_scratch()?;
        }
        let mut next_missing = 0usize;
        let mut wave = 0usize;
        while wave == 0 || next_missing < missing.len() {
            let mut items: Vec<(u32, &[u32])> = Vec::new();
            if wave == 0 {
                items.extend(resident.iter().map(|(sl, p)| (*sl, p.as_slice())));
            }
            if !(split && wave == 0) && next_missing < missing.len() {
                if !scratch_released {
                    self.tide.release_scratch_ring_for_class(cls)?;
                    scratch_released = true;
                }
                let take = (missing.len() - next_missing).min(ring);
                let batch = &missing[next_missing..next_missing + take];
                let ids_to_stage: Vec<ExpertId> = batch.iter().map(|(id, _)| *id).collect();
                let staged = if split && !used_scratch && next_missing == 0 {
                    self.tide.stage_into_scratch_fenced(&ids_to_stage)?
                } else {
                    self.tide.stage_into_scratch(&ids_to_stage)?
                };
                next_missing += take;
                items.extend(staged.iter().zip(batch).map(|(sl, (_, p))| (sl.0, p.as_slice())));
            }
            wave += 1;
            if items.is_empty() {
                continue;
            }
            let mut work: Vec<u32> = Vec::new();
            let mut pairs: Vec<u32> = Vec::new();
            let pairs_per_item = if std::env::var_os("SMALLTIDE_MOE_SCALAR").is_none()
                && std::env::var_os("SMALLTIDE_MOE_WMMA1").is_none()
                && std::env::var_os("SMALLTIDE_MOE_M16").is_none()
            {
                MOE_PAIRS_PER_ITEM_M64
            } else {
                MOE_PAIRS_PER_ITEM
            };
            for (slot, prs) in items {
                for chunk in prs.chunks(pairs_per_item) {
                    work.extend_from_slice(&[slot, pairs.len() as u32, chunk.len() as u32, 0]);
                    pairs.extend_from_slice(chunk);
                }
            }
            let n_work = work.len() / 4;
            let pairs_off = work.len() * 4;
            let mut words = work;
            words.extend_from_slice(&pairs);
            let route_off = words.len() * 4;
            if let Some(rw) = &route_override {
                words.extend_from_slice(rw);
            }
            self.upload_desc(&words)?;
            let desc = self.pf.as_ref().unwrap().desc;
            let route_arg = if route_override.is_some() { desc.slice(route_off, m * 64) } else { route };
            self.backend.moe_grouped(
                out,
                mid,
                tmp,
                x,
                route_arg,
                pool,
                desc.slice(0, words.len() * 4),
                GroupedMoeParams {
                    rows: m,
                    n_work,
                    hidden_size: h,
                    intermediate_size: self.cfg.intermediate_size,
                    expert_stride_bytes: stride,
                    pairs_offset_bytes: pairs_off,
                },
                false,
                s,
            )?;
        }
        self.backend.moe_reduce_rows(out, tmp, h, m, s)?;
        Ok(ids)
    }
}
