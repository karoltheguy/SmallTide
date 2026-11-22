//! Rows-batched chunked prefill for `Inkling2Engine` (up to `PREFILL_CHUNK` tokens per chunk).
//!
//! Same math as `forward_token`, but every projection is a rows GEMM (weights read once per chunk
//! instead of once per token), attention is a multi-row causal kernel over the FP8 KV cache, short
//! convolutions scan the chunk sequentially per channel, and the MoE is expert-grouped: the
//! chunk's (row, k) assignments are grouped by expert on the host and each expert's weights are
//! streamed once for all of its rows. Experts that are not resident in VRAM are staged into the
//! Scratch Rescue slots in waves, so prefill is always exact (no skipped experts).

use crate::engine::{Inkling2Engine, PREFILL_CHUNK};
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
const DESC_BYTES: usize = 160 * 1024;
const DESC_RING: usize = 4;

/// Rows-batched activation buffers (sized for `PREFILL_CHUNK` rows).
pub(crate) struct PrefillScratch {
    cur: DeviceBuf,
    embed: DeviceBuf,
    inp_sa: DeviceBuf,
    ffn_inp: DeviceBuf,
    prelude_cur: DeviceBuf,
    bank_out: DeviceBuf,
    blocks: DeviceBuf,   // [8][M][H]
    bank: DeviceBuf,     // [8][M][H]
    messages: DeviceBuf, // [4][M][H]
    in_qkv: DeviceBuf,   // [3][M][H]
    ctrl_qkv: DeviceBuf, // [3][M][H]
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
    moe_tmp: DeviceBuf,       // [M*8][H]
    moe_out: DeviceBuf,
    desc: DeviceBuf,
    embed_host: PinnedBuf,
    desc_host: Vec<(PinnedBuf, Event, bool)>,
    desc_cursor: usize,
}

impl PrefillScratch {
    pub(crate) fn new<B: ComputeBackend>(b: &B, cfg: &smalltide_core::Inkling2Config) -> Result<Self> {
        let m = PREFILL_CHUNK;
        let h = cfg.hidden_size;
        let qd = cfg.num_attention_heads * cfg.head_dim;
        let kvd = cfg.num_key_value_heads * cfg.head_dim;
        let f = |n: usize| b.alloc_device(n * 4);
        let mut desc_host = Vec::with_capacity(DESC_RING);
        for _ in 0..DESC_RING {
            desc_host.push((b.alloc_pinned(DESC_BYTES)?, b.create_event()?, false));
        }
        Ok(Self {
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
            moe_tmp: f(m * 8 * h)?,
            moe_out: f(m * h)?,
            desc: b.alloc_device(DESC_BYTES)?,
            embed_host: b.alloc_pinned(m * h * 4)?,
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
        let _ = b.free_pinned(self.embed_host);
        for (p, e, _) in self.desc_host {
            let _ = b.free_pinned(p);
            let _ = b.destroy_event(e);
        }
    }
}

/// What the rows of a rows-batched forward are.
#[derive(Clone, Copy)]
pub(crate) enum RowsMode<'a> {
    /// Consecutive tokens of the active lane at positions `pos0..pos0 + m`.
    Prefill { pos0: usize },
    /// One decode token per row: `(lane, pos)`. Lane-specific state (KV caches, short-conv
    /// histories) is per row; everything weight-heavy runs once over all rows.
    Decode { rows: &'a [(usize, usize)] },
}

/// Row `r` of a `[m][n]` f32 buffer.
#[inline]
fn row(buf: DeviceBuf, r: usize, n: usize) -> DeviceBuf {
    buf.slice(r * n * 4, n * 4)
}

/// `[nq][m][H]` slice helper.
#[inline]
fn part(buf: DeviceBuf, idx: usize, m: usize, h: usize) -> DeviceBuf {
    buf.slice(idx * m * h * 4, m * h * 4)
}

impl<B: ComputeBackend> Inkling2Engine<B> {
    /// Prefills `tokens` at positions `pos0..` in chunks of up to `PREFILL_CHUNK` rows. When
    /// `need_logits`, the logits of the last token are left in the logits buffer for sampling.
    pub fn prefill(&mut self, tokens: &[u32], pos0: usize, need_logits: bool) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if self.pf.is_none() {
            self.pf = Some(PrefillScratch::new(&*self.backend, &self.cfg)?);
        }
        // Token boundary bookkeeping (same as forward_token) and a free scratch ring.
        self.run_deferred_tide_step()?;
        let published = self.tide.poll_pending()?;
        self.tide.scheduler.telemetry.stage1_async_published += published as u64;
        self.tide.release_scratch_ring()?;
        self.route_log_inflight = false;
        let mut off = 0usize;
        while off < tokens.len() {
            let m = (tokens.len() - off).min(PREFILL_CHUNK);
            let last = off + m == tokens.len();
            if off > 0 {
                // Prompt-aware loading: swaps queued after the previous chunk that have landed
                // are published now (non-blocking), the next ones are planned from this prompt's
                // routing so far and overlap the next chunk.
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
            off += m;
        }
        Ok(())
    }

    /// Rows-batched forward of `tokens` (one per row) per `mode`. Prefill: logits of the last row
    /// into the logits buffer when `need_logits`. Decode: logits of every row into the batch
    /// logits buffer (`need_logits` ignored).
    pub(crate) fn rows_forward(&mut self, tokens: &[u32], mode: RowsMode, need_logits: bool) -> Result<()> {
        let pos0 = match mode {
            RowsMode::Prefill { pos0 } => pos0,
            RowsMode::Decode { .. } => 0,
        };
        let decode = matches!(mode, RowsMode::Decode { .. });
        self.rows_decode = decode;
        let s = self.tide.compute_stream;
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let m = tokens.len();
        let (cur, embed, embed_host) = {
            let pf = self.pf.as_ref().unwrap();
            (pf.cur, pf.embed, pf.embed_host)
        };
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
        // The pinned embed buffer is reused by the next chunk: make sure this copy has landed.
        self.backend.sync_stream(s)?;
        self.backend.rms_norm(cur, embed, self.dense.embed_norm, m, h, eps, s)?;

        let n_log = self.layer_states.len();
        let dumping = self.dump.is_some() && !decode;
        let mut dump_hidden = dumping.then(|| vec![0f32; n_log * m * h]);
        let mut dump_routes = dumping.then(|| vec![0u32; n_log * m * 8]);

        let start = self.cfg.loop_start_layer;
        let end = self.cfg.loop_end_exclusive;
        let core_len = end - start;
        let mut nb = 0usize;
        let mut run = |eng: &mut Self, il: usize, ip: usize, nb: &mut usize, msgs: Option<DeviceBuf>| -> Result<()> {
            let routes = eng.prefill_layer(il, ip, mode, m, nb, msgs)?;
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
                        dr[(il * m + r) * 8 + k] = routes[r * 8 + k];
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
            let vocab = self.cfg.unpadded_vocab_size;
            let logits = self.batch_logits.expect("batch logits");
            self.backend.gemm_fp8_rows_decode(logits, moe_norm, self.dense.unembed_fp8.data, self.dense.unembed_fp8.row_scales, m, vocab, h, s)?;
            if self.cfg.final_logit_softcapping > 0.0 {
                self.backend.final_logit_softcap(logits, m * vocab, self.cfg.final_logit_softcapping, s)?;
            }
            return Ok(());
        }
        let logits_needed = need_logits || self.dump.is_some();
        if logits_needed {
            // Final mix + norm for all rows (cheap), lm_head for the last row only.
            let pf = self.pf.as_ref().unwrap();
            let (moe_in, moe_norm) = (pf.moe_in, pf.moe_norm);
            let bv = Some(blocks.slice(0, nb * m * h * 4));
            self.backend.attn_res_mix_rows(moe_in, bv, cur, self.dense.output_res_query, nb, 1, h, eps, m, s)?;
            self.backend.rms_norm(moe_norm, moe_in, self.dense.final_norm, m, h, eps, s)?;
            let last = moe_norm.slice((m - 1) * h * 4, h * 4);
            self.backend.gemv_fp8(
                self.scratch.logits,
                last,
                self.dense.unembed_fp8.data,
                self.dense.unembed_fp8.row_scales,
                self.cfg.unpadded_vocab_size,
                h,
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
            self.backend.gemm_fp8_rows_decode(y, x, w.data, w.row_scales, m, w.n_rows, w.k_cols, s)
        } else {
            self.backend.gemm_fp8_rows(y, x, w.data, w.row_scales, m, w.n_rows, w.k_cols, s)
        }
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
        let (pos0, lanes) = match mode {
            RowsMode::Prefill { pos0 } => (pos0, None),
            RowsMode::Decode { rows } => (0, Some(rows)),
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
        b.vec_copy(pf.inp_sa, pf.cur, m * h, s)?;
        let blocks_view = (*nb > 0).then(|| pf.blocks.slice(0, *nb * hb));

        // 1. attention-residual mix
        let (in0, in1, in2) = if let Some(attn_q) = layer.attn_res_query {
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
            for j in 0..3 {
                self.control_rows(c[j], ins[j], part(msgs, j, m, h), &ctrls[j], m, s)?;
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
        let (nq, nk, nv) = if is_local {
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
        self.mm_rows(pf.q, nq, &layer.wq, m, s)?;
        self.mm_rows(pf.k_raw, nk, &layer.wk, m, s)?;
        self.mm_rows(pf.v_raw, nv, &layer.wv, m, s)?;
        self.mm_rows(pf.r, nq, &layer.wr, m, s)?;
        let st = &self.layer_states[il];
        let taps = nkv * cfg.sconv_kernel_size;
        let k_dyn = match layer.k_sconv_dyn_proj {
            Some(w) => {
                b.gemm_f32_rows(pf.k_dyn, nk, w, m, taps, h, s)?;
                Some(pf.k_dyn)
            }
            None => None,
        };
        match lanes {
            None => b.sconv_k4_rows(pf.k, st.sconv_k, pf.k_raw, layer.k_sconv_static, k_dyn, kv_dim, hd, m, taps, s)?,
            Some(rows) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.k, r, kv_dim), ls.sconv_k, row(pf.k_raw, r, kv_dim), layer.k_sconv_static,
                        k_dyn.map(|d| row(d, r, taps)), kv_dim, hd, 1, taps, s)?;
                }
            }
        }
        let v_dyn = match layer.v_sconv_dyn_proj {
            Some(w) => {
                b.gemm_f32_rows(pf.v_dyn, nv, w, m, taps, h, s)?;
                Some(pf.v_dyn)
            }
            None => None,
        };
        match lanes {
            None => b.sconv_k4_rows(pf.v, st.sconv_v, pf.v_raw, layer.v_sconv_static, v_dyn, kv_dim, hd, m, taps, s)?,
            Some(rows) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.v, r, kv_dim), ls.sconv_v, row(pf.v_raw, r, kv_dim), layer.v_sconv_static,
                        v_dyn.map(|d| row(d, r, taps)), kv_dim, hd, 1, taps, s)?;
                }
            }
        }
        // 5. clips, value norm, tau (global layers)
        let tau_at = |p: usize| (!is_local).then_some((p, cfg.log_scaling_alpha, cfg.log_scaling_n_floor as f32));
        match lanes {
            Some(rows) if !is_local => {
                // tau depends on each row's own position.
                for (r, &(_, p)) in rows.iter().enumerate() {
                    b.head_l2_clip_rows(row(pf.q, r, q_dim), Some(layer.q_l2_clip_scale), nqh, hd, nqh, cfg.qk_l2_clip_threshold, tau_at(p), s)?;
                    b.head_l2_clip_rows(row(pf.r, r, nqh * cfg.d_rel), Some(layer.rel_l2_clip_scale), nqh, cfg.d_rel, nqh,
                        cfg.rel_l2_clip_threshold, tau_at(p), s)?;
                }
            }
            _ => {
                b.head_l2_clip_rows(pf.q, Some(layer.q_l2_clip_scale), m * nqh, hd, nqh, cfg.qk_l2_clip_threshold, tau_at(pos0), s)?;
                b.head_l2_clip_rows(pf.r, Some(layer.rel_l2_clip_scale), m * nqh, cfg.d_rel, nqh, cfg.rel_l2_clip_threshold, tau_at(pos0), s)?;
            }
        }
        b.head_l2_clip_rows(pf.k, None, m * nkv, hd, nkv, cfg.qk_l2_clip_threshold, None, s)?;
        b.value_norm_no_gamma(pf.v, m * nkv, hd, eps, s)?;
        // 6. sinks, relative bias, KV store, attention
        b.seg_mark("pf: sinks,rel,kv,attention", s);
        b.gemm_f32_rows(pf.sinks, nq, layer.attn_sink, m, nqh, h, s)?;
        b.rel_table(pf.rel_table, pf.r, layer.rel_logits_proj, m * nqh, cfg.rel_extent, s)?;
        match lanes {
            None => {
                b.kv_store_fp8_rows(st.k_cache, st.k_scales, st.v_cache, st.v_scales, pf.k, pf.v, nkv, st.max_seq, hd, pos0, m, is_local, s)?;
                b.attn_prefill_fp8(
                    pf.attn_out, pf.q, st.k_cache, st.k_scales, st.v_cache, st.v_scales,
                    Some(pf.rel_table), Some(pf.sinks),
                    PrefillAttnParams {
                        n_q_heads: nqh,
                        n_kv_heads: nkv,
                        max_seq: st.max_seq,
                        window: if is_local { st.window } else { 0 },
                        pos0,
                        rows: m,
                        rel_extent: cfg.rel_extent,
                        inv_sqrt_d: 1.0 / (hd as f32).sqrt(),
                    },
                    s,
                )?;
            }
            Some(rows) => {
                // Each row attends over its own lane's cache (decode flash attention).
                for (r, &(lane, p)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.kv_store_fp8_rows(ls.k_cache, ls.k_scales, ls.v_cache, ls.v_scales, row(pf.k, r, kv_dim), row(pf.v, r, kv_dim),
                        nkv, ls.max_seq, hd, p, 1, is_local, s)?;
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
                            n_q_heads: nqh,
                            n_kv_heads: nkv,
                            max_seq: ls.max_seq,
                            q_pos: p,
                            rel_extent: cfg.rel_extent,
                            is_local,
                            num_splits,
                            inv_sqrt_d: 1.0 / (hd as f32).sqrt(),
                            window: if is_local { ls.window } else { 0 },
                        },
                        s,
                    )?;
                }
            }
        }
        // 7. output gate, wo, post-attention conv
        b.seg_mark("pf: gate,wo,conv,mix,router", s);
        self.mm_rows(pf.gate_out, nq, &layer.gate_attn, m, s)?;
        b.gate_attn_sigmoid_mul(pf.attn_out, pf.attn_out, pf.gate_out, m * q_dim, s)?;
        self.mm_rows(pf.o_raw, pf.attn_out, &layer.wo, m, s)?;
        match lanes {
            None => b.sconv_k4_rows(pf.sconv_tmp, st.sconv_attn, pf.o_raw, layer.attn_sconv, None, h, 0, m, 0, s)?,
            Some(rows) => {
                for (r, &(lane, _)) in rows.iter().enumerate() {
                    let ls = self.lane_layer_state(lane, il);
                    b.sconv_k4_rows(row(pf.sconv_tmp, r, h), ls.sconv_attn, row(pf.o_raw, r, h), layer.attn_sconv, None, h, 0, 1, 0, s)?;
                }
            }
        }
        // 8. block boundary
        if ip % cfg.n_layers_per_attn_res_block == 0 {
            b.vec_copy(pf.blocks.slice(*nb * hb, hb), pf.inp_sa, m * h, s)?;
            *nb += 1;
            b.vec_copy(pf.cur, pf.sconv_tmp, m * h, s)?;
        } else {
            b.vec_add(pf.cur, pf.inp_sa, pf.sconv_tmp, m * h, s)?;
        }
        b.vec_copy(pf.ffn_inp, pf.cur, m * h, s)?;
        // 9. MLP input mix -> controller -> norm -> router
        let bv = Some(pf.blocks.slice(0, *nb * hb));
        b.attn_res_mix_rows(pf.moe_in, bv, pf.cur, layer.mlp_res_query, *nb, 1, h, eps, m, s)?;
        let moe_x = if let Some(msgs) = messages {
            let ctrls = layer.controllers.as_ref().unwrap();
            self.control_rows(pf.moe_ctrl, pf.moe_in, part(msgs, 3, m, h), &ctrls[3], m, s)?;
            pf.moe_ctrl
        } else {
            pf.moe_in
        };
        let pf = self.pf.as_ref().unwrap();
        let layer = &self.dense.layers[ip];
        b.rms_norm(pf.moe_norm, moe_x, layer.mlp_norm, m, h, eps, s)?;
        b.gemm_f32_rows(pf.router_logits, pf.moe_norm, layer.router_weight, m, cfg.n_routed_experts, h, s)?;
        b.router_topk8_rows(pf.route, pf.router_logits, Some(layer.router_bias), cfg.route_scale, m, 16, s)?;

        // 10. expert-grouped MoE (host plan from the routed ids)
        b.seg_mark("pf: moe route sync + host plan", s);
        if let Some(rows) = lanes {
            // Batched decode MoE: device-side routing, experts deduplicated across rows, no host
            // sync; per-row renormalized sum + MLP short conv + residual.
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
            b.moe_decode_rows(pf.moe_mid, pf.moe_tmp, pf.moe_norm, pool, slot_table, pf.route, m, params, s)?;
            let log = self.batch_route_log.expect("batch route log");
            b.vec_copy(log.slice(il * crate::engine::MAX_BATCH_ROWS * 64, m * 64), pf.route.slice(0, m * 64), m * 16, s)?;
            let layer = &self.dense.layers[ip];
            for (r, &(lane, _)) in rows.iter().enumerate() {
                let ls = self.lane_layer_state(lane, il);
                b.moe_decode_post(
                    row(pf.cur, r, h),
                    pf.moe_tmp.slice(r * 8 * h * 4, 8 * h * 4),
                    slot_table,
                    pf.route.slice(r * 64, 32),
                    pf.route.slice(r * 64 + 32, 32),
                    cfg.num_experts_per_tok,
                    Some((ls.sconv_mlp, layer.mlp_sconv, row(pf.ffn_inp, r, h))),
                    h,
                    s,
                )?;
            }
            return Ok(Vec::new());
        }
        let ids = self.prefill_moe(ip, m)?;
        b.seg_mark("pf: moe post", s);

        let pf = self.pf.as_ref().unwrap();
        let st = &self.layer_states[il];
        let layer = &self.dense.layers[ip];
        b.sconv_k4_rows(pf.sconv_tmp, st.sconv_mlp, pf.moe_out, layer.mlp_sconv, None, h, 0, m, 0, s)?;
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
        for (e, prs) in by_expert.into_iter().enumerate() {
            if prs.is_empty() {
                continue;
            }
            let id = ExpertId { layer: phys, expert: e as u16 };
            let slot = self.tide.scheduler.cache.gpu_slot_table()[self.tide.scheduler.cache.flat_index(id)];
            if slot != SlotIndex::MISSING.0 && !self.tide.scheduler.cache.entry(id).is_scratch_slot {
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
        let ring = self.tide.scheduler.cache.scratch_ring(cls).1 as usize;

        let (out, mid, tmp, x) = {
            let pf = self.pf.as_ref().unwrap();
            (pf.moe_out, pf.moe_mid, pf.moe_tmp, pf.moe_norm)
        };
        if route_override.is_some() {
            // Skipped pairs are never written by the expert kernels: their tmp rows must be 0.
            self.backend.zero_f32(tmp, m * 8 * h, s)?;
        }
        // Wave 0: resident experts plus the first `ring` missing ones (staged into scratch);
        // further waves stage the next `ring` missing experts each.
        self.backend.seg_mark("pf: moe kernels (+staging)", s);
        let mut next_missing = 0usize;
        let mut first = true;
        while first || next_missing < missing.len() {
            let take = (missing.len() - next_missing).min(ring);
            let batch = &missing[next_missing..next_missing + take];
            next_missing += take;
            let ids_to_stage: Vec<ExpertId> = batch.iter().map(|(id, _)| *id).collect();
            let staged = self.tide.stage_into_scratch(&ids_to_stage)?;
            let mut items: Vec<(u32, &[u32])> = Vec::new();
            if first {
                items.extend(resident.iter().map(|(sl, p)| (*sl, p.as_slice())));
            }
            items.extend(staged.iter().zip(batch).map(|(sl, (_, p))| (sl.0, p.as_slice())));
            first = false;
            if items.is_empty() {
                continue;
            }
            let mut work: Vec<u32> = Vec::new();
            let mut pairs: Vec<u32> = Vec::new();
            for (slot, prs) in items {
                for chunk in prs.chunks(MOE_PAIRS_PER_ITEM) {
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
