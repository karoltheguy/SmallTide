//! Dense text weight loader for `thinkingmachines/Inkling-2-Mini` (`--fp8-dense`).
//!
//! Reads all non-expert `model.llm.*` tensors from `model-00001-of-00001.safetensors`:
//! - Keeps `model.llm.embed.weight` in host RAM (BF16, ~883 MiB) to save VRAM
//! - Quantizes all BF16 linear projections + `model.llm.unembed.weight` (`lm_head`) to
//!   FP8 E4M3FN + per-row F32 scales (~1.5 GiB total vs ~3.0 GiB BF16!)
//! - Loads small F32/BF16 vectors (norms, `attn_res`, `sconv`, `rel_logits_proj`, `clip`, routers) as F32.

use crate::quant::{bf16_to_f32, quantize_bf16_bytes_to_fp8_rows};
use crate::safetensors::{inspect_safetensors_header, SafetensorsHeader};
use smalltide_core::{Inkling2Config, Result, SmallTideError};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A row-major `[n_rows, k_cols]` matrix quantized on load to FP8 E4M3FN + per-row F32 scales.
#[derive(Debug, Clone)]
pub struct Fp8MatrixHost {
    pub n_rows: usize,
    pub k_cols: usize,
    pub data: Vec<u8>,
    pub row_scales: Vec<f32>,
}

impl Fp8MatrixHost {
    pub fn byte_size(&self) -> usize {
        self.data.len() + self.row_scales.len() * 4
    }
}

/// Loop Core rank-64 controller weights for one control channel (`attn`, `k`, `v`, or `mlp`).
#[derive(Debug, Clone)]
pub struct LoopControllerHost {
    pub down: Fp8MatrixHost,   // [64, 4608]
    pub forget: Fp8MatrixHost, // [2304, 64]
    pub gain: Fp8MatrixHost,   // [2304, 64]
}

/// All always-resident dense weights for one physical layer (0..23).
#[derive(Debug, Clone)]
pub struct DenseLayerHost {
    pub physical_layer: usize,
    pub is_local: bool,

    // Attention linear projections (FP8 E4M3FN + row scales)
    pub wq: Fp8MatrixHost,        // [3072, 2304]
    pub wk: Fp8MatrixHost,        // [1024, 2304]
    pub wv: Fp8MatrixHost,        // [1024, 2304]
    pub wo: Fp8MatrixHost,        // [2304, 3072]
    pub gate_attn: Fp8MatrixHost, // [3072, 2304]
    pub wr: Fp8MatrixHost,        // [384, 2304]
    pub attn_sink: Vec<f32>,      // [24, 2304] F32 (tiny 216 KB)

    // Clipping & relative position bias table (F32)
    pub q_l2_clip_scale: Vec<f32>,   // [24 * 128]
    pub rel_l2_clip_scale: Vec<f32>, // [24 * 16]
    pub rel_logits_proj: Vec<f32>,   // [16 * 1024]

    // Short convolutions (k = 4)
    pub k_sconv_static: Vec<f32>,              // [1024 * 4]
    pub v_sconv_static: Vec<f32>,              // [1024 * 4]
    pub k_sconv_dyn_proj: Option<Vec<f32>>,    // global only: [32, 2304]
    pub v_sconv_dyn_proj: Option<Vec<f32>>,    // global only: [32, 2304]
    pub attn_sconv: Vec<f32>,                  // [2304 * 4]
    pub mlp_sconv: Vec<f32>,                   // [2304 * 4]

    // Norms (F32)
    pub attn_norm: Option<Vec<f32>>,    // local layers
    pub q_input_norm: Option<Vec<f32>>, // global layers
    pub k_input_norm: Option<Vec<f32>>, // global layers
    pub v_input_norm: Option<Vec<f32>>, // global layers
    pub mlp_norm: Vec<f32>,

    // MoE Router (F32 for high-precision Top-8 gating!)
    pub router_weight: Vec<f32>, // [128, 2304]
    pub router_bias: Vec<f32>,   // [128]

    // Attention residual queries (F32)
    pub attn_res_query: Option<Vec<f32>>, // None for layer 0; [2304] local, [3*2304] global
    pub mlp_res_query: Vec<f32>,          // [2304]

    // Loop Core controllers & per-layer [4, 2304] cross-pass message queries (`Some` only for physical layers 6..17)
    pub controllers: Option<[LoopControllerHost; 4]>, // [attn, k, v, mlp]
    pub loop_msg_queries: Option<Vec<f32>>,           // [4 * 2304]: [Q/attn, K, V, FFN/mlp]
}

/// Complete always-resident dense text weights of Inkling-2-Mini (ready to upload to VRAM,
/// except `embed_bf16` which stays in Host RAM).
pub struct Inkling2DenseHost {
    /// Kept in Host RAM (`[201024, 2304]` raw BF16 bytes = 883 MiB).
    pub embed_bf16: Vec<u8>,
    pub embed_norm: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub output_res_query: Vec<f32>,
    pub middle_cycle_cross_pass_queries: Vec<f32>,    // [24, 2304]
    pub middle_cycle_cross_pass_kv_queries: Vec<f32>, // [24, 2304]
    /// `lm_head` quantized to FP8 E4M3FN + row scales (`~442 MiB` vs `883 MiB` BF16!).
    pub unembed_fp8: Fp8MatrixHost,
    pub layers: Vec<DenseLayerHost>,
}

impl Inkling2DenseHost {
    /// Total bytes required in VRAM for all always-resident FP8 dense + F32 small weights
    /// (excluding `embed_bf16` which stays in Host RAM).
    pub fn total_vram_bytes(&self) -> usize {
        let mut b = self.unembed_fp8.byte_size()
            + (self.embed_norm.len()
                + self.final_norm.len()
                + self.output_res_query.len()
                + self.middle_cycle_cross_pass_queries.len()
                + self.middle_cycle_cross_pass_kv_queries.len())
                * 4;
        for l in &self.layers {
            b += l.wq.byte_size()
                + l.wk.byte_size()
                + l.wv.byte_size()
                + l.wo.byte_size()
                + l.gate_attn.byte_size()
                + l.wr.byte_size();
            b += (l.attn_sink.len()
                + l.q_l2_clip_scale.len()
                + l.rel_l2_clip_scale.len()
                + l.rel_logits_proj.len()
                + l.k_sconv_static.len()
                + l.v_sconv_static.len()
                + l.attn_sconv.len()
                + l.mlp_sconv.len()
                + l.mlp_norm.len()
                + l.router_weight.len()
                + l.router_bias.len()
                + l.mlp_res_query.len())
                * 4;
            if let Some(v) = &l.k_sconv_dyn_proj { b += v.len() * 4; }
            if let Some(v) = &l.v_sconv_dyn_proj { b += v.len() * 4; }
            if let Some(v) = &l.attn_norm { b += v.len() * 4; }
            if let Some(v) = &l.q_input_norm { b += v.len() * 4; }
            if let Some(v) = &l.k_input_norm { b += v.len() * 4; }
            if let Some(v) = &l.v_input_norm { b += v.len() * 4; }
            if let Some(v) = &l.attn_res_query { b += v.len() * 4; }
            if let Some(ctrls) = &l.controllers {
                for c in ctrls {
                    b += c.down.byte_size() + c.forget.byte_size() + c.gain.byte_size();
                }
            }
            if let Some(q) = &l.loop_msg_queries {
                b += q.len() * 4;
            }
        }
        b
    }

    /// Loads all Inkling-2-Mini dense text tensors from `safetensors_path`, casting BF16 linears -> FP8.
    pub fn load_from_safetensors(cfg: &Inkling2Config, safetensors_path: &Path) -> Result<Self> {
        let hdr = inspect_safetensors_header(safetensors_path)?;
        let data_base = 8u64 + hdr.header_len;
        let mut file = File::open(safetensors_path)?;

        let embed_bf16 = read_raw_bytes(&mut file, &hdr, data_base, "model.llm.embed.weight")?;
        let embed_norm = read_f32_vec(&mut file, &hdr, data_base, "model.llm.embed_norm.weight")?;
        let final_norm = read_f32_vec(&mut file, &hdr, data_base, "model.llm.norm.weight")?;
        let output_res_query =
            read_f32_vec(&mut file, &hdr, data_base, "model.llm.attn_res.output_query")?;
        let middle_cycle_cross_pass_queries = read_f32_vec(
            &mut file,
            &hdr,
            data_base,
            "model.llm.middle_cycle_cross_pass_queries",
        )?;
        let middle_cycle_cross_pass_kv_queries = read_f32_vec(
            &mut file,
            &hdr,
            data_base,
            "model.llm.middle_cycle_cross_pass_kv_queries",
        )?;
        let unembed_fp8 =
            read_fp8_matrix(&mut file, &hdr, data_base, "model.llm.unembed.weight")?;

        let mut layers = Vec::with_capacity(cfg.num_physical_layers);
        for l in 0..cfg.num_physical_layers {
            let p = format!("model.llm.layers.{l}");
            let is_local = cfg.is_local_layer(l);

            let wq = read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.wq_du.weight"))?;
            let wk = read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.wk_dv.weight"))?;
            let wv = read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.wv_dv.weight"))?;
            let wo = read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.wo_ud.weight"))?;
            let gate_attn =
                read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.gate_attn.weight"))?;
            let wr = read_fp8_matrix(&mut file, &hdr, data_base, &format!("{p}.attn.wr_du.weight"))?;
            let attn_sink = read_f32_vec(
                &mut file,
                &hdr,
                data_base,
                &format!("{p}.attn.attn_sink_proj.weight"),
            )?;

            let q_l2_clip_scale =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.q_l2_clip.scale"))?;
            let rel_l2_clip_scale =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.rel_l2_clip.scale"))?;
            let rel_logits_proj_raw = read_f32_vec(
                &mut file,
                &hdr,
                data_base,
                &format!("{p}.attn.rel_logits_proj.proj"),
            )?;
            // Transpose [d_rel=16, rel_extent=1024] -> [rel_extent=1024, d_rel=16] for coalesced dot(R, proj[dist])
            let d_rel = cfg.d_rel;
            let rel_extent = cfg.rel_extent;
            let mut rel_logits_proj = vec![0.0f32; rel_extent * d_rel];
            for r in 0..d_rel {
                for dist in 0..rel_extent {
                    rel_logits_proj[dist * d_rel + r] = rel_logits_proj_raw[r * rel_extent + dist];
                }
            }

            let (k_sconv_static, v_sconv_static, k_sconv_dyn_proj, v_sconv_dyn_proj) = if is_local
            {
                (
                    read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.k_sconv.weight"))?,
                    read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.v_sconv.weight"))?,
                    None,
                    None,
                )
            } else {
                // Global `static_w` is stored as [4, 1024] with lag 0 = current t.
                // Flip axis 0 and transpose to [1024, 4] ([t-3, t-2, t-1, t]) to match `k_sconv.weight`!
                let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
                let k_raw = read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.k_sconv.static_w"))?;
                let v_raw = read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn.v_sconv.static_w"))?;
                let flip_transpose = |raw: &[f32]| -> Vec<f32> {
                    let mut out = vec![0.0f32; kv_dim * 4];
                    for c in 0..kv_dim {
                        for k in 0..4 {
                            out[c * 4 + k] = raw[(3 - k) * kv_dim + c];
                        }
                    }
                    out
                };
                (
                    flip_transpose(&k_raw),
                    flip_transpose(&v_raw),
                    Some(read_f32_vec(
                        &mut file,
                        &hdr,
                        data_base,
                        &format!("{p}.attn.k_sconv.weight_proj"),
                    )?),
                    Some(read_f32_vec(
                        &mut file,
                        &hdr,
                        data_base,
                        &format!("{p}.attn.v_sconv.weight_proj"),
                    )?),
                )
            };

            let attn_sconv =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn_sconv.weight"))?;
            let mlp_sconv =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.mlp_sconv.weight"))?;

            let (attn_norm, q_input_norm, k_input_norm, v_input_norm) = if is_local {
                (
                    Some(read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.attn_norm.weight"))?),
                    None,
                    None,
                    None,
                )
            } else {
                (
                    None,
                    Some(read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.q_input_norm.weight"))?),
                    Some(read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.k_input_norm.weight"))?),
                    Some(read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.v_input_norm.weight"))?),
                )
            };

            let mlp_norm = read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.mlp_norm.weight"))?;
            let router_weight =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.mlp.gate.weight"))?;
            let router_bias =
                read_f32_vec(&mut file, &hdr, data_base, &format!("{p}.mlp.gate.bias"))?;

            let attn_res_query = if l > 0 {
                Some(read_f32_vec(
                    &mut file,
                    &hdr,
                    data_base,
                    &format!("model.llm.attn_res.attn_queries.{}", l - 1),
                )?)
            } else {
                None
            };
            let mlp_res_query = read_f32_vec(
                &mut file,
                &hdr,
                data_base,
                &format!("model.llm.attn_res.mlp_queries.{l}"),
            )?;

            let (controllers, loop_msg_queries) = if cfg.is_loop_core_layer(l) {
                let mut load_ctrl = |name: &str| -> Result<LoopControllerHost> {
                    Ok(LoopControllerHost {
                        down: read_fp8_matrix(
                            &mut file,
                            &hdr,
                            data_base,
                            &format!("{p}.controllers.{name}.down_weight"),
                        )?,
                        forget: read_fp8_matrix(
                            &mut file,
                            &hdr,
                            data_base,
                            &format!("{p}.controllers.{name}.forget_weight"),
                        )?,
                        gain: read_fp8_matrix(
                            &mut file,
                            &hdr,
                            data_base,
                            &format!("{p}.controllers.{name}.gain_weight"),
                        )?,
                    })
                };
                // Per-layer [4, hidden] cross-pass message queries: [Q/attn, K, V, FFN/mlp]
                let i = l - cfg.loop_start_layer;
                let h = cfg.hidden_size;
                let mut mq = Vec::with_capacity(4 * h);
                mq.extend_from_slice(&middle_cycle_cross_pass_queries[(2 * i) * h..(2 * i + 1) * h]);
                mq.extend_from_slice(&middle_cycle_cross_pass_kv_queries[(2 * i) * h..(2 * i + 1) * h]);
                mq.extend_from_slice(&middle_cycle_cross_pass_kv_queries[(2 * i + 1) * h..(2 * i + 2) * h]);
                mq.extend_from_slice(&middle_cycle_cross_pass_queries[(2 * i + 1) * h..(2 * i + 2) * h]);
                (
                    Some([
                        load_ctrl("attn")?,
                        load_ctrl("k")?,
                        load_ctrl("v")?,
                        load_ctrl("mlp")?,
                    ]),
                    Some(mq),
                )
            } else {
                (None, None)
            };

            layers.push(DenseLayerHost {
                physical_layer: l,
                is_local,
                wq,
                wk,
                wv,
                wo,
                gate_attn,
                wr,
                attn_sink,
                q_l2_clip_scale,
                rel_l2_clip_scale,
                rel_logits_proj,
                k_sconv_static,
                v_sconv_static,
                k_sconv_dyn_proj,
                v_sconv_dyn_proj,
                attn_sconv,
                mlp_sconv,
                attn_norm,
                q_input_norm,
                k_input_norm,
                v_input_norm,
                mlp_norm,
                router_weight,
                router_bias,
                attn_res_query,
                mlp_res_query,
                controllers,
                loop_msg_queries,
            });
        }

        Ok(Self {
            embed_bf16,
            embed_norm,
            final_norm,
            output_res_query,
            middle_cycle_cross_pass_queries,
            middle_cycle_cross_pass_kv_queries,
            unembed_fp8,
            layers,
        })
    }
}

pub fn read_raw_bytes(
    file: &mut File,
    hdr: &SafetensorsHeader,
    data_base: u64,
    name: &str,
) -> Result<Vec<u8>> {
    let entry = hdr
        .tensors
        .get(name)
        .ok_or_else(|| SmallTideError::InvalidModel(format!("missing tensor '{name}'")))?;
    let len = entry.data_offsets[1] - entry.data_offsets[0];
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(data_base + entry.data_offsets[0] as u64))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_f32_vec(
    file: &mut File,
    hdr: &SafetensorsHeader,
    data_base: u64,
    name: &str,
) -> Result<Vec<f32>> {
    let entry = hdr
        .tensors
        .get(name)
        .ok_or_else(|| SmallTideError::InvalidModel(format!("missing tensor '{name}'")))?;
    let raw = read_raw_bytes(file, hdr, data_base, name)?;
    match entry.dtype.as_str() {
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect()),
        other => Err(SmallTideError::InvalidModel(format!(
            "unsupported dtype '{other}' for F32 vector '{name}'"
        ))),
    }
}

pub fn read_fp8_matrix(
    file: &mut File,
    hdr: &SafetensorsHeader,
    data_base: u64,
    name: &str,
) -> Result<Fp8MatrixHost> {
    let entry = hdr
        .tensors
        .get(name)
        .ok_or_else(|| SmallTideError::InvalidModel(format!("missing tensor '{name}'")))?;
    if entry.shape.len() != 2 {
        return Err(SmallTideError::InvalidModel(format!(
            "expected 2D matrix for '{name}', got {:?}",
            entry.shape
        )));
    }
    let (n_rows, k_cols) = (entry.shape[0], entry.shape[1]);
    let raw = read_raw_bytes(file, hdr, data_base, name)?;
    match entry.dtype.as_str() {
        "BF16" => {
            let (data, row_scales) = quantize_bf16_bytes_to_fp8_rows(&raw, n_rows, k_cols);
            Ok(Fp8MatrixHost {
                n_rows,
                k_cols,
                data,
                row_scales,
            })
        }
        other => Err(SmallTideError::InvalidModel(format!(
            "expected BF16 for dense matrix '{name}', got '{other}'"
        ))),
    }
}
