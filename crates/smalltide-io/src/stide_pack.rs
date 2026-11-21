//! 4 KB-aligned contiguous single-expert layout (`.stide`) & Inkling-2-Mini safetensors slicer.
//!
//! In upstream `model-00001-of-00001.safetensors`, each layer's 128 experts are stacked into
//! separate tensors (`w13_weight`, `w13_weight_scale`, `w13_weight_outer_scale_denominator`,
//! `w2_weight`, `w2_weight_scale`, `w2_weight_outer_scale_denominator`).
//!
//! Packing each `(layer, expert_id)` contiguously into a 4 KB-aligned slab enables:
//! - Stage 2 (NVMe -> RAM): ONE single `FILE_FLAG_NO_BUFFERING` aligned read per expert (~0.96 ms)
//! - Stage 1 (RAM -> VRAM): ONE single `hipMemcpyAsync` DMA transfer per expert (~0.158 ms)
//! - GPU Kernel: Fixed compile-time byte offsets from `expert_pool_base + slot * expert_stride`!

use crate::safetensors::SafetensorsHeader;
use crate::win_direct_io::{align_up, DirectFile, AlignedBuffer, SECTOR_ALIGN};
use smalltide_core::{DType, ExpertId, Inkling2Config, PhysicalLayerId, Result, SmallTideError};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Fixed 64-byte header at the start of every contiguous expert slab (both in RAM and VRAM).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExpertSlabHeader {
    pub w13_outer_scale: f32,
    pub w2_outer_scale: f32,
    pub w13_weight_offset: u32,
    pub w13_scale_offset: u32,
    pub w2_weight_offset: u32,
    pub w2_scale_offset: u32,
    pub dtype_tag: u32, // 0 = NVFP4, 1 = MXFP8
    pub _reserved: [u32; 9],
}

const _: () = assert!(std::mem::size_of::<ExpertSlabHeader>() == 64);

/// Exact byte layout of a single packed expert slab for a given quantization class.
#[derive(Debug, Clone, Copy)]
pub struct ExpertSlabLayout {
    pub dtype: DType,
    pub header_bytes: usize,
    pub w13_weight_offset: usize,
    pub w13_weight_bytes: usize,
    pub w13_scale_offset: usize,
    pub w13_scale_bytes: usize,
    pub w2_weight_offset: usize,
    pub w2_weight_bytes: usize,
    pub w2_scale_offset: usize,
    pub w2_scale_bytes: usize,
    /// Total 4 KB-aligned stride of one expert slab on disk, in RAM, and in VRAM.
    pub aligned_slab_bytes: usize,
}

impl ExpertSlabLayout {
    pub fn for_inkling2(cfg: &Inkling2Config, dtype: DType) -> Self {
        let d = cfg.hidden_size; // 2304
        let ff = cfg.intermediate_size; // 1152
        // w13: [2 * ff, d] = [2304, 2304] (5,308,416 params)
        // w2:  [d, ff]     = [2304, 1152] (2,654,208 params)
        let w13_params = (2 * ff) * d;
        let w2_params = d * ff;

        let (w13_w, w13_s, w2_w, w2_s) = match dtype {
            // NVFP4: [1, 16] FP8 E4M3 block scales (verified [128, 2304, 144] & [128, 2304, 72])
            DType::NVFP4 => (w13_params / 2, w13_params / 16, w2_params / 2, w2_params / 16),
            // MXFP8: on-disk safetensors stores [128, N, K/32] UE8M0 scales ([128, 2304, 72] & [128, 2304, 36])!
            DType::MXFP8 => (w13_params, w13_params / 32, w2_params, w2_params / 32),
            _ => panic!("ExpertSlabLayout only supports NVFP4 and MXFP8"),
        };

        let header_bytes = 64;
        let w13_weight_offset = align_up(header_bytes, 64);
        let w13_scale_offset = align_up(w13_weight_offset + w13_w, 64);
        let w2_weight_offset = align_up(w13_scale_offset + w13_s, 64);
        let w2_scale_offset = align_up(w2_weight_offset + w2_w, 64);
        let aligned_slab_bytes = align_up(w2_scale_offset + w2_s, SECTOR_ALIGN);

        Self {
            dtype,
            header_bytes,
            w13_weight_offset,
            w13_weight_bytes: w13_w,
            w13_scale_offset,
            w13_scale_bytes: w13_s,
            w2_weight_offset,
            w2_weight_bytes: w2_w,
            w2_scale_offset,
            w2_scale_bytes: w2_s,
            aligned_slab_bytes,
        }
    }
}

/// Byte offsets within `model-00001-of-00001.safetensors` for one `(layer, expert)` slice.
#[derive(Debug, Clone, Copy)]
pub struct SafetensorsExpertSlice {
    pub expert: ExpertId,
    pub dtype: DType,
    pub w13_weight_file_offset: u64,
    pub w13_weight_bytes: usize,
    pub w13_scale_file_offset: u64,
    pub w13_scale_bytes: usize,
    pub w13_amax_file_offset: Option<u64>,
    pub w13_outer_scale_file_offset: Option<u64>,
    pub w2_weight_file_offset: u64,
    pub w2_weight_bytes: usize,
    pub w2_scale_file_offset: u64,
    pub w2_scale_bytes: usize,
    pub w2_amax_file_offset: Option<u64>,
    pub w2_outer_scale_file_offset: Option<u64>,
}

/// Builds a complete index of all 3,072 routed experts from an Inkling-2-Mini safetensors header.
pub fn index_inkling2_experts(
    cfg: &Inkling2Config,
    hdr: &SafetensorsHeader,
) -> Result<Vec<SafetensorsExpertSlice>> {
    let data_base = 8u64 + hdr.header_len;
    let n_exp = cfg.n_routed_experts;
    let mut out = Vec::with_capacity(cfg.num_physical_layers * n_exp);

    for layer in 0..cfg.num_physical_layers {
        let dtype = if cfg.is_mxfp8_layer(layer) {
            DType::MXFP8
        } else {
            DType::NVFP4
        };
        let prefix = format!("model.llm.layers.{layer}.mlp.experts");
        let get = |suffix: &str| -> Result<(u64, usize)> {
            let key = format!("{prefix}.{suffix}");
            let e = hdr.tensors.get(&key).ok_or_else(|| {
                SmallTideError::InvalidModel(format!("missing tensor '{key}' in safetensors"))
            })?;
            let len = e.data_offsets[1]
                .checked_sub(e.data_offsets[0])
                .ok_or_else(|| SmallTideError::InvalidModel(format!("bad offsets for '{key}'")))?;
            Ok((data_base + e.data_offsets[0] as u64, len))
        };

        let (w13_w_off, w13_w_tot) = get("w13_weight")?;
        let (w13_s_off, w13_s_tot) = get("w13_weight_scale")?;
        let (w2_w_off, w2_w_tot) = get("w2_weight")?;
        let (w2_s_off, w2_s_tot) = get("w2_weight_scale")?;

        let w13_amax = get("w13_weight_amax").ok();
        let w13_outer = get("w13_weight_outer_scale_denominator").ok();
        let w2_amax = get("w2_weight_amax").ok();
        let w2_outer = get("w2_weight_outer_scale_denominator").ok();

        let w13_w_per = w13_w_tot / n_exp;
        let w13_s_per = w13_s_tot / n_exp;
        let w2_w_per = w2_w_tot / n_exp;
        let w2_s_per = w2_s_tot / n_exp;

        for e in 0..n_exp {
            let scalar_stride = |opt: Option<(u64, usize)>| -> Option<u64> {
                opt.map(|(base, tot)| {
                    if tot >= n_exp * 4 {
                        base + (e * (tot / n_exp)) as u64
                    } else {
                        base
                    }
                })
            };

            out.push(SafetensorsExpertSlice {
                expert: ExpertId {
                    layer: PhysicalLayerId(layer as u16),
                    expert: e as u16,
                },
                dtype,
                w13_weight_file_offset: w13_w_off + (e * w13_w_per) as u64,
                w13_weight_bytes: w13_w_per,
                w13_scale_file_offset: w13_s_off + (e * w13_s_per) as u64,
                w13_scale_bytes: w13_s_per,
                w13_amax_file_offset: scalar_stride(w13_amax),
                w13_outer_scale_file_offset: scalar_stride(w13_outer),
                w2_weight_file_offset: w2_w_off + (e * w2_w_per) as u64,
                w2_weight_bytes: w2_w_per,
                w2_scale_file_offset: w2_s_off + (e * w2_s_per) as u64,
                w2_scale_bytes: w2_s_per,
                w2_amax_file_offset: scalar_stride(w2_amax),
                w2_outer_scale_file_offset: scalar_stride(w2_outer),
            });
        }
    }

    Ok(out)
}

fn read_f32_at(file: &mut File, offset: u64) -> Result<f32> {
    let mut b = [0u8; 4];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// Reads one expert's scattered slices from `model-00001-of-00001.safetensors` and packs them
/// into a single contiguous 4 KB-aligned slab buffer ready for Tier 1 RAM, Tier 0 VRAM, or `.stide`.
pub fn pack_single_expert_from_safetensors(
    file: &mut File,
    slice: &SafetensorsExpertSlice,
    layout: &ExpertSlabLayout,
    dst: &mut [u8],
) -> Result<()> {
    if dst.len() < layout.aligned_slab_bytes {
        return Err(SmallTideError::InvalidModel(format!(
            "destination slab too small: {} < {}",
            dst.len(),
            layout.aligned_slab_bytes
        )));
    }
    dst[..layout.aligned_slab_bytes].fill(0);

    // ModelOpt NVFP4 scale format verified in hf_quant_config.json: "amax_denominator"
    // => outer_scale = amax / outer_scale_denominator
    let w13_outer_scale = match (slice.w13_amax_file_offset, slice.w13_outer_scale_file_offset) {
        (Some(a_off), Some(d_off)) => {
            let amax = read_f32_at(file, a_off)?;
            let denom = read_f32_at(file, d_off)?;
            if denom.abs() > 1e-12 { amax / denom } else { 1.0 }
        }
        _ => 1.0,
    };
    let w2_outer_scale = match (slice.w2_amax_file_offset, slice.w2_outer_scale_file_offset) {
        (Some(a_off), Some(d_off)) => {
            let amax = read_f32_at(file, a_off)?;
            let denom = read_f32_at(file, d_off)?;
            if denom.abs() > 1e-12 { amax / denom } else { 1.0 }
        }
        _ => 1.0,
    };

    let hdr = ExpertSlabHeader {
        w13_outer_scale,
        w2_outer_scale,
        w13_weight_offset: layout.w13_weight_offset as u32,
        w13_scale_offset: layout.w13_scale_offset as u32,
        w2_weight_offset: layout.w2_weight_offset as u32,
        w2_scale_offset: layout.w2_scale_offset as u32,
        dtype_tag: if slice.dtype == DType::NVFP4 { 0 } else { 1 },
        _reserved: [0; 9],
    };
    let hdr_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            (&hdr as *const ExpertSlabHeader) as *const u8,
            std::mem::size_of::<ExpertSlabHeader>(),
        )
    };
    dst[..64].copy_from_slice(hdr_bytes);

    let read_exact_at = |f: &mut File, off: u64, out: &mut [u8]| -> Result<()> {
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(out)?;
        Ok(())
    };

    read_exact_at(
        file,
        slice.w13_weight_file_offset,
        &mut dst[layout.w13_weight_offset..layout.w13_weight_offset + slice.w13_weight_bytes],
    )?;
    read_exact_at(
        file,
        slice.w13_scale_file_offset,
        &mut dst[layout.w13_scale_offset..layout.w13_scale_offset + slice.w13_scale_bytes],
    )?;
    read_exact_at(
        file,
        slice.w2_weight_file_offset,
        &mut dst[layout.w2_weight_offset..layout.w2_weight_offset + slice.w2_weight_bytes],
    )?;
    read_exact_at(
        file,
        slice.w2_scale_file_offset,
        &mut dst[layout.w2_scale_offset..layout.w2_scale_offset + slice.w2_scale_bytes],
    )?;

    Ok(())
}

pub const STIDE_MAGIC: &[u8; 8] = b"STIDE001";

/// 4 KB-aligned on-disk `.stide` expert pack for 1-op Stage 2 NVMe `NO_BUFFERING` reads.
pub struct StidePackReader {
    direct: DirectFile,
    offsets: Vec<u64>,
    lengths: Vec<usize>,
    n_routed_experts: usize,
}

impl StidePackReader {
    /// One-time repacker: converts Inkling-2-Mini's stacked safetensors into a 4 KB-aligned `.stide` file.
    pub fn build_from_safetensors(
        cfg: &Inkling2Config,
        safetensors_path: &Path,
        out_path: &Path,
    ) -> Result<()> {
        let hdr = crate::safetensors::inspect_safetensors_header(safetensors_path)?;
        let slices = index_inkling2_experts(cfg, &hdr)?;
        let nvfp4_layout = ExpertSlabLayout::for_inkling2(cfg, DType::NVFP4);
        let mxfp8_layout = ExpertSlabLayout::for_inkling2(cfg, DType::MXFP8);

        let mut src = File::open(safetensors_path)?;
        let mut dst = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(out_path)?;

        // File header (4 KB aligned): magic (8) + count (u32) + n_experts_per_layer (u32)
        // + table of (offset: u64, len: u64) per expert, padded to 4 KB.
        let count = slices.len();
        let toc_bytes = align_up(16 + count * 16, SECTOR_ALIGN);
        let mut toc = vec![0u8; toc_bytes];
        toc[0..8].copy_from_slice(STIDE_MAGIC);
        toc[8..12].copy_from_slice(&(count as u32).to_le_bytes());
        toc[12..16].copy_from_slice(&(cfg.n_routed_experts as u32).to_le_bytes());

        let mut cursor = toc_bytes as u64;
        for (i, s) in slices.iter().enumerate() {
            let layout = if s.dtype == DType::NVFP4 { &nvfp4_layout } else { &mxfp8_layout };
            let entry_off = 16 + i * 16;
            toc[entry_off..entry_off + 8].copy_from_slice(&cursor.to_le_bytes());
            toc[entry_off + 8..entry_off + 16]
                .copy_from_slice(&(layout.aligned_slab_bytes as u64).to_le_bytes());
            cursor += layout.aligned_slab_bytes as u64;
        }
        dst.write_all(&toc)?;

        let max_slab = nvfp4_layout.aligned_slab_bytes.max(mxfp8_layout.aligned_slab_bytes);
        let mut scratch = vec![0u8; max_slab];
        for s in &slices {
            let layout = if s.dtype == DType::NVFP4 { &nvfp4_layout } else { &mxfp8_layout };
            pack_single_expert_from_safetensors(
                &mut src,
                s,
                layout,
                &mut scratch[..layout.aligned_slab_bytes],
            )?;
            dst.write_all(&scratch[..layout.aligned_slab_bytes])?;
        }
        dst.sync_all()?;
        Ok(())
    }

    pub fn open(path: &Path) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut head = [0u8; 16];
        f.read_exact(&mut head)?;
        if &head[0..8] != STIDE_MAGIC {
            return Err(SmallTideError::InvalidModel("invalid .stide magic".into()));
        }
        let count = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let n_routed_experts = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let mut toc = vec![0u8; count * 16];
        f.read_exact(&mut toc)?;
        drop(f);

        let mut offsets = Vec::with_capacity(count);
        let mut lengths = Vec::with_capacity(count);
        for i in 0..count {
            let off = u64::from_le_bytes(toc[i * 16..i * 16 + 8].try_into().unwrap());
            let len = u64::from_le_bytes(toc[i * 16 + 8..i * 16 + 16].try_into().unwrap()) as usize;
            offsets.push(off);
            lengths.push(len);
        }

        let direct = DirectFile::open_read(path)?;
        Ok(Self {
            direct,
            offsets,
            lengths,
            n_routed_experts,
        })
    }

    /// Reads an entire 4 KB-aligned expert slab directly from NVMe bypassing Windows OS file cache.
    pub fn read_expert_direct(&self, id: ExpertId, dst: &mut AlignedBuffer) -> Result<usize> {
        let idx = (id.layer.0 as usize) * self.n_routed_experts + (id.expert as usize);
        let off = self.offsets[idx];
        let len = self.lengths[idx];
        self.direct.read_aligned_at(off, dst, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inkling2_slab_layouts_4kb_aligned() {
        let cfg = Inkling2Config::default();
        let nvfp4 = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4);
        let mxfp8 = ExpertSlabLayout::for_inkling2(&cfg, DType::MXFP8);

        assert_eq!(nvfp4.aligned_slab_bytes % SECTOR_ALIGN, 0);
        assert_eq!(mxfp8.aligned_slab_bytes % SECTOR_ALIGN, 0);
        // NVFP4 payload ~4,478,976 B -> aligned slab = 4,481,024 B (~4.273 MiB)
        assert!(nvfp4.aligned_slab_bytes >= 4_478_976 && nvfp4.aligned_slab_bytes < 4_490_000);
        // MXFP8 payload ~8,211,456 B ([N, K/32] scales) -> aligned slab = 8,212,480 B (~7.832 MiB)
        assert!(mxfp8.aligned_slab_bytes >= 8_211_456 && mxfp8.aligned_slab_bytes < 8_225_000);
    }
}
