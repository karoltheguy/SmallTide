//! GPU-resident dense weights for Inkling-2-Mini (`--fp8-dense`), backend-agnostic over `ComputeBackend`.

use smalltide_core::{backend::GemvSegment, ComputeBackend, DeviceBuf, PinnedBuf, QueueKind, Result, Stream};
use smalltide_io::dense_loader::{DenseLayerHost, Fp8MatrixHost, Inkling2DenseHost, LoopControllerHost};
use std::sync::Arc;

pub struct Fp8MatrixGpu {
    pub n_rows: usize,
    pub k_cols: usize,
    pub data: DeviceBuf,
    pub row_scales: DeviceBuf,
    pub is_bf16: bool,
}

impl Fp8MatrixGpu {
    #[inline]
    pub fn seg(&self, x: DeviceBuf, y: DeviceBuf) -> GemvSegment {
        GemvSegment {
            w: self.data,
            row_scales: (!self.is_bf16).then_some(self.row_scales),
            x,
            y,
            n_rows: self.n_rows,
            k_cols: self.k_cols,
            is_bf16: self.is_bf16,
        }
    }

    #[inline]
    pub fn gemv<B: ComputeBackend>(&self, b: &B, out: DeviceBuf, x: DeviceBuf, n_rows: usize, s: Stream) -> Result<()> {
        if self.is_bf16 {
            b.gemv_bf16(out, x, self.data, n_rows, self.k_cols, s)
        } else {
            b.gemv_fp8(out, x, self.data, self.row_scales, n_rows, self.k_cols, s)
        }
    }

    #[inline]
    pub fn gemm_rows<B: ComputeBackend>(&self, b: &B, y: DeviceBuf, x: DeviceBuf, rows: usize, n_rows: usize, s: Stream) -> Result<()> {
        if self.is_bf16 {
            b.gemm_bf16_rows(y, x, self.data, rows, n_rows, self.k_cols, s)
        } else {
            b.gemm_fp8_rows(y, x, self.data, self.row_scales, rows, n_rows, self.k_cols, s)
        }
    }

    #[inline]
    pub fn gemm_rows_decode<B: ComputeBackend>(&self, b: &B, y: DeviceBuf, x: DeviceBuf, rows: usize, n_rows: usize, s: Stream) -> Result<()> {
        if self.is_bf16 {
            b.gemm_bf16_rows(y, x, self.data, rows, n_rows, self.k_cols, s)
        } else {
            b.gemm_fp8_rows_decode(y, x, self.data, self.row_scales, rows, n_rows, self.k_cols, s)
        }
    }
}

pub struct LoopControllerGpu {
    pub down: Fp8MatrixGpu,
    pub forget: Fp8MatrixGpu,
    pub gain: Fp8MatrixGpu,
}

pub struct DenseLayerGpu {
    pub physical_layer: usize,
    pub is_local: bool,

    pub wq: Fp8MatrixGpu,
    pub wk: Fp8MatrixGpu,
    pub wv: Fp8MatrixGpu,
    pub wo: Fp8MatrixGpu,
    pub gate_attn: Fp8MatrixGpu,
    pub wr: Fp8MatrixGpu,
    pub attn_sink: DeviceBuf,

    pub q_l2_clip_scale: DeviceBuf,
    pub rel_l2_clip_scale: DeviceBuf,
    pub rel_logits_proj: DeviceBuf,

    pub k_sconv_static: DeviceBuf,
    pub v_sconv_static: DeviceBuf,
    pub k_sconv_dyn_proj: Option<DeviceBuf>,
    pub v_sconv_dyn_proj: Option<DeviceBuf>,
    pub attn_sconv: DeviceBuf,
    pub mlp_sconv: DeviceBuf,

    pub attn_norm: Option<DeviceBuf>,
    pub q_input_norm: Option<DeviceBuf>,
    pub k_input_norm: Option<DeviceBuf>,
    pub v_input_norm: Option<DeviceBuf>,
    pub mlp_norm: DeviceBuf,

    pub router_weight: DeviceBuf,
    pub router_bias: DeviceBuf,

    pub attn_res_query: Option<DeviceBuf>,
    pub mlp_res_query: DeviceBuf,

    pub controllers: Option<[LoopControllerGpu; 4]>,
    pub loop_msg_queries: Option<DeviceBuf>,
}

/// Complete always-resident Inkling-2-Mini dense model in VRAM (~1.10 GiB with `--fp8-dense`)
/// plus `embed_bf16` in Host RAM (`883 MiB`) for zero-VRAM token lookup.
pub struct Inkling2DenseGpu<B: ComputeBackend> {
    pub backend: Arc<B>,
    pub embed_bf16_host: Vec<u8>,
    pub embed_norm: DeviceBuf,
    pub final_norm: DeviceBuf,
    pub output_res_query: DeviceBuf,
    pub middle_cycle_cross_pass_queries: DeviceBuf,
    pub middle_cycle_cross_pass_kv_queries: DeviceBuf,
    pub unembed_fp8: Fp8MatrixGpu,
    pub layers: Vec<DenseLayerGpu>,
}

pub(crate) struct Uploader<'a, B: ComputeBackend> {
    pub(crate) backend: &'a B,
    pub(crate) staging: PinnedBuf,
    pub(crate) stream: Stream,
}

impl<'a, B: ComputeBackend> Uploader<'a, B> {
    pub(crate) fn upload_bytes(&self, bytes: &[u8]) -> Result<DeviceBuf> {
        let dev = self.backend.alloc_device(bytes.len().max(4))?;
        let chunk_cap = self.staging.size_bytes;
        let mut off = 0usize;
        while off < bytes.len() {
            let take = (bytes.len() - off).min(chunk_cap);
            unsafe {
                let dst = self.backend.pinned_as_mut_slice(self.staging)?;
                dst[..take].copy_from_slice(&bytes[off..off + take]);
            }
            self.backend
                .memcpy_h2d_async(dev, off, self.staging, 0, take, self.stream)?;
            self.backend.sync_stream(self.stream)?;
            off += take;
        }
        Ok(dev)
    }

    pub(crate) fn upload_f32(&self, v: &[f32]) -> Result<DeviceBuf> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        self.upload_bytes(bytes)
    }

    pub(crate) fn upload_fp8(&self, m: &Fp8MatrixHost) -> Result<Fp8MatrixGpu> {
        Ok(Fp8MatrixGpu {
            n_rows: m.n_rows,
            k_cols: m.k_cols,
            data: self.upload_bytes(&m.data)?,
            row_scales: self.upload_f32(&m.row_scales)?,
            is_bf16: m.is_bf16,
        })
    }

    fn upload_ctrl(&self, c: &LoopControllerHost) -> Result<LoopControllerGpu> {
        Ok(LoopControllerGpu {
            down: self.upload_fp8(&c.down)?,
            forget: self.upload_fp8(&c.forget)?,
            gain: self.upload_fp8(&c.gain)?,
        })
    }
}

impl<B: ComputeBackend> Inkling2DenseGpu<B> {
    pub fn upload(backend: Arc<B>, host: Inkling2DenseHost) -> Result<Self> {
        let staging = backend.alloc_pinned(64 * 1024 * 1024)?; // 64 MiB temporary staging
        let stream = backend.create_stream(QueueKind::AsyncCopyH2D)?;
        let up = Uploader {
            backend: backend.as_ref(),
            staging,
            stream,
        };

        let res = (|| -> Result<Self> {
            let embed_norm = up.upload_f32(&host.embed_norm)?;
            let final_norm = up.upload_f32(&host.final_norm)?;
            let output_res_query = up.upload_f32(&host.output_res_query)?;
            let middle_cycle_cross_pass_queries =
                up.upload_f32(&host.middle_cycle_cross_pass_queries)?;
            let middle_cycle_cross_pass_kv_queries =
                up.upload_f32(&host.middle_cycle_cross_pass_kv_queries)?;
            let unembed_fp8 = up.upload_fp8(&host.unembed_fp8)?;

            let mut layers = Vec::with_capacity(host.layers.len());
            for l in &host.layers {
                layers.push(upload_layer(&up, l)?);
            }

            Ok(Self {
                backend: backend.clone(),
                embed_bf16_host: host.embed_bf16,
                embed_norm,
                final_norm,
                output_res_query,
                middle_cycle_cross_pass_queries,
                middle_cycle_cross_pass_kv_queries,
                unembed_fp8,
                layers,
            })
        })();

        let _ = backend.destroy_stream(stream);
        let _ = backend.free_pinned(staging);
        res
    }
}

fn upload_layer<B: ComputeBackend>(
    up: &Uploader<'_, B>,
    l: &DenseLayerHost,
) -> Result<DenseLayerGpu> {
    let opt_f32 = |o: &Option<Vec<f32>>| -> Result<Option<DeviceBuf>> {
        match o {
            Some(v) => Ok(Some(up.upload_f32(v)?)),
            None => Ok(None),
        }
    };
    let controllers = match &l.controllers {
        Some(arr) => Some([
            up.upload_ctrl(&arr[0])?,
            up.upload_ctrl(&arr[1])?,
            up.upload_ctrl(&arr[2])?,
            up.upload_ctrl(&arr[3])?,
        ]),
        None => None,
    };
    Ok(DenseLayerGpu {
        physical_layer: l.physical_layer,
        is_local: l.is_local,
        wq: up.upload_fp8(&l.wq)?,
        wk: up.upload_fp8(&l.wk)?,
        wv: up.upload_fp8(&l.wv)?,
        wo: up.upload_fp8(&l.wo)?,
        gate_attn: up.upload_fp8(&l.gate_attn)?,
        wr: up.upload_fp8(&l.wr)?,
        attn_sink: up.upload_f32(&l.attn_sink)?,
        q_l2_clip_scale: up.upload_f32(&l.q_l2_clip_scale)?,
        rel_l2_clip_scale: up.upload_f32(&l.rel_l2_clip_scale)?,
        rel_logits_proj: up.upload_f32(&l.rel_logits_proj)?,
        k_sconv_static: up.upload_f32(&l.k_sconv_static)?,
        v_sconv_static: up.upload_f32(&l.v_sconv_static)?,
        k_sconv_dyn_proj: opt_f32(&l.k_sconv_dyn_proj)?,
        v_sconv_dyn_proj: opt_f32(&l.v_sconv_dyn_proj)?,
        attn_sconv: up.upload_f32(&l.attn_sconv)?,
        mlp_sconv: up.upload_f32(&l.mlp_sconv)?,
        attn_norm: opt_f32(&l.attn_norm)?,
        q_input_norm: opt_f32(&l.q_input_norm)?,
        k_input_norm: opt_f32(&l.k_input_norm)?,
        v_input_norm: opt_f32(&l.v_input_norm)?,
        mlp_norm: up.upload_f32(&l.mlp_norm)?,
        router_weight: up.upload_f32(&l.router_weight)?,
        router_bias: up.upload_f32(&l.router_bias)?,
        attn_res_query: opt_f32(&l.attn_res_query)?,
        mlp_res_query: up.upload_f32(&l.mlp_res_query)?,
        controllers,
        loop_msg_queries: opt_f32(&l.loop_msg_queries)?,
    })
}
