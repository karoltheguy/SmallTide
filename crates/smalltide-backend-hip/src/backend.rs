//! `HipBackend` implementing the backend-agnostic `smalltide_core::ComputeBackend` trait.
//!
//! All ROCm/HIP handles (`*mut c_void`, `HipStream`, `HipEvent`, `HipGraphExec`) remain strictly
//! internal to this crate behind opaque integer handles (`DeviceBuf`, `PinnedBuf`, `Stream`, `Event`, `Graph`).

use crate::ffi::{
    check_hip, CoreKernels, HipEvent, HipGraph, HipGraphExec, HipRuntime, HipStream,
    HIP_ERROR_NOT_READY, HIP_EVENT_DISABLE_TIMING, HIP_HOST_MALLOC_DEFAULT,
    HIP_MEMCPY_DEVICE_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE, HIP_STREAM_CAPTURE_MODE_GLOBAL, HIP_STREAM_NON_BLOCKING,
};
use smalltide_core::backend::{AttnPrepParams, FlashDecodeParams, FusedMixParams, GemvSegment, GroupedMoeParams, MoeDispatchParams, PrefillAttnParams};
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, DeviceInfo, Event, Graph, PinnedBuf, QueueKind, Result,
    SmallTideError, Stream,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

struct DeviceAlloc {
    ptr: *mut c_void,
    size_bytes: usize,
    /// Device view of host-mapped pinned memory (owned by a PinnedBuf; never hipFree'd).
    borrowed: bool,
}
unsafe impl Send for DeviceAlloc {}

struct PinnedAlloc {
    ptr: *mut c_void,
    size_bytes: usize,
}
unsafe impl Send for PinnedAlloc {}

struct GraphEntry {
    graph: HipGraph,
    exec: HipGraphExec,
}
unsafe impl Send for GraphEntry {}

struct StreamEntry(HipStream);
unsafe impl Send for StreamEntry {}

struct EventEntry(HipEvent);
unsafe impl Send for EventEntry {}

pub struct HipBackend {
    rt: HipRuntime,
    kernels: CoreKernels,
    device_id: i32,
    next_id: AtomicU64,
    dev_bufs: Mutex<HashMap<u64, DeviceAlloc>>,
    pinned_bufs: Mutex<HashMap<u64, PinnedAlloc>>,
    streams: Mutex<HashMap<u64, StreamEntry>>,
    events: Mutex<HashMap<u64, EventEntry>>,
    graphs: Mutex<HashMap<u64, GraphEntry>>,
    /// Host-side instrumentation (see `HipBackend::take_host_stats`).
    stat_launches: AtomicU64,
    stat_launch_ns: AtomicU64,
    stat_sync_ns: AtomicU64,
    stat_syncs: AtomicU64,
    kernel_timer: Mutex<Option<KernelTimer>>,
    seg_timer: Mutex<Option<SegTimer>>,
    #[cfg(windows)]
    wddm: Option<crate::wddm::Adapter>,
}

/// Coarse GPU timeline segments (timing events at `seg_mark` points); dev profiling only.
struct SegTimer {
    events: Vec<HipEvent>,
    marks: Vec<&'static str>,
    totals: HashMap<&'static str, (u64, f64)>,
}
unsafe impl Send for SegTimer {}

/// Per-kernel GPU timing (timing events around every launch); dev/profiling only.
struct KernelTimer {
    events: Vec<HipEvent>,
    cursor: usize,
    pending: Vec<(&'static str, usize)>,
    totals: HashMap<&'static str, (u64, f64)>,
}
unsafe impl Send for KernelTimer {}

/// Host-side cost counters since the last `take_host_stats`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostStats {
    pub launches: u64,
    pub launch_ms: f64,
    pub syncs: u64,
    pub sync_wait_ms: f64,
}

unsafe impl Send for HipBackend {}
unsafe impl Sync for HipBackend {}

impl HipBackend {
    /// Imports a shared D3D12 buffer (NT handle) as device memory. The returned buffer is
    /// borrowed: the D3D12 resource must outlive it.
    #[cfg(windows)]
    pub fn import_d3d12_buffer(&self, buf: &crate::d3d12::D3d12Buffer) -> Result<DeviceBuf> {
        #[repr(C)]
        struct HandleDesc {
            ty: i32,
            handle: *mut c_void,
            name: *const c_void,
            size: u64,
            flags: u32,
            reserved: [u32; 16],
        }
        #[repr(C)]
        struct BufferDesc {
            offset: u64,
            size: u64,
            flags: u32,
            reserved: [u32; 16],
        }
        type FnImport = unsafe extern "C" fn(*mut *mut c_void, *const HandleDesc) -> i32;
        type FnMap = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const BufferDesc) -> i32;
        let lib = unsafe { libloading::Library::new("amdhip64_7.dll") }
            .map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("reload amdhip64_7.dll: {e}") })?;
        let (import, map): (FnImport, FnMap) = unsafe {
            let a = lib.get::<FnImport>(b"hipImportExternalMemory\0").map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("{e}") })?;
            let b = lib.get::<FnMap>(b"hipExternalMemoryGetMappedBuffer\0").map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("{e}") })?;
            (*a, *b)
        };
        std::mem::forget(lib); // already loaded by the runtime; keep the symbols valid
        const D3D12_RESOURCE: i32 = 5;
        const DEDICATED: u32 = 1;
        let hd = HandleDesc { ty: D3D12_RESOURCE, handle: buf.handle, name: ptr::null(), size: buf.size, flags: DEDICATED, reserved: [0; 16] };
        let mut ext: *mut c_void = ptr::null_mut();
        unsafe { check_hip(import(&mut ext, &hd), "hipImportExternalMemory")? };
        let bd = BufferDesc { offset: 0, size: buf.size, flags: 0, reserved: [0; 16] };
        let mut dptr: *mut c_void = ptr::null_mut();
        unsafe { check_hip(map(&mut dptr, ext, &bd), "hipExternalMemoryGetMappedBuffer")? };
        let size_bytes = buf.size as usize;
        let did = self.next_handle();
        self.dev_bufs.lock().unwrap().insert(did, DeviceAlloc { ptr: dptr, size_bytes, borrowed: true });
        Ok(DeviceBuf { id: did, offset_bytes: 0, size_bytes })
    }

    pub fn new(device_id: i32) -> Result<Self> {
        let rt = HipRuntime::load()?;
        let _ = rt.probe_device(device_id)?;
        let kernels = rt.load_core_kernels()?;
        Ok(Self {
            rt,
            kernels,
            device_id,
            next_id: AtomicU64::new(1),
            dev_bufs: Mutex::new(HashMap::new()),
            pinned_bufs: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            stat_launches: AtomicU64::new(0),
            stat_launch_ns: AtomicU64::new(0),
            stat_sync_ns: AtomicU64::new(0),
            stat_syncs: AtomicU64::new(0),
            kernel_timer: Mutex::new(None),
            seg_timer: Mutex::new(None),
            #[cfg(windows)]
            wddm: {
                let a = crate::wddm::Adapter::open_amd();
                // Ask the OS to keep what it can of our working set resident (best effort).
                if let Some(a) = &a {
                    if let Some(i) = a.local_info() {
                        let _ = a.reserve_local(i.available_for_reservation);
                    }
                }
                a
            },
        })
    }

    /// Windows: `(process local usage, reservable/resident budget)` in bytes from DXGI. On this
    /// driver, device allocations beyond roughly the reservable amount are silently backed by
    /// system memory (GPU reads over PCIe, ~28 GB/s) although `hipMemGetInfo` counts them as VRAM.
    pub fn resident_vram_info(&self) -> Option<(u64, u64)> {
        #[cfg(windows)]
        {
            let i = self.wddm.as_ref()?.local_info()?;
            Some((i.current_usage, i.current_reservation.max(i.available_for_reservation)))
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Enables coarse segment timing (see `ComputeBackend::seg_mark`).
    pub fn set_segment_timing(&self, on: bool) -> Result<()> {
        let mut g = self.seg_timer.lock().unwrap();
        if on && g.is_none() {
            let mut events = Vec::with_capacity(4096);
            for _ in 0..4096 {
                let mut e: HipEvent = ptr::null_mut();
                unsafe { check_hip((self.rt.hip_event_create_with_flags)(&mut e, 0), "hipEventCreateWithFlags")? };
                events.push(e);
            }
            *g = Some(SegTimer { events, marks: Vec::new(), totals: HashMap::new() });
        }
        Ok(())
    }

    fn flush_segments(&self, t: &mut SegTimer) {
        let n = t.marks.len();
        if n >= 2 {
            unsafe { let _ = (self.rt.hip_event_synchronize)(t.events[n - 1]); }
            for i in 0..n - 1 {
                let mut ms = 0.0f32;
                unsafe { let _ = (self.rt.hip_event_elapsed_time)(&mut ms, t.events[i], t.events[i + 1]); }
                let e = t.totals.entry(t.marks[i]).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += ms as f64;
            }
        }
        // keep the last mark as the start of the next batch
        if n > 0 {
            let last = t.marks[n - 1];
            t.events.swap(0, n - 1);
            t.marks.clear();
            t.marks.push(last);
        }
    }

    /// Returns and resets `(segment, count, total GPU ms)` sorted by time (drops the open segment).
    pub fn take_segment_times(&self) -> Vec<(&'static str, u64, f64)> {
        let mut g = self.seg_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return Vec::new() };
        self.flush_segments(t);
        t.marks.clear();
        let mut v: Vec<_> = t.totals.drain().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    fn kernel_name(&self, f: crate::ffi::HipFunction) -> &'static str {
        let k = &self.kernels;
        macro_rules! names {
            ($($field:ident),*) => {{ $( if f == k.$field { return stringify!($field); } )* "unknown" }};
        }
        names!(noop, rmsnorm_f32, gemv_fp8_e4m3_f32, gemv_nvfp4_f32, gemv_mxfp8_f32,
            moe_w13_swiglu_indirect, moe_w2_accum_indirect, router_topk8_f32, sconv_k4_decode_f32,
            kv_store_fp8_decode, flash_decode_fp8_split, flash_decode_fp8_reduce, head_l2_clip_f32,
            value_norm_no_gamma_f32, gate_attn_sigmoid_mul_f32, attn_res_mix_f32, gemv_f32,
            rel_table_f32, loop_control_prep_f32, loop_control_apply_f32, vec_add_f32, vec_scale_f32,
            vec_copy_f32, zero_f32, final_logit_softcap_f32, argmax_partial_f32)
    }

    /// Enables/disables per-kernel GPU timing (adds 2 timing events per launch).
    pub fn set_kernel_timing(&self, on: bool) -> Result<()> {
        let mut g = self.kernel_timer.lock().unwrap();
        if on && g.is_none() {
            let mut events = Vec::with_capacity(8192);
            for _ in 0..8192 {
                let mut e: HipEvent = ptr::null_mut();
                unsafe { check_hip((self.rt.hip_event_create_with_flags)(&mut e, 0), "hipEventCreateWithFlags")? };
                events.push(e);
            }
            *g = Some(KernelTimer { events, cursor: 0, pending: Vec::new(), totals: HashMap::new() });
        } else if !on {
            if let Some(t) = g.take() {
                for e in t.events {
                    unsafe { let _ = (self.rt.hip_event_destroy)(e); }
                }
            }
        }
        Ok(())
    }

    fn flush_timer(&self, t: &mut KernelTimer) {
        if let Some(&(_, last)) = t.pending.last() {
            unsafe { let _ = (self.rt.hip_event_synchronize)(t.events[last + 1]); }
        }
        for (name, i) in t.pending.drain(..) {
            let mut ms = 0.0f32;
            unsafe { let _ = (self.rt.hip_event_elapsed_time)(&mut ms, t.events[i], t.events[i + 1]); }
            let e = t.totals.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms as f64;
        }
        t.cursor = 0;
    }

    /// Returns and resets accumulated `(kernel, launches, total GPU ms)`, sorted by time.
    pub fn take_kernel_times(&self) -> Vec<(&'static str, u64, f64)> {
        let mut g = self.kernel_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return Vec::new() };
        self.flush_timer(t);
        let mut v: Vec<_> = t.totals.drain().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    /// Launches `f` with the given argument pointers (each must point at the argument value).
    fn launch(
        &self,
        f: crate::ffi::HipFunction,
        grid: (u32, u32, u32),
        block: u32,
        args: &mut [*mut c_void],
        stream: Stream,
        what: &'static str,
    ) -> Result<()> {
        let s = self.raw_stream(stream)?;
        unsafe {
            check_hip(
                self.launch_raw(f, grid.0, grid.1, grid.2, block, 1, 1, 0, s, args.as_mut_ptr(), ptr::null_mut()),
                what,
            )
        }
    }

    /// Returns and resets host-side launch / sync-wait counters.
    pub fn take_host_stats(&self) -> HostStats {
        HostStats {
            launches: self.stat_launches.swap(0, Ordering::Relaxed),
            launch_ms: self.stat_launch_ns.swap(0, Ordering::Relaxed) as f64 * 1e-6,
            syncs: self.stat_syncs.swap(0, Ordering::Relaxed),
            sync_wait_ms: self.stat_sync_ns.swap(0, Ordering::Relaxed) as f64 * 1e-6,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    unsafe fn launch_raw(
        &self,
        f: crate::ffi::HipFunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shmem: u32,
        s: HipStream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> crate::ffi::HipError {
        let t0 = std::time::Instant::now();
        let mut timer = self.kernel_timer.lock().unwrap();
        let slot = timer.as_mut().map(|t| {
            if t.cursor + 2 > t.events.len() {
                self.flush_timer(t);
            }
            let i = t.cursor;
            t.cursor += 2;
            let _ = (self.rt.hip_event_record)(t.events[i], s);
            i
        });
        let r = (self.rt.hip_module_launch_kernel)(f, gx, gy, gz, bx, by, bz, shmem, s, params, extra);
        if let (Some(t), Some(i)) = (timer.as_mut(), slot) {
            let _ = (self.rt.hip_event_record)(t.events[i + 1], s);
            t.pending.push((self.kernel_name(f), i));
        }
        drop(timer);
        self.stat_launch_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stat_launches.fetch_add(1, Ordering::Relaxed);
        r
    }

    fn next_handle(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn dev_ptr(&self, buf: DeviceBuf) -> Result<*mut c_void> {
        let guard = self.dev_bufs.lock().unwrap();
        let a = guard.get(&buf.id).ok_or_else(|| SmallTideError::Backend {
            backend: "hip",
            message: format!("invalid DeviceBuf id={}", buf.id),
        })?;
        if buf.offset_bytes + buf.size_bytes > a.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: format!(
                    "DeviceBuf slice out of bounds: {} + {} > {}",
                    buf.offset_bytes, buf.size_bytes, a.size_bytes
                ),
            });
        }
        Ok(unsafe { (a.ptr as *mut u8).add(buf.offset_bytes) as *mut c_void })
    }

    fn opt_dev_ptr(&self, buf: Option<DeviceBuf>) -> Result<*mut c_void> {
        match buf {
            Some(b) => self.dev_ptr(b),
            None => Ok(ptr::null_mut()),
        }
    }

    fn pinned_entry(&self, buf: PinnedBuf) -> Result<(*mut c_void, usize)> {
        let guard = self.pinned_bufs.lock().unwrap();
        guard
            .get(&buf.id)
            .map(|a| (a.ptr, a.size_bytes))
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid PinnedBuf id={}", buf.id),
            })
    }

    fn raw_stream(&self, stream: Stream) -> Result<HipStream> {
        if stream.0 == 0 {
            return Ok(ptr::null_mut());
        }
        let guard = self.streams.lock().unwrap();
        guard
            .get(&stream.0)
            .map(|s| s.0)
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid Stream id={}", stream.0),
            })
    }

    fn raw_event(&self, event: Event) -> Result<HipEvent> {
        let guard = self.events.lock().unwrap();
        guard
            .get(&event.0)
            .map(|e| e.0)
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid Event id={}", event.0),
            })
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        for (_, g) in self.graphs.get_mut().unwrap().drain() {
            unsafe {
                let _ = (self.rt.hip_graph_exec_destroy)(g.exec);
                let _ = (self.rt.hip_graph_destroy)(g.graph);
            }
        }
        for (_, e) in self.events.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_event_destroy)(e.0); }
        }
        for (_, s) in self.streams.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_stream_destroy)(s.0); }
        }
        for (_, p) in self.pinned_bufs.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_host_free)(p.ptr); }
        }
        for (_, d) in self.dev_bufs.get_mut().unwrap().drain() {
            if !d.borrowed {
                unsafe { let _ = (self.rt.hip_free)(d.ptr); }
            }
        }
        unsafe {
            let _ = (self.rt.hip_module_unload)(self.kernels.module);
        }
    }
}

impl ComputeBackend for HipBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        let r = self.rt.probe_device(self.device_id)?;
        Ok(DeviceInfo {
            backend_name: "hip",
            device_name: r.device_name,
            arch_name: "gfx1201".into(),
            total_vram_bytes: r.total_vram_bytes,
            free_vram_bytes: r.free_vram_bytes,
            supports_fp8_wmma: true,
            supports_graphs: true,
        })
    }

    fn alloc_device(&self, size_bytes: usize) -> Result<DeviceBuf> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check_hip((self.rt.hip_malloc)(&mut ptr, size_bytes), "hipMalloc")? };
        let id = self.next_handle();
        self.dev_bufs
            .lock()
            .unwrap()
            .insert(id, DeviceAlloc { ptr, size_bytes, borrowed: false });
        Ok(DeviceBuf { id, offset_bytes: 0, size_bytes })
    }

    fn free_device(&self, buf: DeviceBuf) -> Result<()> {
        if buf.offset_bytes != 0 {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "cannot free a DeviceBuf sub-slice".into(),
            });
        }
        if let Some(a) = self.dev_bufs.lock().unwrap().remove(&buf.id) {
            if !a.borrowed {
                unsafe { check_hip((self.rt.hip_free)(a.ptr), "hipFree")? };
            }
        }
        Ok(())
    }

    fn alloc_host_mapped(&self, size_bytes: usize) -> Result<(DeviceBuf, PinnedBuf)> {
        const HIP_HOST_MALLOC_PORTABLE: u32 = 0x1;
        const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;
        let mut hptr: *mut c_void = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_host_malloc)(&mut hptr, size_bytes, HIP_HOST_MALLOC_PORTABLE | HIP_HOST_MALLOC_MAPPED),
                "hipHostMalloc(mapped)",
            )?;
        }
        let mut dptr: *mut c_void = ptr::null_mut();
        let r = unsafe { (self.rt.hip_host_get_device_pointer)(&mut dptr, hptr, 0) };
        if r != 0 || dptr.is_null() {
            unsafe { let _ = (self.rt.hip_host_free)(hptr); }
            check_hip(r, "hipHostGetDevicePointer")?;
            return Err(SmallTideError::Backend { backend: "hip", message: "null device pointer for mapped host memory".into() });
        }
        let pid = self.next_handle();
        self.pinned_bufs.lock().unwrap().insert(pid, PinnedAlloc { ptr: hptr, size_bytes });
        let did = self.next_handle();
        self.dev_bufs
            .lock()
            .unwrap()
            .insert(did, DeviceAlloc { ptr: dptr, size_bytes, borrowed: true });
        Ok((DeviceBuf { id: did, offset_bytes: 0, size_bytes }, PinnedBuf { id: pid, size_bytes }))
    }

    fn alloc_pinned(&self, size_bytes: usize) -> Result<PinnedBuf> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_host_malloc)(&mut ptr, size_bytes, HIP_HOST_MALLOC_DEFAULT),
                "hipHostMalloc",
            )?;
        }
        let id = self.next_handle();
        self.pinned_bufs
            .lock()
            .unwrap()
            .insert(id, PinnedAlloc { ptr, size_bytes });
        Ok(PinnedBuf { id, size_bytes })
    }

    fn free_pinned(&self, buf: PinnedBuf) -> Result<()> {
        if let Some(a) = self.pinned_bufs.lock().unwrap().remove(&buf.id) {
            unsafe { check_hip((self.rt.hip_host_free)(a.ptr), "hipHostFree")? };
        }
        Ok(())
    }

    unsafe fn pinned_as_mut_slice(&self, buf: PinnedBuf) -> Result<&mut [u8]> {
        let (ptr, len) = self.pinned_entry(buf)?;
        Ok(std::slice::from_raw_parts_mut(ptr as *mut u8, len))
    }

    fn create_stream(&self, _kind: QueueKind) -> Result<Stream> {
        let mut s: HipStream = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_stream_create_with_flags)(&mut s, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags",
            )?;
        }
        let id = self.next_handle();
        self.streams.lock().unwrap().insert(id, StreamEntry(s));
        Ok(Stream(id))
    }

    fn destroy_stream(&self, stream: Stream) -> Result<()> {
        if let Some(s) = self.streams.lock().unwrap().remove(&stream.0) {
            unsafe { check_hip((self.rt.hip_stream_destroy)(s.0), "hipStreamDestroy")? };
        }
        Ok(())
    }

    fn create_event(&self) -> Result<Event> {
        let mut e: HipEvent = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_event_create_with_flags)(&mut e, HIP_EVENT_DISABLE_TIMING),
                "hipEventCreateWithFlags",
            )?;
        }
        let id = self.next_handle();
        self.events.lock().unwrap().insert(id, EventEntry(e));
        Ok(Event(id))
    }

    fn destroy_event(&self, event: Event) -> Result<()> {
        if let Some(e) = self.events.lock().unwrap().remove(&event.0) {
            unsafe { check_hip((self.rt.hip_event_destroy)(e.0), "hipEventDestroy")? };
        }
        Ok(())
    }

    fn record_event(&self, event: Event, stream: Stream) -> Result<()> {
        let e = self.raw_event(event)?;
        let s = self.raw_stream(stream)?;
        unsafe { check_hip((self.rt.hip_event_record)(e, s), "hipEventRecord") }
    }

    fn stream_wait_event(&self, stream: Stream, event: Event) -> Result<()> {
        let e = self.raw_event(event)?;
        let s = self.raw_stream(stream)?;
        unsafe { check_hip((self.rt.hip_stream_wait_event)(s, e, 0), "hipStreamWaitEvent") }
    }

    fn sync_event(&self, event: Event) -> Result<()> {
        let e = self.raw_event(event)?;
        unsafe { check_hip((self.rt.hip_event_synchronize)(e), "hipEventSynchronize") }
    }

    fn query_event(&self, event: Event) -> Result<bool> {
        let e = self.raw_event(event)?;
        let code = unsafe { (self.rt.hip_event_query)(e) };
        match code {
            0 => Ok(true),
            HIP_ERROR_NOT_READY => Ok(false),
            other => check_hip(other, "hipEventQuery").map(|_| false),
        }
    }

    fn seg_mark(&self, label: &'static str, stream: Stream) {
        let mut g = self.seg_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return };
        if t.marks.len() + 1 >= t.events.len() {
            self.flush_segments(t);
        }
        if let Ok(s) = self.raw_stream(stream) {
            let i = t.marks.len();
            unsafe { let _ = (self.rt.hip_event_record)(t.events[i], s); }
            t.marks.push(label);
        }
    }

    fn sync_stream(&self, stream: Stream) -> Result<()> {
        let s = self.raw_stream(stream)?;
        let t0 = std::time::Instant::now();
        let r = unsafe { check_hip((self.rt.hip_stream_synchronize)(s), "hipStreamSynchronize") };
        self.stat_sync_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stat_syncs.fetch_add(1, Ordering::Relaxed);
        r
    }

    fn memcpy_d2h_async(
        &self,
        dst: PinnedBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        let d = self.dev_ptr(src)?;
        let (h, h_len) = self.pinned_entry(dst)?;
        if src_offset + len > src.size_bytes || dst_offset + len > h_len {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2h_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let src_p = (d as *const u8).add(src_offset) as *const c_void;
            let dst_p = (h as *mut u8).add(dst_offset) as *mut c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_DEVICE_TO_HOST, s),
                "hipMemcpyAsync(D2H)",
            )
        }
    }

    fn memcpy_d2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let d = self.dev_ptr(dst)?;
        let sp = self.dev_ptr(src)?;
        if src_offset + len > src.size_bytes || dst_offset + len > dst.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2d_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let src_p = (sp as *const u8).add(src_offset) as *const c_void;
            let dst_p = (d as *mut u8).add(dst_offset) as *mut c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_DEVICE_TO_DEVICE, s),
                "hipMemcpyAsync(D2D)",
            )
        }
    }

    fn memcpy_h2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: PinnedBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        let d = self.dev_ptr(dst)?;
        let (h, h_len) = self.pinned_entry(src)?;
        if dst_offset + len > dst.size_bytes || src_offset + len > h_len {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_h2d_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let dst_p = (d as *mut u8).add(dst_offset) as *mut c_void;
            let src_p = (h as *const u8).add(src_offset) as *const c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_HOST_TO_DEVICE, s),
                "hipMemcpyAsync(H2D)",
            )
        }
    }

    fn memcpy_d2h_sync(
        &self,
        dst: &mut [u8],
        src: DeviceBuf,
        src_offset: usize,
        stream: Stream,
    ) -> Result<()> {
        let d = self.dev_ptr(src)?;
        if src_offset + dst.len() > src.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2h_sync out of bounds".into(),
            });
        }
        self.sync_stream(stream)?;
        unsafe {
            let src_p = (d as *const u8).add(src_offset) as *const c_void;
            check_hip(
                (self.rt.hip_memcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    src_p,
                    dst.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                ),
                "hipMemcpy(D2H)",
            )
        }
    }

    fn begin_graph_capture(&self, stream: Stream) -> Result<()> {
        let s = self.raw_stream(stream)?;
        unsafe {
            check_hip(
                (self.rt.hip_stream_begin_capture)(s, HIP_STREAM_CAPTURE_MODE_GLOBAL),
                "hipStreamBeginCapture",
            )
        }
    }

    fn end_graph_capture(&self, stream: Stream) -> Result<Graph> {
        let s = self.raw_stream(stream)?;
        let mut graph: HipGraph = ptr::null_mut();
        let mut exec: HipGraphExec = ptr::null_mut();
        unsafe {
            check_hip((self.rt.hip_stream_end_capture)(s, &mut graph), "hipStreamEndCapture")?;
            check_hip(
                (self.rt.hip_graph_instantiate)(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0),
                "hipGraphInstantiate",
            )?;
        }
        let id = self.next_handle();
        self.graphs
            .lock()
            .unwrap()
            .insert(id, GraphEntry { graph, exec });
        Ok(Graph(id))
    }

    fn launch_graph(&self, graph: Graph, stream: Stream) -> Result<()> {
        let s = self.raw_stream(stream)?;
        let exec = {
            let guard = self.graphs.lock().unwrap();
            guard
                .get(&graph.0)
                .map(|g| g.exec)
                .ok_or_else(|| SmallTideError::Backend {
                    backend: "hip",
                    message: format!("invalid Graph id={}", graph.0),
                })?
        };
        unsafe { check_hip((self.rt.hip_graph_launch)(exec, s), "hipGraphLaunch") }
    }

    fn destroy_graph(&self, graph: Graph) -> Result<()> {
        if let Some(g) = self.graphs.lock().unwrap().remove(&graph.0) {
            unsafe {
                let _ = (self.rt.hip_graph_exec_destroy)(g.exec);
                let _ = (self.rt.hip_graph_destroy)(g.graph);
            }
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        rows: usize,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 5] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.rmsnorm_f32, rows as u32, 1, 1, if rows == 1 { 1024 } else { 256 }, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(rmsnorm_f32)",
            )
        }
    }

    fn matmul(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        weight_dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        stream: Stream,
    ) -> Result<()> {
        if m != 1 || weight_dtype != DType::FP8E4M3 {
            return Err(SmallTideError::Unsupported(format!(
                "HipBackend::matmul currently implements m=1 FP8E4M3 (got m={m}, dtype={weight_dtype:?})"
            )));
        }
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let mut d_s: *mut c_void = ptr::null_mut();
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut k_arg = k as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_s as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_fp8_e4m3_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_fp8_e4m3_f32)",
            )
        }
    }

    fn gemv_fp8(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_fp8: DeviceBuf,
        row_scales: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(w_fp8)?;
        let mut d_s = self.dev_ptr(row_scales)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n_rows as u32;
        let mut k_arg = k_cols as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_s as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_fp8_e4m3_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_fp8_e4m3_f32)",
            )
        }
    }

    fn gemv_f32(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_f32: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(w_f32)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n_rows as u32;
        let mut k_arg = k_cols as u32;
        let mut params: [*mut c_void; 5] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_f32)",
            )
        }
    }

    fn moe_swiglu_indirect(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()> {
        let mid_buf = self.alloc_device(params.top_k * params.intermediate_size * 4)?;
        let res = self.moe_swiglu_indirect_with_scratch(
            out, mid_buf, x, expert_pool, slot_table, topk_ids, topk_weights, params, stream,
        );
        let _ = self.sync_stream(stream);
        let _ = self.free_device(mid_buf);
        res
    }

    fn moe_swiglu_indirect_with_scratch(
        &self,
        out: DeviceBuf,
        mid_scratch: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()> {
        if params.batch_size != 1 {
            return Err(SmallTideError::Unsupported(
                "batched MoE GEMM (prefill/dspark) lands in Phase 2b".into(),
            ));
        }
        let mut d_mid = self.dev_ptr(mid_scratch)?;
        let mut d_out = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_pool = self.dev_ptr(expert_pool)?;
        let mut d_slots = self.dev_ptr(slot_table)?;
        let mut d_ids = self.dev_ptr(topk_ids)?;
        let mut d_w = self.dev_ptr(topk_weights)?;
        let s = self.raw_stream(stream)?;

        let mut stride_arg = params.expert_stride_bytes as u64;
        let mut h_arg = params.hidden_size as u32;
        let mut ff_arg = params.intermediate_size as u32;
        let mut gcap_arg: f32 = 4.0;
        let mut ucap_arg: f32 = 25.0;
        let mut k_arg = params.top_k as u32;

        let mut p1: [*mut c_void; 10] = [
            &mut d_mid as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_pool as *mut _ as *mut c_void,
            &mut d_slots as *mut _ as *mut c_void,
            &mut d_ids as *mut _ as *mut c_void,
            &mut stride_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut ff_arg as *mut _ as *mut c_void,
            &mut gcap_arg as *mut _ as *mut c_void,
            &mut ucap_arg as *mut _ as *mut c_void,
        ];
        let mut p2: [*mut c_void; 10] = [
            &mut d_out as *mut _ as *mut c_void,
            &mut d_mid as *mut _ as *mut c_void,
            &mut d_pool as *mut _ as *mut c_void,
            &mut d_slots as *mut _ as *mut c_void,
            &mut d_ids as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut stride_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut ff_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];

        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.moe_w13_swiglu_indirect, ff_arg.div_ceil(4), k_arg, 1, 128, 1, 1, 0, s, p1.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(moe_w13_swiglu_indirect)",
            )?;
            check_hip(
                self.launch_raw(
                    self.kernels.moe_w2_accum_indirect, h_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, p2.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(moe_w2_accum_indirect)",
            )
        }
    }

    fn router_topk8(
        &self,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        x: DeviceBuf,
        gate_w: DeviceBuf,
        gate_b: Option<DeviceBuf>,
        hidden_size: usize,
        route_scale: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_ids = self.dev_ptr(topk_ids)?;
        let mut d_w = self.dev_ptr(topk_weights)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_gw = self.dev_ptr(gate_w)?;
        let mut d_gb = self.opt_dev_ptr(gate_b)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden_size as u32;
        let mut rs_arg = route_scale;
        let mut params: [*mut c_void; 7] = [
            &mut d_ids as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_gw as *mut _ as *mut c_void,
            &mut d_gb as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut rs_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.router_topk8_f32, 1, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(router_topk8_f32)",
            )
        }
    }

    fn sconv_k4_decode(
        &self,
        out: DeviceBuf,
        state: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>,
        dim: usize,
        head_dim: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_st = self.dev_ptr(state)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let mut d_dyn = self.opt_dev_ptr(dyn_taps)?;
        let s = self.raw_stream(stream)?;
        let mut dim_arg = dim as u32;
        let mut hd_arg = head_dim as u32;
        let mut params: [*mut c_void; 7] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_st as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_dyn as *mut _ as *mut c_void,
            &mut dim_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.sconv_k4_decode_f32, dim_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(sconv_k4_decode_f32)",
            )
        }
    }

    fn head_l2_clip(
        &self,
        x: DeviceBuf,
        scale: Option<DeviceBuf>,
        n_heads: usize,
        head_dim: usize,
        threshold: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_x = self.dev_ptr(x)?;
        let mut d_sc = self.opt_dev_ptr(scale)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_heads as u32;
        let mut hd_arg = head_dim as u32;
        let mut thr_arg = threshold;
        let mut params: [*mut c_void; 5] = [
            &mut d_x as *mut _ as *mut c_void,
            &mut d_sc as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut thr_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.head_l2_clip_f32, nh_arg, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(head_l2_clip_f32)",
            )
        }
    }

    fn value_norm_no_gamma(
        &self,
        v: DeviceBuf,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_v = self.dev_ptr(v)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_kv_heads as u32;
        let mut hd_arg = head_dim as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 4] = [
            &mut d_v as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.value_norm_no_gamma_f32, nh_arg, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(value_norm_no_gamma_f32)",
            )
        }
    }

    fn rel_table(
        &self,
        table: DeviceBuf,
        r: DeviceBuf,
        rel_proj: DeviceBuf,
        n_heads: usize,
        rel_extent: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_t = self.dev_ptr(table)?;
        let mut d_r = self.dev_ptr(r)?;
        let mut d_p = self.dev_ptr(rel_proj)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_heads as u32;
        let mut ext_arg = rel_extent as u32;
        let mut params: [*mut c_void; 5] = [
            &mut d_t as *mut _ as *mut c_void,
            &mut d_r as *mut _ as *mut c_void,
            &mut d_p as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut ext_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.rel_table_f32, ext_arg.div_ceil(128), nh_arg, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(rel_table_f32)",
            )
        }
    }

    fn kv_store_fp8_decode(
        &self,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        k_in: DeviceBuf,
        v_in: DeviceBuf,
        n_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        slot_pos: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_kc = self.dev_ptr(k_cache)?;
        let mut d_ks = self.dev_ptr(k_scales)?;
        let mut d_vc = self.dev_ptr(v_cache)?;
        let mut d_vs = self.dev_ptr(v_scales)?;
        let mut d_ki = self.dev_ptr(k_in)?;
        let mut d_vi = self.dev_ptr(v_in)?;
        let s = self.raw_stream(stream)?;
        let mut nkv_arg = n_kv_heads as u32;
        let mut max_arg = max_seq as u32;
        let mut hd_arg = head_dim as u32;
        let mut slot_arg = slot_pos as u32;
        let mut params: [*mut c_void; 10] = [
            &mut d_kc as *mut _ as *mut c_void,
            &mut d_ks as *mut _ as *mut c_void,
            &mut d_vc as *mut _ as *mut c_void,
            &mut d_vs as *mut _ as *mut c_void,
            &mut d_ki as *mut _ as *mut c_void,
            &mut d_vi as *mut _ as *mut c_void,
            &mut nkv_arg as *mut _ as *mut c_void,
            &mut max_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut slot_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.kv_store_fp8_decode, nkv_arg, 1, 1, hd_arg, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(kv_store_fp8_decode)",
            )
        }
    }

    fn flash_decode_fp8(
        &self,
        out: DeviceBuf,
        partial_out: DeviceBuf,
        partial_m: DeviceBuf,
        partial_l: DeviceBuf,
        q: DeviceBuf,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        rel_table: Option<DeviceBuf>,
        sink_logit: Option<DeviceBuf>,
        params: FlashDecodeParams,
        stream: Stream,
    ) -> Result<()> {
        let mut d_out = self.dev_ptr(out)?;
        let mut d_po = self.dev_ptr(partial_out)?;
        let mut d_pm = self.dev_ptr(partial_m)?;
        let mut d_pl = self.dev_ptr(partial_l)?;
        let mut d_q = self.dev_ptr(q)?;
        let mut d_kc = self.dev_ptr(k_cache)?;
        let mut d_ks = self.dev_ptr(k_scales)?;
        let mut d_vc = self.dev_ptr(v_cache)?;
        let mut d_vs = self.dev_ptr(v_scales)?;
        let mut d_rel = self.opt_dev_ptr(rel_table)?;
        let mut d_sink = self.opt_dev_ptr(sink_logit)?;
        let s = self.raw_stream(stream)?;

        let mut nq_arg = params.n_q_heads as u32;
        let mut nkv_arg = params.n_kv_heads as u32;
        let mut max_arg = params.max_seq as u32;
        let mut qpos_arg = params.q_pos as u32;
        let mut rext_arg = params.rel_extent as u32;
        let mut loc_arg: u32 = if params.is_local { 1 } else { 0 };
        let mut splits_arg = params.num_splits as u32;
        let mut inv_sqrt_d = params.inv_sqrt_d;
        let mut window_arg = params.window as u32;

        let mut p1: [*mut c_void; 18] = [
            &mut d_po as *mut _ as *mut c_void,
            &mut d_pm as *mut _ as *mut c_void,
            &mut d_pl as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut d_kc as *mut _ as *mut c_void,
            &mut d_ks as *mut _ as *mut c_void,
            &mut d_vc as *mut _ as *mut c_void,
            &mut d_vs as *mut _ as *mut c_void,
            &mut d_rel as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut nkv_arg as *mut _ as *mut c_void,
            &mut max_arg as *mut _ as *mut c_void,
            &mut qpos_arg as *mut _ as *mut c_void,
            &mut rext_arg as *mut _ as *mut c_void,
            &mut loc_arg as *mut _ as *mut c_void,
            &mut splits_arg as *mut _ as *mut c_void,
            &mut inv_sqrt_d as *mut _ as *mut c_void,
            &mut window_arg as *mut _ as *mut c_void,
        ];
        let mut p2: [*mut c_void; 7] = [
            &mut d_out as *mut _ as *mut c_void,
            &mut d_po as *mut _ as *mut c_void,
            &mut d_pm as *mut _ as *mut c_void,
            &mut d_pl as *mut _ as *mut c_void,
            &mut d_sink as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut splits_arg as *mut _ as *mut c_void,
        ];

        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.flash_decode_fp8_split, nkv_arg, splits_arg, 1, 128, 1, 1, 0, s, p1.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(flash_decode_fp8_split)",
            )?;
            check_hip(
                self.launch_raw(
                    self.kernels.flash_decode_fp8_reduce, nq_arg, 1, 1, 32, 1, 1, 0, s, p2.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(flash_decode_fp8_reduce)",
            )
        }
    }

    fn gate_attn_sigmoid_mul(
        &self,
        out: DeviceBuf,
        attn: DeviceBuf,
        gate: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_a = self.dev_ptr(attn)?;
        let mut d_g = self.dev_ptr(gate)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_a as *mut _ as *mut c_void,
            &mut d_g as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gate_attn_sigmoid_mul_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gate_attn_sigmoid_mul_f32)",
            )
        }
    }

    fn attn_res_mix(
        &self,
        out: DeviceBuf,
        blocks: Option<DeviceBuf>,
        cur: DeviceBuf,
        queries: DeviceBuf,
        num_blocks: usize,
        num_queries: usize,
        hidden: usize,
        eps: f32,
        normalize_values: bool,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_b = self.opt_dev_ptr(blocks)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_q = self.dev_ptr(queries)?;
        let s = self.raw_stream(stream)?;
        let mut nb_arg = num_blocks as u32;
        let mut nq_arg = num_queries as u32;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut nv_arg: u32 = if normalize_values { 1 } else { 0 };
        let mut params: [*mut c_void; 9] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut nb_arg as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
            &mut nv_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.attn_res_mix_f32, nq_arg, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(attn_res_mix_f32)",
            )
        }
    }

    fn loop_control_prep(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_m = self.dev_ptr(msg)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 5] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_m as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.loop_control_prep_f32, 2, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(loop_control_prep_f32)",
            )
        }
    }

    fn loop_control_apply(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        gain_raw: DeviceBuf,
        forget_raw: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_m = self.dev_ptr(msg)?;
        let mut d_g = self.dev_ptr(gain_raw)?;
        let mut d_f = self.dev_ptr(forget_raw)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_m as *mut _ as *mut c_void,
            &mut d_g as *mut _ as *mut c_void,
            &mut d_f as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.loop_control_apply_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(loop_control_apply_f32)",
            )
        }
    }

    fn vec_add(
        &self,
        out: DeviceBuf,
        a: DeviceBuf,
        b: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_a = self.dev_ptr(a)?;
        let mut d_b = self.dev_ptr(b)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_a as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_add_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_add_f32)",
            )
        }
    }

    fn vec_scale(&self, x: DeviceBuf, alpha: f32, n: usize, stream: Stream) -> Result<()> {
        let mut d_x = self.dev_ptr(x)?;
        let s = self.raw_stream(stream)?;
        let mut a_arg = alpha;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 3] = [
            &mut d_x as *mut _ as *mut c_void,
            &mut a_arg as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_scale_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_scale_f32)",
            )
        }
    }

    fn vec_copy(&self, dst: DeviceBuf, src: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let mut d_dst = self.dev_ptr(dst)?;
        let mut d_src = self.dev_ptr(src)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 3] = [
            &mut d_dst as *mut _ as *mut c_void,
            &mut d_src as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_copy_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_copy_f32)",
            )
        }
    }

    fn zero_f32(&self, dst: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let mut d_dst = self.dev_ptr(dst)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 2] = [
            &mut d_dst as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.zero_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(zero_f32)",
            )
        }
    }

    fn final_logit_softcap(
        &self,
        logits: DeviceBuf,
        vocab_size: usize,
        cap: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_l = self.dev_ptr(logits)?;
        let s = self.raw_stream(stream)?;
        let mut v_arg = vocab_size as u32;
        let mut c_arg = cap;
        let mut params: [*mut c_void; 3] = [
            &mut d_l as *mut _ as *mut c_void,
            &mut v_arg as *mut _ as *mut c_void,
            &mut c_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.final_logit_softcap_f32, v_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(final_logit_softcap_f32)",
            )
        }
    }

    fn argmax_partial_f32(
        &self,
        out_vals: DeviceBuf,
        out_idx: DeviceBuf,
        x: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_v = self.dev_ptr(out_vals)?;
        let mut d_i = self.dev_ptr(out_idx)?;
        let mut d_x = self.dev_ptr(x)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_v as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.argmax_partial_f32, 64, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(argmax_partial_f32)",
            )
        }
    }
    fn gemm_fp8_rows(&self, y: DeviceBuf, x: DeviceBuf, w_fp8: DeviceBuf, row_scales: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w_fp8)?, self.dev_ptr(row_scales)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut m), p(&mut n), p(&mut k)];
        self.launch(self.kernels.gemm_fp8_e4m3_batched_f32, (n.div_ceil(4), m.div_ceil(8), 1), 128, &mut args, stream, "launch(gemm_fp8_rows)")
    }

    fn gemm_f32_rows(&self, y: DeviceBuf, x: DeviceBuf, w: DeviceBuf, rows: usize, n_rows: usize,
        k_cols: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut m), p(&mut n), p(&mut k)];
        self.launch(self.kernels.gemm_f32_rows, (n.div_ceil(4), m.div_ceil(8), 1), 128, &mut args, stream, "launch(gemm_f32_rows)")
    }

    fn attn_res_mix_rows(&self, out: DeviceBuf, blocks: Option<DeviceBuf>, cur: DeviceBuf,
        queries: DeviceBuf, num_blocks: usize, num_queries: usize, hidden: usize, eps: f32,
        rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d) = (self.dev_ptr(out)?, self.opt_dev_ptr(blocks)?, self.dev_ptr(cur)?, self.dev_ptr(queries)?);
        let (mut nb, mut nq, mut h, mut e, mut nv, mut m) = (num_blocks as u32, num_queries as u32, hidden as u32, eps, 1u32, rows as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut nb), p(&mut nq), p(&mut h), p(&mut e), p(&mut nv), p(&mut m)];
        let threads = if m == 1 { 1024 } else { 256 };
        self.launch(self.kernels.attn_res_mix_rows_f32, (nq, m, 1), threads, &mut args, stream, "launch(attn_res_mix_rows)")
    }

    fn head_l2_clip_rows(&self, x: DeviceBuf, scale: Option<DeviceBuf>, total_heads: usize,
        head_dim: usize, heads_per_row: usize, threshold: f32, tau: Option<(usize, f32, f32)>,
        stream: Stream) -> Result<()> {
        let (mut a, mut b) = (self.dev_ptr(x)?, self.opt_dev_ptr(scale)?);
        let (pos0, alpha, floor) = tau.unwrap_or((0, 0.0, 1.0));
        let (mut th, mut hd, mut hpr, mut thr, mut p0, mut al, mut fl) =
            (total_heads as u32, head_dim as u32, heads_per_row as u32, threshold, pos0 as u32, alpha, floor);
        let mut args = [p(&mut a), p(&mut b), p(&mut th), p(&mut hd), p(&mut hpr), p(&mut thr), p(&mut p0), p(&mut al), p(&mut fl)];
        self.launch(self.kernels.head_l2_clip_rows_f32, (th, 1, 1), 128, &mut args, stream, "launch(head_l2_clip_rows)")
    }

    fn sconv_k4_rows(&self, out: DeviceBuf, state: DeviceBuf, x: DeviceBuf, weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>, dim: usize, head_dim: usize, rows: usize, dyn_stride: usize,
        stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e) =
            (self.dev_ptr(out)?, self.dev_ptr(state)?, self.dev_ptr(x)?, self.dev_ptr(weight)?, self.opt_dev_ptr(dyn_taps)?);
        let (mut di, mut hd, mut m, mut ds) = (dim as u32, head_dim as u32, rows as u32, dyn_stride as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut di), p(&mut hd), p(&mut m), p(&mut ds)];
        self.launch(self.kernels.sconv_k4_rows_f32, (di.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(sconv_k4_rows)")
    }

    fn kv_store_fp8_rows(&self, k_cache: DeviceBuf, k_scales: DeviceBuf, v_cache: DeviceBuf,
        v_scales: DeviceBuf, k_in: DeviceBuf, v_in: DeviceBuf, n_kv_heads: usize, max_seq: usize,
        head_dim: usize, pos0: usize, rows: usize, is_local: bool, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e, mut f) = (self.dev_ptr(k_cache)?, self.dev_ptr(k_scales)?,
            self.dev_ptr(v_cache)?, self.dev_ptr(v_scales)?, self.dev_ptr(k_in)?, self.dev_ptr(v_in)?);
        let (mut nkv, mut ms, mut hd, mut p0, mut loc) = (n_kv_heads as u32, max_seq as u32, head_dim as u32, pos0 as u32, is_local as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut nkv), p(&mut ms), p(&mut hd), p(&mut p0), p(&mut loc)];
        self.launch(self.kernels.kv_store_fp8_rows, (nkv, rows as u32, 1), hd, &mut args, stream, "launch(kv_store_fp8_rows)")
    }

    fn attn_prefill_fp8(&self, out: DeviceBuf, q: DeviceBuf, k_cache: DeviceBuf, k_scales: DeviceBuf,
        v_cache: DeviceBuf, v_scales: DeviceBuf, rel_table: Option<DeviceBuf>, sinks: Option<DeviceBuf>,
        pp: PrefillAttnParams, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (self.dev_ptr(out)?, self.dev_ptr(q)?,
            self.dev_ptr(k_cache)?, self.dev_ptr(k_scales)?, self.dev_ptr(v_cache)?, self.dev_ptr(v_scales)?,
            self.opt_dev_ptr(rel_table)?, self.opt_dev_ptr(sinks)?);
        let (mut nq, mut nkv, mut ms, mut win, mut p0, mut re, mut isd) = (pp.n_q_heads as u32, pp.n_kv_heads as u32,
            pp.max_seq as u32, pp.window as u32, pp.pos0 as u32, pp.rel_extent as u32, pp.inv_sqrt_d);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut g), p(&mut h),
            p(&mut nq), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut p0), p(&mut re), p(&mut isd)];
        self.launch(self.kernels.attn_prefill_fp8, (nkv, pp.rows as u32, 1), 128, &mut args, stream, "launch(attn_prefill_fp8)")
    }

    fn router_topk8_rows(&self, route: DeviceBuf, logits: DeviceBuf, bias: Option<DeviceBuf>,
        route_scale: f32, rows: usize, row_stride_words: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(route)?, self.dev_ptr(logits)?, self.opt_dev_ptr(bias)?);
        let (mut rs, mut st) = (route_scale, row_stride_words as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut rs), p(&mut st)];
        self.launch(self.kernels.router_topk8_rows, (rows as u32, 1, 1), 128, &mut args, stream, "launch(router_topk8_rows)")
    }

    fn loop_control_prep_rows(&self, out: DeviceBuf, cur: DeviceBuf, msg: DeviceBuf, hidden: usize,
        eps: f32, rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(out)?, self.dev_ptr(cur)?, self.dev_ptr(msg)?);
        let (mut h, mut e) = (hidden as u32, eps);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut h), p(&mut e)];
        let threads = if rows == 1 { 1024 } else { 256 };
        self.launch(self.kernels.loop_control_prep_rows_f32, (2, rows as u32, 1), threads, &mut args, stream, "launch(loop_control_prep_rows)")
    }

    fn moe_grouped(&self, out: DeviceBuf, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, route: DeviceBuf,
        pool: DeviceBuf, desc: DeviceBuf, gp: GroupedMoeParams, reduce: bool, stream: Stream) -> Result<()> {
        if gp.n_work > 0 {
            let work = desc.slice(0, gp.n_work * 16);
            let pairs = desc.slice(gp.pairs_offset_bytes, desc.size_bytes - gp.pairs_offset_bytes);
            let (mut a, mut b, mut c, mut d, mut e) =
                (self.dev_ptr(mid)?, self.dev_ptr(x)?, self.dev_ptr(pool)?, self.dev_ptr(work)?, self.dev_ptr(pairs)?);
            let (mut st, mut h, mut i, mut gc, mut uc) = (gp.expert_stride_bytes as u64, gp.hidden_size as u32, gp.intermediate_size as u32, 4.0f32, 25.0f32);
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
            self.launch(self.kernels.moe_w13_grouped, (gp.n_work as u32, i.div_ceil(64), 1), 256, &mut args, stream, "launch(moe_w13_grouped)")?;
            let (mut a, mut b, mut c, mut d, mut e, mut f) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?, self.dev_ptr(pool)?,
                self.dev_ptr(work)?, self.dev_ptr(pairs)?, self.dev_ptr(route)?);
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut st), p(&mut h), p(&mut i)];
            self.launch(self.kernels.moe_w2_grouped, (gp.n_work as u32, h.div_ceil(64), 1), 256, &mut args, stream, "launch(moe_w2_grouped)")?;
        }
        if reduce {
            self.moe_reduce_rows(out, tmp, gp.hidden_size, gp.rows, stream)?;
        }
        Ok(())
    }

    fn moe_decode_experts(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, pool: DeviceBuf,
        slot_table: DeviceBuf, topk_ids: DeviceBuf, mp: MoeDispatchParams, stream: Stream) -> Result<()> {
        let (mut a, mut bx, mut c, mut d, mut e) =
            (self.dev_ptr(mid)?, self.dev_ptr(x)?, self.dev_ptr(pool)?, self.dev_ptr(slot_table)?, self.dev_ptr(topk_ids)?);
        let (mut st, mut h, mut i, mut gc, mut uc) =
            (mp.expert_stride_bytes as u64, mp.hidden_size as u32, mp.intermediate_size as u32, 4.0f32, 25.0f32);
        let k = mp.top_k as u32;
        let mut args = [p(&mut a), p(&mut bx), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
        let v2 = std::env::var_os("SMALLTIDE_MOE_V3").is_none();
        if v2 {
            self.launch(self.kernels.moe_w13_decode, (i.div_ceil(8), k, 1), 256, &mut args, stream, "launch(moe_w13_decode)")?;
        } else {
            self.launch(self.kernels.moe_w13_decode3, (i.div_ceil(32), k, 1), 256, &mut args, stream, "launch(moe_w13_decode3)")?;
        }
        let (mut a, mut bm) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?);
        let mut args = [p(&mut a), p(&mut bm), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i)];
        if v2 {
            self.launch(self.kernels.moe_w2_decode, (h.div_ceil(8), k, 1), 256, &mut args, stream, "launch(moe_w2_decode)")
        } else {
            self.launch(self.kernels.moe_w2_decode3, (h.div_ceil(32), k, 1), 256, &mut args, stream, "launch(moe_w2_decode3)")
        }
    }

    fn moe_decode_post(&self, out: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, topk_ids: DeviceBuf,
        topk_weights: DeviceBuf, top_k: usize, sconv: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
        hidden: usize, stream: Stream) -> Result<()> {
        let (mut a, mut bt, mut c, mut d, mut e) = (self.dev_ptr(out)?, self.dev_ptr(tmp)?, self.dev_ptr(slot_table)?,
            self.dev_ptr(topk_ids)?, self.dev_ptr(topk_weights)?);
        let (mut f, mut g, mut r) = match sconv {
            Some((state, w, residual)) => (self.dev_ptr(state)?, self.dev_ptr(w)?, self.dev_ptr(residual)?),
            None => (ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
        };
        let (mut k, mut h) = (top_k as u32, hidden as u32);
        let mut args = [p(&mut a), p(&mut bt), p(&mut c), p(&mut d), p(&mut e), p(&mut k), p(&mut f), p(&mut g), p(&mut r), p(&mut h)];
        self.launch(self.kernels.moe_post_decode, (h.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(moe_post_decode)")
    }

    fn attn_prep_decode(&self, pp: AttnPrepParams, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ArgsC {
            q: *mut c_void,
            r: *mut c_void,
            k_raw: *mut c_void,
            v_raw: *mut c_void,
            k_dyn: *mut c_void,
            v_dyn: *mut c_void,
            sconv_k: *mut c_void,
            sconv_v: *mut c_void,
            k_w: *mut c_void,
            v_w: *mut c_void,
            q_scale: *mut c_void,
            r_scale: *mut c_void,
            rel_proj: *mut c_void,
            rel_table: *mut c_void,
            k_cache: *mut c_void,
            k_scales: *mut c_void,
            v_cache: *mut c_void,
            v_scales: *mut c_void,
            n_q: u32,
            n_kv: u32,
            max_seq: u32,
            slot: u32,
            rel_extent: u32,
            qk_thr: f32,
            rel_thr: f32,
            tau: f32,
            eps: f32,
        }
        if pp.n_q_heads == 0 || pp.n_kv_heads == 0 {
            return Err(SmallTideError::Backend { backend: "hip", message: "attn_prep_decode: bad head counts".into() });
        }
        let mut a = ArgsC {
            q: self.dev_ptr(pp.q)?,
            r: self.dev_ptr(pp.r)?,
            k_raw: self.dev_ptr(pp.k_raw)?,
            v_raw: self.dev_ptr(pp.v_raw)?,
            k_dyn: self.opt_dev_ptr(pp.k_dyn)?,
            v_dyn: self.opt_dev_ptr(pp.v_dyn)?,
            sconv_k: self.dev_ptr(pp.sconv_k)?,
            sconv_v: self.dev_ptr(pp.sconv_v)?,
            k_w: self.dev_ptr(pp.k_w)?,
            v_w: self.dev_ptr(pp.v_w)?,
            q_scale: self.dev_ptr(pp.q_scale)?,
            r_scale: self.dev_ptr(pp.r_scale)?,
            rel_proj: self.dev_ptr(pp.rel_proj)?,
            rel_table: self.dev_ptr(pp.rel_table)?,
            k_cache: self.dev_ptr(pp.k_cache)?,
            k_scales: self.dev_ptr(pp.k_scales)?,
            v_cache: self.dev_ptr(pp.v_cache)?,
            v_scales: self.dev_ptr(pp.v_scales)?,
            n_q: pp.n_q_heads as u32,
            n_kv: pp.n_kv_heads as u32,
            max_seq: pp.max_seq as u32,
            slot: pp.slot as u32,
            rel_extent: pp.rel_extent as u32,
            qk_thr: pp.qk_threshold,
            rel_thr: pp.rel_threshold,
            tau: pp.tau,
            eps: pp.eps,
        };
        let blocks = (pp.n_kv_heads + 2 * pp.n_q_heads) as u32;
        let mut args = [p(&mut a)];
        self.launch(self.kernels.attn_prep_decode, (blocks, 1, 1), 128, &mut args, stream, "launch(attn_prep_decode)")
    }

    fn fused_mix_decode(&self, pp: FusedMixParams, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ArgsC {
            cur: *mut c_void,
            o_raw: *mut c_void,
            sconv_state: *mut c_void,
            sconv_w: *mut c_void,
            residual: *mut c_void,
            blocks: *mut c_void,
            copy_out: *mut c_void,
            queries: *mut c_void,
            mix_out: *mut c_void,
            norm_w0: *mut c_void,
            norm_w1: *mut c_void,
            norm_w2: *mut c_void,
            norm_out: *mut c_void,
            nb: u32,
            boundary: u32,
            nq: u32,
            hidden: u32,
            eps: f32,
        }
        if pp.hidden > 3 * 1024 || pp.nq == 0 || pp.nq > 3 || pp.nb + 1 >= 16 {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: format!("fused_mix_decode: unsupported hidden={} nq={} nb={}", pp.hidden, pp.nq, pp.nb),
            });
        }
        let (o_raw, st, w) = match pp.sconv {
            Some((x, st, w)) => (Some(x), Some(st), Some(w)),
            None => (None, None, None),
        };
        let nw = if pp.norm_w[0].is_some() { pp.norm_w } else { [None; 3] };
        let mut a = ArgsC {
            cur: self.dev_ptr(pp.cur)?,
            o_raw: self.opt_dev_ptr(o_raw)?,
            sconv_state: self.opt_dev_ptr(st)?,
            sconv_w: self.opt_dev_ptr(w)?,
            residual: self.dev_ptr(pp.residual)?,
            blocks: self.dev_ptr(pp.blocks)?,
            copy_out: self.opt_dev_ptr(pp.copy_out)?,
            queries: self.opt_dev_ptr(pp.queries)?,
            mix_out: self.opt_dev_ptr(pp.mix_out)?,
            norm_w0: self.opt_dev_ptr(nw[0])?,
            norm_w1: self.opt_dev_ptr(nw[1])?,
            norm_w2: self.opt_dev_ptr(nw[2])?,
            norm_out: self.dev_ptr(pp.norm_out)?,
            nb: pp.nb as u32,
            boundary: pp.boundary as u32,
            nq: pp.nq as u32,
            hidden: pp.hidden as u32,
            eps: pp.eps,
        };
        let mut args = [p(&mut a)];
        self.launch(self.kernels.fused_mix_decode, (1, 1, 1), 1024, &mut args, stream, "launch(fused_mix_decode)")
    }

    fn gemv_multi(&self, segs: &[GemvSegment], stream: Stream) -> Result<()> {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct SegC {
            w: *const c_void,
            scale: *const c_void,
            x: *const c_void,
            y: *mut c_void,
            n_rows: u32,
            k_cols: u32,
            is_f32: u32,
            row_start: u32,
        }
        #[repr(C)]
        struct ArgsC {
            seg: [SegC; 8],
            n_seg: u32,
            total_rows: u32,
        }
        let empty = SegC { w: ptr::null(), scale: ptr::null(), x: ptr::null(), y: ptr::null_mut(), n_rows: 0, k_cols: 0, is_f32: 0, row_start: u32::MAX };
        for chunk in segs.chunks(8) {
            let mut a = ArgsC { seg: [empty; 8], n_seg: chunk.len() as u32, total_rows: 0 };
            for (i, sg) in chunk.iter().enumerate() {
                if sg.k_cols % 16 != 0 {
                    return Err(SmallTideError::Backend { backend: "hip", message: format!("gemv_multi: k_cols {} not a multiple of 16", sg.k_cols) });
                }
                a.seg[i] = SegC {
                    w: self.dev_ptr(sg.w)? as *const c_void,
                    scale: match sg.row_scales { Some(r) => self.dev_ptr(r)? as *const c_void, None => ptr::null() },
                    x: self.dev_ptr(sg.x)? as *const c_void,
                    y: self.dev_ptr(sg.y)?,
                    n_rows: sg.n_rows as u32,
                    k_cols: sg.k_cols as u32,
                    is_f32: sg.row_scales.is_none() as u32,
                    row_start: a.total_rows,
                };
                a.total_rows += sg.n_rows as u32;
            }
            let mut args = [p(&mut a)];
            self.launch(self.kernels.gemv_multi, (a.total_rows.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_multi)")?;
        }
        Ok(())
    }

    fn moe_reduce_rows(&self, out: DeviceBuf, tmp: DeviceBuf, hidden: usize, rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b) = (self.dev_ptr(out)?, self.dev_ptr(tmp)?);
        let mut h = hidden as u32;
        let mut args = [p(&mut a), p(&mut b), p(&mut h)];
        self.launch(self.kernels.moe_reduce_rows, (h.div_ceil(256), rows as u32, 1), 256, &mut args, stream, "launch(moe_reduce_rows)")
    }
}

/// Kernel-argument pointer helper.
#[inline]
fn p<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}
