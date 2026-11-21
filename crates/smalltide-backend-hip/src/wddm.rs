//! WDDM video-memory budget / reservation via DXGI (Windows only).
//!
//! On Windows the AMD driver silently backs device allocations that do not fit the process's
//! *resident* local video memory with system memory (read by the GPU over PCIe at ~28 GB/s)
//! while `hipMemGetInfo` still reports them as consumed VRAM. A process can ask the OS to keep
//! a given amount resident with `IDXGIAdapter3::SetVideoMemoryReservation`; this module queries
//! the budget and makes that reservation for the AMD adapter with the most dedicated memory.

use std::ffi::c_void;

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct VideoMemoryInfo {
    pub budget: u64,
    pub current_usage: u64,
    pub available_for_reservation: u64,
    pub current_reservation: u64,
}

#[repr(C)]
struct AdapterDesc1 {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    sub_sys_id: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    adapter_luid: u64,
    flags: u32,
}

#[repr(C)]
struct Guid([u8; 16]);

const IID_IDXGI_FACTORY1: Guid = Guid([
    0x78, 0xae, 0x0a, 0x77, 0x6f, 0xf2, 0xba, 0x4d, 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87,
]);
const IID_IDXGI_ADAPTER3: Guid = Guid([
    0xa4, 0x67, 0x59, 0x64, 0x92, 0x13, 0x10, 0x43, 0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd,
]);

type Hresult = i32;

unsafe fn vtbl(obj: *mut c_void, idx: usize) -> *const c_void {
    let table = *(obj as *const *const *const c_void);
    *table.add(idx)
}

unsafe fn release(obj: *mut c_void) {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vtbl(obj, 2));
    f(obj);
}

/// The AMD adapter (largest dedicated memory) as an `IDXGIAdapter3*`.
pub struct Adapter {
    _lib: libloading::Library,
    a3: *mut c_void,
    pub dedicated_bytes: usize,
}

unsafe impl Send for Adapter {}
unsafe impl Sync for Adapter {}

impl Adapter {
    pub fn open_amd() -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("dxgi.dll").ok()?;
            let create: libloading::Symbol<unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> Hresult> =
                lib.get(b"CreateDXGIFactory1\0").ok()?;
            let mut fac: *mut c_void = std::ptr::null_mut();
            if create(&IID_IDXGI_FACTORY1, &mut fac) != 0 || fac.is_null() {
                return None;
            }
            let enum1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> Hresult =
                std::mem::transmute(vtbl(fac, 12));
            let mut best: Option<(*mut c_void, usize)> = None;
            let mut i = 0u32;
            loop {
                let mut ad: *mut c_void = std::ptr::null_mut();
                if enum1(fac, i, &mut ad) != 0 {
                    break;
                }
                let get_desc1: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> Hresult =
                    std::mem::transmute(vtbl(ad, 10));
                let mut d: AdapterDesc1 = std::mem::zeroed();
                let ok = get_desc1(ad, &mut d) == 0 && d.vendor_id == 0x1002;
                if ok && best.map_or(true, |(_, m)| d.dedicated_video_memory > m) {
                    if let Some((old, _)) = best.take() {
                        release(old);
                    }
                    best = Some((ad, d.dedicated_video_memory));
                } else {
                    release(ad);
                }
                i += 1;
            }
            release(fac);
            let (ad, dedicated) = best?;
            let qi: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult =
                std::mem::transmute(vtbl(ad, 0));
            let mut a3: *mut c_void = std::ptr::null_mut();
            let hr = qi(ad, &IID_IDXGI_ADAPTER3, &mut a3);
            release(ad);
            if hr != 0 || a3.is_null() {
                return None;
            }
            Some(Self { _lib: lib, a3, dedicated_bytes: dedicated })
        }
    }

    /// Local (dedicated) segment info for this process.
    pub fn local_info(&self) -> Option<VideoMemoryInfo> {
        unsafe {
            let q: unsafe extern "system" fn(*mut c_void, u32, i32, *mut VideoMemoryInfo) -> Hresult =
                std::mem::transmute(vtbl(self.a3, 14));
            let mut info = VideoMemoryInfo::default();
            (q(self.a3, 0, 0, &mut info) == 0).then_some(info)
        }
    }

    /// Asks the OS to keep `bytes` of this process's local video memory resident.
    pub fn reserve_local(&self, bytes: u64) -> bool {
        unsafe {
            let set: unsafe extern "system" fn(*mut c_void, u32, i32, u64) -> Hresult =
                std::mem::transmute(vtbl(self.a3, 15));
            set(self.a3, 0, 0, bytes) == 0
        }
    }
}

impl Drop for Adapter {
    fn drop(&mut self) {
        unsafe { release(self.a3) };
    }
}
