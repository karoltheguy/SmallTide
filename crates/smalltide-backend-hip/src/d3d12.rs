//! D3D12-allocated device memory imported into HIP (Windows only).
//!
//! The HIP runtime's own allocations past ~5.5 GiB are placed in system memory by the driver at
//! allocation time and never migrate back, while D3D12 committed resources (what games use) are
//! managed by the OS residency manager against the full local budget. This module creates
//! shared D3D12 buffers on the AMD adapter, exports them as NT handles and imports them into HIP
//! (`hipImportExternalMemory` + `hipExternalMemoryGetMappedBuffer`).

use std::ffi::c_void;

#[repr(C)]
struct Guid([u8; 16]);

const fn guid(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> Guid {
    let a = d1.to_le_bytes();
    let b = d2.to_le_bytes();
    let c = d3.to_le_bytes();
    Guid([a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d4[0], d4[1], d4[2], d4[3], d4[4], d4[5], d4[6], d4[7]])
}

const IID_ID3D12_DEVICE: Guid = guid(0x189819f1, 0x1db6, 0x4b57, [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7]);
const IID_ID3D12_DEVICE1: Guid = guid(0x77acce80, 0x638e, 0x4e65, [0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e]);
const IID_ID3D12_RESOURCE: Guid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);

#[repr(C)]
struct HeapProperties {
    ty: u32,
    cpu_page_property: u32,
    memory_pool_preference: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
}

#[repr(C)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

type Hresult = i32;

unsafe fn vtbl(obj: *mut c_void, idx: usize) -> *const c_void {
    let table = *(obj as *const *const *const c_void);
    *table.add(idx)
}

unsafe fn release(obj: *mut c_void) {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(vtbl(obj, 2));
    f(obj);
}

/// A D3D12 device on the AMD adapter that hands out shared buffer allocations.
pub struct D3d12Device {
    _lib: libloading::Library,
    dev: *mut c_void,
    dev1: *mut c_void,
}

unsafe impl Send for D3d12Device {}
unsafe impl Sync for D3d12Device {}

/// One shared committed buffer (kept alive while HIP uses its import).
pub struct D3d12Buffer {
    res: *mut c_void,
    pub handle: *mut c_void,
    pub size: u64,
}

unsafe impl Send for D3d12Buffer {}
unsafe impl Sync for D3d12Buffer {}

impl Drop for D3d12Buffer {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                close_handle(self.handle);
            }
            if !self.res.is_null() {
                release(self.res);
            }
        }
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn CloseHandle(h: *mut c_void) -> i32;
}

unsafe fn close_handle(h: *mut c_void) {
    CloseHandle(h);
}

impl D3d12Device {
    /// `adapter` is an `IDXGIAdapter*` (see `wddm::Adapter::raw`).
    pub fn create(adapter: *mut c_void) -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("d3d12.dll").ok()?;
            let create: libloading::Symbol<
                unsafe extern "system" fn(*mut c_void, u32, *const Guid, *mut *mut c_void) -> Hresult,
            > = lib.get(b"D3D12CreateDevice\0").ok()?;
            let mut dev: *mut c_void = std::ptr::null_mut();
            const FL_11_0: u32 = 0xb000;
            if create(adapter, FL_11_0, &IID_ID3D12_DEVICE, &mut dev) != 0 || dev.is_null() {
                return None;
            }
            let qi: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult =
                std::mem::transmute(vtbl(dev, 0));
            let mut dev1: *mut c_void = std::ptr::null_mut();
            if qi(dev, &IID_ID3D12_DEVICE1, &mut dev1) != 0 {
                dev1 = std::ptr::null_mut();
            }
            Some(Self { _lib: lib, dev, dev1 })
        }
    }

    /// Committed, shared, UAV-capable buffer in the default (local video memory) heap.
    pub fn alloc_shared_buffer(&self, size: u64) -> Result<D3d12Buffer, String> {
        let size = size.div_ceil(65536) * 65536;
        unsafe {
            let hp = HeapProperties { ty: 1, cpu_page_property: 0, memory_pool_preference: 0, creation_node_mask: 1, visible_node_mask: 1 };
            let desc = ResourceDesc {
                dimension: 1,
                alignment: 0,
                width: size,
                height: 1,
                depth_or_array_size: 1,
                mip_levels: 1,
                format: 0,
                sample_count: 1,
                sample_quality: 0,
                layout: 1,
                flags: 0x4,
            };
            const HEAP_FLAG_SHARED: u32 = 0x1;
            let ccr: unsafe extern "system" fn(
                *mut c_void,
                *const HeapProperties,
                u32,
                *const ResourceDesc,
                u32,
                *const c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = std::mem::transmute(vtbl(self.dev, 27));
            let mut res: *mut c_void = std::ptr::null_mut();
            let hr = ccr(self.dev, &hp, HEAP_FLAG_SHARED, &desc, 0, std::ptr::null(), &IID_ID3D12_RESOURCE, &mut res);
            if hr != 0 || res.is_null() {
                return Err(format!("CreateCommittedResource({size}) hr=0x{:08x}", hr as u32));
            }
            let csh: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, u32, *const u16, *mut *mut c_void) -> Hresult =
                std::mem::transmute(vtbl(self.dev, 31));
            let mut handle: *mut c_void = std::ptr::null_mut();
            const GENERIC_ALL: u32 = 0x1000_0000;
            let hr = csh(self.dev, res, std::ptr::null(), GENERIC_ALL, std::ptr::null(), &mut handle);
            if hr != 0 || handle.is_null() {
                release(res);
                return Err(format!("CreateSharedHandle hr=0x{:08x}", hr as u32));
            }
            if !self.dev1.is_null() && std::env::var_os("SMALLTIDE_D3D12_NO_PRIORITY").is_none() {
                // ID3D12Device1::SetResidencyPriority(1, &obj, &MAXIMUM)
                let srp: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, *const u32) -> Hresult =
                    std::mem::transmute(vtbl(self.dev1, 46));
                const PRIORITY_MAXIMUM: u32 = 0xC800_0000;
                let objs = [res];
                let pri = [PRIORITY_MAXIMUM];
                let _ = srp(self.dev1, 1, objs.as_ptr(), pri.as_ptr());
            }
            Ok(D3d12Buffer { res, handle, size })
        }
    }
}

impl Drop for D3d12Device {
    fn drop(&mut self) {
        unsafe {
            if !self.dev1.is_null() {
                release(self.dev1);
            }
            release(self.dev);
        }
    }
}
