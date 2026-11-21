//! Windows unbuffered, overlapped direct NVMe I/O (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`).
//!
//! Windows equivalent of Linux `O_DIRECT`: bypasses the Windows OS file cache so Stage 2
//! (Disk -> RAM) expert streaming never competes with our Tier 1 RAM pool for the 32 GB RAM.

use smalltide_core::{Result, SmallTideError};
use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

pub const SECTOR_ALIGN: usize = 4096;

pub const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;
    pub const INVALID_HANDLE_VALUE: Handle = (-1isize) as Handle;

    pub const GENERIC_READ: u32 = 0x8000_0000;
    pub const FILE_SHARE_READ: u32 = 0x0000_0001;
    pub const OPEN_EXISTING: u32 = 3;
    pub const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    pub const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

    pub const MEM_COMMIT: u32 = 0x0000_1000;
    pub const MEM_RESERVE: u32 = 0x0000_2000;
    pub const MEM_RELEASE: u32 = 0x0000_8000;
    pub const PAGE_READWRITE: u32 = 0x04;

    pub const ERROR_IO_PENDING: u32 = 997;

    #[repr(C)]
    pub struct Overlapped {
        pub internal: usize,
        pub internal_high: usize,
        pub offset: u32,
        pub offset_high: u32,
        pub h_event: Handle,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: u32,
            dwShareMode: u32,
            lpSecurityAttributes: *mut c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes: u32,
            hTemplateFile: Handle,
        ) -> Handle;
        pub fn CloseHandle(hObject: Handle) -> i32;
        pub fn CreateEventW(
            lpEventAttributes: *mut c_void,
            bManualReset: i32,
            bInitialState: i32,
            lpName: *const u16,
        ) -> Handle;
        pub fn ReadFile(
            hFile: Handle,
            lpBuffer: *mut c_void,
            nNumberOfBytesToRead: u32,
            lpNumberOfBytesRead: *mut u32,
            lpOverlapped: *mut Overlapped,
        ) -> i32;
        pub fn GetOverlappedResult(
            hFile: Handle,
            lpOverlapped: *mut Overlapped,
            lpNumberOfBytesTransferred: *mut u32,
            bWait: i32,
        ) -> i32;
        pub fn GetLastError() -> u32;
        pub fn VirtualAlloc(
            lpAddress: *mut c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(lpAddress: *mut c_void, dwSize: usize, dwFreeType: u32) -> i32;
    }
}

/// Page-aligned (4 KB) host buffer required by `FILE_FLAG_NO_BUFFERING`.
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    pub fn alloc(size_bytes: usize) -> Result<Self> {
        let len = align_up(size_bytes.max(SECTOR_ALIGN), SECTOR_ALIGN);
        #[cfg(windows)]
        {
            let ptr = unsafe {
                win32::VirtualAlloc(
                    ptr::null_mut(),
                    len,
                    win32::MEM_COMMIT | win32::MEM_RESERVE,
                    win32::PAGE_READWRITE,
                )
            };
            if ptr.is_null() {
                return Err(SmallTideError::Io(std::io::Error::last_os_error()));
            }
            Ok(Self {
                ptr: ptr as *mut u8,
                len,
            })
        }
        #[cfg(not(windows))]
        {
            Err(SmallTideError::Unsupported(
                "AlignedBuffer currently implemented for Windows".into(),
            ))
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            let _ = win32::VirtualFree(self.ptr as *mut _, 0, win32::MEM_RELEASE);
        }
    }
}

/// Unbuffered overlapped file handle (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`).
pub struct DirectFile {
    #[cfg(windows)]
    handle: win32::Handle,
}

unsafe impl Send for DirectFile {}
unsafe impl Sync for DirectFile {}

impl DirectFile {
    pub fn open_read(path: &Path) -> Result<Self> {
        #[cfg(windows)]
        {
            let wide: Vec<u16> = OsStr::new(path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let handle = unsafe {
                win32::CreateFileW(
                    wide.as_ptr(),
                    win32::GENERIC_READ,
                    win32::FILE_SHARE_READ,
                    ptr::null_mut(),
                    win32::OPEN_EXISTING,
                    win32::FILE_FLAG_NO_BUFFERING | win32::FILE_FLAG_OVERLAPPED,
                    ptr::null_mut(),
                )
            };
            if handle == win32::INVALID_HANDLE_VALUE {
                return Err(SmallTideError::Io(std::io::Error::last_os_error()));
            }
            Ok(Self { handle })
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            Err(SmallTideError::Unsupported(
                "DirectFile currently implemented for Windows".into(),
            ))
        }
    }

    /// Reads `len` bytes (must be a multiple of 4096) at `offset` (multiple of 4096)
    /// into an aligned buffer using overlapped direct I/O.
    pub fn read_aligned_at(&self, offset: u64, buf: &mut AlignedBuffer, len: usize) -> Result<usize> {
        if offset % (SECTOR_ALIGN as u64) != 0 || len % SECTOR_ALIGN != 0 || len > buf.len() {
            return Err(SmallTideError::InvalidModel(format!(
                "direct read requires 4KB alignment: offset={offset}, len={len}, cap={}",
                buf.len()
            )));
        }
        #[cfg(windows)]
        unsafe {
            let event = win32::CreateEventW(ptr::null_mut(), 1, 0, ptr::null());
            if event.is_null() {
                return Err(SmallTideError::Io(std::io::Error::last_os_error()));
            }
            let mut ov = win32::Overlapped {
                internal: 0,
                internal_high: 0,
                offset: (offset & 0xFFFF_FFFF) as u32,
                offset_high: (offset >> 32) as u32,
                h_event: event,
            };
            let mut transferred: u32 = 0;
            let ok = win32::ReadFile(
                self.handle,
                buf.ptr as *mut _,
                len as u32,
                &mut transferred,
                &mut ov,
            );
            if ok == 0 {
                let err = win32::GetLastError();
                if err != win32::ERROR_IO_PENDING {
                    let _ = win32::CloseHandle(event);
                    return Err(SmallTideError::Io(std::io::Error::from_raw_os_error(
                        err as i32,
                    )));
                }
                let wait_ok = win32::GetOverlappedResult(self.handle, &mut ov, &mut transferred, 1);
                if wait_ok == 0 {
                    let err2 = win32::GetLastError();
                    let _ = win32::CloseHandle(event);
                    return Err(SmallTideError::Io(std::io::Error::from_raw_os_error(
                        err2 as i32,
                    )));
                }
            }
            let _ = win32::CloseHandle(event);
            Ok(transferred as usize)
        }
        #[cfg(not(windows))]
        {
            let _ = (offset, buf, len);
            Err(SmallTideError::Unsupported("Windows only".into()))
        }
    }
}

impl Drop for DirectFile {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            let _ = win32::CloseHandle(self.handle);
        }
    }
}
