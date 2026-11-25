//! Linux-only pieces of the HIP backend (the Windows equivalents live inline in `ffi.rs`,
//! `wddm.rs` and `d3d12.rs`).

/// HIP runtime shared objects tried in order by `HipRuntime::load`: the ROCm 7 / 6 sonames on
/// the loader path (distro packages, `LD_LIBRARY_PATH`), then the default `/opt/rocm` install.
pub const HIP_RUNTIME_CANDIDATES: &[&str] = &[
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.7",
    "/opt/rocm/lib/libamdhip64.so.6",
    "/opt/rocm/lib/libamdhip64.so",
];

/// Install hint appended when no HIP runtime could be loaded.
pub const HIP_RUNTIME_HINT: &str =
    "Install the ROCm HIP runtime (6.x/7.x with RDNA 4 support) or add its lib directory to LD_LIBRARY_PATH.";
