use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_hipcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("SMALLTIDE_HIPCC") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
        let candidate = Path::new(&rocm).join("bin").join("hipcc.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Search standard AMD ROCm Windows install paths (newest version first)
    let rocm_root = Path::new(r"C:\Program Files\AMD\ROCm");
    if let Ok(entries) = std::fs::read_dir(rocm_root) {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        versions.reverse();
        for dir in versions {
            let candidate = dir.join("bin").join("hipcc.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/core_ops.hip");
    println!("cargo:rerun-if-changed=kernels/moe_ops.hip");
    println!("cargo:rerun-if-changed=kernels/attn_ops.hip");
    println!("cargo:rerun-if-changed=kernels/inkling2_ops.hip");
    println!("cargo:rerun-if-changed=kernels/batched_ops.hip");
    println!("cargo:rerun-if-changed=kernels/prefill_ops.hip");
    println!("cargo:rerun-if-env-changed=SMALLTIDE_HIPCC");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=SMALLTIDE_HIP_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let hsaco_out = out_dir.join("smalltide_gfx1201.hsaco");
    let arch = env::var("SMALLTIDE_HIP_ARCH").unwrap_or_else(|_| "gfx1201".to_string());

    let Some(hipcc) = find_hipcc() else {
        // Allow compiling the workspace even on machines without ROCm SDK installed;
        // emit an empty marker file and fail gracefully at runtime if kernels are invoked.
        println!("cargo:warning=hipcc not found; building smalltide-backend-hip without compiled .hsaco kernels");
        std::fs::write(&hsaco_out, []).expect("write empty hsaco placeholder");
        return;
    };

    let status = Command::new(&hipcc)
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "-ffast-math",
            "kernels/core_ops.hip",
            "-o",
        ])
        .arg(&hsaco_out)
        .status()
        .unwrap_or_else(|e| panic!("failed to execute {}: {e}", hipcc.display()));

    if !status.success() {
        panic!(
            "hipcc failed with status {status} compiling kernels/core_ops.hip for {arch}"
        );
    }
}
