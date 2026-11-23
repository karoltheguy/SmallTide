//! Dev tool: how long a host-to-card copy from page-locked memory takes, by size and by how it
//! is split into commands. A rescue copies one expert (4.27 or 7.83 MiB) per command.
//!
//!   cargo run --release -p smalltide-backend-hip --example h2d_timing
use smalltide_backend_hip::HipBackend;
use smalltide_core::backend::{ComputeBackend, QueueKind};
use std::time::Instant;

fn main() -> smalltide_core::Result<()> {
    let b = HipBackend::new_auto()?;
    const MIB: usize = 1 << 20;
    let total = 64 * MIB;
    let dev = b.alloc_device(total)?;
    let pin = b.alloc_pinned(total)?;
    unsafe {
        let s = b.pinned_as_mut_slice(pin)?;
        for (i, x) in s.iter_mut().enumerate() {
            *x = (i * 31) as u8;
        }
    }
    let stream = b.create_stream(QueueKind::AsyncCopyH2D)?;
    let ev = b.create_event()?;
    // warm up
    for _ in 0..20 {
        b.memcpy_h2d_async(dev, 0, pin, 0, total, stream)?;
        b.sync_stream(stream)?;
    }
    let reps = 300;
    println!("one command of N bytes, then wait:");
    for kib in [4usize, 64, 512, 1024, 2187, 4373, 8018, 16384, 32768] {
        let n = kib * 1024;
        let t = Instant::now();
        for _ in 0..reps {
            b.memcpy_h2d_async(dev, 0, pin, 0, n, stream)?;
            b.sync_stream(stream)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("  {:>7.2} MiB: {:.3} ms  ({:.1} GB/s)", n as f64 / MIB as f64, ms, n as f64 / 1e9 / (ms * 1e-3));
    }
    let slab = 4373 * 1024;
    println!("three 4.27 MiB slabs:");
    for mode in ["3 commands, one wait", "3 commands, started as queued", "1 command of 3 slabs"] {
        let t = Instant::now();
        for _ in 0..reps {
            match mode {
                "3 commands, one wait" => {
                    for i in 0..3 {
                        b.memcpy_h2d_async(dev, i * slab, pin, i * slab, slab, stream)?;
                    }
                }
                "3 commands, started as queued" => {
                    for i in 0..3 {
                        b.memcpy_h2d_async(dev, i * slab, pin, i * slab, slab, stream)?;
                        b.record_event(ev, stream)?;
                        let _ = b.query_event(ev)?;
                    }
                }
                _ => b.memcpy_h2d_async(dev, 0, pin, 0, 3 * slab, stream)?,
            }
            b.sync_stream(stream)?;
        }
        println!("  {mode}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
    }
    Ok(())
}
