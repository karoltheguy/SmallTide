//! Times pinned host allocations (hipHostMalloc): `pinned_alloc [MiB each] [count]`. Stops early,
//! freeing everything, as soon as one allocation takes longer than a second.
use smalltide_backend_hip::HipBackend;
use smalltide_core::ComputeBackend;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mib: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    let b = HipBackend::new_auto().expect("init HipBackend");
    let mut held = Vec::new();
    for i in 0..n {
        let t = Instant::now();
        let p = b.alloc_pinned(mib << 20).expect("alloc_pinned");
        let t_alloc = t.elapsed().as_secs_f64();
        let t = Instant::now();
        unsafe { b.pinned_as_mut_slice(p).expect("slice").fill(1) };
        let t_touch = t.elapsed().as_secs_f64();
        held.push(p);
        if t_alloc + t_touch > 0.15 || i + 1 == n {
            println!("alloc {i} (total {} MiB): {t_alloc:.3} s, first touch {t_touch:.3} s", (i + 1) * mib);
        }
        if t_alloc + t_touch > 1.0 {
            println!("stopping: an allocation took longer than a second");
            break;
        }
    }
    let t = Instant::now();
    for p in held {
        b.free_pinned(p).expect("free");
    }
    println!("freed all in {:.3} s", t.elapsed().as_secs_f64());
}
