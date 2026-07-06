// Benchmark: candle LS-NMF multiplicative update at realistic ESAT sizes.
//
// Compares three strategies for running B independent NMF solves:
//   (a) CPU + Accelerate BLAS        -> B problems, one at a time (2D)
//   (b) Metal GPU, single            -> B problems, one at a time (2D)
//   (c) Metal GPU, batched           -> all B problems in one 3D tensor
//
// The update loop mirrors esat/model/ls_nmf.py exactly:
//   WeV = We * V                 (precomputed)
//   WH   = W @ H
//   H    = H * (Wt@WeV) / (Wt@(We*WH))
//   WH   = W @ H
//   W    = W * (WeV@Ht) / ((We*WH)@Ht)
//
// candle's matmul and .t() operate on the last two dims, so the SAME kernel
// runs 2D (single) or 3D (batched over the leading dim).

use candle_core::{Device, Tensor, Result};
use std::time::Instant;

const EPS: f64 = 1e-9;

// One LS-NMF multiplicative update. Works for 2D (n,m) or 3D (B,n,m) tensors.
fn nmf_update(we: &Tensor, wev: &Tensor, w: &Tensor, h: &Tensor) -> Result<(Tensor, Tensor)> {
    // ---- update H ----
    let wh = w.matmul(h)?;                                  // (.,n,m)
    let wt = w.t()?;                                        // (.,k,n)
    let h_num = wt.matmul(wev)?;                            // (.,k,m)
    let h_den = wt.matmul(&we.mul(&wh)?)?.affine(1.0, EPS)?;
    let h = h.mul(&h_num.div(&h_den)?)?;
    // ---- update W ----
    let wh = w.matmul(&h)?;
    let ht = h.t()?;                                        // (.,m,k)
    let w_num = wev.matmul(&ht)?;                           // (.,n,k)
    let w_den = we.mul(&wh)?.matmul(&ht)?.affine(1.0, EPS)?;
    let w = w.mul(&w_num.div(&w_den)?)?;
    Ok((w, h))
}

fn rand_pos(shape: &[usize], dev: &Device) -> Result<Tensor> {
    // uniform in [0.1, 1.0], f32
    Ok(Tensor::rand(0.1f32, 1.0f32, shape, dev)?)
}

// Force the queued GPU work to actually complete (readback a scalar).
fn sync(t: &Tensor) -> Result<()> {
    let _ = t.sum_all()?.to_scalar::<f32>()?;
    Ok(())
}

// B problems run one at a time (2D). Used for CPU and Metal-single.
fn bench_sequential(dev: &Device, b: usize, n: usize, m: usize, k: usize, iters: usize) -> Result<f64> {
    let t0 = Instant::now();
    for _ in 0..b {
        let v = rand_pos(&[n, m], dev)?;
        let we = rand_pos(&[n, m], dev)?;
        let mut w = rand_pos(&[n, k], dev)?;
        let mut h = rand_pos(&[k, m], dev)?;
        let wev = we.mul(&v)?;
        for _ in 0..iters {
            let (w2, h2) = nmf_update(&we, &wev, &w, &h)?;
            w = w2;
            h = h2;
        }
        sync(&w)?; // realistic: result read back per solve
    }
    Ok(t0.elapsed().as_secs_f64())
}

// All B problems batched into one 3D tensor (B,n,m).
fn bench_batched(dev: &Device, b: usize, n: usize, m: usize, k: usize, iters: usize) -> Result<f64> {
    let t0 = Instant::now();
    let v = rand_pos(&[b, n, m], dev)?;
    let we = rand_pos(&[b, n, m], dev)?;
    let mut w = rand_pos(&[b, n, k], dev)?;
    let mut h = rand_pos(&[b, k, m], dev)?;
    let wev = we.mul(&v)?;
    for _ in 0..iters {
        let (w2, h2) = nmf_update(&we, &wev, &w, &h)?;
        w = w2;
        h = h2;
    }
    sync(&w)?;
    Ok(t0.elapsed().as_secs_f64())
}

fn run_case(cpu: &Device, metal: Option<&Device>, label: &str,
            b: usize, n: usize, m: usize, k: usize, iters: usize) -> Result<()> {
    // warmup (Metal compiles kernels lazily on first use)
    let _ = bench_batched(cpu, 2, n, m, k, 3)?;
    if let Some(g) = metal { let _ = bench_batched(g, 2, n, m, k, 3)?; }

    let cpu_seq = bench_sequential(cpu, b, n, m, k, iters)?;
    let cpu_bat = bench_batched(cpu, b, n, m, k, iters)?;
    let (gpu_seq, gpu_bat) = match metal {
        Some(g) => (bench_sequential(g, b, n, m, k, iters)?, bench_batched(g, b, n, m, k, iters)?),
        None => (f64::NAN, f64::NAN),
    };

    println!("\n== {label} :  V=({n}x{m})  k={k}  B={b}  iters={iters} ==");
    println!("  (a) CPU+Accelerate  sequential : {:8.3} s", cpu_seq);
    println!("      CPU+Accelerate  batched    : {:8.3} s", cpu_bat);
    println!("  (b) Metal GPU       sequential : {:8.3} s   ({:.2}x vs CPU-seq)", gpu_seq, cpu_seq / gpu_seq);
    println!("  (c) Metal GPU       batched    : {:8.3} s   ({:.2}x vs CPU-seq)", gpu_bat, cpu_seq / gpu_bat);
    Ok(())
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0).ok();
    println!("Metal available: {}", metal.is_some());

    let g = metal.as_ref();
    let iters = 500;

    // Realistic ESAT: Baltimore dataset is 307x41, factors 3-7.
    // run_case args: (b, n, m, k, iters)  ->  B solves of V=(n x m), rank k
    run_case(&cpu, g, "ESAT real size, few solves",     10, 307, 41, 6, iters)?;
    run_case(&cpu, g, "ESAT real size, bootstrap",     100, 307, 41, 6, iters)?;
    run_case(&cpu, g, "ESAT real size, big bootstrap",1000, 307, 41, 6, iters)?;

    // Large matrices: where GPU normally wins on a single solve.
    run_case(&cpu, g, "LARGE single solve",             1, 2000, 500, 20, iters)?;
    run_case(&cpu, g, "LARGE batched",                 50, 2000, 500, 20, iters)?;

    Ok(())
}
