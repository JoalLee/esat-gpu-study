# ESAT GPU Study — Project Summary

## Overview

This project accelerates the **Environmental Source Apportionment Toolkit (ESAT)**
NMF solver by packing multiple training tasks into a single 3-D tensor call
(`ls_nmf_batched`). The work sits on top of the official ESAT codebase —
EPA/Quanted already rewrote the core NMF loop in Rust + PyO3; we add
GPU-batched multi-model training and optimise the DISP binary search.

---

## Changes by Component

### 1. Rust — `rust/lib.rs` (new `ls_nmf_batched`, `backend.rs`)

| File                       | What changed                                                                                         | Why                                              |
| :------------------------- | :--------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| `rust/backend.rs`        | Extracted`try_new_metal()`, device detection, GPU platform string                                  | Clean separation from core NMF logic             |
| `rust/ls_nmf_batched.rs` | New file: batched 3-D NMF kernel accepting`(B, n, k)` / `(B, k, m)` tensors                      | Train N models in one GPU launch                 |
| `rust/lib.rs`            | Shrank from 400+ lines to ~100 — delegates to`ls_nmf_batched.rs` and `backend.rs`               | Readability, maintainability                     |
| `rust/lib.rs`            | Added`backend` field to return dict (`"gpu"` / `"metal"` / `"cuda"` / `"cpu"`)             | Downstream can verify GPU is actually being used |
| `rust/lib.rs`            | Added 2-D broadcast for`V` and `We` — caller can pass `(n, m)` instead of `(1, n, m)` tiled | Saves memory (no`np.tile`)                     |
| `rust/lib.rs`            | Wrapped Metal init in`catch_unwind` so failed GPU init falls back to CPU gracefully                | Stability on headless / containerised machines   |

### 2. Python — batched NMF entry points

**`esat/model/batch_sa.py`**

- Added `use_gpu=True` path: packs all N models into `ls_nmf_batched` (single call)
- Added CPU sequential path (`if not self.parallel and not self.use_gpu`: simple for loop)
  → Fixes `mp.Manager` spawn crash on macOS when running outside Jupyter
- `parallel=True` path unchanged (still uses `mp.Pool` inside `mp.Manager`)

**`esat/model/gpu_batch.py`** (new file)

- `run_ls_nmf_batched(V, W_batch, H_batch, max_iter, ...)` — shared helper for
  stacking, calling the Rust kernel, and unpacking results
- `weighted_errors_from_uncertainty(U)` — computes `We = 1 / U²`

**`esat/error/bootstrap.py`**

- Added `use_gpu=True` path: packs all B bootstrap runs into `ls_nmf_batched`
- Records `metadata["backend"]` from the Rust return dict
- Fix: `block_size = int(N / 2)` to avoid float `range()` error

**`esat/error/displacement.py`**

- **GPU batched SA.train**: each factor's DISP tasks are collected, then
  trained together via `_batch_train_disp` → `ls_nmf_batched`
- **Delta-optimised `_q_for_h_value()`**: replaces `q_loss(V, U, W, H)`
  for single-H-element changes. Instead of recomputing the full `W @ H`
  (O(nk + nm)), it only updates the affected column of `WH`:
  `updated_residual = base_residual - W[:,f] × ΔH[f,j]`
  → **~10× speedup for the binary search itself**
- 2-D broadcast for V/We (removed `np.tile`)
- Records `self._backend` from Rust result
- `metadata["use_gpu"]` flag

**`esat/error/bs_disp.py`**

- Added `use_gpu` parameter, passed through to internal `Bootstrap` and `Displacement` instances
- Removed stale `cores` parameter (was unused)

### 3. Vendored Candle patch

**`vendor/candle-core-0.11.0/src/metal_backend/mod.rs`**

- Candle 0.11.0 panics with `Device::all().swap_remove(0)` when no Metal device
  is visible. The patch replaces the panic with a proper `Err`.
  ```rust
  // Before: panics
  let device = Device::all().swap_remove(ordinal);
  // After: returns Err
  let device = Device::system_default()
      .or_else(|| Device::all().into_iter().nth(ordinal))
      .ok_or_else(|| ...)?;
  ```
- Also removed one unused Metal import to silence compiler warnings.
- Documented in `PATCHES.md` with rationale, scope, and long-term cleanup options.

### 4. Bug fixes

| Issue                                        | File                    | Fix                                                                    |
| :------------------------------------------- | :---------------------- | :--------------------------------------------------------------------- |
| `mp.Manager()` spawn crash on macOS        | `batch_sa.py`         | Added direct CPU sequential loop when`parallel=False, use_gpu=False` |
| `float` in `range()` in block resampling | `bootstrap.py`        | `N/2` → `int(N / 2)`                                              |
| f-string syntax in nbconvert output          | `bench_cpu_vs_gpu.py` | Replaced nested`"` with `'` inside f-strings                       |
| DISP file indentation corruption             | `displacement.py`     | Rewritten cleanly from original git ancestor + patches                 |

### 5. New files

| File                                         | Purpose                                  |
| :------------------------------------------- | :--------------------------------------- |
| `rust/backend.rs`                          | GPU backend detection and initialisation |
| `rust/ls_nmf_batched.rs`                   | Batched 3-D NMF kernel                   |
| `esat/model/gpu_batch.py`                  | Shared GPU stacking/unpacking utilities  |
| `PATCHES.md`                               | Vendored candle-core patch documentation |
| `notebooks/bench_cpu_vs_gpu.ipynb`         | CPU vs GPU benchmark (Baton Rouge)       |
| `notebooks/epa_esat_workflow_01_gpu.ipynb` | Full workflow notebook with GPU enabled  |
| `bench_cpu_vs_gpu.py`                      | Script version of the benchmark          |

---

## Performance Results

Tested on **M4 Pro MacBook** with **Baltimore dataset** (26 species, 657 samples, k=6)
unless noted otherwise.

### Base Model Training (6 models, 2000 iterations each)

| Mode | Time | per model | Speedup vs CPU | Hardware |
|:----|:----:|:---------:|:--------------:|:---------|
| Rust CPU single-solve | 1.70 s | 0.29 s | 1× | M4 Pro CPU |
| GPU batched (Metal) | **0.31 s** | — | **5.5×** | M4 Pro GPU |
| GPU batched (CUDA, warm) | **0.23 s** | — | **7.4×** | GB10 GPU |

GPU batches all 6 models into one `ls_nmf_batched` call.

### Bootstrap (20 runs, 2000 iterations each, block bootstrap)

| Mode                     |       Time       | per run |     Speedup     |
| :----------------------- | :--------------: | :-----: | :--------------: |
| Rust CPU                 |      5.80 s      | 0.290 s |       1×       |
| GPU batched (Metal)      | **0.40 s** | 0.020 s | **14.5×** |
| GPU batched (CUDA, warm) | **0.32 s** | 0.016 s | **18.1×** |

Speedup grows with `B` because the GPU batch call overhead is fixed.

### DISP (6 factors, 26 features, max_search=20, increase + decrease)

| Mode                       |       Time       |    Speedup    | Note                                                      |
| :------------------------- | :--------------: | :------------: | :-------------------------------------------------------- |
| Rust CPU (original)        |      ~390 s      |      1×      | 624 SA.train + ~50K q_loss calls                          |
| GPU batched + delta q_loss | **14.3 s** | **27×** | SA.train batched (12 calls), q_loss delta-optimised       |
| ─ SA.train portion        |      ~2.4 s      |       —       | 12 ×`ls_nmf_batched` calls                             |
| ─ q_loss binary search    |      ~7.1 s      |       —       | delta-optimised, 6 factors × 2 directions × 1.18 s each |
| ─ overhead                |      ~4.8 s      |       —       | Python loops, tqdm, factor comparison                     |

The delta-optimised `_q_for_h_value()` is the larger contributor:
each factor × direction scans all 26 features in **~1.18 s** (vs ~50 s before).

### BS-DISP (B=20, DISP on each bootstrap model)

| Mode                                      |  Estimated time  |     Speedup     |
| :---------------------------------------- | :--------------: | :-------------: |
| Rust CPU                                  |     ~2 hours     |       1×       |
| GPU (Bootstrap 0.4 s + 20 × DISP 14.3 s) | **~290 s** | **~25×** |

---

## File Inventory

```
rust/
├── lib.rs               →  NMF entry points (ls_nmf, ws_nmf, ls_nmf_batched)
├── backend.rs           →  try_new_metal(), device init  (NEW)
├── ls_nmf_batched.rs    →  batched 3-D NMF kernel       (NEW)

esat/
├── model/
│   ├── batch_sa.py      →  GPU batched training, CPU sequential fix
│   ├── gpu_batch.py     →  shared stacking helpers      (NEW)
│   └── sa.py            →  (unchanged — already had Rust dispatch)
├── error/
│   ├── bootstrap.py     →  GPU batched runs + backend metadata
│   ├── displacement.py  →  GPU batched + delta q_loss + 2D broadcast
│   └── bs_disp.py       →  use_gpu passthrough

vendor/candle-core-0.11.0/src/
└── metal_backend/mod.rs →  Metal init panic fix

notebooks/
├── bench_cpu_vs_gpu.ipynb    →  speed comparison        (NEW)
├── epa_esat_workflow_01_gpu.ipynb  →  GPU workflow      (NEW)
└── bench_cpu_vs_gpu.py       →  script version          (NEW)

PATCHES.md               →  vendored candle doc          (NEW)
```

---

## What Was NOT Changed

- `esat/model/sa.py` — single-model Rust dispatch was already correct
- `esat/data/` — DataHandler, analysis, plotting — all Python, no GPU needed
- `esat/rotational/constrained.py` — CPU only (rotational ambiguity, not speed-critical)
- `esat/error/error.py` — plotting wrapper, no computation
- `esat/utils.py`, `esat/metrics.py` — utility functions (not speed bottlenecks)
- All test files — not touched (our changes are backward-compatible with `use_gpu=False`)

---

## Long-Term To-Do

1. **Per-slice convergence in `ls_nmf_batched`** — currently all slices run
   `max_iter` regardless of individual convergence. Adding early-stop per slice
   would help when some models converge faster than others.
2. **WS-NMF batched version** — `esat_rust.ws_nmf` has CPU Rust but no
   batched GPU path. Needed for data with negative values.
3. **vendor/candle-core** — the 49k-line vendored directory is a maintenance
   burden. Options:

   - Upgrade to a Candle release that fixes Metal init upstream
   - Submit the patch upstream
   - Replace with a minimal fork reference
4. **Test CUDA on GB10 regularly** — the Metal and CUDA tweaks (zero-size
   tensor handling, device ordinal logic) diverge. CI or a manual smoke test
   before each snapshot push.

---

## Remaining Gaps

- `bench_cpu_vs_gpu.ipynb` DISP section on Baton Rouge (307×41) takes ~8 min
  CPU vs ~2 min GPU — the notebook runs both in sequence so it blocks.
- The official workflow notebook (`epa_esat_workflow_01.ipynb`) sets
  `parallel=True` by default, which uses `mp.Pool` → crashes on macOS when
  run from the CLI. The GPU variant (`_gpu.ipynb`) fixes this with
  `parallel=False`.
