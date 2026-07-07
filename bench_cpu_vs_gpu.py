import os, sys, time, json, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, "/Users/joallee/Project/esat-gpu-study")
from esat.model.batch_sa import BatchSA
from esat.error.bootstrap import Bootstrap
from esat.error.displacement import Displacement

from esat.data.datahandler import DataHandler

data_dir = "/Users/joallee/Project/esat-gpu-study/data"
dh = DataHandler(
    input_path=os.path.join(data_dir, "Dataset-BatonRouge-con.csv"),
    uncertainty_path=os.path.join(data_dir, "Dataset-BatonRouge-unc.csv"),
    index_col="Date"
)
V, U = dh.get_data()
print(f"V = {V.shape}  ({len(dh.features)} features, {V.shape[0]} samples)")
print(f"Factors: 6  Models: 20  Max iter: 2000  Seed: 42")

FACTORS, MODELS, MAX_ITER, SEED = 6, 6, 2000, 42

# --- CPU (official Rust single-solve) ---
t0 = time.time()
bsa_cpu = BatchSA(V=V, U=U, factors=FACTORS, models=MODELS, method="ls-nmf",
                  max_iter=MAX_ITER, seed=SEED, verbose=False,
                  parallel=False, use_gpu=False)
ok_cpu, _ = bsa_cpu.train()
t_cpu = time.time() - t0
sa_cpu = bsa_cpu.results[bsa_cpu.best_model]

# --- GPU (batched) ---
t0 = time.time()
bsa_gpu = BatchSA(V=V, U=U, factors=FACTORS, models=MODELS, method="ls-nmf",
                  max_iter=MAX_ITER, seed=SEED, verbose=False,
                  use_gpu=True, parallel=False)
ok_gpu, _ = bsa_gpu.train()
t_gpu = time.time() - t0
sa_gpu = bsa_gpu.results[bsa_gpu.best_model]

print()
print(f"Base Model Training ({MODELS} models, {MAX_ITER} iter):")
print(f"  {'':>15s} {'Time':>8s} {'Q(true)':>10s} {'Best':>5s}")
print(f"  {'CPU (Rust)':>15s} {t_cpu:>7.3f}s {sa_cpu.Qtrue:>9.0f} {bsa_cpu.best_model+1:>4d}")
print(f"  {'GPU (Metal)':>15s} {t_gpu:>7.3f}s {sa_gpu.Qtrue:>9.0f} {bsa_gpu.best_model+1:>4d}")
if t_gpu > 0:
    print(f"  {'Speedup':>15s} {t_cpu/t_gpu:>7.1f}x")

sa_cpu.metadata["converge_delta"] = 0.1
sa_cpu.metadata["converge_n"] = 100
sa_gpu.metadata["converge_delta"] = 0.1
sa_gpu.metadata["converge_n"] = 100

BS_N, BS_BLOCK, BS_THRESH = 20, 10, 0.6

# CPU Bootstrap
t0 = time.time()
bs_cpu = Bootstrap(sa=sa_cpu, feature_labels=dh.features,
                   bootstrap_n=BS_N, block_size=BS_BLOCK, threshold=BS_THRESH,
                   parallel=False, use_gpu=False)
bs_cpu.run(keep_H=True, block=True)
t_bs_cpu = time.time() - t0
q_cpu = [bs_cpu.bs_results[k]["model"].Qtrue for k in bs_cpu.bs_results]

# GPU Bootstrap
t0 = time.time()
bs_gpu = Bootstrap(sa=sa_gpu, feature_labels=dh.features,
                   bootstrap_n=BS_N, block_size=BS_BLOCK, threshold=BS_THRESH,
                   parallel=False, use_gpu=True)
bs_gpu.run(keep_H=True, block=True)
t_bs_gpu = time.time() - t0
q_gpu = [bs_gpu.bs_results[k]["model"].Qtrue for k in bs_gpu.bs_results]

print(f"Bootstrap ({BS_N} runs):")
print(f"  {'':>15s} {'Time':>8s} {'/run':>8s} {'Q range':>15s}")
print(f"  {'CPU (Rust)':>15s} {t_bs_cpu:>7.3f}s {t_bs_cpu/BS_N:>7.4f}s [{min(q_cpu):.0f}, {max(q_cpu):.0f}]")
print(f"  {'GPU (Metal)':>15s} {t_bs_gpu:>7.3f}s {t_bs_gpu/BS_N:>7.4f}s [{min(q_gpu):.0f}, {max(q_gpu):.0f}]")
if t_bs_gpu > 0:
    print(f"  {'Speedup':>15s} {t_bs_cpu/t_bs_gpu:>7.1f}x")

MAX_SEARCH, THRESH_DQ = 20, 0.1

# CPU DISP
t0 = time.time()
disp_cpu = Displacement(sa=sa_cpu, feature_labels=dh.features,
                        max_search=MAX_SEARCH, threshold_dQ=THRESH_DQ,
                        parallel=False, use_gpu=False)
disp_cpu.run()
t_disp_cpu = time.time() - t0

# GPU DISP
t0 = time.time()
disp_gpu = Displacement(sa=sa_gpu, feature_labels=dh.features,
                        max_search=MAX_SEARCH, threshold_dQ=THRESH_DQ,
                        parallel=False, use_gpu=True)
disp_gpu.run()
t_disp_gpu = time.time() - t0

print(f"DISP (6 factors, all features, max_search={MAX_SEARCH}):")
print(f"  {'':>15s} {'Time':>8s}  {'inc factors':>11s}  {'dec factors':>11s}")
print(f"  {'CPU (Rust)':>15s} {t_disp_cpu:>7.3f}s  {len(disp_cpu.increase_results):>10d}  {len(disp_cpu.decrease_results):>10d}")
print(f"  {'GPU (Metal)':>15s} {t_disp_gpu:>7.3f}s  {len(disp_gpu.increase_results):>10d}  {len(disp_gpu.decrease_results):>10d}")
if t_disp_gpu > 0:
    print(f"  {'Speedup':>15s} {t_disp_cpu/t_disp_gpu:>7.1f}x")

print("=" * 68)
print(f"{'':25s} {'CPU (Rust)':>12s} {'GPU (Metal)':>12s} {'Speedup':>8s}")
print("-" * 68)
print(f"{'Base model':25s} {t_cpu:>9.3f}s {t_gpu:>9.3f}s {t_cpu/t_gpu:>7.1f}x")
print(f"{'Bootstrap':25s} {t_bs_cpu:>9.3f}s {t_bs_gpu:>9.3f}s {t_bs_cpu/t_bs_gpu:>7.1f}x")
print(f"{'DISP':25s} {t_disp_cpu:>9.3f}s {t_disp_gpu:>9.3f}s {t_disp_cpu/t_disp_gpu:>7.1f}x")
print("-" * 68)
total_cpu = t_cpu + t_bs_cpu + t_disp_cpu
total_gpu = t_gpu + t_bs_gpu + t_disp_gpu
print(f"{'Total':25s} {total_cpu:>9.3f}s {total_gpu:>9.3f}s {total_cpu/total_gpu:>7.1f}x")
print("=" * 68)
print(f"Hardware:        M4 Pro MacBook")
bk = bs_gpu.metadata.get("backend", "?") if hasattr(bs_gpu, "metadata") else "?"
print(f"Rust backend:    {bk}")
print(f"Dataset:         Baton Rouge ({V.shape[0]} features, {V.shape[1]} samples)")
