# ESAT GPU Study Repo Guide

這份文件是給實驗室同仁快速理解本 repo 用的。官方 ESAT 背景請先看
`README.md`；GPU 開發細節請看 `GPU_DEV_NOTES.md` 和
`GPU_STUDY_SUMMARY.md`。本文件重點是：每個資料夾放什麼、哪些檔案可以用、
哪些只是開發產物，以及第一次 clone 後怎麼跑。

## 先看這幾個檔案

| 檔案                     | 用途                                                            |
| ------------------------ | --------------------------------------------------------------- |
| `README.md`            | 官方 ESAT 介紹、基本 API、原始 workflow 背景                    |
| `LAB_REPO_GUIDE.md`    | 本文件，repo 地圖與實驗室使用說明                               |
| `GPU_DEV_NOTES.md`     | GPU 分支的 build 指令、backend 檢查、開發限制                   |
| `GPU_STUDY_SUMMARY.md` | GPU 加速修改摘要與 benchmark 結果整理                           |
| `PATCHES.md`           | 為什麼 repo 內有 vendored`candle-core` patch                  |
| `pyproject.toml`       | Python package 設定、CLI entry point、Rust extension build 設定 |
| `Cargo.toml`           | Rust/PyO3 extension 設定、CUDA/Metal feature 開關               |

## 第一次使用

建議每個人都用獨立 virtual environment，不要直接裝到系統 Python。

```bash
cd /path/to/esat-gpu-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install maturin pytest jupyterlab
python -m pip install -r requirements.txt
```

Apple Silicon / Metal:

```bash
maturin develop --release --features metal
```

NVIDIA CUDA:

```bash
maturin develop --release --features cuda
```

CPU-only fallback:

```bash
maturin develop --release
```

檢查目前 Python import 到哪個 `esat_rust`：

```bash
python -c 'import esat_rust; print(esat_rust.__file__)'
```

如果從 repo 根目錄 import 到 `./esat_rust.cpython-*.so`，代表根目錄有 stale
extension artifact，可能蓋掉 `.venv` 裡剛 build 的版本。正常開發時應該 import
到 `.venv/lib/python.../site-packages/esat_rust/` 底下。

檢查 GPU backend 是否真的有跑：

```bash
python -c 'import numpy as np, esat_rust; V=np.ones((2,3)); We=np.ones((2,3)); W=np.ones((1,2,2)); H=np.ones((1,2,3)); print(esat_rust.ls_nmf_batched(V, We, W, H, 1, False, True)["backend"])'
```

不要只看 `use_gpu=True`。如果回傳是 `cpu`，代表它 fallback 到 CPU。

## 主要資料夾

| 路徑                           | 類型                      | 用途                                                     | 同仁通常要不要改                         |
| ------------------------------ | ------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| `esat/`                      | 核心 Python package       | ESAT 的主要 API、資料處理、NMF 模型、error estimation    | 只有開發功能或修 bug 時才改              |
| `rust/`                      | Rust/PyO3 extension       | Rust NMF kernel、GPU batched kernel、backend 初始化      | 只有 GPU/kernel 開發時才改               |
| `vendor/candle-core-0.11.0/` | vendored dependency patch | 本 repo 暫時 patch Candle 0.11 的 Metal 初始化問題       | 不要隨便改，先看`PATCHES.md`           |
| `tests/`                     | pytest 測試               | Python/Rust 整合與功能 regression tests                  | 改功能時要補或跑                         |
| `data/`                      | 範例與實驗資料            | 官方範例資料、PMF_data FS0610 資料                       | 可讀取；新增大資料前先確認是否要進 git   |
| `scripts/`                   | 可重跑腳本                | 目前主要是 FS0610 PMF 分析腳本                           | 建議新增可重現分析放這裡                 |
| `notebooks/`                 | Notebook 範例與探索       | 官方 workflow、GPU workflow、benchmark notebook、舊探索  | 可用，但不要把所有 notebook 都當正式流程 |
| `eval/`                      | 評估/模擬工具             | factor comparison、perturbation、runtime test、simulator | 進階驗證時用                             |
| `docs/`                      | Sphinx 文件               | 產生 API 文件的`.rst` 與 HTML 產物                     | 一般使用者不用改                         |
| `paper/`                     | 論文素材                  | JOSS paper markdown / bib                                | 通常不用改                               |
| `output/`                    | 本機輸出                  | 跑 FS0610 或 benchmark 產出的結果                        | 不應當作 source code                     |
| `test_candle/`               | Rust probe project        | 單獨測 Candle/Metal/CUDA 行為                            | 只在 debug backend 時用                  |

## `esat/` 內部結構

| 路徑                           | 用途                                                                  |
| ------------------------------ | --------------------------------------------------------------------- |
| `esat/data/`                 | 讀取濃度/不確定性資料、資料分析與 imputation 工具                     |
| `esat/model/sa.py`           | 單一 SA/NMF model 物件                                                |
| `esat/model/batch_sa.py`     | 多模型訓練入口；`BatchSA(use_gpu=True)` 會走 GPU batched LS-NMF     |
| `esat/model/gpu_batch.py`    | Python 端包裝 Rust`ls_nmf_batched` 的 helper                        |
| `esat/model/ls_nmf.py`       | LS-NMF update 邏輯                                                    |
| `esat/model/ws_nmf.py`       | WS-NMF update 邏輯；目前沒有 GPU batched 版本                         |
| `esat/error/bootstrap.py`    | Bootstrap error estimation；目前有 GPU batched LS-NMF 路徑            |
| `esat/error/displacement.py` | DISP error estimation；有 GPU batched training 與 delta Q search 優化 |
| `esat/error/bs_disp.py`      | Bootstrap + DISP 組合流程                                             |
| `esat/rotational/`           | constrained / rotational ambiguity 相關功能                           |
| `esat/cli/`                  | `esat` command line 入口                                            |
| `esat/metrics.py`            | Q loss / robust Q loss 等評估函式                                     |
| `esat/utils.py`              | 共用工具函式                                                          |

## `rust/` 內部結構

| 檔案                       | 用途                                                   |
| -------------------------- | ------------------------------------------------------ |
| `rust/lib.rs`            | PyO3 module 入口，暴露 Python 可呼叫的 Rust functions  |
| `rust/backend.rs`        | CPU/Metal/CUDA device 初始化與 fallback 管理           |
| `rust/ls_nmf_batched.rs` | GPU batched LS-NMF kernel；接受 2-D/3-D`V` 和 `We` |

目前 GPU 主要支援 LS-NMF batched path。WS-NMF GPU batched 尚未完成。

## Notebook 怎麼看

| Notebook                                     | 建議用途                                    |
| -------------------------------------------- | ------------------------------------------- |
| `notebooks/epa_esat_workflow_01.ipynb`     | 官方 ESAT workflow 參考                     |
| `notebooks/epa_esat_workflow_01_gpu.ipynb` | GPU 版 workflow 參考                        |
| `notebooks/bench_cpu_vs_gpu.ipynb`         | CPU/GPU benchmark 探索                      |
| `notebooks/epa_esat_simulator_01.ipynb`    | synthetic data simulator 範例               |
| `notebooks/exp/`                           | 開發探索，不保證乾淨或最新                  |
| `notebooks/old/`                           | 舊 notebook，除非追歷史，否則不要從這裡開始 |
| `notebooks/proto/`                         | prototype notebook，通常不是正式流程        |

如果只是要跑一個可重現分析，優先用 `scripts/` 裡的 `.py` 腳本，不要只依賴
notebook cell execution state。

## Data 與 output

`data/` 放輸入資料：

| 路徑                                 | 內容                                                   |
| ------------------------------------ | ------------------------------------------------------ |
| `data/Dataset-Baltimore_*`         | 官方 Baltimore 範例資料                                |
| `data/Dataset-BatonRouge-*`        | 官方 Baton Rouge 範例資料                              |
| `data/Dataset-StLouis-*`           | 官方 St. Louis 範例資料                                |
| `data/PMF_data/species_FS0610.csv` | FS0610 species/concentration 資料 (TClab, MingHan提供) |
| `data/PMF_data/unc_FS0610.csv`     | FS0610 uncertainty 資料 (TClab, MingHan提供)          |

`output/` 放跑出來的結果，例如 `output/pmf_fs0610*/`。這些結果可以用來檢查，
但不要把它們當成程式碼來源。若要分享正式結果，建議說明產生它的 script、
commit、參數和 backend。

## 常用指令

跑測試：

```bash
pytest tests -q
```

只跑 GPU batched LS-NMF 相關測試：

```bash
pytest tests/model/test_ls_nmf_batched.py -q
```

只跑 Bootstrap / BS-DISP 相關測試：

```bash
pytest tests/error/test_bootstrap.py tests/error/test_bs_disp.py -q
```

跑 FS0610 分析腳本：

```bash
python scripts/run_pmf_fs0610.py
```

如果要確認目前 repo 狀態：

```bash
git status --short --branch
```

## 不要提交的東西

以下通常是本機產物或暫存，不應該進 git：

- `.venv/`, `venv/`
- `target/`, `test_candle/target/`
- `__pycache__/`, `.pytest_cache/`
- `*.so`, `*.pyc`, `*.prof`, `*.log`
- `esat.egg-info/`, `dist/`, `build/`
- notebook checkpoints: `.ipynb_checkpoints/`
- 大型 benchmark output、臨時 profiling output
- 沒有說明來源與參數的 `output/` 結果

根目錄如果出現 `esat_rust.cpython-*.so`，尤其要小心。它可能讓 Python import
到舊的 extension，導致剛 build 的 Metal/CUDA 版本沒有生效。

## 開發規則

1. 修改核心功能時，同步補測試。
2. 不要只靠 notebook 證明結果；重要流程要能用 script 重跑。
3. GPU 是否真的啟用，要看 Rust 回傳的 `backend`，不要只看 `use_gpu=True`。
4. `vendor/candle-core-0.11.0/` 是暫時方案；除非在修 Candle backend，不要改。
5. WS-NMF 和 LS-NMF 不等價；目前 GPU batched 工作主要在 LS-NMF。
6. 若要把結果給其他人，至少附上 commit hash、build feature (`metal`/`cuda`/CPU)、
   `max_iter`、模型數、bootstrap 數，以及輸入資料版本。

## 目前已知限制

- GPU batched LS-NMF 會跑滿 `max_iter`，收斂追蹤不完全等同原本逐模型路徑。
- WS-NMF 還沒有 GPU batched 實作。
- DISP 的搜尋流程仍有 Python-side loop；GPU 主要加速重訓練部分。
- `vendor/candle-core-0.11.0/` 造成 repo 比較大，長期應改成 upstream fix 或小 patch。
- notebook 可能含有舊 output 或探索狀態；正式分析請優先找 script。

## 同仁如果只想跑現成流程

建議順序：

1. 看 `README.md` 理解 ESAT/PMF 的基本概念。
2. 看本文件理解 repo 結構。
3. 建 `.venv` 並用 `maturin develop --release --features metal` 或
   `--features cuda` build。
4. 跑 backend smoke test，確認不是 fallback 到 CPU。
5. 跑 `pytest tests/model/test_ls_nmf_batched.py -q`。
6. 從 `notebooks/epa_esat_workflow_01_gpu.ipynb` 或
   `scripts/run_pmf_fs0610.py` 開始。
