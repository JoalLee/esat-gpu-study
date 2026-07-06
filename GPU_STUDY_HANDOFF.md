# ESAT GPU 加速研究 — 交接報告

> 目的：把 ESAT 中「可以用 GPU 的部分」實作為 GPU 版本。
> 本報告記錄目前調查結論、benchmark 實證、以及可直接執行的下一步規劃。
> 平台：macOS / Apple Silicon（Metal）。GPU backend：`candle-core` 0.11。

---

## 1. 一句話結論

**不要**為單一 NMF solve 加 GPU 路徑（已實測是負收益）；
**要**做的是一個**批次化（batched）的 NMF kernel**，把大量獨立的小 solve 堆成一個 3D tensor 一次算完。
在真實 ESAT 尺寸下，批次 GPU 有 **5–6× 加速**，單次 GPU 反而比 CPU **慢 2.6×**。

---

## 2. 背景：ESAT 裡哪些東西能上 GPU

ESAT 的核心是兩個 NMF 迭代 kernel（Rust 端）：
- `ls_nmf()` — least-squares NMF（`esat/model/ls_nmf.py` 的 Python 對照）
- `ws_nmf()` — weighted-semi NMF

`SA.train()`（`esat/model/sa.py:398`）在 optimized 模式下，把 V/U/We/W/H 轉成 float32，連同 `use_gpu` flag 一次丟進 Rust，整個 `max_iter` 迴圈都在 Rust 內完成。

### GPU 適用性分層（已用實際 call site 驗證）

| 層級 | 內容 | 能否 GPU |
|------|------|---------|
| **A：真正的 kernel** | `ls_nmf.update`（純 matmul + elementwise）、`ws_nmf.update`（matmul + 每 factor 一次 `pinv`） | ✅ LS 乾淨；⚠️ WS 的 `pinv` 是難點 |
| **B：間接受益（迴圈留在 Python）** | `BatchSA`（N models）、`Bootstrap`（resample→refit）、`Displacement`、`BS-DISP`、`Estimator`（rank selection, MC refit） | 🔁 只是「重複呼叫 kernel」，要快必須**批次化** |
| **C：無法 GPU** | resample 索引、correlation mapping、p-value、MSE、masking、preprocessing、所有 plot | ❌ 永遠 CPU |

**關鍵洞察**：Tier B 全部是「大量獨立的小 NMF solve」。它們不是靠「單次 solve 變快」得利，而是靠「一次算一大批」得利。

---

## 3. 兩個決定成敗的限制條件

1. **GPU vs. multiprocessing 互斥**
   ESAT 現有提速靠 CPU 多核（`esat/model/batch_sa.py:255` 的 `mp.Pool`）。這跟「單一 GPU」衝突：
   - 多 process 搶同一張 Metal GPU → 無效益。
   - **macOS 上 `fork` + Metal = 直接 crash**（Obj-C runtime 不能跨 fork）。
   - → 走 GPU 就要放棄 process 級平行，改由**單一 process 獨佔 GPU 並批次化**。

2. **矩陣太小**
   真實資料 `data/Dataset-Baltimore` = **307 samples × 41 features**，factors 3–7。
   GPU 只在大 matmul 上贏；小矩陣時 kernel launch + host↔device 傳輸開銷蓋過收益。
   → 單次 solve 在 GPU 上必輸給 CPU 的 Accelerate BLAS（已實測）。

---

## 4. Benchmark 實證

程式：`test_candle/src/nmf_bench.rs`（複製 `ls_nmf.py:39-49` 的乘法更新迴圈）。
重跑方式：
```bash
cd test_candle && cargo run --release --bin nmf_bench
```
設定：Apple Metal，f32，每個 solve 跑 500 iters。三種策略：
- (a) CPU + Accelerate BLAS，逐一 solve
- (b) Metal GPU，逐一 solve（= 目前 `prefer_gpu` flag 會走的路）
- (c) Metal GPU，批次（B 個 solve 堆成一個 3D tensor）

### 真實 ESAT 尺寸 V=(307×41), k=6

| B (solves) | (a) CPU+Accelerate | (b) Metal 單次 | (c) Metal 批次 |
|---:|---:|---:|---:|
| 10  | 0.101 s | 0.270 s ❌ 0.38× | 0.037 s ✅ **2.7×** |
| 100 (bootstrap) | 0.982 s | 2.597 s ❌ 0.38× | 0.164 s ✅ **6.0×** |
| 1000 (big bootstrap) | 9.894 s | 25.8 s ❌ 0.38× | 1.928 s ✅ **5.1×** |

### 大矩陣 V=(2000×500), k=20（對照組）

| B | CPU+Accelerate | Metal 單次 | Metal 批次 |
|---:|---:|---:|---:|
| 1  | 0.371 s | 0.185 s ✅ 2.0× | 0.179 s ✅ 2.1× |
| 50 | 18.8 s  | 8.74 s ✅ 2.15× | 7.16 s ✅ 2.6× |

（倍率相對 CPU 逐一；❌ = 比 CPU 慢）

### 讀出來的三件事
1. **單次 GPU 在 ESAT 尺寸穩定 0.38×（慢 2.6 倍）** → 目前的 GPU flag 路徑若 ship 就是效能退步。
2. **批次 GPU 是唯一贏法**，5–6× 且隨 batch 放大。同樣 B=1000 工作量，GPU 單次逐一 25.8s vs GPU 批次 1.93s → 光批次化就差 **13×**。
3. GPU 只在大矩陣才單次就贏，但 ESAT 矩陣不大，此路不通。

---

## 5. 目前程式碼狀態（未 commit 的 working changes）

- `rust/lib.rs`
  - `ls_nmf` / `ws_nmf` 的 device 目前被寫死 `Device::Cpu`（不是真 Metal），`use_gpu = prefer_gpu`。
  - tensor dtype 已從 f64 改為 **f32**（正確：Apple GPU 無 f64 硬體支援）。
  - 殘留 **DEBUG 探針**（`PRE-UPDATE w*h FAILED` 等），需清除。
  - 有一批未使用的手刻線代數函式（`calculate_inverse`/`determinant`/`adjoint`/`qr_decomposition`）—— 為 WS-NMF 的求逆準備，尚未接上。
- `Cargo.toml`：主 crate 的 candle 是 `default-features = false`（純 CPU，**沒有 metal、沒有 accelerate**）。
- `test_candle/`：獨立沙盒，candle features = `["metal", "accelerate"]`，Metal matmul 已驗證可用（`src/main.rs`），benchmark 在 `src/nmf_bench.rs`。
- `esat_rust.cpython-312-darwin.so`：已編譯（目前是 CPU 版）。
- 待清理：`Cargo.toml.bak`。

**已驗證的事實**：`candle Metal matmul (f32)` 在 pure-Rust binary 完全正常。前面「Metal 在 PyO3 下無法初始化」的說法不成立——真正踩到的錯是 Metal 無 f64（硬體限制），已用 f32 解掉。

---

## 6. 下一步規劃（交給 Agent 執行）

### 里程碑 M1 — LS-NMF 批次 kernel 原型（最高優先，收益明確）
1. 在 `rust/lib.rs` 新增函式 `ls_nmf_batched`，輸入 3D tensor：
   - `V, We: (B, n, m)`、`W: (B, n, k)`、`H: (B, k, m)`。
   - 更新式沿用 `nmf_bench.rs` 的 `nmf_update`（candle `matmul` / `.t()` 對最後兩維自動批次）。
2. Device：用 `Device::new_metal(0)`（真 Metal，非現在的 `Device::Cpu`）。全程 f32。
3. **正確性驗證優先於效能**：對同一組輸入，批次 kernel 每個 slice 的結果，要和現有逐一 `ls_nmf` 數值一致（設容差，如相對誤差 < 1e-4）。寫成一個 Rust 或 Python 測試。
4. 再測效能，對照本報告 §4 的表確認 5–6×。

### 里程碑 M2 — 收斂處理
- 批次內不同 solve 收斂速度不同。先實作最簡單版：**全部跑到 `max_iter`**（已足夠打敗 CPU）。
- 進階（可選）：對已收斂的 slice 做 mask 凍結，減少浪費。

### 里程碑 M3 — Python 接線 + 放棄多核衝突
1. 在 `esat/model/batch_sa.py`：當 `use_gpu=True` 時，走「單 process + 批次 GPU」路徑，**繞過 `mp.Pool`**（避免 fork+Metal crash 與 GPU 爭用）。
2. 讓 `Bootstrap` / `Estimator` 在 `use_gpu=True` 時，把它們的 N 個 solve 收集成一個批次呼叫 `ls_nmf_batched`，而不是迴圈呼叫 `ls_nmf`。
3. 保留現有 CPU 多核路徑作為 `use_gpu=False` 的預設。

### 里程碑 M4 — WS-NMF（硬骨頭，最後做）
- WS-NMF 每 factor 一次 `pinv`（`esat/model/ws_nmf.py:59`）。
- candle Metal 的 batched 矩陣求逆支援有限、且 f32 數值風險高。
- 選項：(a) 只把 matmul 部分上 GPU、`pinv` 拉回 CPU；(b) 用 batched Cholesky/解線性系統取代顯式 inverse；(c) WS-NMF 暫不上 GPU。
- **建議先評估再實作，不要一開始就投入。**

### 清理任務（順手）
- 移除 `rust/lib.rs` 的 DEBUG 探針（`PRE-UPDATE ... FAILED`）。
- 決定主 `Cargo.toml` 是否要加入 `metal` + `accelerate` features（M1 需要 metal）。
- 刪除 `Cargo.toml.bak`。

---

## 7. 風險與待決事項

| 項目 | 說明 |
|------|------|
| f32 精度 | Apple GPU 無 f64。NMF 是迭代放大的乘法更新，需驗證 f32 對最終 Q/MSE 的影響是否可接受。 |
| PyO3 + Metal 交集 | pure-Rust Metal 已驗證 OK；PyO3 `.so` + Metal 尚未實測（M1 會第一次驗證）。理論上可行（PyTorch MPS 即是先例）。 |
| WS-NMF pinv | 見 M4，最大不確定性。 |
| 收斂數不齊 | 批次全跑滿 vs. mask，效能/複雜度取捨。 |
| 記憶體 | B 很大時 3D tensor 佔用；B=1000, 307×41 沒問題，需注意 WS-NMF 大 batch。 |

---

## 8. 關鍵檔案索引

- `test_candle/src/nmf_bench.rs` — 效能 benchmark（可重跑、可改尺寸）
- `test_candle/src/main.rs` — Metal matmul 健全性測試
- `rust/lib.rs` — `ls_nmf` / `ws_nmf` 現有實作（`ls_nmf_update_gpu` 在 ~L420、`ws_nmf_update_gpu` 在 ~L795）
- `esat/model/ls_nmf.py` / `ws_nmf.py` — Python 對照的更新式（正確性驗證基準）
- `esat/model/sa.py:398` — kernel 呼叫點
- `esat/model/batch_sa.py:255` — 現有多核平行（M3 要繞過）
- `esat/error/bootstrap.py`、`esat/estimator.py` — Tier B 批次化目標
