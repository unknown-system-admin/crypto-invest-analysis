# 動能趨勢分析設計（Report Momentum Trend）

日期：2026-08-08

## 目標

手動觸發日報時，用長週期框架（預設 1d）重現「前兩次」的動能狀態，評估出動能的演進方向（例如「動能偏多，但開始減弱」）。**不儲存歷史記錄**，純按當下觸發即時回推計算。

## 背景決策

- 原本想用排程每 4 小時記錄一次 `direction + strength` 比對 → 被否定（需常駐儲存、受排程驅動）
- 使用者偏好：單次觸發、無狀態、每次重算（資料自然最新）
- 持有週期為日/週級（BTC、山寨幣），故預設框架用 1d K 線

## API 設計

```
GET /report?symbol=<SYMBOL>&tf=1d&step=1
```

- `symbol`（可選）：指定單一 symbol；省略則跑 `config.yaml` 全部 symbols
- `tf`（可選，預設 `1d`）：時間框架，可為 `1h` / `4h` / `1d` / `1w`
- `step`（可選，預設 `1`）：比較間隔的 bar 數（整數 ≥ 1）

由 `config.yaml` 提供：
```yaml
reports:
  default_tf: "1d"
  default_step: 1
```
查詢參數覆蓋 config 預設。

## 動能狀態（momentum state）

每次計算得出 `{ direction, strength }`：

- `direction`：偏多 / 偏空 / 震盪 — 沿用現有 `analyze_signals()` 的 `direction` 欄位
- `strength`：強 / 中 / 弱
  - `bullish_count/total >= 0.75` → 強
  - `0.5 <= bullish_count/total < 0.75` → 中
  - `bullish_count/total < 0.5` → 弱

## 三個時間點（同一份 DataFrame offset）

| 時間點 | 索引 | 對應 |
|--------|------|------|
| 本次 | `df.iloc[-1]` | 現在 |
| 前一次 | `df.iloc[-1-step]` | `step` bars 前 |
| 前兩次 | `df.iloc[-1-2*step]` | `2*step` bars 前 |

同一份 `tf` 的 OHLCV DataFrame 套用三種 offset，各算一次 momentum state。

## 演進判斷

依三個時間點的方向與強度變化歸類。以**本次（最新）時間點為基準**，比較「前一次」與「前兩次」：

- **方向改變**（本次方向 ≠ 前一次或前兩次方向）→「**方向反轉**」
- **方向不變**（本次、前一次、前兩次方向皆相同）：
  - 強度遞減（強→中/弱、中→弱，沿時間軸）→「**減弱中**」
  - 強度遞增（弱→中/強、中→強，沿時間軸）→「**增強中**」
  - 強度不變 →「**維持**」

### 文案範例
- `偏多(強) → 偏多(中) → 偏多(動) — 減弱中`
- `偏空(弱) → 偏空(中) → 偏空(強) — 增強中`
- `偏多(強) → 偏多(強) → 偏多(強) — 維持`
- `偏空 → 偏多 → 偏多 — 方向反轉`

## 呈現

在 `build_report_embed` 中增加「⚡ 動能演進」欄：
```
⚡ 動能演進: 偏多(強) → 偏多(中) → 偏多(弱) — 減弱中
```

## 檔案影響

- `analysis/momentum.py`（新增）：momentum state 計算 + 演進判斷
- `monitor/main.py`：`/report` 支援 `tf`/`step` 參數、接入 momentum
- `monitor/notifier.py`：`build_report_embed` 加入「⚡ 動能演進」文字
- `monitor/config.yaml`：新增 `reports:` 區段
- `tests/test_momentum.py`：新增測試（state 計算、演進判斷、API 參數）

## 測試重點（驗證意圖）

- 不與日系框架：給特定的 `bullish_count/total`，驗證 direction/strength 正確 mapping
- 給三組 state，驗證演進歸「增/減/維/反轉」符合預期
- `/report?tf=4h&step=4` 能正確套用 offset 並回傳趨勢文字
- 省略參數時使用 config 預設