# HFT 訊號改善 v2 設計（資料源 + ATR 停損 + 波動率過濾）

日期：2026-09-06

## 背景

HFT 機器人 v1（M1 校準）結果：混合訊號在 OKX 14 天 15m 資料上無明顯 edge（BTC -0.7% / SOL +0.4%）。兩個根因：
1. OKX 短週期歷史只有 1440 根（15m=14 天），樣本太小，參數校準無統計意義
2. 訊號本身在短週期成本磨損嚴重（低波動盤整假訊號多）

## 目標

1. 用 Binance 長歷史（6-12 個月 15m/1h）重做校準，確認/否決訊號 edge
2. 加入 ATR 追蹤停損 + 波動率過濾器，直接針對成本磨損問題
3. 找到淨報酬 > 0 且有統計意義（>100 筆交易）的參數

## 變更

### 1. `bot/binance_data.py`（新增）— 長歷史資料源

- `ccxt.binance()`，分頁抓取 15m/1h 永續合約歷史（BTC/USDT:USDT, SOL/USDT:USDT），CSV 快取
- 供校準用（`bot/backtest_calibration.py`）；機器人執行仍走 OKX demo
- 目標資料長度：6-12 個月

### 2. `bot/risk.py`（修改）— ATR 追蹤停損

- 新增 `check_trailing_stop(entry, peak, atr, side, k)`：停損價 = `max(固定停損價, 峰值 ∓ k×ATR)`
- 固定 -2.5% 保留為最低保險（雙層）

### 3. `bot/signals.py`（修改）— 波動率過濾器

- 進場前檢查 LTF `ATR/close ≥ min_atr_pct`（預設 0.0015），低波動不進場

### 4. `bot/state.py`（修改）

- position 加 `peak` 欄位（追蹤停損用）

### 5. `bot/loop.py`（修改）

- 接 ATR 追蹤停損；每 tick 更新 position.peak

### 6. `bot/config.py`（修改）

- 新增 `atr_stop_mult`（預設 2.5）、`min_atr_pct`（預設 0.0015）

### 7. `bot/backtest_calibration.py`（修改）

- 改用 Binance 長歷史；grid 掃：HTF 門檻 × 停損模式（固定/追蹤）× ATR 倍數 × 波動率門檻
- 報表含交易數、淨報酬、MaxDD

## 驗證標準

- 校準結果：最佳參數淨報酬 > 0 且交易數 > 100
- 全部單元測試通過

## 非目標

- 不修改 OKX demo 執行路徑
- 不做新的策略家族（均值回歸等，留待後續）