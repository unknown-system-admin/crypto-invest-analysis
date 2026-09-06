# 日內均值回歸策略 v3 設計

日期：2026-09-06

## 背景

動能（追趨勢）在 15m 日內兩次校準均無 edge（v1 OKX 14天、v2 Binance 3個月）。動能死於盤整市。均值回歸（賭反彈）與動能互補，靠盤整賺錢。用已建好的 Binance 長歷史資料源 + ATR 停損 + 校準基建直接驗證。

## 策略

### 訊號（15m LTF + 1h HTF）

```
HTF (1h) 趨勢過濾：close > SMA_50 → 上升（只做多）；close < SMA_50 → 下降（只做空）
LTF (15m) 進場（RSI 交叉觸發，避免連續超賣重複進場）：
    prev_RSI > oversold 且 curr_RSI <= oversold（fresh 超賣）+ 上升趨勢 → 做多
    prev_RSI < overbought 且 curr_RSI >= overbought（fresh 超買）+ 下降趨勢 → 做空
出場：
    RSI >= exit_long_rsi (55) → 平多（均值回歸完成）
    RSI <= exit_short_rsi (45) → 平空
    或 ATR 追蹤停損（k×ATR）+ 2.5% 硬底（沿用 v2 risk.py）
```

### 參數（grid 校準）

- `rsi_oversold`: 20 / 25 / 30
- `rsi_overbought`: 70 / 75 / 80
- `htf_filter`: on / off（SMA_50 過濾要不要）
- 停損: hard（2.5%）/ trailing（ATR k=2/3/4）
- `rsi_exit_long=55`, `rsi_exit_short=45`（固定，不掃）

## 檔案變更

```
bot/meanreversion.py            # 新增：均值回歸訊號
bot/backtest_calibration.py     # 改：可切換動能/均值回歸訊號 + 對應 grid
bot/config.py                   # 新增 mean_reversion 參數區（含動能模式開關）
bot/loop.py                     # 改：依 config 選用動能或均值回歸訊號
```

## 驗證標準

- Binance 3 個月資料，均值回歸最佳配置：交易數 >= 100 且兩資產淨報酬 > 0（對比動能的 17-26 筆）
- 全部測試通過

## 非目標

- 不改 OKX demo 執行路徑
- 不做動能與均值回歸的組合（先單獨驗證均值回歸）