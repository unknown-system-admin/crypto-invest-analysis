# 資金費率當均值回歸濾網 v4 設計

日期：2026-09-06

## 背景

三次日內價格訊號（動能 v1/v2、均值回歸 v3）無法通過統計驗證。資金費率是加密特有的「情緒」訊號：正費率 = 多頭擁擠、負費率 = 空頭擁擠。極端費率預測反轉（contrarian）。OKX 實測：BTC 83% 資金期為正、均值回歸 v3 兩資產皆正 — 兩者疊加值得驗證。

## 策略

### 訊號：均值回歸（v3） + 資金費率 contrarian 濾網

```
進場（需同時滿足）：
    做多：RSI(15m) fresh 超賣（v3）+ 資金費率 < -threshold（空頭擁擠 → 賭反彈）
    做空：RSI(15m) fresh 超買（v3）+ 資金費率 > +threshold（多頭擁擠 → 賭回調）
出場：RSI 回中（55/45）+ ATR 追蹤停損（沿用）
```

- 資金費率門檻用**絕對值**（threshold > 0，兩側對稱）grid 校準
- 資金費率在 8h 資金期之間 forward-fill 對齊 15m K 線
- HTF 趨勢過濾（SMA_50）沿用 v3

### 校準

- Binance 價格長歷史 + funding 歷史（Binance 優先，OKX 95 天為後備）
- grid：RSI over/overbought × funding_threshold × htf_filter × stop
- 門檻：交易數 ≥ 100 且兩資產淨報酬 > 0

## 檔案變更

```
bot/funding.py                    # 新增：funding 歷史抓取 + bias 判斷
bot/meanreversion.py              # 改：evaluate 支援 funding bias 參數
bot/backtest_calibration.py       # 改：--signal meanreversion 支援 funding 濾網
bot/config.py                     # 改：funding_threshold
bot/loop.py                       # 改：live 抓 funding rate 當濾網
```

## 非目標

- 不做 delta-neutral 純收割（留待未來，本版先驗證「費率當訊號」）
- 不改 OKX demo 執行路徑