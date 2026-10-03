# Crypto Invest Analysis

加密貨幣投資分析系統 — 技術分析儀表板、策略回測、動能分析、Discord 自動報告。

## 線上訊號規格（2026-10-03）

`GET /signal` 是通知，不是回測重播，也不下單。

看什麼：

- 價格：OKX `fetch_ticker` 最新成交價。不用日線收盤，也不用歷史進場價。
- 趨勢：最新一根 1d 動能分數與一日變化。權重 RSI 0.3 / MACD 0.1 / SMA20 0.4 / SMA50 0.2。
- 不看持倉、不看冷卻、不重播歷史成交。

判斷：

- 做多：分數 > 0.05 且變化 > 0
- 做空：分數 < -0.30 且變化 < 0
- 其余：趨勢消失（平區）

給什麼：

- BTC 與 SOL 合併成一則 Discord。
- 只有某一檔從做多/做空翻轉，或從做多/做空落出門檻（趨勢消失），才發送。
- 兩邊都沒變、或第一次看到就是平區：不通知。
- 掃描頻率來自外部排程（目前盤中每小時，加上日報時點），服務本身不輪詢。

這與下方 walk-forward 回測不是同一套。回測有持倉與冷卻，數字仍有效，但不再當通知語意。

## 系統架構

單容器：Streamlit 8501 對外，FastAPI monitor 8000 內部。Discord 經 Cloudflare Worker 轉發。

端點：`/health` `/report` `/check` `/signal`。

## 動能分析

- 動能分數：RSI 0.3、MACD 0.1、SMA20 0.4、SMA50 0.2，範圍約 -1 ~ +1
- 報告顯示最近 4 個分數、變化、加速度

## 觸發

```bash
curl https://crypto-invest-analysis.onrender.com/signal
curl "https://crypto-invest-analysis.onrender.com/report?tf=1d&step=1"
```

冷啟動約 60–90 秒。

## 策略驗證結果（2026-09，多折 walk-forward）

回測配置（不是線上通知規則）：`buy=0.05, sell=-0.30, short_entry=-0.30, cooldown=3, trend_filter=True, strong_filter=False, min_holding=0, dd_stop=50, pos=95%`

| 資產 | 固定配置（OOS 3 折複合） | B&H | 逐折重選 grid |
|------|----------------------|-----|--------------|
| BTC | **+15.41%** | -17.53% | -28.37% |
| SOL | **+11.86%** | -38.80% | -30.25% |

參數凍結。逐折重選會過擬合。SOL 是跨資產轉移驗證。

已否決：`short_entry=-0.15`、`strong_filter=True`。日內 bot 四次校準皆未過交易數 >= 100，不上真錢。
