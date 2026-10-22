# Live signal and report spec (2026-10-22)

## GET /signal

Notification only. Not a backtest replay and not an order.

### Inputs

- Price: OKX `fetch_ticker` last trade. Not the daily close. Not a historical fill.
- Trend: latest 1d momentum score and one-day delta. Weights RSI 0.3 / MACD 0.1 / SMA20 0.4 / SMA50 0.2.
- No position, no cooldown, no replay of old trades.

### Regime

- long: score > 0.05 and delta > 0
- short: score < -0.30 and delta < 0
- flat: otherwise (trend disappeared)

### Output

- One Discord message containing BTC and SOL.
- Send only when a symbol enters long/short, flips, or leaves a long/short regime.
- No message if both are unchanged, or if the first observation is already flat.
- Scan cadence is external (hourly 09:00-23:00 Asia/Taipei, plus daily report times). The service does not poll itself.

## GET /report

Market daily report. Always sends. One Discord message for BTC and SOL.

- Title: `市場日報 — BTC / SOL`
- Body: Taiwan timestamp, then each symbol's summary, multi-timeframe, momentum path, last 4 scores, delta, acceleration.
- Response includes `combined: true` and `reports: ["BTC/USDT", "SOL/USDT"]`.
- Triggered by the six Grok daily times (08:30 / 12:00 / 15:00 / 18:00 / 21:00 / 23:00 Asia/Taipei) and by a manual GET.

Walk-forward results (buy=0.05, sell=-0.30, cooldown=3, position-aware) remain the research record. They are not the live notification rule.
