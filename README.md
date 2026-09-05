# Crypto Invest Analysis

加密貨幣投資分析系統 — 技術分析儀表板、策略回測、動能分析、Discord 自動報告。

## 系統架構

```
                        ┌─────────────────────────────────────────┐
                        │   Render (crypto-dashboard service)     │
                        │   https://crypto-invest-analysis.onrender.com
                        │                                         │
   瀏覽器 ──────────────▶│  [Docker 容器] start.sh                 │
                        │   ├─ Streamlit 儀表板 (port 8501, 對外) │
                        │   │   └ app.py — 技術分析/回測/交易 UI   │
                        │   │                                     │
                        │   └─ FastAPI monitor (port 8000, 內部)   │
                        │       └ monitor/main.py                  │
                        │           ├─ GET /report  (市場日報)     │
                        │           ├─ GET /check   (定時告警檢查) │
                        │           ├─ GET /health  (健康檢查)      │
                        │           └─ GET /debug/* (除錯端點)     │
                        └───────────────────┬─────────────────────┘
                                            │ Bot REST API
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │  Cloudflare Worker (反向代理)            │
                        │  https://crypto-analysis-report.        │
                        │  stevenwang890207.workers.dev           │
                        │                                         │
                        │  用途： bypass Discord 對 Render IP 的   │
                        │  全域封鎖 (429 global rate limit)       │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │  Discord Bot「Crypto Report#8575」       │
                        │  → 伺服器 $$ 的 #常規 頻道               │
                        └─────────────────────────────────────────┘
```

### 單容器雙服務

Render 免費版只允許一個 service，所以 FastAPI 和 Streamlit 跑在同一個容器：

- `start.sh` — 啟動腳本：背景跑 uvicorn (8000)，前景跑 streamlit (8501)
- `Dockerfile.app` — 映像檔定義，EXPOSE 8000 + 8501
- `render.yaml` — Render 部署設定（單一 `crypto-dashboard` service）

### Discord 通知鏈

```
/report 請求 → 產生報告 embed → send_bot_message()
    → Cloudflare Worker (?path=/channels/{id}/messages)
    → Discord API v10 → #常規 頻道
```

**為什麼需要 Cloudflare Worker？**

Discord 封鎖了 Render 免費版的共享 IP 段（太多免費服務打 Discord API 觸發全域 429），
webhook 和 Bot REST API 都會被擋。Cloudflare Worker 的 IP 乾淨未被封鎖，
扮演「轉發站」：Render → Worker → Discord。免費額度每天 10 萬次請求，遠超需求。

**為什麼 token 用 base64 存在 config.yaml？**

Render 環境變數設定後未正確注入容器（原因不明），
改把 `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` base64 編碼後存於
`monitor/config.yaml` 的 `bot_token_b64` / `channel_id_b64`，
程式啟動時解碼。環境變數仍優先（如有設定）。

## 專案結構

```
├── app.py                  # Streamlit 儀表板（技術分析/回測/投資組合 UI）
├── start.sh                # 容器啟動腳本（uvicorn 背景 + streamlit 前景）
├── Dockerfile.app          # 單容器映像（Render 部署用）
├── render.yaml             # Render 服務定義
│
├── monitor/                # FastAPI 通知服務
│   ├── main.py             #   端點：/report /check /health /debug/*
│   ├── notifier.py         #   Discord Bot REST API（經 Worker 代理）+ webhook fallback
│   ├── config.yaml         #   Discord 設定（token/challenge base64）、告警、策略參數
│   └── checker.py / state.py
│
├── feature_engine/         # 特徵引擎
│   ├── indicators.py       #   9 個技術指標（SMA20/50/200, EMA26, ATR, RSI, MACD, MFI, OBV）
│   ├── momentum.py         #   動能分數（RSI 0.3 / MACD 0.1 / SMA20 0.4 / SMA50 0.2）
│   ├── labels.py           #   動能標籤（防未來函數洩漏）
│   └── builder.py          #   組裝特徵矩陣
│
├── backtest_engine/        # 回測引擎
│   ├── engine.py           #   回測主體（含放空保證金機制）
│   ├── rule_strategy.py    #   動能門檻策略（buy=0.08 / sell=-0.07）
│   ├── model_strategy.py   #   ML 模型策略（RF/XGBoost）
│   ├── short_strategy.py   #   放空策略（MomentumShort / MLShort）
│   └── metrics.py          #   績效指標
│
├── data/                   # OKX 資料抓取（ccxt）
├── analysis/               # 技術分析（多空訊號、動能趨勢、支撐壓力）
├── trading/                # 模擬交易（paper trading）
├── train_model.py          # 本地模型訓練 CLI
├── data_cache/             # K 線 CSV 快取（BTC 2 年 1h 資料）
├── tests/                  # 測試（pytest）
└── reports/                # 視覺化回測報告（PNG）
```

## 動能分析（報告核心）

- **動能分數**：加權組合（RSI 0.3、MACD 0.1、SMA20 0.4、SMA50 0.2），範圍 -1 ~ +1
- **趨勢判定**：嚴格單調（不允許持平），`中→中→強` 視為「穩定/維持」
- **報告顯示**：最近 4 個分數、1 階導數（變化率）、2 階導數（加速度）+ 中文解讀
- **時間戳**：台灣時間（UTC+8）`🕐 YYYY/MM/DD HH:MM`

## 觸發報告

```bash
# 手動觸發（冷啟動需等 60-90 秒）
curl https://crypto-invest-analysis.onrender.com/report

# 指定參數
curl "https://crypto-invest-analysis.onrender.com/report?tf=1d&step=1"
```

## 本地開發

```bash
# 環境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 跑儀表板
streamlit run app.py

# 跑 monitor
uvicorn monitor.main:app --port 8000

# 訓練模型
.venv/bin/python train_model.py --model rf --optimize

# 測試
.venv/bin/python -m pytest
```

## 已知限制

- OKX API 無 API key 限 ~300 根 K 線（快取 + 分批抓取緩解）
- Render 免費版 15 分鐘無流量休眠，首次請求需等冷啟動
- 策略閾值已校準（2 年資料 BTC +34.7%，但動能策略因交易成本仍為負，持續優化中）
