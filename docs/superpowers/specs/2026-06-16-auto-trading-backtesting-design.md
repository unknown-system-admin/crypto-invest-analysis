# Stage 3: 自動交易與策略回測 — 設計文件

## 目標
在現有加密貨幣技術分析系統上，加入策略回測、紙上交易與實盤自動交易功能，一條龍整合至 Streamlit 儀表板與 FastAPI Monitor。

## 範圍
- 策略引擎：可組合的多策略加權投票系統
- 歷史回測：逐根 K 線模擬，產出績效報表與圖表
- 紙上交易：Monitor /check 定時評估，模擬下單與持倉追蹤
- 實盤交易：經 ccxt 下市價單至 Binance
- 全整合至現有 Streamlit 儀表板（新分頁）

## 非範圍
- 進階委託簿模擬（滑價用固定比例）
- WebSocket 即時報價（仍用 30 分輪詢）
- 多交易所支援（僅 Binance）
- 更複雜的風控（基本即可）

## 系統架構

```
Streamlit 儀表板 (新頁面)
    │ 回測請求       │ 讀取狀態
    ▼               ▼
trading/ 套件 ──► state.json / records.json
    │
    ▼
Monitor /check (策略評估 + 下單)
```

### 模組職責

#### `trading/strategy.py`
策略基底類別與五種內建策略實作：
- **MA Cross** — EMA12/26 或 SMA20/50 交叉
- **RSI Threshold** — RSI > 70 空 / < 30 多
- **MACD Cross** — MACD 上穿/下穿訊號線
- **Multi-Indicator Composite** — 直接沿用 `analysis/summary.py` 的 `analyze_signals()` 多空訊號系統
- **Custom** — YAML 設定任選以上策略組合，加權投票決定方向

#### `trading/backtest.py`
接收策略設定、交易對、時間範圍、初始資金，逐根 K 線跑：
1. OHLCV → 指標計算 → 策略評估 → 訊號產生
2. 訊號觸發 → Portfolio 檢查 → 執行掛單
3. 回測結束 → 計算績效指標

績效輸出：
- 總報酬率 (%)
- CAGR (年化報酬率)
- 最大回撤 (Max Drawdown %)
- Sharpe Ratio
- 勝率 (Win Rate %)
- 總交易次數
- Equity Curve (Plotly)
- 交易明細表
- 每月報酬熱力圖

#### `trading/portfolio.py`
持倉與訂單管理，紙上交易和實盤共用：
- `Position`：交易對、方向、數量、開倉價、當前價、浮動盈虧
- `Order`：時間、方向、數量、價格、手續費、狀態
- `Portfolio`：資金餘額、持倉列表、訂單歷史、總損益
- 狀態序列化至 JSON，支援重啟復原

#### `trading/executor.py`
訂單執行層，兩種實作：
- **PaperExecutor**：模擬成交（最新價 + 滑價），不發真實訂單
- **LiveExecutor**：透過 ccxt `create_market_buy/sell_order` 下單，`fetch_order` 確認成交

#### `trading/config.py`
```
strategies:
  ma_cross:
    enabled: true
    params: { fast: 12, slow: 26, type: "ema" }
    weight: 1
  rsi:
    enabled: true
    params: { period: 14, overbought: 70, oversold: 30 }
    weight: 1
  macd:
    enabled: false
    params: { fast: 12, slow: 26, signal: 9 }
    weight: 1
  composite:
    enabled: true
    params: {}
    weight: 2
```

各策略回傳方向（偏多/偏空/中立），加權投票總分超過閾值（可設定 0.6）才執行。

### 紙上交易流程
```
Monitor /check (每30分)
  → fetch_ohlcv
  → compute_all + analyze_signals
  → strategy.py 評估所有啟用策略
  → 加權總分 > 閾值 ?
     → PaperExecutor 模擬成交
     → portfolio.py 更新持倉
  → 發 Discord 通知 (開倉/平倉/損益)
```

### 實盤交易流程
同紙上交，但 `LiveExecutor` 改為：
1. 檢查風控（單筆 ≤ 25%、非反向、日限額）
2. ccxt `create_market_buy_order` / `create_market_sell_order`
3. `fetch_order` 確認成交
4. 更新 portfolio
5. Discord 通知（含實際成交價）

### 回測操作介面（Streamlit 新頁面）
側邊欄：
- 交易對、時間框架、回測區間、初始資金、手續費率、滑價
- 策略啟用開關與參數調整

主面板：
- 績效摘要指標卡（4-6 個 KPI）
- Equity Curve 圖表
- 交易明細表格
- 每月報酬熱力圖

### 紙上/實盤儀表板（Streamlit 新分頁）
- 當前持倉列表（交易對、數量、均價、浮盈、ROI%）
- 歷史訂單記錄
- 帳戶總損益折線圖
- 啟用策略一覽

### 狀態持久化
檔案路徑：`trading/state.json`
結構：
```json
{
  "portfolio": { "cash": 10000, "positions": [...], "orders": [...] },
  "backtest_results": { "btc-usdt": { ... } },
  "config": { "strategies": {...}, "risk": {...} }
}
```

### 風控設定
```yaml
risk:
  max_position_pct: 25        # 單筆 ≤ 資金 25%
  min_bars_hold: 1            # 持倉至少 1 根 K 線
  max_daily_trades: 10        # 每日最多交易次數
  max_drawdown_stop: 30       # 回撤 > 30% 停止交易
```

## 與現有系統的關係
- `analysis/summary.py` — `analyze_signals()` 提供 composite 策略的訊號
- `data/fetcher.py` — `fetch_ohlcv()` 提供回測與即時資料
- `monitor/main.py` /check — 整合策略評估與交易執行
- `monitor/notifier.py` — 交易通知復用現有 Discord 機制
- `monitor/config.yaml` — 擴充加入 trading 相關設定

## 測試策略
- `tests/test_trading_strategy.py` — 五種策略的訊號產生邏輯
- `tests/test_trading_backtest.py` — 回測引擎正確性（含 edge cases）
- `tests/test_trading_portfolio.py` — 持倉增刪、損益計算、序列化
- `tests/test_trading_executor.py` — 紙上成交模擬
- 不測試實盤 executor（需要真實 API key）

## 部署考量
- 無需新服務：Streamlit 已部署，Monitor 已部署
- 紙上交易狀態寫入 `trading/state.json`（需要確保 Render 磁碟可寫）
- 實盤 API key 透過 Render 環境變數注入
- `render.yaml` 不需修改（共用現有 app/monitor 服務）
