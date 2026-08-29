# 本地回測系統設計 Spec ✅ 已完成

## 目標

建立統一特徵層 + 雙模式回測系統，支援：
1. 動能量化（連續分數 + 導數）
2. 擴充技術指標至 18 個
3. Colab ML 管線（特徵選擇 + 模型訓練）

## 架構：統一特徵層 + 雙模式回測

### 1. 特徵引擎（feature_engine/）

#### 1.1 indicators.py — 擴充技術指標（18 個）

| 類別 | 指標 | 參數 |
|------|------|------|
| 趨勢 | SMA 20/50/200 | 固定窗口 |
| | EMA 12/26 | 固定窗口 |
| | ADX | 14期 |
| | Ichimoku Cloud | 9/26/52 |
| 波動率 | Bollinger Bands | 20期, 2倍標準差 |
| | ATR | 14期 |
| | Keltner Channels | 20期, 2倍ATR |
| 動量 | RSI | 14期 |
| | MACD | 12/26/9 |
| | Stochastic %K/%D | 14期, 3平滑 |
| | CCI | 20期 |
| | Williams %R | 14期 |
| | ROC | 12期 |
| | MFI | 14期 |
| 成交量 | OBV | — |
| | VWAP | — |
| | CMF | 20期 |

使用 `ta` 函式庫計算，所有指標一次性產出。

#### 1.2 momentum.py — 動能量化

**momentum_score**：連續值 [-1, 1]

| 子指標 | 歸一化方式 | 權重 |
|--------|-----------|------|
| RSI | `(rsi - 50) / 50` | 0.3 |
| MACD histogram | `tanh(histogram / std)` | 0.3 |
| 價格 vs SMA20 | `(close - sma20) / sma20` | 0.2 |
| 價格 vs SMA50 | `(close - sma50) / sma50` | 0.2 |

公式：
```
score = w1*rsi_norm + w2*macd_norm + w3*sma20_norm + w4*sma50_norm
score = clip(score, -1, 1)
```

**分數對應**：

| 範圍 | 意義 |
|------|------|
| 0.6 ~ 1.0 | 強勢上漲 |
| 0.2 ~ 0.6 | 溫和上漲 |
| -0.2 ~ 0.2 | 震盪/持平 |
| -0.6 ~ -0.2 | 溫和下跌 |
| -1.0 ~ -0.6 | 強勢下跌 |

**導數**：
- `delta(t) = score(t) - score(t-1)`：一階導數（動能方向）
- `acceleration(t) = delta(t) - delta(t-1)`：二階導數（趨勢變化）

#### 1.3 labels.py — 標籤生成

- `future_return(df, n_bars)`：未來 N 根 K 棒的報酬率
- `binary_label(df, n_bars, threshold)`：漲（1）/ 跌（0）分類標籤
- 預設 `n_bars=5`，`threshold=0`（正=漲、負=跌）

#### 1.4 builder.py — 特徵矩陣組裝

- `build_feature_matrix(df, n_bars=5)` → 完整特徵 DataFrame + 標籤
- 每行是一個時間點，每列是一個特徵
- 自動處理 NaN（丟棄前 N 期）

---

### 2. 回測引擎（backtest_engine/）

#### 2.1 engine.py — 核心回測（共用）

- 逐 bar 迭代、倉位管理、風險控制
- 沿用現有邏輯：倉位比例上限、每日交易上限、最大回撤停損
- 接收 `Strategy` 介面，不關心信號來源

#### 2.2 策略介面

```python
class Strategy(ABC):
    def evaluate(self, features: pd.Series) -> Signal:
        """輸入特徵，回傳 Signal"""
        ...
```

#### 2.3 rule_strategy.py — 規則模式

- 用動能分數 + 導數的門檻組合
- 例如：`score > 0.6 and delta > 0` → 做多
- 參數可調整（從 optimizer 找最佳組合）

#### 2.4 model_strategy.py — 模型模式

- 匯入 Colab 訓練好的模型（pickle/ONNX/JSON 規則）
- `model.predict(features)` → 做多/做空/觀望
- 支援 XGBoost、Random Forest、或簡單規則模型

#### 2.5 metrics.py — 績效指標

- 總報酬、年化報酬、Sharpe、最大回撤、勝率
- 沿用現有計算方式

---

### 3. Colab ML 管線

#### 3.1 資料準備

- 從本地匯出特徵矩陣 CSV（`feature_engine/builder.py` 產出）
  - 檔案格式：每行一個時間點，每列一個特徵，最後一列為標籤（0/1）
  - 檔案命名：`{symbol}_{timeframe}_features.csv`，例如 `BTC_USDT_1h_features.csv`
- 或直接在 Colab 計算（共用 `indicators.py`、`momentum.py`）
  - 需要將 `feature_engine/` 目錄上傳到 Colab 或從 GitHub clone

#### 3.2 特徵選擇

- `feature_importance`：用 Random Forest / XGBoost 計算特徵重要性
- `correlation_matrix`：檢查特徵間共線性，移除高相關特徵
- 輸出：關鍵特徵清單（如 top 10 特徵）

#### 3.3 模型訓練

- 訓練 XGBoost / LightGBM 分類模型
- 目標：預測未來 N 棒漲跌（二分類）
- 交叉驗證 + 超參數調整
- 輸出：訓練好的模型 + 績效報告

#### 3.4 模型匯出

- 首選：pickle / joblib（最簡單）
- 後續考量：ONNX、JSON 規則

#### 3.5 結果視覺化

- 特徵重要性排序圖
- 混淆矩陣
- 累積報酬曲線對比（規則 vs 模型）

---

### 4. 目錄結構

```
crypto-invest-analysis/
├── feature_engine/           # 新增：特徵引擎
│   ├── __init__.py
│   ├── indicators.py         # 18個技術指標
│   ├── momentum.py           # 動能量化（-1~1 + 導數）
│   ├── labels.py             # 標籤生成
│   └── builder.py            # 組裝特徵矩陣
│
├── backtest_engine/          # 新增：回測引擎
│   ├── __init__.py
│   ├── engine.py             # 核心回測（共用）
│   ├── rule_strategy.py      # 規則模式
│   ├── model_strategy.py     # 模型模式
│   └── metrics.py            # 績效指標
│
├── colab/                    # 新增：Colab notebooks
│   └── momentum_ml.ipynb
│
├── indicators/               # 現有（保留，功能遷移到 feature_engine）
├── trading/                  # 現有（保留，回測邏輯重構）
└── ...
```

### 5. 整合方式

1. 現有 `indicators/calculator.py` 的功能遷移到 `feature_engine/indicators.py`
2. 現有 `trading/backtest.py` 的回測邏輯重構到 `backtest_engine/engine.py`
3. 現有 `trading/strategy.py` 的策略介面統一到 `backtest_engine/` 的策略類別
4. 保留舊模組的 API 相容性：在舊模組中加入 import re-export，確保現有 import 路徑不變
   - 例：`indicators/calculator.py` 保留 `compute_all()` 函數，內部轉呼叫 `feature_engine.indicators.compute_all()`
   - 例：`trading/strategy.py` 保留 `CustomComposite` 等類別，內部轉呼叫 `backtest_engine/` 的對應類別

### 6. 不動的部分

- `data/fetcher.py`：資料抓取不變
- `monitor/`：監控模組不變
- `app.py`：Streamlit 儀表板可之後再整合新功能

---

## 成功標準

1. `feature_engine` 可獨立計算 18 個指標 + 動能量化分數
2. `backtest_engine` 支援規則模式和模型模式
3. Colab notebook 可完成特徵選擇 + 模型訓練
4. 所有現有測試繼續通過
5. 新增模組的單元測試覆盖率 > 80%
