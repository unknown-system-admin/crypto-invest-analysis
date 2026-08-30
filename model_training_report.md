# 模型訓練報告

生成時間: 2026-08-30 20:16

## 訓練參數
| 參數 | 數值 |
|------|------|
| 模型類型 | RF |
| n_estimators | 50 |
| max_depth | 3 |
| min_samples_split | 2 |
| min_samples_leaf | 5 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | 1h |
| 總筆數 | 1440 |
| 訓練集 | 1108 |
| 測試集 | 277 |

## 訓練結果
| 指標 | 數值 |
|------|------|
| 訓練精度 | 68.59% |
| 測試精度 | 45.49% |
| F1 Score | 0.2011 |
| AUC | 0.4666 |

## 特徵重要性
| 排名 | 特徵 | 重要性 |
|------|------|--------|
| 1 | close | 0.2504 |
| 2 | SMA_20 | 0.1372 |
| 3 | MFI | 0.1259 |
| 4 | RSI | 0.1091 |
| 5 | momentum_score | 0.0753 |
| 6 | OBV | 0.0741 |
| 7 | ATR | 0.0716 |
| 8 | SMA_50 | 0.0646 |
| 9 | MACD | 0.0573 |
| 10 | momentum_delta | 0.0344 |

## 使用方式
```python
from joblib import load

model = load('model_rf_optimized.pkl')
scaler = load('scaler.pkl')
```
