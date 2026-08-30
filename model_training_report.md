# 模型訓練報告

生成時間: 2026-08-30 18:44

## 訓練參數
| 參數 | 數值 |
|------|------|
| 模型類型 | RF |
| n_estimators | 100 |
| max_depth | 7 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

## 資料摘要
| 項目 | 數值 |
|------|------|
| 時間框架 | 1h |
| 總筆數 | 300 |
| 訓練集 | 196 |
| 測試集 | 49 |

## 訓練結果
| 指標 | 數值 |
|------|------|
| 訓練精度 | 96.43% |
| 測試精度 | 59.18% |
| F1 Score | 0.6552 |
| AUC | 0.6621 |

## 特徵重要性
| 排名 | 特徵 | 重要性 |
|------|------|--------|
| 1 | OBV | 0.2060 |
| 2 | close | 0.1749 |
| 3 | SMA_20 | 0.1394 |
| 4 | MACD | 0.0984 |
| 5 | SMA_50 | 0.0895 |
| 6 | ATR | 0.0639 |
| 7 | MFI | 0.0623 |
| 8 | momentum_score | 0.0581 |
| 9 | RSI | 0.0579 |
| 10 | momentum_delta | 0.0496 |

## 使用方式
```python
from joblib import load

model = load('model.pkl')
scaler = load('scaler.pkl')
```
