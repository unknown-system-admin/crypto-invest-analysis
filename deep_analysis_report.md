# Deep Analysis Report



**Generated:** 2026-08-30 17:14:10

**Symbol:** BTC/USDT



---



## A. Timeframe Analysis

Comparing backtest results across 1h, 4h, and 1d timeframes using BTC/USDT.

| Timeframe | Samples | Rule Return% | Rule Sharpe | Rule WinRate | ML Return% | ML Sharpe |
|-----------|---------|--------------|-------------|--------------|------------|-----------|
| 1h | 96 | -0.4 | -6.62 | 0.0% | 1.1 | 22.12 |
| 4h | 96 | 0.1 | 0.66 | 100.0% | 2.3 | 11.21 |
| 1d | 96 | 0.1 | 0.13 | 100.0% | 6.1 | 3.60 |

**Best rule-based timeframe:** 4h (Sharpe 0.66)
**Best ML timeframe:** 1h (Sharpe 22.12)

## B. Threshold Search

Tested 9 x 9 = 81 combinations.

### Top 5 by Sharpe Ratio

| Buy | Sell | Return% | Sharpe | Trades | WinRate | MaxDD% |
|-----|------|---------|--------|--------|---------|--------|
| 0.25 | -0.15 | 2.1 | 1.51 | 1 | 100.0% | 3.9% |
| 0.25 | -0.10 | 2.1 | 1.51 | 1 | 100.0% | 3.9% |
| 0.15 | -0.15 | 2.1 | 1.46 | 1 | 100.0% | 3.9% |
| 0.15 | -0.10 | 2.1 | 1.46 | 1 | 100.0% | 3.9% |
| 0.20 | -0.15 | 2.1 | 1.46 | 1 | 100.0% | 3.9% |

**Best combination:** buy=0.25, sell=-0.15
- Return: 2.1%
- Sharpe: 1.51
- Win Rate: 100.0%
- Max Drawdown: 3.9%

## C. Cross-Validation

5-fold cross-validation on 96 samples with 31 features.

| Fold | Accuracy |
|------|----------|
| 1 | 0.5000 |
| 2 | 0.2105 |
| 3 | 0.4211 |
| 4 | 0.4211 |
| 5 | 0.4737 |
| **Mean** | **0.4053** |
| **Std** | **0.1021** |

**Holdout test accuracy:** 0.9000

**Interpretation:** CV mean 0.4053 ± 0.1021 provides a more robust estimate than the single holdout split (0.9000).

## D. Feature Correlation Matrix

Computed correlation matrix for 30 features.

### Highly Correlated Pairs (|r| > 0.8)

| Feature A | Feature B | Correlation |
|-----------|-----------|-------------|
| SMA_20 | BB_middle | 1.0 |
| STOCH_K | Williams_R | 1.0 |
| SMA_20 | KC_upper | 0.999 |
| SMA_20 | KC_lower | 0.999 |
| BB_middle | KC_upper | 0.999 |
| BB_middle | KC_lower | 0.999 |
| KC_upper | KC_lower | 0.996 |
| SMA_20 | EMA_26 | 0.991 |
| EMA_26 | BB_middle | 0.991 |
| EMA_26 | KC_lower | 0.991 |
| EMA_26 | KC_upper | 0.987 |
| EMA_12 | ICHIMOKU_A | 0.985 |
| SMA_200 | VWAP | 0.967 |
| ICHIMOKU_A | BB_lower | 0.967 |
| EMA_26 | ICHIMOKU_A | 0.964 |
| MACD_signal | OBV | 0.96 |
| MACD | MACD_signal | 0.957 |
| MACD_histogram | momentum_score | 0.954 |
| EMA_12 | BB_lower | 0.953 |
| SMA_20 | ICHIMOKU_A | 0.952 |
| ICHIMOKU_A | BB_middle | 0.952 |
| ICHIMOKU_A | KC_upper | 0.948 |
| ROC | momentum_score | 0.948 |
| STOCH_D | CCI | 0.947 |
| EMA_12 | EMA_26 | 0.946 |
| ICHIMOKU_A | KC_lower | 0.946 |
| SMA_20 | EMA_12 | 0.933 |
| EMA_12 | BB_middle | 0.933 |
| STOCH_K | STOCH_D | 0.932 |
| STOCH_K | CCI | 0.932 |
| STOCH_D | Williams_R | 0.932 |
| CCI | Williams_R | 0.932 |
| EMA_12 | KC_upper | 0.93 |
| SMA_20 | BB_lower | 0.928 |
| BB_middle | BB_lower | 0.928 |
| BB_lower | KC_upper | 0.927 |
| EMA_26 | BB_lower | 0.925 |
| EMA_12 | KC_lower | 0.924 |
| STOCH_D | momentum_score | 0.923 |
| BB_lower | KC_lower | 0.922 |
| CCI | momentum_score | 0.917 |
| EMA_12 | OBV | 0.898 |
| RSI | CCI | 0.897 |
| RSI | CMF | 0.895 |
| STOCH_K | momentum_score | 0.893 |
| Williams_R | momentum_score | 0.893 |
| MACD_histogram | ROC | 0.891 |
| MACD | OBV | 0.889 |
| RSI | MACD | 0.887 |
| BB_lower | OBV | 0.879 |
| RSI | ROC | 0.876 |
| MACD | CMF | 0.873 |
| STOCH_D | ROC | 0.869 |
| EMA_12 | MACD_signal | 0.866 |
| RSI | momentum_score | 0.864 |
| CCI | ROC | 0.863 |
| ICHIMOKU_A | OBV | 0.86 |
| SMA_50 | SMA_200 | -0.847 |
| BB_upper | KC_lower | 0.843 |
| STOCH_K | ROC | 0.843 |
| Williams_R | ROC | 0.843 |
| SMA_20 | BB_upper | 0.836 |
| BB_upper | BB_middle | 0.836 |
| BB_upper | KC_upper | 0.836 |
| MACD_histogram | STOCH_D | 0.831 |
| MFI | momentum_score | 0.83 |
| RSI | STOCH_D | 0.828 |
| ADX | VWAP | -0.823 |
| EMA_26 | BB_upper | 0.822 |
| RSI | STOCH_K | 0.822 |
| RSI | Williams_R | 0.822 |
| BB_lower | MACD_signal | 0.812 |
| ICHIMOKU_A | MACD_signal | 0.807 |
| MACD_signal | CMF | 0.806 |
| MACD_histogram | CCI | 0.802 |

### Suggested Features to Remove

Based on average correlation with other features, consider removing:

- `ADX`
- `BB_middle`
- `CCI`
- `EMA_12`
- `EMA_26`
- `ICHIMOKU_A`
- `KC_lower`
- `KC_upper`
- `MACD`
- `MACD_histogram`
- `MACD_signal`
- `MFI`
- `OBV`
- `ROC`
- `RSI`
- `SMA_20`
- `SMA_200`
- `SMA_50`
- `STOCH_D`
- `STOCH_K`
- `momentum_score`

### Full Correlation Summary

- Total features: 30
- Highly correlated pairs: 75
- Suggested removals: 21

## E. Model Comparison

Comparing Random Forest, XGBoost, and LightGBM on the same train/test split.

| Model | Accuracy | Top Feature 1 | Top Feature 2 | Top Feature 3 |
|-------|----------|---------------|---------------|---------------|
| RandomForest **BEST** | 0.9000 | VWAP (0.062) | SMA_200 (0.058) | MFI (0.056) |
| XGBoost | 0.8500 | ATR (0.134) | MFI (0.128) | SMA_200 (0.103) |
| LightGBM | 0.8000 | BB_lower (26.000) | momentum_acceleration (26.000) | CMF (25.000) |

**Best model:** RandomForest with accuracy 0.9000