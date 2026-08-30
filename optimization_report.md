# Crypto Investment Analysis - Optimization Report

## Summary

This report documents the results of 5 optimization techniques applied to the crypto investment analysis model.

---

## A. Remove Redundant Features

### Method
- Removed features with |correlation| > 0.8
- Kept the feature with higher importance from each correlated pair

### Results
- **Baseline accuracy:** 0.9792
- **Reduced model accuracy:** 0.9000
- **Accuracy change:** -0.0792
- **Features removed:** 21

### Correlated Pairs Found
- SMA_50 <-> SMA_200: r=0.847
- SMA_20 <-> EMA_12: r=0.933
- SMA_20 <-> EMA_26: r=0.991
- EMA_12 <-> EMA_26: r=0.946
- SMA_20 <-> ICHIMOKU_A: r=0.952
- EMA_12 <-> ICHIMOKU_A: r=0.985
- EMA_26 <-> ICHIMOKU_A: r=0.964
- SMA_20 <-> BB_upper: r=0.836
- EMA_26 <-> BB_upper: r=0.822
- SMA_20 <-> BB_middle: r=1.000
- EMA_12 <-> BB_middle: r=0.933
- EMA_26 <-> BB_middle: r=0.991
- ICHIMOKU_A <-> BB_middle: r=0.952
- BB_upper <-> BB_middle: r=0.836
- SMA_20 <-> BB_lower: r=0.928
- EMA_12 <-> BB_lower: r=0.953
- EMA_26 <-> BB_lower: r=0.925
- ICHIMOKU_A <-> BB_lower: r=0.967
- BB_middle <-> BB_lower: r=0.928
- SMA_20 <-> KC_upper: r=0.999
- EMA_12 <-> KC_upper: r=0.930
- EMA_26 <-> KC_upper: r=0.987
- ICHIMOKU_A <-> KC_upper: r=0.948
- BB_upper <-> KC_upper: r=0.836
- BB_middle <-> KC_upper: r=0.999
- BB_lower <-> KC_upper: r=0.927
- SMA_20 <-> KC_lower: r=0.999
- EMA_12 <-> KC_lower: r=0.924
- EMA_26 <-> KC_lower: r=0.991
- ICHIMOKU_A <-> KC_lower: r=0.946
- BB_upper <-> KC_lower: r=0.843
- BB_middle <-> KC_lower: r=0.999
- BB_lower <-> KC_lower: r=0.922
- KC_upper <-> KC_lower: r=0.996
- RSI <-> MACD: r=0.887
- EMA_12 <-> MACD_signal: r=0.866
- ICHIMOKU_A <-> MACD_signal: r=0.807
- BB_lower <-> MACD_signal: r=0.812
- MACD <-> MACD_signal: r=0.957
- RSI <-> STOCH_K: r=0.822
- RSI <-> STOCH_D: r=0.828
- MACD_histogram <-> STOCH_D: r=0.831
- STOCH_K <-> STOCH_D: r=0.932
- RSI <-> CCI: r=0.897
- MACD_histogram <-> CCI: r=0.802
- STOCH_K <-> CCI: r=0.932
- STOCH_D <-> CCI: r=0.947
- RSI <-> Williams_R: r=0.822
- STOCH_K <-> Williams_R: r=1.000
- STOCH_D <-> Williams_R: r=0.932
- CCI <-> Williams_R: r=0.932
- RSI <-> ROC: r=0.876
- MACD_histogram <-> ROC: r=0.891
- STOCH_K <-> ROC: r=0.843
- STOCH_D <-> ROC: r=0.869
- CCI <-> ROC: r=0.863
- Williams_R <-> ROC: r=0.843
- EMA_12 <-> OBV: r=0.898
- ICHIMOKU_A <-> OBV: r=0.860
- BB_lower <-> OBV: r=0.879
- MACD <-> OBV: r=0.889
- MACD_signal <-> OBV: r=0.960
- SMA_200 <-> VWAP: r=0.967
- ADX <-> VWAP: r=0.823
- RSI <-> CMF: r=0.895
- MACD <-> CMF: r=0.873
- MACD_signal <-> CMF: r=0.806
- RSI <-> momentum_score: r=0.864
- MACD_histogram <-> momentum_score: r=0.954
- STOCH_K <-> momentum_score: r=0.893
- STOCH_D <-> momentum_score: r=0.923
- CCI <-> momentum_score: r=0.917
- Williams_R <-> momentum_score: r=0.893
- ROC <-> momentum_score: r=0.948
- MFI <-> momentum_score: r=0.830

### Features Removed
ADX, BB_lower, BB_middle, BB_upper, CCI, CMF, EMA_12, ICHIMOKU_A, KC_lower, KC_upper, MACD_signal, OBV, ROC, RSI, SMA_20, SMA_50, STOCH_D, STOCH_K, VWAP, Williams_R, momentum_score

---

## B. Regularize Model

### Method
- Tested combinations of max_depth, min_samples_split, min_samples_leaf
- Used 5-fold cross-validation to select best parameters

### Results
- **Best parameters:** {'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 2}
- **Best CV accuracy:** 0.8025
- **Best test accuracy:** 0.9000
- **Baseline test accuracy:** 0.9000

### Top 5 Configurations
| 7 | 2 | 2 | 0.8025 | 0.9000 |
| 10 | 2 | 2 | 0.8025 | 0.9000 |
| None | 2 | 2 | 0.8025 | 0.9000 |
| 7 | 2 | 1 | 0.8017 | 0.9000 |
| 7 | 5 | 1 | 0.8017 | 0.9000 |

---

## C. Feature Engineering

### Method
- Created combination features: RSI_MACD, Price_SMA_ratio, BB_position, Momentum_RSI
- Tested if engineered features improve model performance

### Results
- **Baseline accuracy:** 0.9000
- **Engineered model accuracy:** 0.9000
- **Accuracy change:** +0.0000
- **New features added:** 5

### New Features
close, RSI_MACD, Price_SMA_ratio, BB_position, Momentum_RSI

---

## D. Time Series Cross-Validation

### Method
- Compared standard 5-fold CV with TimeSeriesSplit (5 splits)
- Also tested temporal 80/20 train/test split

### Results
- **Standard 5-Fold CV:** 0.4158 (+/- 0.1058)
- **Time Series 5-Fold CV:** 0.4875 (+/- 0.1212)
- **Temporal Split Accuracy:** 0.5500

### Fold Scores
| Fold | Standard | Time Series |
|------|----------|-------------|
| 1 | 0.5000 | 0.5000 |
| 2 | 0.2105 | 0.6875 |
| 3 | 0.4737 | 0.3125 |
| 4 | 0.4211 | 0.5000 |
| 5 | 0.4737 | 0.4375 |

### Key Insight
Standard CV may overestimate performance by -0.0717 compared to time series CV.

---

## E. Optimize Momentum Formula

### Method
- Tested different weight combinations for momentum_score
- Weights: RSI, MACD, SMA20, SMA50 (must sum to 1.0)

### Results
- **Current weights:** RSI=0.3, MACD=0.3, SMA20=0.2, SMA50=0.2
- **Current CV accuracy:** 0.4875
- **Best weights:** RSI=0.3, MACD=0.1, SMA20=0.4, SMA50=0.2
- **Best CV accuracy:** 0.5000
- **Improvement:** +0.0125

### Top 5 Weight Combinations
| RSI=0.3 | MACD=0.1 | SMA20=0.4 | SMA50=0.2 | 0.5000 |
| RSI=0.3 | MACD=0.1 | SMA20=0.5 | SMA50=0.1 | 0.5000 |
| RSI=0.4 | MACD=0.2 | SMA20=0.1 | SMA50=0.3 | 0.5000 |
| RSI=0.4 | MACD=0.2 | SMA20=0.2 | SMA50=0.2 | 0.5000 |
| RSI=0.1 | MACD=0.1 | SMA20=0.3 | SMA50=0.5 | 0.4875 |

---

## Recommendations

1. **Feature Selection:** Remove 21 redundant features to simplify the model
2. **Regularization:** Use max_depth=7 to prevent overfitting
3. **Feature Engineering:** Engineered features did not improve performance
4. **Validation:** Always use time series cross-validation for financial data
5. **Momentum:** Consider updating weights to 0.3/0.1/0.4/0.2
