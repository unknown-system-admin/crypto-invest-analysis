import pandas as pd
import numpy as np
from scipy import stats
from feature_engine.labels import future_return
from backtest_engine.rule_strategy import MomentumRuleStrategy
from backtest_engine.engine import BacktestEngine


class MomentumAnalyzer:
    def analyze_correlation(self, df: pd.DataFrame, forward_days: int = 5) -> dict:
        future_ret = future_return(df, n_bars=forward_days)
        valid_idx = df["momentum_score"].dropna().index.intersection(future_ret.dropna().index)
        x = df.loc[valid_idx, "momentum_score"]
        y = future_ret.loc[valid_idx]
        corr, p_value = stats.pearsonr(x, y)
        return {
            "correlation": round(corr, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
        }

    def backtest_thresholds(
        self,
        df: pd.DataFrame,
        buy_thresholds: list,
        sell_thresholds: list,
    ) -> list:
        results = []
        for buy_thresh in buy_thresholds:
            for sell_thresh in sell_thresholds:
                strategy = MomentumRuleStrategy(
                    buy_threshold=buy_thresh,
                    sell_threshold=sell_thresh,
                )
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=10000,
                    symbol="BTC/USDT",
                    timeframe="1d",
                )
                features = df[["close", "momentum_score", "momentum_delta"]].copy()
                features = features.dropna()
                result = engine.run(features)
                results.append({
                    "buy_threshold": buy_thresh,
                    "sell_threshold": sell_thresh,
                    "total_trades": result.total_trades,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "win_rate": result.win_rate,
                })
        return results
