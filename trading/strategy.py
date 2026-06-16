from typing import Optional, Callable
from dataclasses import dataclass
import pandas as pd


@dataclass
class Signal:
    direction: str
    confidence: float
    source: str


class Strategy:
    name: str = ""

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        raise NotImplementedError


class MACross(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 12, slow: int = 26, ma_type: str = "ema"):
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        col_fast = f"{self.ma_type.upper()}_{self.fast}"
        col_slow = f"{self.ma_type.upper()}_{self.slow}"
        if col_fast not in overlay.columns or col_slow not in overlay.columns:
            return Signal("中立", 0.0, self.name)
        prev_fast = overlay[col_fast].iloc[-2]
        prev_slow = overlay[col_slow].iloc[-2]
        cur_fast = overlay[col_fast].iloc[-1]
        cur_slow = overlay[col_slow].iloc[-1]
        if prev_fast <= prev_slow and cur_fast > cur_slow:
            return Signal("偏多", 0.8, self.name)
        if prev_fast >= prev_slow and cur_fast < cur_slow:
            return Signal("偏空", 0.8, self.name)
        return Signal("中立", 0.5, self.name)


class RSIThreshold(Strategy):
    name = "rsi"

    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        rsi = subplots.get("rsi", pd.DataFrame())
        if rsi.empty or "RSI" not in rsi.columns:
            return Signal("中立", 0.0, self.name)
        val = rsi["RSI"].iloc[-1]
        prev = rsi["RSI"].iloc[-2] if len(rsi) > 1 else val
        if prev >= self.oversold and val < self.oversold:
            return Signal("偏空", 0.7, self.name)
        if prev <= self.overbought and val > self.overbought:
            return Signal("偏多", 0.7, self.name)
        if val <= self.oversold:
            return Signal("偏多", 0.6, self.name)
        if val >= self.overbought:
            return Signal("偏空", 0.6, self.name)
        return Signal("中立", 0.5, self.name)


class MACDCross(Strategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        macd = subplots.get("macd", pd.DataFrame())
        if macd.empty or "MACD" not in macd.columns or "Signal" not in macd.columns:
            return Signal("中立", 0.0, self.name)
        prev_macd = macd["MACD"].iloc[-2]
        prev_sig = macd["Signal"].iloc[-2]
        cur_macd = macd["MACD"].iloc[-1]
        cur_sig = macd["Signal"].iloc[-1]
        if prev_macd <= prev_sig and cur_macd > cur_sig:
            return Signal("偏多", 0.8, self.name)
        if prev_macd >= prev_sig and cur_macd < cur_sig:
            return Signal("偏空", 0.8, self.name)
        return Signal("中立", 0.5, self.name)


class CompositeStrategy(Strategy):
    name = "composite"

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Signal:
        from analysis.summary import analyze_signals
        sig = analyze_signals(overlay, subplots)
        direction = sig.get("direction", "中立")
        total = sig.get("total", 1)
        bullish = sig.get("bullish_count", 0)
        bearish = sig.get("bearish_count", 0)
        confidence = max(bullish, bearish) / total if total > 0 else 0
        return Signal(direction, confidence, self.name)


class CustomComposite:
    def __init__(self, strategy_configs: dict, threshold: float = 0.6):
        self.strategies = []
        self.weights = []
        strategy_map = {
            "ma_cross": MACross,
            "rsi": RSIThreshold,
            "macd": MACDCross,
            "composite": CompositeStrategy,
        }
        param_key_map = {
            "ma_cross": {"type": "ma_type"},
        }
        for name, cfg in strategy_configs.items():
            if cfg.get("enabled") and name in strategy_map:
                cls = strategy_map[name]
                params = cfg.get("params", {}).copy()
                for old_key, new_key in param_key_map.get(name, {}).items():
                    if old_key in params:
                        params[new_key] = params.pop(old_key)
                weight = cfg.get("weight", 1)
                self.strategies.append(cls(**params))
                self.weights.append(weight)
        self.threshold = threshold

    def evaluate(self, overlay: pd.DataFrame, subplots: dict) -> Optional[Signal]:
        total_weight = sum(self.weights)
        if total_weight == 0:
            return None
        score_bullish = 0.0
        score_bearish = 0.0
        for s, w in zip(self.strategies, self.weights):
            sig = s.evaluate(overlay, subplots)
            if sig.direction == "偏多":
                score_bullish += sig.confidence * w
            elif sig.direction == "偏空":
                score_bearish += sig.confidence * w
        bull_ratio = score_bullish / total_weight
        bear_ratio = score_bearish / total_weight
        if bull_ratio > self.threshold and bull_ratio > bear_ratio:
            return Signal("偏多", bull_ratio, "custom_composite")
        if bear_ratio > self.threshold and bear_ratio > bull_ratio:
            return Signal("偏空", bear_ratio, "custom_composite")
        return Signal("中立", max(bull_ratio, bear_ratio), "custom_composite")
