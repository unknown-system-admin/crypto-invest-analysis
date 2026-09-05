from backtest_engine.strategy import Strategy, Signal


class MeanReversionStrategy(Strategy):
    def __init__(self, **params):
        # TODO: Add strategy parameters
        for k, v in params.items():
            setattr(self, k, v)

    def evaluate(self, features) -> Signal:
        """
        Features available:
        - momentum_score, momentum_delta
        - SMA_20, SMA_50, SMA_200, EMA_26
        - RSI, MACD, ATR, MFI, OBV
        - close, volume, high, low
        """
        # TODO: Implement strategy logic
        score = features.get("momentum_score", 0)
        delta = features.get("momentum_delta", 0)
        
        # Example logic - replace with your strategy
        if score > 0.05 and delta > 0:
            return Signal("偏多", abs(score), "meanreversionstrategy")
        elif score < -0.05 and delta < 0:
            return Signal("偏空", abs(score), "meanreversionstrategy")
        else:
            return Signal("中立", 0.5, "meanreversionstrategy")
