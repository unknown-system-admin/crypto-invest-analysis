from backtest_engine.strategy import Strategy, Signal


class MomentumShortStrategy(Strategy):
    def __init__(self, buy_threshold: float = 0.01, sell_threshold: float = -0.01):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def evaluate(self, features) -> Signal:
        score = features.get("momentum_score", 0)
        delta = features.get("momentum_delta", 0)

        if score > self.buy_threshold and delta > 0:
            return Signal("偏多", abs(score), "momentum_short")
        elif score < self.sell_threshold and delta < 0:
            return Signal("偏空", abs(score), "momentum_short")
        else:
            return Signal("中立", 0.5, "momentum_short")


class MLShortStrategy(Strategy):
    def __init__(self, model, scaler, threshold: float = 0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def evaluate(self, features) -> Signal:
        feature_cols = ["SMA_20", "SMA_50", "RSI", "MACD", "ATR", "MFI", "OBV", "close", "momentum_score", "momentum_delta"]
        X = [[features.get(col, 0) for col in feature_cols]]
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0]
        max_prob = max(prob)

        if pred == 1 and max_prob > self.threshold:
            return Signal("偏多", max_prob, "ml_short")
        elif pred == 0 and max_prob > self.threshold:
            return Signal("偏空", max_prob, "ml_short")
        else:
            return Signal("中立", 0.5, "ml_short")
