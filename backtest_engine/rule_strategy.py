from backtest_engine.strategy import Strategy, Signal


class MomentumRuleStrategy(Strategy):
    def __init__(self, buy_threshold: float = 0.08, sell_threshold: float = -0.30,
                 short_entry_threshold: float = None):
        """Momentum rule with split thresholds.

        buy_threshold:      long entry / short exit threshold (score > this)
        sell_threshold:     long exit threshold (score < this) — deep, holds pullbacks
        short_entry_threshold: short entry threshold (score < this). Defaults to
                            sell_threshold for backward compatibility (single-gate).
                            Set higher (less negative) to short earlier in bears.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.short_entry_threshold = short_entry_threshold if short_entry_threshold is not None else sell_threshold

    def evaluate(self, features, side: str = "flat") -> Signal:
        score = features.get("momentum_score", 0)
        delta = features.get("momentum_delta", 0)

        if side == "long":
            # Hold through pullbacks; exit only on deep reversal
            if score < self.sell_threshold:
                return Signal("偏空", abs(score), "momentum_rule")
            return Signal("中立", 0.5, "momentum_rule")

        if side == "short":
            # Cover on strong bounce
            if score > self.buy_threshold:
                return Signal("偏多", abs(score), "momentum_rule")
            return Signal("中立", 0.5, "momentum_rule")

        # Flat: entries
        if score > self.buy_threshold and delta > 0:
            return Signal("偏多", abs(score), "momentum_rule")
        elif score < self.short_entry_threshold and delta < 0:
            return Signal("偏空", abs(score), "momentum_rule")
        else:
            return Signal("中立", 0.5, "momentum_rule")