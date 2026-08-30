from backtest_engine.strategy import Strategy, Signal


class MomentumRuleStrategy(Strategy):
    def __init__(self, buy_threshold: float = 0.25, sell_threshold: float = -0.15):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def evaluate(self, features) -> Signal:
        score = features.get("momentum_score", 0)
        delta = features.get("momentum_delta", 0)
        
        if score > self.buy_threshold and delta > 0:
            return Signal("偏多", abs(score), "momentum_rule")
        elif score < self.sell_threshold and delta < 0:
            return Signal("偏空", abs(score), "momentum_rule")
        else:
            return Signal("中立", 0.5, "momentum_rule")
