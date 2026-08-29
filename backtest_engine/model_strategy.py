import numpy as np
from backtest_engine.strategy import Strategy, Signal


class ModelStrategy(Strategy):
    def __init__(self, model, feature_columns: list, confidence_threshold: float = 0.5):
        self.model = model
        self.feature_columns = feature_columns
        self.confidence_threshold = confidence_threshold
    
    def evaluate(self, features) -> Signal:
        try:
            X = np.array([[features[col] for col in self.feature_columns]])
            prediction = self.model.predict(X)[0]
            
            # Try to get probability if available
            try:
                proba = self.model.predict_proba(X)[0]
                confidence = max(proba)
            except (AttributeError, TypeError):
                confidence = 1.0
            
            if confidence < self.confidence_threshold:
                return Signal("中立", confidence, "model")
            
            if prediction == 1:
                return Signal("偏多", confidence, "model")
            else:
                return Signal("偏空", confidence, "model")
                
        except Exception:
            return Signal("中立", 0.0, "model")