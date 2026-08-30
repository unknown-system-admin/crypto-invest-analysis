import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FeatureSelector:
    def __init__(self):
        self.model = None
        self.feature_importance = {}

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        label_column: str,
    ) -> dict:
        X = df[feature_columns]
        y = df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.feature_importance = dict(zip(
            feature_columns,
            self.model.feature_importances_
        ))

        return {
            "accuracy": round(accuracy, 4),
            "feature_importance": self.feature_importance,
        }

    def get_top_features(self, n: int = 10) -> list:
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [f[0] for f in sorted_features[:n]]
