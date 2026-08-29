from dataclasses import dataclass
import abc
import pandas as pd


@dataclass
class Signal:
    direction: str  # "偏多", "偏空", "中立"
    confidence: float
    source: str


class Strategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, features: pd.Series) -> Signal:
        ...
