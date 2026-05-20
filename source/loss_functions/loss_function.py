from typing import List
from abc import ABC, abstractmethod

from source.expressions.expression import Expression


class LossFunction(ABC):
    @abstractmethod
    def compute(self, predictions: List[Expression], targets: List[float]) -> Expression:
        raise NotImplementedError("Must be implemented")
