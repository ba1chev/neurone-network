import numpy as np
from abc import ABC, abstractmethod

from source.expressions.expression import Expression


class LossFunction(ABC):
    @abstractmethod
    def compute(self, predictions: Expression, targets: np.ndarray) -> Expression:
        raise NotImplementedError("Must be implemented")
