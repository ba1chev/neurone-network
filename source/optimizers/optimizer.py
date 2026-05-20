from typing import List
from abc import ABC, abstractmethod

from source.expressions.variable import Variable
from source.constants import ERROR_LEARNING_RATE_NON_POSITIVE, ERROR_OPTIMIZER_NO_PARAMETERS


class Optimizer(ABC):
    def __init__(self, parameters: List[Variable], learning_rate: float) -> None:
        if len(parameters) == 0:
            raise ValueError(ERROR_OPTIMIZER_NO_PARAMETERS)
        if learning_rate <= 0:
            raise ValueError(ERROR_LEARNING_RATE_NON_POSITIVE)

        self._parameters: List[Variable] = parameters
        self._learning_rate: float = learning_rate

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError("Must be implemented")

    def zero_gradients(self) -> None:
        for parameter in self._parameters:
            parameter.zero_gradient()
