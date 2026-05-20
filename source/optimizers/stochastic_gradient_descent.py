from typing import List

from source.expressions.variable import Variable
from source.optimizers.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters: List[Variable], learning_rate: float) -> None:
        super().__init__(parameters, learning_rate)

    def step(self) -> None:
        for parameter in self._parameters:
            parameter.value = parameter.value - self._learning_rate * parameter.gradient
