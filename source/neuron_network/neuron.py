import random
from typing import List, Optional

import source.expressions.operators  # noqa: F401  registers operators on Expression
from source.expressions.variable import Variable
from source.expressions.expression import Expression
from source.constants import ACTIVATIONS, INITIAL_BIAS


class Neuron:
    def __init__(self, num_inputs: int, activation: Optional[str] = "sigmoid") -> None:
        if num_inputs <= 0:
            raise ValueError("Neuron must have at least one input")
        if activation is not None and activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'")

        self._num_inputs: int = num_inputs
        self._activation: Optional[str] = activation
        self._weights: List[Variable] = [
            Variable(random.uniform(-1.0, 1.0), f"w{i}") for i in range(num_inputs)
        ]
        self._bias: Variable = Variable(INITIAL_BIAS, "b")

    def forward(self, inputs: List[Expression]) -> Expression:
        if len(inputs) != self._num_inputs:
            raise ValueError(
                f"Expected {self._num_inputs} inputs, got {len(inputs)}"
            )

        weighted_sum: Expression = self._weights[0] * inputs[0]
        for weight, x in zip(self._weights[1:], inputs[1:]):
            weighted_sum = weighted_sum + weight * x
        weighted_sum = weighted_sum + self._bias

        if self._activation is None:
            return weighted_sum
        return ACTIVATIONS[self._activation](weighted_sum)

    def parameters(self) -> List[Variable]:
        return [*self._weights, self._bias]
