import math
import numpy as np
from typing import List, Optional

import source.expressions.operators  # registers operators on Expression
from source.expressions.variable import Variable
from source.expressions.expression import Expression
from source.constants import (
    ACTIVATIONS, ERROR_UNKNOWN_INITIALIZATION, INITIALIZATIONS,
    INITIAL_BIAS, UNIFORM_INITIALIZATION, XAVIER_INITIALIZATION
)


class DenseLayer:
    def __init__(self, num_inputs: int, num_neurons: int, activation: Optional[str] = "sigmoid", initialization: str = UNIFORM_INITIALIZATION) -> None:
        if num_inputs <= 0:
            raise ValueError("Layer must have at least one input")
        if num_neurons <= 0:
            raise ValueError("Layer must have at least one neuron")
        if activation is not None and activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'")
        if initialization not in INITIALIZATIONS:
            raise ValueError(f"{ERROR_UNKNOWN_INITIALIZATION}: '{initialization}'")

        self._num_inputs: int = num_inputs
        self._num_neurons: int = num_neurons
        self._activation: Optional[str] = activation
        self._weights: Variable = Variable(
            _sample_weights(initialization, num_inputs, num_neurons), "W"
        )
        self._bias: Variable = Variable(
            np.full(num_neurons, INITIAL_BIAS), "b"
        )

    def forward(self, inputs: Expression) -> Expression:
        pre_activation: Expression = inputs @ self._weights + self._bias
        if self._activation is None:
            return pre_activation
        return ACTIVATIONS[self._activation](pre_activation)

    def parameters(self) -> List[Variable]:
        return [self._weights, self._bias]


def _sample_weights(scheme: str, num_inputs: int, num_neurons: int) -> np.ndarray:
    shape: tuple = (num_inputs, num_neurons)
    if scheme == UNIFORM_INITIALIZATION:
        return np.random.uniform(-1.0, 1.0, size=shape)
    if scheme == XAVIER_INITIALIZATION:
        limit: float = math.sqrt(6.0 / (num_inputs + num_neurons))
        return np.random.uniform(-limit, limit, size=shape)
    raise ValueError(f"{ERROR_UNKNOWN_INITIALIZATION}: '{scheme}'")
