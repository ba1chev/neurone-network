from typing import List, Optional

from source.expressions.variable import Variable
from source.expressions.expression import Expression
from source.neuron_network.neuron import Neuron, make_weight_initializer
from source.constants import UNIFORM_INITIALIZATION


class NeuronLayer:
    def __init__(self, num_inputs: int, num_neurons: int, activation: Optional[str] = "sigmoid", initialization: str = UNIFORM_INITIALIZATION) -> None:
        if num_inputs <= 0:
            raise ValueError("Layer must have at least one input")
        if num_neurons <= 0:
            raise ValueError("Layer must have at least one neuron")

        self._num_inputs: int = num_inputs
        self._num_neurons: int = num_neurons
        weight_initializer = make_weight_initializer(
            initialization, fan_in=num_inputs, fan_out=num_neurons
        )
        self._neurons: List[Neuron] = [
            Neuron(num_inputs, activation, weight_initializer)
            for _ in range(num_neurons)
        ]

    def forward(self, inputs: List[Expression]) -> List[Expression]:
        if len(inputs) != self._num_inputs:
            raise ValueError(
                f"Expected {self._num_inputs} inputs, got {len(inputs)}"
            )
        return [neuron.forward(inputs) for neuron in self._neurons]

    def parameters(self) -> List[Variable]:
        parameters: List[Variable] = []
        for current_neuron in self._neurons:
            parameters.extend(current_neuron.parameters())
        return parameters
