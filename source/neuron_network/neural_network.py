from typing import List, Optional

from source.expressions.expression import Expression
from source.neuron_network.dense_layer import DenseLayer
from source.expressions.variable import Variable
from source.constants import UNIFORM_INITIALIZATION


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: Optional[List[Optional[str]]] = None, initialization: str = UNIFORM_INITIALIZATION) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least one input and one output layer")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive")

        num_layers: int = len(layer_sizes) - 1
        if activations is None:
            activations = ["sigmoid"] * num_layers
        if len(activations) != num_layers:
            raise ValueError(
                f"Expected {num_layers} activations, got {len(activations)}"
            )

        self._layer_sizes: List[int] = layer_sizes
        self._layers: List[DenseLayer] = [
            DenseLayer(
                layer_sizes[i], layer_sizes[i + 1],
                activations[i], initialization
            )
            for i in range(num_layers)
        ]

    def forward(self, inputs: Expression) -> Expression:
        current: Expression = inputs
        for layer in self._layers:
            current = layer.forward(current)
        return current

    def parameters(self) -> List[Variable]:
        parameters: List[Variable] = []
        for current_layer in self._layers:
            parameters.extend(current_layer.parameters())
        return parameters
