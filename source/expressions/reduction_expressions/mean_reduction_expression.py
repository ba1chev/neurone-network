import numpy as np
from typing import Optional, Union, Tuple

from source.expressions.expression import Expression
from source.expressions.reduction_expressions.reduction_expression import ReductionExpression


class MeanReductionExpression(ReductionExpression):
    def __init__(self, expr: Expression, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> None:
        super().__init__(expr, axis)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        input_value: np.ndarray = self._expr.forward()
        self._input_shape: Tuple[int, ...] = input_value.shape
        self._divisor: float = float(self._compute_divisor(input_value.shape))
        return input_value.sum(axis=self._axis) / self._divisor

    def backward(self, gradient: np.ndarray) -> None:
        # mean = sum / divisor; grad of input is broadcast(grad / divisor) over reduced axes
        expanded: np.ndarray = self._expand_to_input_shape(gradient / self._divisor)
        self._expr.backward(np.broadcast_to(expanded, self._input_shape).copy())

    def _compute_divisor(self, input_shape: Tuple[int, ...]) -> int:
        if self._axis is None:
            return int(np.prod(input_shape)) if input_shape else 1
        axes: Tuple[int, ...] = (self._axis,) if isinstance(self._axis, int) else self._axis
        divisor: int = 1
        for axis in axes:
            divisor *= input_shape[axis]
        return divisor

    def _expand_to_input_shape(self, gradient: np.ndarray) -> np.ndarray:
        if self._axis is None:
            return gradient.reshape((1,) * len(self._input_shape))
        axes: Tuple[int, ...] = (self._axis,) if isinstance(self._axis, int) else self._axis
        target_shape: list = list(self._input_shape)
        for axis in axes:
            target_shape[axis] = 1
        return gradient.reshape(target_shape)
