import numpy as np
from typing import Optional, Union, Tuple

from source.expressions.expression import Expression
from source.expressions.reduction_expressions.reduction_expression import ReductionExpression


class LogSumExpReductionExpression(ReductionExpression):
    def __init__(self, expr: Expression, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> None:
        super().__init__(expr, axis)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        input_value: np.ndarray = self._expr.forward()
        self._input_shape: Tuple[int, ...] = input_value.shape

        # log-sum-exp trick: subtract max to keep exponentials in range, then add it back
        max_value: np.ndarray = np.max(input_value, axis=self._axis, keepdims=True)
        shifted: np.ndarray = input_value - max_value
        exps: np.ndarray = np.exp(shifted)
        sum_exps: np.ndarray = exps.sum(axis=self._axis, keepdims=True)

        # softmax falls out for free; cache it for backward
        self._softmax: np.ndarray = exps / sum_exps
        return (max_value + np.log(sum_exps)).reshape(self._reduced_shape())

    def backward(self, gradient: np.ndarray) -> None:
        # d log-sum-exp / d x_i = softmax(x)_i ; broadcast upstream gradient over reduced axes
        expanded: np.ndarray = self._expand_to_input_shape(gradient)
        self._expr.backward(expanded * self._softmax)

    def _reduced_shape(self) -> Tuple[int, ...]:
        if self._axis is None:
            return ()
        axes: Tuple[int, ...] = (self._axis,) if isinstance(self._axis, int) else self._axis
        return tuple(size for axis, size in enumerate(self._input_shape) if axis not in axes)

    def _expand_to_input_shape(self, gradient: np.ndarray) -> np.ndarray:
        if self._axis is None:
            return gradient.reshape((1,) * len(self._input_shape))
        axes: Tuple[int, ...] = (self._axis,) if isinstance(self._axis, int) else self._axis
        target_shape: list = list(self._input_shape)
        for axis in axes:
            target_shape[axis] = 1
        return gradient.reshape(target_shape)
