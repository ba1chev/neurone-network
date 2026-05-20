import numpy as np

from source.expressions.expression import Expression
from source.expressions.broadcasting import unbroadcast
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class MinBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._left_value: np.ndarray = self._left_expr.forward()
        self._right_value: np.ndarray = self._right_expr.forward()
        return np.minimum(self._left_value, self._right_value)

    def backward(self, gradient: np.ndarray) -> None:
        # gradient flows entirely to the smaller side, ties split evenly
        left_strictly_smaller: np.ndarray = self._left_value < self._right_value
        right_strictly_smaller: np.ndarray = self._right_value < self._left_value
        tie: np.ndarray = ~(left_strictly_smaller | right_strictly_smaller)
        left_local: np.ndarray = np.where(left_strictly_smaller, gradient, 0.0) + np.where(tie, gradient / 2.0, 0.0)
        right_local: np.ndarray = np.where(right_strictly_smaller, gradient, 0.0) + np.where(tie, gradient / 2.0, 0.0)
        self._left_expr.backward(unbroadcast(left_local, self._left_value.shape))
        self._right_expr.backward(unbroadcast(right_local, self._right_value.shape))
