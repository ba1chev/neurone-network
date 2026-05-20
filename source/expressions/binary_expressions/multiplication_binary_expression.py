import numpy as np

from source.expressions.expression import Expression
from source.expressions.broadcasting import unbroadcast
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class MultiplicationBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._left_value: np.ndarray = self._left_expr.forward()
        self._right_value: np.ndarray = self._right_expr.forward()
        return self._left_value * self._right_value

    def backward(self, gradient: np.ndarray) -> None:
        self._left_expr.backward(unbroadcast(self._right_value * gradient, self._left_value.shape))
        self._right_expr.backward(unbroadcast(self._left_value * gradient, self._right_value.shape))
