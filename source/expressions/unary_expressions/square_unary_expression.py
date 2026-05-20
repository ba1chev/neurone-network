import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class SquareUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._value: np.ndarray = self._expr.forward()
        return self._value * self._value

    def backward(self, gradient: np.ndarray) -> None:
        self._expr.backward(2.0 * self._value * gradient)
