import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class ReluUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._value: np.ndarray = self._expr.forward()
        return np.maximum(self._value, 0.0)

    def backward(self, gradient: np.ndarray) -> None:
        local: np.ndarray = np.where(self._value > 0.0, gradient, 0.0)
        self._expr.backward(local)
