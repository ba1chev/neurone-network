import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression
from source.constants import ERROR_LOG_NON_POSITIVE


class LogUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._value: np.ndarray = self._expr.forward()
        if np.any(self._value <= 0.0):
            raise RuntimeError(ERROR_LOG_NON_POSITIVE)
        return np.log(self._value)

    def backward(self, gradient: np.ndarray) -> None:
        self._expr.backward((1.0 / self._value) * gradient)
