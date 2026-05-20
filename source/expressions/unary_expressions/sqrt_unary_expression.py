import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression
from source.constants import ERROR_SQRT_GRAD_AT_ZERO, ERROR_SQRT_NEGATIVE


class SqrtUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        value: np.ndarray = self._expr.forward()
        if np.any(value < 0.0):
            raise RuntimeError(ERROR_SQRT_NEGATIVE)
        # pre-caching for better performance
        self._result: np.ndarray = np.sqrt(value)
        return self._result

    def backward(self, gradient: np.ndarray) -> None:
        if np.any(self._result == 0.0):
            raise RuntimeError(ERROR_SQRT_GRAD_AT_ZERO)
        self._expr.backward((1.0 / (2.0 * self._result)) * gradient)
