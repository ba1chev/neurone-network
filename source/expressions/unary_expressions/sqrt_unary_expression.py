import math

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression
from source.constants import ERROR_SQRT_GRAD_AT_ZERO, ERROR_SQRT_NEGATIVE


class SqrtUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        value = self._expr.forward()
        if value < 0:
            raise RuntimeError(ERROR_SQRT_NEGATIVE)
        # pre-caching for better performance
        self._result = math.sqrt(value)
        return self._result

    def backward(self, gradient: float) -> None:
        if self._result == 0:
            raise RuntimeError(ERROR_SQRT_GRAD_AT_ZERO)
        self._expr.backward((1 / (2 * self._result)) * gradient)
