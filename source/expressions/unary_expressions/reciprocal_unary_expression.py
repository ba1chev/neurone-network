from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression
from source.constants import ERROR_RECIPROCAL_OF_ZERO


class ReciprocalUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._value = self._expr.forward()
        if self._value == 0:
            raise RuntimeError(ERROR_RECIPROCAL_OF_ZERO)
        return 1 / self._value

    def backward(self, gradient: float) -> None:
        self._expr.backward((-1 / (self._value * self._value)) * gradient)
