import math

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class SigmoidUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._result = 1 / (1 + math.exp(-self._expr.forward()))
        return self._result

    def backward(self, gradient: float) -> None:
        local = self._result * (1 - self._result)
        self._expr.backward(local * gradient)
