from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class ReluUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._value = self._expr.forward()
        return self._value if self._value > 0 else 0.0

    def backward(self, gradient: float) -> None:
        local = gradient if self._value > 0 else 0.0
        self._expr.backward(local)
