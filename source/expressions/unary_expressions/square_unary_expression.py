from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class SquareUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._value = self._expr.forward()
        return self._value * self._value

    def backward(self, gradient: float) -> None:
        self._expr.backward(2 * self._value * gradient)
