from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class AbsUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._value = self._expr.forward()
        return abs(self._value)

    def backward(self, gradient: float) -> None:
        if self._value > 0:
            sign = 1.0
        elif self._value < 0:
            sign = -1.0
        else:
            sign = 0.0
        self._expr.backward(sign * gradient)
