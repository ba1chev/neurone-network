from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class NegationUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> float:
        return -self._expr.forward()

    def backward(self, gradient: float) -> None:
        self._expr.backward(-gradient)
