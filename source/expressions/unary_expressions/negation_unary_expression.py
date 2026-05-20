import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class NegationUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        return -self._expr.forward()

    def backward(self, gradient: np.ndarray) -> None:
        self._expr.backward(-gradient)
