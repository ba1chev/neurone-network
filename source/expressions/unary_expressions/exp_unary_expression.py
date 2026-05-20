import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class ExpUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._result: np.ndarray = np.exp(self._expr.forward())
        return self._result

    def backward(self, gradient: np.ndarray) -> None:
        self._expr.backward(self._result * gradient)
