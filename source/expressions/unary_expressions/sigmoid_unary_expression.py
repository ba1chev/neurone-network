import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class SigmoidUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._result: np.ndarray = 1.0 / (1.0 + np.exp(-self._expr.forward()))
        return self._result

    def backward(self, gradient: np.ndarray) -> None:
        local: np.ndarray = self._result * (1.0 - self._result)
        self._expr.backward(local * gradient)
