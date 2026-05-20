import numpy as np

from source.expressions.expression import Expression
from source.expressions.unary_expressions.unary_expression import UnaryExpression


class AbsUnaryExpression(UnaryExpression):
    def __init__(self, expr: Expression) -> None:
        super().__init__(expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._value: np.ndarray = self._expr.forward()
        return np.abs(self._value)

    def backward(self, gradient: np.ndarray) -> None:
        # gradient flows with the sign of the input; zero entries contribute zero
        sign: np.ndarray = np.sign(self._value)
        self._expr.backward(sign * gradient)
