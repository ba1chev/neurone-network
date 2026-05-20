from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class MultiplicationBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._left_value = self._left_expr.forward()
        self._right_value = self._right_expr.forward()
        return self._left_value * self._right_value

    def backward(self, gradient: float) -> None:
        self._left_expr.backward(self._right_value * gradient)
        self._right_expr.backward(self._left_value * gradient)