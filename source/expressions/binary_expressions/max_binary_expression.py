from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class MaxBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._left_value = self._left_expr.forward()
        self._right_value = self._right_expr.forward()
        return max(self._left_value, self._right_value)

    def backward(self, gradient: float) -> None:
        # gradient flows entirely to the larger side, ties split evenly
        if self._left_value > self._right_value:
            self._left_expr.backward(gradient)
            self._right_expr.backward(0.0)
        elif self._right_value > self._left_value:
            self._left_expr.backward(0.0)
            self._right_expr.backward(gradient)
        else:
            half = gradient / 2
            self._left_expr.backward(half)
            self._right_expr.backward(half)
