from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class DivisionBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._left_value = self._left_expr.forward()
        self._right_value = self._right_expr.forward()
        if self._right_value == 0:
            raise RuntimeError("Cannot divide by zero")
        return self._left_value / self._right_value

    def backward(self, gradient: float) -> None:
        self._left_expr.backward((1 / self._right_value) * gradient)
        self._right_expr.backward((-self._left_value / self._right_value ** 2) * gradient)