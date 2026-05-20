from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class AdditionBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> float:
        return self._left_expr.forward() + self._right_expr.forward()
    
    def backward(self, gradient: float) -> None:
        self._left_expr.backward(gradient)
        self._right_expr.backward(gradient)