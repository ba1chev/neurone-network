from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class AdditionBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def evaluate(self) -> float:
        return self._left_expr.evaluate() + self._right_expr.evaluate()