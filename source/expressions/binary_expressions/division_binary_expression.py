from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression


class DivisionBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def evaluate(self) -> float:
        right_value = self._right_expr.evaluate()
        if right_value == 0:
            raise RuntimeError("Cannot divide by zero")
        return self._left_expr.evaluate() / right_value