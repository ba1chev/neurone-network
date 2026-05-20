from source.expressions.expression import Expression


class BinaryExpression(Expression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        self._left_expr: Expression = left_expr
        self._right_expr: Expression = right_expr