from source.expressions.expression import Expression


class UnaryExpression(Expression):
    def __init__(self, expr: Expression) -> None:
        self._expr: Expression = expr