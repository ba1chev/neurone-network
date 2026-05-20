from source.expressions.expression import Expression


class UnaryExpression(Expression):
    def __init__(self, expr: Expression) -> None:
        self._expr: Expression = expr

    def _collect_variables(self, out: list, seen: set) -> None:
        self._expr._collect_variables(out, seen)
