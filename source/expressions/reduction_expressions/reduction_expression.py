from typing import Optional, Union, Tuple

from source.expressions.expression import Expression


class ReductionExpression(Expression):
    def __init__(self, expr: Expression, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> None:
        self._expr: Expression = expr
        self._axis: Optional[Union[int, Tuple[int, ...]]] = axis

    def _collect_variables(self, out: list, seen: set) -> None:
        self._expr._collect_variables(out, seen)
