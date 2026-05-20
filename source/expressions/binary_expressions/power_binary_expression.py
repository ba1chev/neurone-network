import math

from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression
from source.constants import ERROR_POW_NEGATIVE_BASE, ERROR_POW_ZERO_NEGATIVE_EXP


class PowerBinaryExpression(BinaryExpression):
    def __init__(self, base_expr: Expression, exponent_expr: Expression) -> None:
        super().__init__(base_expr, exponent_expr)

    def forward(self) -> float:
        # pre-caching for better performance
        self._left_value = self._left_expr.forward()
        self._right_value = self._right_expr.forward()
        if self._left_value < 0 and not float(self._right_value).is_integer():
            raise RuntimeError(ERROR_POW_NEGATIVE_BASE)
        if self._left_value == 0 and self._right_value < 0:
            raise RuntimeError(ERROR_POW_ZERO_NEGATIVE_EXP)
        self._result = self._left_value ** self._right_value
        return self._result

    def backward(self, gradient: float) -> None:
        base_gradient = self._right_value * (self._left_value ** (self._right_value - 1))
        self._left_expr.backward(base_gradient * gradient)

        if self._left_value > 0:
            exponent_gradient = self._result * math.log(self._left_value)
        else:
            exponent_gradient = 0.0
        self._right_expr.backward(exponent_gradient * gradient)
