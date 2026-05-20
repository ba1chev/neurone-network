import numpy as np

from source.expressions.expression import Expression
from source.expressions.broadcasting import unbroadcast
from source.expressions.binary_expressions.binary_expression import BinaryExpression
from source.constants import ERROR_POW_NEGATIVE_BASE, ERROR_POW_ZERO_NEGATIVE_EXP


class PowerBinaryExpression(BinaryExpression):
    def __init__(self, base_expr: Expression, exponent_expr: Expression) -> None:
        super().__init__(base_expr, exponent_expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._left_value: np.ndarray = self._left_expr.forward()
        self._right_value: np.ndarray = self._right_expr.forward()
        non_integer_exponent: np.ndarray = self._right_value != np.floor(self._right_value)
        if np.any((self._left_value < 0.0) & non_integer_exponent):
            raise RuntimeError(ERROR_POW_NEGATIVE_BASE)
        if np.any((self._left_value == 0.0) & (self._right_value < 0.0)):
            raise RuntimeError(ERROR_POW_ZERO_NEGATIVE_EXP)
        self._result: np.ndarray = self._left_value ** self._right_value
        return self._result

    def backward(self, gradient: np.ndarray) -> None:
        base_local: np.ndarray = self._right_value * (self._left_value ** (self._right_value - 1.0))
        # log of non-positive base is undefined; gradient through exponent is zero there
        safe_log_base: np.ndarray = np.where(self._left_value > 0.0, np.log(np.where(self._left_value > 0.0, self._left_value, 1.0)), 0.0)
        exponent_local: np.ndarray = self._result * safe_log_base
        self._left_expr.backward(unbroadcast(base_local * gradient, self._left_value.shape))
        self._right_expr.backward(unbroadcast(exponent_local * gradient, self._right_value.shape))
