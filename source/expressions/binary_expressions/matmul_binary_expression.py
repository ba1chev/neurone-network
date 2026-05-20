import numpy as np

from source.expressions.expression import Expression
from source.expressions.binary_expressions.binary_expression import BinaryExpression
from source.constants import ERROR_MATMUL_DIMENSION_MISMATCH


class MatmulBinaryExpression(BinaryExpression):
    def __init__(self, left_expr: Expression, right_expr: Expression) -> None:
        super().__init__(left_expr, right_expr)

    def forward(self) -> np.ndarray:
        # pre-caching for better performance
        self._left_value: np.ndarray = self._left_expr.forward()
        self._right_value: np.ndarray = self._right_expr.forward()
        if self._left_value.shape[-1] != self._right_value.shape[0]:
            raise RuntimeError(
                f"{ERROR_MATMUL_DIMENSION_MISMATCH}: "
                f"{self._left_value.shape} @ {self._right_value.shape}"
            )
        return self._left_value @ self._right_value

    def backward(self, gradient: np.ndarray) -> None:
        # d(A @ B) / dA = grad @ B.T ; d(A @ B) / dB = A.T @ grad
        self._left_expr.backward(gradient @ self._right_value.T)
        self._right_expr.backward(self._left_value.T @ gradient)
