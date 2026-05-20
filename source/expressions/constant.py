import numpy as np

from source.expressions.expression import Expression


class Constant(Expression):
    def __init__(self, value: np.ndarray) -> None:
        self._value: np.ndarray = np.asarray(value, dtype=np.float64)

    def forward(self) -> np.ndarray:
        return self._value

    def backward(self, gradient: np.ndarray) -> None:
        return

    def _collect_variables(self, out: list, seen: set) -> None:
        return

    def __repr__(self) -> str:
        return f"Const(shape={self._value.shape})"
