import numpy as np

from source.expressions.expression import Expression


class Variable(Expression):
    def __init__(self, value: np.ndarray, name: str) -> None:
        self._name: str = name
        self._value: np.ndarray = np.asarray(value, dtype=np.float64)
        self._gradient: np.ndarray = np.zeros_like(self._value)

    def forward(self) -> np.ndarray:
        return self._value

    def backward(self, gradient: np.ndarray) -> None:
        self._gradient = self._gradient + gradient

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def gradient(self) -> np.ndarray:
        return self._gradient

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._value = np.asarray(value, dtype=np.float64)

    @gradient.setter
    def gradient(self, gradient: np.ndarray) -> None:
        self._gradient = np.asarray(gradient, dtype=np.float64)

    def zero_gradient(self) -> None:
        self._gradient = np.zeros_like(self._value)

    def _collect_variables(self, out: list, seen: set) -> None:
        if id(self) in seen:
            return
        seen.add(id(self))
        out.append(self)

    def __repr__(self) -> str:
        return f"Var(shape={self._value.shape}, name={self._name})"
