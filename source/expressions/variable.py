from source.expressions.expression import Expression


class Variable(Expression):
    def __init__(self, value: float, name: str) -> None:
        self._name: str = name
        self._value: float = value
        self._gradient: float = 0.0

    def forward(self) -> float:
        return self._value

    def backward(self, gradient: float) -> None:
        self._gradient += gradient
    
    @property
    def value(self) -> float:
        return self._value
    
    @property
    def gradient(self) -> float:
        return self._gradient
    
    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    @gradient.setter
    def gradient(self, gradient: float) -> None:
        self._gradient = gradient

    def zero_gradient(self):
        self._gradient = 0.0

    def __repr__(self) -> str:
        return f"Var(value={self._value}, name={self._name})"