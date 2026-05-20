from source.expressions.expression import Expression


class Variable(Expression):
    def __init__(self, value: float, name: str) -> None:
        self._value = value
        self._name = name

    def evaluate(self) -> float:
        return self._value
    
    @property
    def value(self) -> float:
        return self._value
    
    @property.setter
    def value(self, value: float) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Var(value={self._value}, name={self._name})"