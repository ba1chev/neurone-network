from source.expressions.expression import Expression


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value
    
    def forward(self) -> float:
        return self._value
    
    def backward(self, gradient: float) -> None:
        return
    
    def __repr__(self) -> str:
        return f"Const(value={self._value})"