from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    def forward(self) -> float:
        raise NotImplementedError("Must be implemented")
    
    @abstractmethod
    def backward(self, gradient: float) -> None:
        raise NotImplementedError("Must be implemented")