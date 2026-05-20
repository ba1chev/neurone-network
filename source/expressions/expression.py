from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        raise NotImplementedError("Must be implemented")