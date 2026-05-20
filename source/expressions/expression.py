from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    def forward(self) -> float:
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def backward(self, gradient: float) -> None:
        raise NotImplementedError("Must be implemented")

    @abstractmethod
    def _collect_variables(self, out: list, seen: set) -> None:
        raise NotImplementedError("Must be implemented")

    def get_variables(self) -> list:
        out: list = []
        seen: set = set()
        self._collect_variables(out, seen)
        return out

    def zero_all_gradients(self) -> None:
        for current_variable in self.get_variables():
            current_variable.zero_gradient()
