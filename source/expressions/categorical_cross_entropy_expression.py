import math
from typing import List

from source.expressions.expression import Expression


class CategoricalCrossEntropyExpression(Expression):
    def __init__(self, logits_per_sample: List[List[Expression]], targets: List[int]) -> None:
        self._logits_per_sample: List[List[Expression]] = logits_per_sample
        self._targets: List[int] = targets
        self._softmax_per_sample: List[List[float]] = []

    def forward(self) -> float:
        self._softmax_per_sample = []
        total_loss: float = 0.0
        batch_size: int = len(self._logits_per_sample)

        for logit_expressions, target in zip(self._logits_per_sample, self._targets):
            logit_values: List[float] = [expr.forward() for expr in logit_expressions]
            max_logit: float = max(logit_values)
            shifted: List[float] = [value - max_logit for value in logit_values]
            exps: List[float] = [math.exp(value) for value in shifted]
            denominator: float = sum(exps)
            log_sum_exp: float = max_logit + math.log(denominator)
            total_loss += -logit_values[target] + log_sum_exp
            self._softmax_per_sample.append([value / denominator for value in exps])

        return total_loss / batch_size

    def backward(self, gradient: float) -> None:
        batch_size: int = len(self._logits_per_sample)
        scale: float = gradient / batch_size

        for logit_expressions, softmax, target in zip(
            self._logits_per_sample, self._softmax_per_sample, self._targets
        ):
            for index, logit_expression in enumerate(logit_expressions):
                indicator: float = 1.0 if index == target else 0.0
                logit_expression.backward((softmax[index] - indicator) * scale)

    def _collect_variables(self, out: list, seen: set) -> None:
        for logit_expressions in self._logits_per_sample:
            for logit_expression in logit_expressions:
                logit_expression._collect_variables(out, seen)
