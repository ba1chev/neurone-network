import numpy as np

from source.expressions.expression import Expression
from source.constants import (
    ERROR_CCE_NO_LOGITS, ERROR_CCE_TARGET_OUT_OF_RANGE,
    ERROR_LOSS_EMPTY, ERROR_LOSS_LENGTH_MISMATCH
)


class SoftmaxCrossEntropyExpression(Expression):
    def __init__(self, logits_expression: Expression, targets: np.ndarray) -> None:
        self._logits_expression: Expression = logits_expression
        self._targets: np.ndarray = np.asarray(targets, dtype=np.int64)

    def forward(self) -> np.ndarray:
        logits: np.ndarray = self._logits_expression.forward()
        if logits.ndim != 2:
            raise ValueError(ERROR_CCE_NO_LOGITS)
        batch_size, num_classes = logits.shape
        if batch_size == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        if num_classes == 0:
            raise ValueError(ERROR_CCE_NO_LOGITS)
        if self._targets.shape != (batch_size,):
            raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)
        if np.any(self._targets < 0) or np.any(self._targets >= num_classes):
            raise ValueError(ERROR_CCE_TARGET_OUT_OF_RANGE)

        # log-sum-exp trick: shift by max for numerical stability
        max_per_row: np.ndarray = np.max(logits, axis=1, keepdims=True)
        shifted: np.ndarray = logits - max_per_row
        exps: np.ndarray = np.exp(shifted)
        sum_exps: np.ndarray = exps.sum(axis=1, keepdims=True)
        # softmax falls out for free; cache it for backward
        self._softmax: np.ndarray = exps / sum_exps
        self._batch_size: int = batch_size

        log_sum_exp: np.ndarray = (max_per_row + np.log(sum_exps)).reshape(batch_size)
        target_logits: np.ndarray = logits[np.arange(batch_size), self._targets]
        return np.array((log_sum_exp - target_logits).mean())

    def backward(self, gradient: np.ndarray) -> None:
        # closed-form: d L / d logits = (softmax - one_hot) / batch_size
        one_hot: np.ndarray = np.zeros_like(self._softmax)
        one_hot[np.arange(self._batch_size), self._targets] = 1.0
        scale: np.ndarray = gradient / self._batch_size
        self._logits_expression.backward((self._softmax - one_hot) * scale)

    def _collect_variables(self, out: list, seen: set) -> None:
        self._logits_expression._collect_variables(out, seen)
