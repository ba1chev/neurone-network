from typing import List

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.expressions.unary_expressions.log_unary_expression import LogUnaryExpression
from source.constants import ERROR_LOSS_EMPTY, ERROR_LOSS_LENGTH_MISMATCH


class BinaryCrossEntropy(LossFunction):
    def compute(self, predictions: List[Expression], targets: List[float]) -> Expression:
        if len(predictions) == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        if len(predictions) != len(targets):
            raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)

        total: Expression = self._term(predictions[0], targets[0])
        for prediction, target in zip(predictions[1:], targets[1:]):
            total = total + self._term(prediction, target)
        return -total / Constant(float(len(predictions)))

    def _term(self, prediction: Expression, target: float) -> Expression:
        return (
            Constant(target) * LogUnaryExpression(prediction)
            + Constant(1.0 - target) * LogUnaryExpression(Constant(1.0) - prediction)
        )
