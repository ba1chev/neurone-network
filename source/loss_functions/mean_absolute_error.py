from typing import List

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.constants import ERROR_LOSS_EMPTY, ERROR_LOSS_LENGTH_MISMATCH


class MeanAbsoluteError(LossFunction):
    def compute(self, predictions: List[Expression], targets: List[float]) -> Expression:
        if len(predictions) == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        if len(predictions) != len(targets):
            raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)

        absolute_errors: Expression = abs(predictions[0] - Constant(targets[0]))
        for prediction, target in zip(predictions[1:], targets[1:]):
            absolute_errors = absolute_errors + abs(prediction - Constant(target))
        return absolute_errors / Constant(float(len(predictions)))
