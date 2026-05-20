from typing import List

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.constants import ERROR_LOSS_EMPTY, ERROR_LOSS_LENGTH_MISMATCH


class MeanSquaredError(LossFunction):
    def compute(self, predictions: List[Expression], targets: List[float]) -> Expression:
        if len(predictions) == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        if len(predictions) != len(targets):
            raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)

        squared_errors: Expression = (predictions[0] - Constant(targets[0])) ** Constant(2.0)
        for prediction, target in zip(predictions[1:], targets[1:]):
            squared_errors = squared_errors + (prediction - Constant(target)) ** Constant(2.0)
        return squared_errors / Constant(float(len(predictions)))
