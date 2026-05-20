import numpy as np

import source.expressions.operators  # registers operators on Expression
from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.expressions.constant import Constant
from source.expressions.reduction_expressions.mean_reduction_expression import MeanReductionExpression
from source.constants import ERROR_LOSS_EMPTY


class MeanAbsoluteError(LossFunction):
    def compute(self, predictions: Expression, targets: np.ndarray) -> Expression:
        targets_array: np.ndarray = np.asarray(targets, dtype=np.float64)
        if targets_array.size == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        difference: Expression = predictions - Constant(targets_array)
        return MeanReductionExpression(abs(difference))
