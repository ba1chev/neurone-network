import numpy as np

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.expressions.unary_expressions.log_unary_expression import LogUnaryExpression
from source.expressions.reduction_expressions.mean_reduction_expression import MeanReductionExpression
from source.constants import ERROR_LOSS_EMPTY


class BinaryCrossEntropy(LossFunction):
    def compute(self, predictions: Expression, targets: np.ndarray) -> Expression:
        targets_array: np.ndarray = np.asarray(targets, dtype=np.float64)
        if targets_array.size == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        target_constant: Expression = Constant(targets_array)
        one_constant: Expression = Constant(np.array(1.0))
        per_sample: Expression = (
            target_constant * LogUnaryExpression(predictions)
            + (one_constant - target_constant) * LogUnaryExpression(one_constant - predictions)
        )
        return -MeanReductionExpression(per_sample)
