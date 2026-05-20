import numpy as np

from source.expressions.expression import Expression
from source.loss_functions.loss_function import LossFunction
from source.expressions.softmax_cross_entropy_expression import SoftmaxCrossEntropyExpression


class CategoricalCrossEntropy(LossFunction):
    def compute(self, predictions: Expression, targets: np.ndarray) -> Expression:
        return SoftmaxCrossEntropyExpression(predictions, targets)
