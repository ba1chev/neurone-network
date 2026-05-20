from typing import List, Optional

from source.expressions.expression import Expression
from source.expressions.categorical_cross_entropy_expression import CategoricalCrossEntropyExpression
from source.constants import (
    ERROR_CCE_NO_LOGITS, ERROR_CCE_TARGET_OUT_OF_RANGE,
    ERROR_LOSS_EMPTY, ERROR_LOSS_LENGTH_MISMATCH
)


class CategoricalCrossEntropy:
    def compute(self, logits_per_sample: List[List[Expression]], targets: List[int]) -> Expression:
        if len(logits_per_sample) == 0:
            raise ValueError(ERROR_LOSS_EMPTY)
        if len(logits_per_sample) != len(targets):
            raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)

        num_classes: Optional[int] = None
        for sample_logits, target in zip(logits_per_sample, targets):
            if len(sample_logits) == 0:
                raise ValueError(ERROR_CCE_NO_LOGITS)
            if num_classes is None:
                num_classes = len(sample_logits)
            elif len(sample_logits) != num_classes:
                raise ValueError(ERROR_LOSS_LENGTH_MISMATCH)
            if target < 0 or target >= len(sample_logits):
                raise ValueError(ERROR_CCE_TARGET_OUT_OF_RANGE)

        return CategoricalCrossEntropyExpression(logits_per_sample, targets)
