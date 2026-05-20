from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.tanh_unary_expression import TanhUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


ACTIVATIONS = {
    "sigmoid": SigmoidUnaryExpression,
    "relu": ReluUnaryExpression,
    "tanh": TanhUnaryExpression
}

INITIAL_GRADIENT: float = 0.0
INITIAL_BIAS: float = 0.0

UNIFORM_INITIALIZATION: str = "uniform"
XAVIER_INITIALIZATION: str = "xavier"
INITIALIZATIONS = {UNIFORM_INITIALIZATION, XAVIER_INITIALIZATION}

ERROR_DIVIDE_BY_ZERO = "Cannot divide by zero"
ERROR_LOG_NON_POSITIVE = "Cannot take log of non-positive value"
ERROR_SQRT_NEGATIVE = "Cannot take sqrt of negative value"
ERROR_SQRT_GRAD_AT_ZERO = "sqrt gradient is undefined at zero"
ERROR_RECIPROCAL_OF_ZERO = "Cannot take reciprocal of zero"
ERROR_POW_NEGATIVE_BASE = "Cannot raise negative base to non-integer exponent"
ERROR_POW_ZERO_NEGATIVE_EXP = "Cannot raise zero to negative exponent"

ERROR_LOSS_EMPTY = "Loss requires at least one prediction"
ERROR_LOSS_LENGTH_MISMATCH = "Predictions and targets must have the same length"
ERROR_BCE_PROBABILITY_OUT_OF_RANGE = "BinaryCrossEntropy predictions must lie in (0, 1)"
ERROR_CCE_TARGET_OUT_OF_RANGE = "CategoricalCrossEntropy target index out of range"
ERROR_CCE_NO_LOGITS = "CategoricalCrossEntropy requires at least one logit per sample"

ERROR_LEARNING_RATE_NON_POSITIVE = "Learning rate must be positive"
ERROR_OPTIMIZER_NO_PARAMETERS = "Optimizer requires at least one parameter"

ERROR_UNKNOWN_INITIALIZATION = "Unknown weight initialization scheme"
