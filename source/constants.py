from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.tanh_unary_expression import TanhUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


ACTIVATIONS = {
    "sigmoid": SigmoidUnaryExpression,
    "relu": ReluUnaryExpression,
    "tanh": TanhUnaryExpression,
}

INITIAL_GRADIENT: float = 0.0
INITIAL_BIAS: float = 0.0

ERROR_DIVIDE_BY_ZERO = "Cannot divide by zero"
ERROR_LOG_NON_POSITIVE = "Cannot take log of non-positive value"
ERROR_SQRT_NEGATIVE = "Cannot take sqrt of negative value"
ERROR_SQRT_GRAD_AT_ZERO = "sqrt gradient is undefined at zero"
ERROR_RECIPROCAL_OF_ZERO = "Cannot take reciprocal of zero"
ERROR_POW_NEGATIVE_BASE = "Cannot raise negative base to non-integer exponent"
ERROR_POW_ZERO_NEGATIVE_EXP = "Cannot raise zero to negative exponent"
