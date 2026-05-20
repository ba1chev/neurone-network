from typing import Union

from source.expressions.constant import Constant
from source.expressions.expression import Expression
from source.expressions.unary_expressions.abs_unary_expression import AbsUnaryExpression
from source.expressions.binary_expressions.power_binary_expression import PowerBinaryExpression
from source.expressions.unary_expressions.negation_unary_expression import NegationUnaryExpression
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression
from source.expressions.binary_expressions.division_binary_expression import DivisionBinaryExpression
from source.expressions.binary_expressions.subtraction_binary_expression import SubtractionBinaryExpression
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


def wrap(other: Union[Expression, int, float]):
    if isinstance(other, Expression):
        return other
    if isinstance(other, (int, float)):
        return Constant(float(other))
    return NotImplemented


def add_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return AdditionBinaryExpression(self, wrapped)


def radd_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return AdditionBinaryExpression(wrapped, self)


def sub_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return SubtractionBinaryExpression(self, wrapped)


def rsub_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return SubtractionBinaryExpression(wrapped, self)


def mul_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return MultiplicationBinaryExpression(self, wrapped)


def rmul_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return MultiplicationBinaryExpression(wrapped, self)


def truediv_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return DivisionBinaryExpression(self, wrapped)


def rtruediv_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return DivisionBinaryExpression(wrapped, self)


def pow_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return PowerBinaryExpression(self, wrapped)


def rpow_operator(self, other):
    wrapped = wrap(other)
    if wrapped is NotImplemented:
        return NotImplemented
    return PowerBinaryExpression(wrapped, self)


def neg_operator(self):
    return NegationUnaryExpression(self)


def abs_operator(self):
    return AbsUnaryExpression(self)


# Mounted on Expression after the fact to break a circular dependency:
# Expression is the base class of every concrete operator, so it cannot
# import them at class-definition time.
Expression.__add__ = add_operator
Expression.__radd__ = radd_operator
Expression.__sub__ = sub_operator
Expression.__rsub__ = rsub_operator
Expression.__mul__ = mul_operator
Expression.__rmul__ = rmul_operator
Expression.__truediv__ = truediv_operator
Expression.__rtruediv__ = rtruediv_operator
Expression.__pow__ = pow_operator
Expression.__rpow__ = rpow_operator
Expression.__neg__ = neg_operator
Expression.__abs__ = abs_operator
