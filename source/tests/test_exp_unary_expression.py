import math
import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.exp_unary_expression import ExpUnaryExpression


class TestExpUnaryExpression(unittest.TestCase):
    def test_forward_zero(self):
        self.assertAlmostEqual(ExpUnaryExpression(Variable(0.0, "x")).forward(), 1.0)

    def test_forward_one(self):
        self.assertAlmostEqual(ExpUnaryExpression(Variable(1.0, "x")).forward(), math.e)

    def test_forward_negative(self):
        self.assertAlmostEqual(ExpUnaryExpression(Variable(-1.0, "x")).forward(), 1 / math.e)

    def test_backward(self):
        x = Variable(2.0, "x")
        expr = ExpUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, math.exp(2.0))

    def test_backward_with_upstream(self):
        x = Variable(0.0, "x")
        expr = ExpUnaryExpression(x)
        expr.forward()
        expr.backward(3.0)
        self.assertAlmostEqual(x.gradient, 3.0)


if __name__ == "__main__":
    unittest.main()
