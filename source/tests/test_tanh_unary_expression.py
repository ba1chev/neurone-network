import math
import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.tanh_unary_expression import TanhUnaryExpression


class TestTanhUnaryExpression(unittest.TestCase):
    def test_forward_zero(self):
        self.assertAlmostEqual(TanhUnaryExpression(Variable(0.0, "x")).forward(), 0.0)

    def test_forward_known_value(self):
        self.assertAlmostEqual(TanhUnaryExpression(Variable(1.0, "x")).forward(), math.tanh(1.0))

    def test_forward_large_positive_approaches_one(self):
        self.assertAlmostEqual(TanhUnaryExpression(Variable(20.0, "x")).forward(), 1.0, places=5)

    def test_forward_large_negative_approaches_minus_one(self):
        self.assertAlmostEqual(TanhUnaryExpression(Variable(-20.0, "x")).forward(), -1.0, places=5)

    def test_backward_at_zero(self):
        x = Variable(0.0, "x")
        expr = TanhUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 1.0)

    def test_backward_known(self):
        x = Variable(1.0, "x")
        expr = TanhUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        expected = 1 - math.tanh(1.0) ** 2
        self.assertAlmostEqual(x.gradient, expected)


if __name__ == "__main__":
    unittest.main()
