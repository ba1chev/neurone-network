import math
import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


class TestSigmoidUnaryExpression(unittest.TestCase):
    def test_forward_zero_is_half(self):
        self.assertAlmostEqual(SigmoidUnaryExpression(Variable(0.0, "x")).forward(), 0.5)

    def test_forward_large_positive_approaches_one(self):
        self.assertAlmostEqual(SigmoidUnaryExpression(Variable(20.0, "x")).forward(), 1.0, places=5)

    def test_forward_large_negative_approaches_zero(self):
        self.assertAlmostEqual(SigmoidUnaryExpression(Variable(-20.0, "x")).forward(), 0.0, places=5)

    def test_forward_known_value(self):
        expected = 1 / (1 + math.exp(-1))
        self.assertAlmostEqual(SigmoidUnaryExpression(Variable(1.0, "x")).forward(), expected)

    def test_backward_at_zero(self):
        x = Variable(0.0, "x")
        expr = SigmoidUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 0.25)

    def test_backward_with_upstream(self):
        x = Variable(0.0, "x")
        expr = SigmoidUnaryExpression(x)
        expr.forward()
        expr.backward(4.0)
        self.assertAlmostEqual(x.gradient, 1.0)


if __name__ == "__main__":
    unittest.main()
