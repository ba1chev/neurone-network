import math
import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.log_unary_expression import LogUnaryExpression


class TestLogUnaryExpression(unittest.TestCase):
    def test_forward_one(self):
        self.assertAlmostEqual(LogUnaryExpression(Variable(1.0, "x")).forward(), 0.0)

    def test_forward_e(self):
        self.assertAlmostEqual(LogUnaryExpression(Variable(math.e, "x")).forward(), 1.0)

    def test_forward_zero_raises(self):
        with self.assertRaises(RuntimeError):
            LogUnaryExpression(Variable(0.0, "x")).forward()

    def test_forward_negative_raises(self):
        with self.assertRaises(RuntimeError):
            LogUnaryExpression(Variable(-1.0, "x")).forward()

    def test_backward(self):
        x = Variable(2.0, "x")
        expr = LogUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 0.5)

    def test_backward_with_upstream(self):
        x = Variable(4.0, "x")
        expr = LogUnaryExpression(x)
        expr.forward()
        expr.backward(2.0)
        self.assertAlmostEqual(x.gradient, 0.5)


if __name__ == "__main__":
    unittest.main()
