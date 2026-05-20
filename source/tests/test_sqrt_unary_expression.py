import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.sqrt_unary_expression import SqrtUnaryExpression


class TestSqrtUnaryExpression(unittest.TestCase):
    def test_forward(self):
        self.assertAlmostEqual(SqrtUnaryExpression(Variable(9.0, "x")).forward(), 3.0)

    def test_forward_zero(self):
        self.assertAlmostEqual(SqrtUnaryExpression(Variable(0.0, "x")).forward(), 0.0)

    def test_forward_negative_raises(self):
        with self.assertRaises(RuntimeError):
            SqrtUnaryExpression(Variable(-1.0, "x")).forward()

    def test_backward(self):
        x = Variable(9.0, "x")
        expr = SqrtUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 1 / 6)

    def test_backward_with_upstream(self):
        x = Variable(4.0, "x")
        expr = SqrtUnaryExpression(x)
        expr.forward()
        expr.backward(8.0)
        self.assertAlmostEqual(x.gradient, 2.0)

    def test_backward_at_zero_raises(self):
        x = Variable(0.0, "x")
        expr = SqrtUnaryExpression(x)
        expr.forward()
        with self.assertRaises(RuntimeError):
            expr.backward(1.0)


if __name__ == "__main__":
    unittest.main()
