import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.reciprocal_unary_expression import ReciprocalUnaryExpression


class TestReciprocalUnaryExpression(unittest.TestCase):
    def test_forward(self):
        self.assertAlmostEqual(ReciprocalUnaryExpression(Variable(4.0, "x")).forward(), 0.25)

    def test_forward_negative(self):
        self.assertAlmostEqual(ReciprocalUnaryExpression(Variable(-2.0, "x")).forward(), -0.5)

    def test_forward_zero_raises(self):
        with self.assertRaises(RuntimeError):
            ReciprocalUnaryExpression(Variable(0.0, "x")).forward()

    def test_backward(self):
        x = Variable(2.0, "x")
        expr = ReciprocalUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, -0.25)

    def test_backward_with_upstream(self):
        x = Variable(2.0, "x")
        expr = ReciprocalUnaryExpression(x)
        expr.forward()
        expr.backward(4.0)
        self.assertAlmostEqual(x.gradient, -1.0)


if __name__ == "__main__":
    unittest.main()
