import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.abs_unary_expression import AbsUnaryExpression


class TestAbsUnaryExpression(unittest.TestCase):
    def test_forward_positive(self):
        self.assertEqual(AbsUnaryExpression(Variable(3.0, "x")).forward(), 3.0)

    def test_forward_negative(self):
        self.assertEqual(AbsUnaryExpression(Variable(-3.0, "x")).forward(), 3.0)

    def test_forward_zero(self):
        self.assertEqual(AbsUnaryExpression(Variable(0.0, "x")).forward(), 0.0)

    def test_backward_positive(self):
        x = Variable(3.0, "x")
        expr = AbsUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, 1.0)

    def test_backward_negative(self):
        x = Variable(-3.0, "x")
        expr = AbsUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, -1.0)

    def test_backward_zero(self):
        x = Variable(0.0, "x")
        expr = AbsUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, 0.0)

    def test_backward_with_upstream(self):
        x = Variable(-5.0, "x")
        expr = AbsUnaryExpression(x)
        expr.forward()
        expr.backward(2.0)
        self.assertEqual(x.gradient, -2.0)


if __name__ == "__main__":
    unittest.main()
