import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.negation_unary_expression import NegationUnaryExpression


class TestNegationUnaryExpression(unittest.TestCase):
    def test_forward_positive(self):
        self.assertEqual(NegationUnaryExpression(Variable(3.0, "x")).forward(), -3.0)

    def test_forward_negative(self):
        self.assertEqual(NegationUnaryExpression(Variable(-4.0, "x")).forward(), 4.0)

    def test_forward_zero(self):
        self.assertEqual(NegationUnaryExpression(Variable(0.0, "x")).forward(), 0.0)

    def test_backward(self):
        x = Variable(3.0, "x")
        expr = NegationUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, -1.0)

    def test_backward_with_upstream(self):
        x = Variable(3.0, "x")
        expr = NegationUnaryExpression(x)
        expr.forward()
        expr.backward(2.5)
        self.assertEqual(x.gradient, -2.5)


if __name__ == "__main__":
    unittest.main()
