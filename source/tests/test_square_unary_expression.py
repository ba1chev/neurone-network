import unittest

from source.expressions.variable import Variable
from source.expressions.unary_expressions.square_unary_expression import SquareUnaryExpression


class TestSquareUnaryExpression(unittest.TestCase):
    def test_forward(self):
        self.assertEqual(SquareUnaryExpression(Variable(3.0, "x")).forward(), 9.0)

    def test_forward_negative_input(self):
        self.assertEqual(SquareUnaryExpression(Variable(-4.0, "x")).forward(), 16.0)

    def test_forward_zero(self):
        self.assertEqual(SquareUnaryExpression(Variable(0.0, "x")).forward(), 0.0)

    def test_backward_chain_rule(self):
        x = Variable(3.0, "x")
        expr = SquareUnaryExpression(x)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, 6.0)

    def test_backward_with_upstream(self):
        x = Variable(4.0, "x")
        expr = SquareUnaryExpression(x)
        expr.forward()
        expr.backward(0.5)
        self.assertEqual(x.gradient, 4.0)

    def test_backward_requires_forward(self):
        x = Variable(3.0, "x")
        expr = SquareUnaryExpression(x)
        with self.assertRaises(AttributeError):
            expr.backward(1.0)


if __name__ == "__main__":
    unittest.main()
