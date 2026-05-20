import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


class TestMultiplicationBinaryExpression(unittest.TestCase):
    def test_forward_two_constants(self):
        expr = MultiplicationBinaryExpression(Constant(3.0), Constant(4.0))
        self.assertEqual(expr.forward(), 12.0)

    def test_forward_with_zero(self):
        expr = MultiplicationBinaryExpression(Constant(0.0), Constant(99.0))
        self.assertEqual(expr.forward(), 0.0)

    def test_forward_with_negatives(self):
        expr = MultiplicationBinaryExpression(Constant(-2.0), Constant(3.0))
        self.assertEqual(expr.forward(), -6.0)
        expr2 = MultiplicationBinaryExpression(Constant(-2.0), Constant(-3.0))
        self.assertEqual(expr2.forward(), 6.0)

    def test_backward_chain_rule(self):
        a = Variable(3.0, "a")
        b = Variable(4.0, "b")
        expr = MultiplicationBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        # d(ab)/da = b = 4, d(ab)/db = a = 3
        self.assertEqual(a.gradient, 4.0)
        self.assertEqual(b.gradient, 3.0)

    def test_backward_with_upstream_gradient(self):
        a = Variable(2.0, "a")
        b = Variable(5.0, "b")
        expr = MultiplicationBinaryExpression(a, b)
        expr.forward()
        expr.backward(2.0)
        self.assertEqual(a.gradient, 10.0)  # 5 * 2
        self.assertEqual(b.gradient, 4.0)   # 2 * 2

    def test_backward_requires_forward_first(self):
        a = Variable(2.0, "a")
        b = Variable(5.0, "b")
        expr = MultiplicationBinaryExpression(a, b)
        # forward caches operand values; without it backward should fail
        with self.assertRaises(AttributeError):
            expr.backward(1.0)

    def test_backward_with_zero_operand(self):
        a = Variable(0.0, "a")
        b = Variable(7.0, "b")
        expr = MultiplicationBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        # d(ab)/da = b = 7, d(ab)/db = a = 0
        self.assertEqual(a.gradient, 7.0)
        self.assertEqual(b.gradient, 0.0)

    def test_backward_same_variable_on_both_sides(self):
        x = Variable(3.0, "x")
        expr = MultiplicationBinaryExpression(x, x)
        self.assertEqual(expr.forward(), 9.0)
        expr.backward(1.0)
        # d(x^2)/dx = 2x = 6 → accumulated from both sides
        self.assertEqual(x.gradient, 6.0)


if __name__ == "__main__":
    unittest.main()
