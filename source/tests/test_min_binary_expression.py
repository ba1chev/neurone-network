import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.min_binary_expression import MinBinaryExpression


class TestMinBinaryExpression(unittest.TestCase):
    def test_forward_left_smaller(self):
        expr = MinBinaryExpression(Constant(2.0), Constant(7.0))
        self.assertEqual(expr.forward(), 2.0)

    def test_forward_right_smaller(self):
        expr = MinBinaryExpression(Constant(5.0), Constant(3.0))
        self.assertEqual(expr.forward(), 3.0)

    def test_forward_equal(self):
        expr = MinBinaryExpression(Constant(4.0), Constant(4.0))
        self.assertEqual(expr.forward(), 4.0)

    def test_forward_with_negatives(self):
        expr = MinBinaryExpression(Constant(-5.0), Constant(-2.0))
        self.assertEqual(expr.forward(), -5.0)

    def test_backward_left_wins(self):
        a = Variable(2.0, "a")
        b = Variable(7.0, "b")
        expr = MinBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 1.0)
        self.assertEqual(b.gradient, 0.0)

    def test_backward_right_wins(self):
        a = Variable(5.0, "a")
        b = Variable(3.0, "b")
        expr = MinBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 0.0)
        self.assertEqual(b.gradient, 1.0)

    def test_backward_tie_splits_evenly(self):
        a = Variable(4.0, "a")
        b = Variable(4.0, "b")
        expr = MinBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 0.5)
        self.assertEqual(b.gradient, 0.5)

    def test_backward_with_upstream(self):
        a = Variable(2.0, "a")
        b = Variable(7.0, "b")
        expr = MinBinaryExpression(a, b)
        expr.forward()
        expr.backward(4.0)
        self.assertEqual(a.gradient, 4.0)
        self.assertEqual(b.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
