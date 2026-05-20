import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.subtraction_binary_expression import SubtractionBinaryExpression


class TestSubtractionBinaryExpression(unittest.TestCase):
    def test_forward_two_constants(self):
        expr = SubtractionBinaryExpression(Constant(5.0), Constant(3.0))
        self.assertEqual(expr.forward(), 2.0)

    def test_forward_negative_result(self):
        expr = SubtractionBinaryExpression(Constant(1.0), Constant(4.0))
        self.assertEqual(expr.forward(), -3.0)

    def test_forward_two_variables(self):
        a = Variable(7.0, "a")
        b = Variable(2.0, "b")
        expr = SubtractionBinaryExpression(a, b)
        self.assertEqual(expr.forward(), 5.0)

    def test_backward_propagates_positive_to_left_negative_to_right(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = SubtractionBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 1.0)
        self.assertEqual(b.gradient, -1.0)

    def test_backward_with_arbitrary_gradient(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = SubtractionBinaryExpression(a, b)
        expr.forward()
        expr.backward(2.5)
        self.assertEqual(a.gradient, 2.5)
        self.assertEqual(b.gradient, -2.5)

    def test_backward_with_negative_upstream_gradient(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = SubtractionBinaryExpression(a, b)
        expr.forward()
        expr.backward(-3.0)
        self.assertEqual(a.gradient, -3.0)
        self.assertEqual(b.gradient, 3.0)

    def test_backward_same_variable_on_both_sides_cancels(self):
        x = Variable(4.0, "x")
        expr = SubtractionBinaryExpression(x, x)
        self.assertEqual(expr.forward(), 0.0)
        expr.backward(1.0)
        self.assertEqual(x.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
