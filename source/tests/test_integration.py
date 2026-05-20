import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression
from source.expressions.binary_expressions.division_binary_expression import DivisionBinaryExpression
from source.expressions.binary_expressions.subtraction_binary_expression import SubtractionBinaryExpression
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


class TestExpressionIntegration(unittest.TestCase):
    def test_compound_forward_add_then_multiply(self):
        a, b, c = Variable(1.0, "a"), Variable(2.0, "b"), Variable(3.0, "c")
        expr = MultiplicationBinaryExpression(
            AdditionBinaryExpression(a, b), c
        )
        self.assertEqual(expr.forward(), 9.0)

    def test_compound_backward_add_then_multiply(self):
        a, b, c = Variable(1.0, "a"), Variable(2.0, "b"), Variable(3.0, "c")
        expr = MultiplicationBinaryExpression(
            AdditionBinaryExpression(a, b), c
        )
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 3.0)
        self.assertEqual(b.gradient, 3.0)
        self.assertEqual(c.gradient, 3.0)

    def test_compound_forward_subtraction_and_division(self):
        a, b, c = Variable(10.0, "a"), Variable(4.0, "b"), Variable(2.0, "c")
        expr = DivisionBinaryExpression(
            SubtractionBinaryExpression(a, b), c
        )
        self.assertEqual(expr.forward(), 3.0)

    def test_compound_backward_subtraction_and_division(self):
        a, b, c = Variable(10.0, "a"), Variable(4.0, "b"), Variable(2.0, "c")
        expr = DivisionBinaryExpression(
            SubtractionBinaryExpression(a, b), c
        )
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(a.gradient, 0.5)
        self.assertAlmostEqual(b.gradient, -0.5)
        self.assertAlmostEqual(c.gradient, -1.5)

    def test_shared_variable_accumulates_across_subgraph(self):
        x = Variable(2.0, "x")
        expr = AdditionBinaryExpression(
            MultiplicationBinaryExpression(x, x), x
        )
        self.assertEqual(expr.forward(), 6.0)
        expr.backward(1.0)
        self.assertEqual(x.gradient, 5.0)

    def test_zero_gradient_between_two_backward_passes(self):
        a = Variable(3.0, "a")
        b = Variable(4.0, "b")
        expr = MultiplicationBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 4.0)
        self.assertEqual(b.gradient, 3.0)

        a.zero_gradient()
        b.zero_gradient()
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 4.0)
        self.assertEqual(b.gradient, 3.0)

    def test_constants_do_not_track_gradient_in_compound(self):
        a = Variable(7.0, "a")
        expr = AdditionBinaryExpression(
            MultiplicationBinaryExpression(a, Constant(5.0)),
            Constant(2.0)
        )
        self.assertEqual(expr.forward(), 37.0)
        expr.backward(1.0)
        self.assertEqual(a.gradient, 5.0)

    def test_deep_nesting_forward_and_backward(self):
        a = Variable(3.0, "a")
        b = Variable(1.0, "b")
        numerator = MultiplicationBinaryExpression(
            AdditionBinaryExpression(a, b),
            SubtractionBinaryExpression(a, b)
        )
        expr = DivisionBinaryExpression(numerator, a)
        self.assertAlmostEqual(expr.forward(), 8.0 / 3.0)
        expr.backward(1.0)
        self.assertAlmostEqual(a.gradient, 10.0 / 9.0)
        self.assertAlmostEqual(b.gradient, -2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
