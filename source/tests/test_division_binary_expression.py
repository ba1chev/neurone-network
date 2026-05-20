import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.division_binary_expression import DivisionBinaryExpression


class TestDivisionBinaryExpression(unittest.TestCase):
    def test_forward_two_constants(self):
        expr = DivisionBinaryExpression(Constant(10.0), Constant(2.0))
        self.assertEqual(expr.forward(), 5.0)

    def test_forward_non_integer_result(self):
        expr = DivisionBinaryExpression(Constant(1.0), Constant(4.0))
        self.assertEqual(expr.forward(), 0.25)

    def test_forward_with_negatives(self):
        expr = DivisionBinaryExpression(Constant(-6.0), Constant(2.0))
        self.assertEqual(expr.forward(), -3.0)
        expr2 = DivisionBinaryExpression(Constant(-6.0), Constant(-2.0))
        self.assertEqual(expr2.forward(), 3.0)

    def test_forward_division_by_zero_raises(self):
        expr = DivisionBinaryExpression(Constant(1.0), Constant(0.0))
        with self.assertRaises(RuntimeError) as ctx:
            expr.forward()
        self.assertIn("zero", str(ctx.exception).lower())

    def test_backward_chain_rule(self):
        a = Variable(8.0, "a")
        b = Variable(2.0, "b")
        expr = DivisionBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(a.gradient, 0.5)
        self.assertAlmostEqual(b.gradient, -2.0)

    def test_backward_with_upstream_gradient(self):
        a = Variable(8.0, "a")
        b = Variable(2.0, "b")
        expr = DivisionBinaryExpression(a, b)
        expr.forward()
        expr.backward(3.0)
        self.assertAlmostEqual(a.gradient, 1.5)
        self.assertAlmostEqual(b.gradient, -6.0)

    def test_backward_requires_forward_first(self):
        a = Variable(8.0, "a")
        b = Variable(2.0, "b")
        expr = DivisionBinaryExpression(a, b)
        with self.assertRaises(AttributeError):
            expr.backward(1.0)

    def test_backward_with_negative_numerator(self):
        a = Variable(-4.0, "a")
        b = Variable(2.0, "b")
        expr = DivisionBinaryExpression(a, b)
        self.assertEqual(expr.forward(), -2.0)
        expr.backward(1.0)
        self.assertAlmostEqual(a.gradient, 0.5)
        self.assertAlmostEqual(b.gradient, 1.0)


if __name__ == "__main__":
    unittest.main()
