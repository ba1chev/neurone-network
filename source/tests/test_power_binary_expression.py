import math
import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.power_binary_expression import PowerBinaryExpression


class TestPowerBinaryExpression(unittest.TestCase):
    def test_forward_integer_exponent(self):
        expr = PowerBinaryExpression(Constant(2.0), Constant(3.0))
        self.assertEqual(expr.forward(), 8.0)

    def test_forward_zero_exponent(self):
        expr = PowerBinaryExpression(Constant(7.0), Constant(0.0))
        self.assertEqual(expr.forward(), 1.0)

    def test_forward_fractional_exponent(self):
        expr = PowerBinaryExpression(Constant(9.0), Constant(0.5))
        self.assertAlmostEqual(expr.forward(), 3.0)

    def test_forward_negative_exponent(self):
        expr = PowerBinaryExpression(Constant(2.0), Constant(-2.0))
        self.assertAlmostEqual(expr.forward(), 0.25)

    def test_forward_negative_base_integer_exponent(self):
        expr = PowerBinaryExpression(Constant(-2.0), Constant(3.0))
        self.assertEqual(expr.forward(), -8.0)

    def test_forward_negative_base_non_integer_raises(self):
        expr = PowerBinaryExpression(Constant(-2.0), Constant(0.5))
        with self.assertRaises(RuntimeError):
            expr.forward()

    def test_forward_zero_base_negative_exponent_raises(self):
        expr = PowerBinaryExpression(Constant(0.0), Constant(-1.0))
        with self.assertRaises(RuntimeError):
            expr.forward()

    def test_backward_base_only(self):
        x = Variable(3.0, "x")
        expr = PowerBinaryExpression(x, Constant(2.0))
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 6.0)

    def test_backward_exponent_only(self):
        y = Variable(3.0, "y")
        expr = PowerBinaryExpression(Constant(2.0), y)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(y.gradient, 8.0 * math.log(2.0))

    def test_backward_both_variables(self):
        x = Variable(2.0, "x")
        y = Variable(3.0, "y")
        expr = PowerBinaryExpression(x, y)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 12.0)
        self.assertAlmostEqual(y.gradient, 8.0 * math.log(2.0))

    def test_backward_with_upstream(self):
        x = Variable(2.0, "x")
        expr = PowerBinaryExpression(x, Constant(3.0))
        expr.forward()
        expr.backward(0.5)
        self.assertAlmostEqual(x.gradient, 6.0)

    def test_backward_negative_base_exponent_grad_is_zero(self):
        x = Variable(-2.0, "x")
        y = Variable(3.0, "y")
        expr = PowerBinaryExpression(x, y)
        expr.forward()
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 12.0)
        self.assertEqual(y.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
