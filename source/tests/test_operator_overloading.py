import unittest

import source.expressions.operators  # registers operators on Expression

from source.expressions.variable import Variable
from source.expressions.unary_expressions.abs_unary_expression import AbsUnaryExpression
from source.expressions.binary_expressions.power_binary_expression import PowerBinaryExpression
from source.expressions.unary_expressions.negation_unary_expression import NegationUnaryExpression
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression
from source.expressions.binary_expressions.division_binary_expression import DivisionBinaryExpression
from source.expressions.binary_expressions.subtraction_binary_expression import SubtractionBinaryExpression
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


class TestOperatorOverloading(unittest.TestCase):
    def test_add_returns_addition(self):
        self.assertIsInstance(Variable(1.0, "x") + Variable(2.0, "y"), AdditionBinaryExpression)

    def test_sub_returns_subtraction(self):
        self.assertIsInstance(Variable(1.0, "x") - Variable(2.0, "y"), SubtractionBinaryExpression)

    def test_mul_returns_multiplication(self):
        self.assertIsInstance(Variable(1.0, "x") * Variable(2.0, "y"), MultiplicationBinaryExpression)

    def test_truediv_returns_division(self):
        self.assertIsInstance(Variable(1.0, "x") / Variable(2.0, "y"), DivisionBinaryExpression)

    def test_pow_returns_power(self):
        self.assertIsInstance(Variable(2.0, "x") ** Variable(3.0, "y"), PowerBinaryExpression)

    def test_neg_returns_negation(self):
        self.assertIsInstance(-Variable(1.0, "x"), NegationUnaryExpression)

    def test_abs_returns_abs(self):
        self.assertIsInstance(abs(Variable(-1.0, "x")), AbsUnaryExpression)

    def test_add_forward(self):
        self.assertEqual((Variable(2.0, "a") + Variable(3.0, "b")).forward(), 5.0)

    def test_sub_forward(self):
        self.assertEqual((Variable(5.0, "a") - Variable(3.0, "b")).forward(), 2.0)

    def test_mul_forward(self):
        self.assertEqual((Variable(4.0, "a") * Variable(3.0, "b")).forward(), 12.0)

    def test_div_forward(self):
        self.assertEqual((Variable(8.0, "a") / Variable(2.0, "b")).forward(), 4.0)

    def test_pow_forward(self):
        self.assertEqual((Variable(2.0, "a") ** Variable(3.0, "b")).forward(), 8.0)

    def test_neg_forward(self):
        self.assertEqual((-Variable(3.0, "a")).forward(), -3.0)

    def test_abs_forward(self):
        self.assertEqual(abs(Variable(-7.0, "a")).forward(), 7.0)

    def test_left_scalar_add(self):
        expr = 2 + Variable(3.0, "x")
        self.assertEqual(expr.forward(), 5.0)

    def test_right_scalar_add(self):
        expr = Variable(3.0, "x") + 2
        self.assertEqual(expr.forward(), 5.0)

    def test_left_scalar_sub(self):
        expr = 10 - Variable(3.0, "x")
        self.assertEqual(expr.forward(), 7.0)

    def test_right_scalar_sub(self):
        expr = Variable(3.0, "x") - 1
        self.assertEqual(expr.forward(), 2.0)

    def test_left_scalar_mul(self):
        x = Variable(4.0, "x")
        expr = 3 * x
        self.assertEqual(expr.forward(), 12.0)
        expr.backward(1.0)
        self.assertEqual(x.gradient, 3.0)

    def test_right_scalar_div(self):
        expr = Variable(8.0, "x") / 4
        self.assertEqual(expr.forward(), 2.0)

    def test_left_scalar_pow(self):
        expr = 2 ** Variable(3.0, "y")
        self.assertAlmostEqual(expr.forward(), 8.0)

    def test_right_scalar_pow(self):
        expr = Variable(3.0, "x") ** 2
        self.assertEqual(expr.forward(), 9.0)

    def test_compound_forward_and_backward(self):
        a, b, c = Variable(1.0, "a"), Variable(2.0, "b"), Variable(3.0, "c")
        expr = (a + b) * c
        self.assertEqual(expr.forward(), 9.0)
        expr.backward(1.0)
        self.assertEqual(a.gradient, 3.0)
        self.assertEqual(b.gradient, 3.0)
        self.assertEqual(c.gradient, 3.0)

    def test_polynomial_via_overloading(self):
        x = Variable(4.0, "x")
        expr = 2 * x ** 2 + 3 * x + 1
        self.assertAlmostEqual(expr.forward(), 45.0)
        expr.backward(1.0)
        self.assertAlmostEqual(x.gradient, 19.0)

    def test_neg_in_compound(self):
        a = Variable(2.0, "a")
        b = Variable(3.0, "b")
        expr = -(a * b)
        self.assertEqual(expr.forward(), -6.0)
        expr.backward(1.0)
        self.assertEqual(a.gradient, -3.0)
        self.assertEqual(b.gradient, -2.0)

    def test_unsupported_type_falls_through(self):
        x = Variable(1.0, "x")
        with self.assertRaises(TypeError):
            x + "not a number"

    def test_constant_stays_constant_in_overload(self):
        x = Variable(2.0, "x")
        expr = x * 5
        self.assertEqual(expr.forward(), 10.0)
        expr.backward(1.0)
        self.assertEqual(x.gradient, 5.0)


if __name__ == "__main__":
    unittest.main()
