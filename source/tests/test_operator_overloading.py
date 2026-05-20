import unittest
import numpy as np

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.unary_expressions.abs_unary_expression import AbsUnaryExpression
from source.expressions.binary_expressions.matmul_binary_expression import MatmulBinaryExpression
from source.expressions.unary_expressions.negation_unary_expression import NegationUnaryExpression
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


class TestArithmeticOperators(unittest.TestCase):
    def test_add_two_variables(self):
        a = Variable(np.array([1.0, 2.0]), "a")
        b = Variable(np.array([3.0, 4.0]), "b")
        expression = a + b
        self.assertIsInstance(expression, AdditionBinaryExpression)
        np.testing.assert_array_equal(expression.forward(), np.array([4.0, 6.0]))

    def test_add_variable_and_int_wraps_constant(self):
        a = Variable(np.array([1.0]), "a")
        expression = a + 5
        np.testing.assert_array_equal(expression.forward(), np.array([6.0]))

    def test_add_variable_and_ndarray_wraps_constant(self):
        a = Variable(np.array([1.0, 2.0]), "a")
        expression = a + np.array([10.0, 20.0])
        np.testing.assert_array_equal(expression.forward(), np.array([11.0, 22.0]))

    def test_radd_with_python_scalar(self):
        a = Variable(np.array([1.0]), "a")
        expression = 5 + a
        np.testing.assert_array_equal(expression.forward(), np.array([6.0]))

    def test_subtraction(self):
        a = Variable(np.array([5.0]), "a")
        np.testing.assert_array_equal((a - 2).forward(), np.array([3.0]))
        np.testing.assert_array_equal((10 - a).forward(), np.array([5.0]))

    def test_multiplication(self):
        a = Variable(np.array([3.0]), "a")
        expression = a * 4
        self.assertIsInstance(expression, MultiplicationBinaryExpression)
        np.testing.assert_array_equal(expression.forward(), np.array([12.0]))

    def test_division(self):
        a = Variable(np.array([10.0]), "a")
        np.testing.assert_array_equal((a / 2).forward(), np.array([5.0]))
        np.testing.assert_array_equal((20 / a).forward(), np.array([2.0]))

    def test_power(self):
        a = Variable(np.array([3.0]), "a")
        np.testing.assert_array_equal((a ** 2).forward(), np.array([9.0]))


class TestMatmulOperator(unittest.TestCase):
    def test_matmul_two_variables(self):
        a = Variable(np.array([[1.0, 2.0]]), "a")
        b = Variable(np.array([[3.0], [4.0]]), "b")
        expression = a @ b
        self.assertIsInstance(expression, MatmulBinaryExpression)
        np.testing.assert_array_equal(expression.forward(), np.array([[11.0]]))

    def test_matmul_with_ndarray_on_right(self):
        a = Variable(np.array([[1.0, 2.0]]), "a")
        result = (a @ np.array([[3.0], [4.0]])).forward()
        np.testing.assert_array_equal(result, np.array([[11.0]]))


class TestUnaryOperators(unittest.TestCase):
    def test_negation(self):
        a = Variable(np.array([1.0, -2.0]), "a")
        expression = -a
        self.assertIsInstance(expression, NegationUnaryExpression)
        np.testing.assert_array_equal(expression.forward(), np.array([-1.0, 2.0]))

    def test_abs(self):
        a = Variable(np.array([-3.0, 4.0]), "a")
        expression = abs(a)
        self.assertIsInstance(expression, AbsUnaryExpression)
        np.testing.assert_array_equal(expression.forward(), np.array([3.0, 4.0]))


class TestBackwardThroughOperators(unittest.TestCase):
    def test_compound_expression_backprops_correctly(self):
        a = Variable(np.array([2.0]), "a")
        b = Variable(np.array([3.0]), "b")
        expression = (a + b) * (a - b)
        expression.forward()
        expression.backward(np.array([1.0]))
        # f = (a+b)(a-b) = a^2 - b^2 ; df/da = 2a, df/db = -2b
        np.testing.assert_allclose(a.gradient, np.array([4.0]))
        np.testing.assert_allclose(b.gradient, np.array([-6.0]))


class TestWrapHandlesNumpyScalars(unittest.TestCase):
    def test_numpy_float_wraps_to_constant(self):
        a = Variable(np.array([1.0]), "a")
        expression = a + np.float64(5.0)
        np.testing.assert_array_equal(expression.forward(), np.array([6.0]))


if __name__ == "__main__":
    unittest.main()
