import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.max_binary_expression import MaxBinaryExpression
from source.expressions.binary_expressions.min_binary_expression import MinBinaryExpression
from source.expressions.binary_expressions.power_binary_expression import PowerBinaryExpression
from source.expressions.binary_expressions.matmul_binary_expression import MatmulBinaryExpression
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression
from source.expressions.binary_expressions.division_binary_expression import DivisionBinaryExpression
from source.expressions.binary_expressions.subtraction_binary_expression import SubtractionBinaryExpression
from source.expressions.binary_expressions.multiplication_binary_expression import MultiplicationBinaryExpression


class TestAdditionBinaryExpression(unittest.TestCase):
    def test_forward_elementwise(self):
        a = Variable(np.array([1.0, 2.0]), "a")
        b = Variable(np.array([3.0, 4.0]), "b")
        np.testing.assert_array_equal(AdditionBinaryExpression(a, b).forward(), np.array([4.0, 6.0]))

    def test_backward_distributes_gradient(self):
        a = Variable(np.array([1.0, 2.0]), "a")
        b = Variable(np.array([3.0, 4.0]), "b")
        expression = AdditionBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([5.0, 7.0]))
        np.testing.assert_array_equal(a.gradient, np.array([5.0, 7.0]))
        np.testing.assert_array_equal(b.gradient, np.array([5.0, 7.0]))

    def test_backward_unbroadcasts_bias_pattern(self):
        # matmul-output (batch=4, features=3) plus bias (3,)
        weighted = Variable(np.zeros((4, 3)), "weighted")
        bias = Variable(np.zeros(3), "bias")
        expression = AdditionBinaryExpression(weighted, bias)
        expression.forward()
        expression.backward(np.ones((4, 3)))
        np.testing.assert_array_equal(weighted.gradient, np.ones((4, 3)))
        np.testing.assert_array_equal(bias.gradient, np.full(3, 4.0))


class TestSubtractionBinaryExpression(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array([5.0, 3.0]), "a")
        b = Variable(np.array([1.0, 2.0]), "b")
        np.testing.assert_array_equal(SubtractionBinaryExpression(a, b).forward(), np.array([4.0, 1.0]))

    def test_backward_signs(self):
        a = Variable(np.array([5.0, 3.0]), "a")
        b = Variable(np.array([1.0, 2.0]), "b")
        expression = SubtractionBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(a.gradient, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(b.gradient, np.array([-1.0, -1.0]))

    def test_backward_unbroadcasts_negation_too(self):
        a = Variable(np.zeros((2, 3)), "a")
        b = Variable(np.zeros(3), "b")
        expression = SubtractionBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.ones((2, 3)))
        np.testing.assert_array_equal(b.gradient, np.full(3, -2.0))


class TestMultiplicationBinaryExpression(unittest.TestCase):
    def test_forward_elementwise(self):
        a = Variable(np.array([2.0, 3.0]), "a")
        b = Variable(np.array([4.0, 5.0]), "b")
        np.testing.assert_array_equal(MultiplicationBinaryExpression(a, b).forward(), np.array([8.0, 15.0]))

    def test_backward_chain_rule(self):
        a = Variable(np.array([2.0, 3.0]), "a")
        b = Variable(np.array([4.0, 5.0]), "b")
        expression = MultiplicationBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(a.gradient, np.array([4.0, 5.0]))
        np.testing.assert_array_equal(b.gradient, np.array([2.0, 3.0]))

    def test_backward_with_broadcast_scalar(self):
        a = Variable(np.array([2.0, 3.0]), "a")
        scalar = Variable(np.array(5.0), "scalar")
        expression = MultiplicationBinaryExpression(a, scalar)
        expression.forward()
        expression.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(a.gradient, np.array([5.0, 5.0]))
        np.testing.assert_array_equal(scalar.gradient, np.array(5.0))


class TestDivisionBinaryExpression(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array([8.0, 9.0]), "a")
        b = Variable(np.array([2.0, 3.0]), "b")
        np.testing.assert_array_equal(DivisionBinaryExpression(a, b).forward(), np.array([4.0, 3.0]))

    def test_backward(self):
        a = Variable(np.array([8.0]), "a")
        b = Variable(np.array([2.0]), "b")
        expression = DivisionBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([1.0]))
        np.testing.assert_allclose(a.gradient, np.array([0.5]))
        np.testing.assert_allclose(b.gradient, np.array([-2.0]))

    def test_zero_denominator_rejected(self):
        a = Variable(np.array([1.0]), "a")
        b = Variable(np.array([0.0]), "b")
        with self.assertRaises(RuntimeError):
            DivisionBinaryExpression(a, b).forward()


class TestPowerBinaryExpression(unittest.TestCase):
    def test_forward(self):
        base = Variable(np.array([2.0, 3.0]), "base")
        exponent = Constant(np.array(2.0))
        np.testing.assert_array_equal(PowerBinaryExpression(base, exponent).forward(), np.array([4.0, 9.0]))

    def test_backward_through_base(self):
        base = Variable(np.array([2.0, 3.0]), "base")
        exponent = Constant(np.array(2.0))
        expression = PowerBinaryExpression(base, exponent)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_allclose(base.gradient, np.array([4.0, 6.0]))

    def test_negative_base_non_integer_exponent_rejected(self):
        base = Variable(np.array([-2.0]), "base")
        exponent = Constant(np.array(0.5))
        with self.assertRaises(RuntimeError):
            PowerBinaryExpression(base, exponent).forward()


class TestMaxBinaryExpression(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array([1.0, 5.0]), "a")
        b = Variable(np.array([3.0, 2.0]), "b")
        np.testing.assert_array_equal(MaxBinaryExpression(a, b).forward(), np.array([3.0, 5.0]))

    def test_backward_routes_to_larger(self):
        a = Variable(np.array([1.0, 5.0]), "a")
        b = Variable(np.array([3.0, 2.0]), "b")
        expression = MaxBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_array_equal(a.gradient, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(b.gradient, np.array([1.0, 0.0]))

    def test_backward_splits_ties_evenly(self):
        a = Variable(np.array([2.0]), "a")
        b = Variable(np.array([2.0]), "b")
        expression = MaxBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([1.0]))
        np.testing.assert_allclose(a.gradient, np.array([0.5]))
        np.testing.assert_allclose(b.gradient, np.array([0.5]))


class TestMinBinaryExpression(unittest.TestCase):
    def test_forward(self):
        a = Variable(np.array([1.0, 5.0]), "a")
        b = Variable(np.array([3.0, 2.0]), "b")
        np.testing.assert_array_equal(MinBinaryExpression(a, b).forward(), np.array([1.0, 2.0]))

    def test_backward_routes_to_smaller(self):
        a = Variable(np.array([1.0, 5.0]), "a")
        b = Variable(np.array([3.0, 2.0]), "b")
        expression = MinBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_array_equal(a.gradient, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(b.gradient, np.array([0.0, 1.0]))


class TestMatmulBinaryExpression(unittest.TestCase):
    def test_forward_2d_at_2d(self):
        a = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "a")
        b = Variable(np.array([[5.0, 6.0], [7.0, 8.0]]), "b")
        result = MatmulBinaryExpression(a, b).forward()
        np.testing.assert_array_equal(result, np.array([[19.0, 22.0], [43.0, 50.0]]))

    def test_forward_batch_at_weights(self):
        # input (batch=4, features=3) @ weights (3, 2) -> (4, 2)
        a = Variable(np.ones((4, 3)), "a")
        w = Variable(np.ones((3, 2)), "w")
        result = MatmulBinaryExpression(a, w).forward()
        np.testing.assert_array_equal(result, np.full((4, 2), 3.0))

    def test_backward_uses_transpose(self):
        a = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "a")
        b = Variable(np.array([[1.0], [1.0]]), "b")
        expression = MatmulBinaryExpression(a, b)
        expression.forward()
        expression.backward(np.array([[1.0], [1.0]]))
        # dA = grad @ B.T -> shape (2,2) of ones
        np.testing.assert_array_equal(a.gradient, np.array([[1.0, 1.0], [1.0, 1.0]]))
        # dB = A.T @ grad -> (2,1)
        np.testing.assert_array_equal(b.gradient, np.array([[4.0], [6.0]]))

    def test_dimension_mismatch_rejected(self):
        a = Variable(np.ones((4, 3)), "a")
        b = Variable(np.ones((5, 2)), "b")
        with self.assertRaises(RuntimeError):
            MatmulBinaryExpression(a, b).forward()


class TestBinaryCollectVariables(unittest.TestCase):
    def test_collects_from_both_sides(self):
        a = Variable(np.array([1.0]), "a")
        b = Variable(np.array([2.0]), "b")
        expression = AdditionBinaryExpression(a, b)
        variables = expression.get_variables()
        self.assertIn(a, variables)
        self.assertIn(b, variables)

    def test_dedups_shared_variable(self):
        x = Variable(np.array([1.0]), "x")
        expression = MultiplicationBinaryExpression(x, x)
        variables = expression.get_variables()
        self.assertEqual(len(variables), 1)


if __name__ == "__main__":
    unittest.main()
