import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.unary_expressions.abs_unary_expression import AbsUnaryExpression
from source.expressions.unary_expressions.exp_unary_expression import ExpUnaryExpression
from source.expressions.unary_expressions.log_unary_expression import LogUnaryExpression
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.sqrt_unary_expression import SqrtUnaryExpression
from source.expressions.unary_expressions.tanh_unary_expression import TanhUnaryExpression
from source.expressions.unary_expressions.square_unary_expression import SquareUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression
from source.expressions.unary_expressions.negation_unary_expression import NegationUnaryExpression
from source.expressions.unary_expressions.reciprocal_unary_expression import ReciprocalUnaryExpression


class TestSigmoidUnaryExpression(unittest.TestCase):
    def test_forward_elementwise(self):
        x = Variable(np.array([0.0, 1.0, -1.0]), "x")
        result = SigmoidUnaryExpression(x).forward()
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0])))
        np.testing.assert_allclose(result, expected)

    def test_backward_elementwise(self):
        x = Variable(np.array([0.0, 1.0, -1.0]), "x")
        expression = SigmoidUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(3))
        sigmoid = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0])))
        np.testing.assert_allclose(x.gradient, sigmoid * (1.0 - sigmoid))


class TestReluUnaryExpression(unittest.TestCase):
    def test_forward_clamps_negatives(self):
        x = Variable(np.array([-2.0, 0.0, 3.0]), "x")
        result = ReluUnaryExpression(x).forward()
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 3.0]))

    def test_backward_zero_for_non_positive(self):
        x = Variable(np.array([-2.0, 0.0, 3.0]), "x")
        expression = ReluUnaryExpression(x)
        expression.forward()
        expression.backward(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(x.gradient, np.array([0.0, 0.0, 1.0]))


class TestTanhUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([0.0, 1.0]), "x")
        result = TanhUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.tanh(np.array([0.0, 1.0])))

    def test_backward(self):
        x = Variable(np.array([0.0, 1.0]), "x")
        expression = TanhUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(2))
        tanh_values = np.tanh(np.array([0.0, 1.0]))
        np.testing.assert_allclose(x.gradient, 1.0 - tanh_values * tanh_values)


class TestExpUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([0.0, 1.0]), "x")
        result = ExpUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.exp(np.array([0.0, 1.0])))

    def test_backward_equals_forward_value(self):
        x = Variable(np.array([0.0, 1.0, 2.0]), "x")
        expression = ExpUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(3))
        np.testing.assert_allclose(x.gradient, np.exp(np.array([0.0, 1.0, 2.0])))


class TestLogUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([1.0, np.e]), "x")
        result = LogUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([0.0, 1.0]))

    def test_backward_is_reciprocal(self):
        x = Variable(np.array([2.0, 4.0]), "x")
        expression = LogUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_allclose(x.gradient, np.array([0.5, 0.25]))

    def test_non_positive_value_rejected(self):
        x = Variable(np.array([1.0, 0.0]), "x")
        with self.assertRaises(RuntimeError):
            LogUnaryExpression(x).forward()


class TestSqrtUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([4.0, 9.0]), "x")
        result = SqrtUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([2.0, 3.0]))

    def test_backward(self):
        x = Variable(np.array([4.0, 9.0]), "x")
        expression = SqrtUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_allclose(x.gradient, np.array([0.25, 1.0 / 6.0]))

    def test_negative_value_rejected(self):
        x = Variable(np.array([-1.0]), "x")
        with self.assertRaises(RuntimeError):
            SqrtUnaryExpression(x).forward()

    def test_zero_gradient_rejected(self):
        x = Variable(np.array([0.0]), "x")
        expression = SqrtUnaryExpression(x)
        expression.forward()
        with self.assertRaises(RuntimeError):
            expression.backward(np.array([1.0]))


class TestSquareUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([2.0, -3.0]), "x")
        result = SquareUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([4.0, 9.0]))

    def test_backward(self):
        x = Variable(np.array([2.0, -3.0]), "x")
        expression = SquareUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_allclose(x.gradient, np.array([4.0, -6.0]))


class TestAbsUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([-2.0, 0.0, 3.0]), "x")
        result = AbsUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([2.0, 0.0, 3.0]))

    def test_backward_uses_sign(self):
        x = Variable(np.array([-2.0, 0.0, 3.0]), "x")
        expression = AbsUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(3))
        np.testing.assert_allclose(x.gradient, np.array([-1.0, 0.0, 1.0]))


class TestNegationUnaryExpression(unittest.TestCase):
    def test_forward_negates(self):
        x = Variable(np.array([1.0, -2.0]), "x")
        result = NegationUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([-1.0, 2.0]))

    def test_backward_negates_gradient(self):
        x = Variable(np.array([1.0, -2.0]), "x")
        expression = NegationUnaryExpression(x)
        expression.forward()
        expression.backward(np.array([0.5, 0.25]))
        np.testing.assert_allclose(x.gradient, np.array([-0.5, -0.25]))


class TestReciprocalUnaryExpression(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([2.0, 4.0]), "x")
        result = ReciprocalUnaryExpression(x).forward()
        np.testing.assert_allclose(result, np.array([0.5, 0.25]))

    def test_backward(self):
        x = Variable(np.array([2.0, 4.0]), "x")
        expression = ReciprocalUnaryExpression(x)
        expression.forward()
        expression.backward(np.ones(2))
        np.testing.assert_allclose(x.gradient, np.array([-0.25, -1.0 / 16.0]))

    def test_zero_value_rejected(self):
        x = Variable(np.array([0.0]), "x")
        with self.assertRaises(RuntimeError):
            ReciprocalUnaryExpression(x).forward()


class TestUnaryExpressionsCollectVariables(unittest.TestCase):
    def test_all_unaries_propagate_collect(self):
        x = Variable(np.array([1.0]), "x")
        wrapping_classes = [
            SigmoidUnaryExpression, ReluUnaryExpression,
            TanhUnaryExpression, ExpUnaryExpression,
            LogUnaryExpression, SqrtUnaryExpression,
            SquareUnaryExpression, AbsUnaryExpression,
            NegationUnaryExpression, ReciprocalUnaryExpression,
        ]
        for wrapper in wrapping_classes:
            variables = wrapper(x).get_variables()
            self.assertIn(x, variables)

    def test_constant_input_yields_no_variables(self):
        constant = Constant(np.array([1.0, 2.0]))
        self.assertEqual(SigmoidUnaryExpression(constant).get_variables(), [])


if __name__ == "__main__":
    unittest.main()
