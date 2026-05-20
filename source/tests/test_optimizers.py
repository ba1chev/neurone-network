import unittest
import numpy as np

from source.expressions.variable import Variable
from source.optimizers.stochastic_gradient_descent import StochasticGradientDescent


class TestOptimizerValidation(unittest.TestCase):
    def test_no_parameters_rejected(self):
        with self.assertRaises(ValueError):
            StochasticGradientDescent([], learning_rate=0.1)

    def test_non_positive_learning_rate_rejected(self):
        parameter = Variable(np.array([1.0]), "p")
        with self.assertRaises(ValueError):
            StochasticGradientDescent([parameter], learning_rate=0.0)
        with self.assertRaises(ValueError):
            StochasticGradientDescent([parameter], learning_rate=-0.1)


class TestStochasticGradientDescentStep(unittest.TestCase):
    def test_subtracts_lr_times_gradient(self):
        parameter = Variable(np.array([1.0, 2.0, 3.0]), "p")
        parameter.gradient = np.array([10.0, 10.0, 10.0])
        optimizer = StochasticGradientDescent([parameter], learning_rate=0.1)
        optimizer.step()
        np.testing.assert_allclose(parameter.value, np.array([0.0, 1.0, 2.0]))

    def test_zero_gradient_leaves_value_unchanged(self):
        parameter = Variable(np.array([5.0, 5.0]), "p")
        optimizer = StochasticGradientDescent([parameter], learning_rate=0.1)
        optimizer.step()
        np.testing.assert_array_equal(parameter.value, np.array([5.0, 5.0]))

    def test_step_preserves_shape(self):
        parameter = Variable(np.zeros((3, 4)), "W")
        parameter.gradient = np.ones((3, 4))
        optimizer = StochasticGradientDescent([parameter], learning_rate=0.5)
        optimizer.step()
        self.assertEqual(parameter.value.shape, (3, 4))


class TestZeroGradients(unittest.TestCase):
    def test_zero_gradients_resets_all_to_zeros(self):
        first = Variable(np.array([1.0]), "first")
        second = Variable(np.zeros((2, 2)), "second")
        first.gradient = np.array([5.0])
        second.gradient = np.full((2, 2), 7.0)
        optimizer = StochasticGradientDescent([first, second], learning_rate=0.1)
        optimizer.zero_gradients()
        np.testing.assert_array_equal(first.gradient, np.zeros(1))
        np.testing.assert_array_equal(second.gradient, np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
