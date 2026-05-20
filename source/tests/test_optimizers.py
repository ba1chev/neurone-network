import unittest

import source.expressions.operators  # egisters operators on Expression
from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.optimizers.stochastic_gradient_descent import StochasticGradientDescent


class TestStochasticGradientDescentConstruction(unittest.TestCase):
    def test_empty_parameters_rejected(self):
        with self.assertRaises(ValueError):
            StochasticGradientDescent(parameters=[], learning_rate=0.1)

    def test_zero_learning_rate_rejected(self):
        with self.assertRaises(ValueError):
            StochasticGradientDescent(parameters=[Variable(1.0, "x")], learning_rate=0.0)

    def test_negative_learning_rate_rejected(self):
        with self.assertRaises(ValueError):
            StochasticGradientDescent(parameters=[Variable(1.0, "x")], learning_rate=-0.1)


class TestStochasticGradientDescentStep(unittest.TestCase):
    def test_step_updates_value_using_gradient(self):
        x = Variable(5.0, "x")
        x.gradient = 2.0
        StochasticGradientDescent([x], learning_rate=0.1).step()
        self.assertAlmostEqual(x.value, 4.8)

    def test_step_with_zero_gradient_leaves_value_unchanged(self):
        x = Variable(3.0, "x")
        StochasticGradientDescent([x], learning_rate=0.5).step()
        self.assertEqual(x.value, 3.0)

    def test_step_updates_all_parameters(self):
        a = Variable(10.0, "a")
        b = Variable(-4.0, "b")
        a.gradient = 1.0
        b.gradient = -2.0
        StochasticGradientDescent([a, b], learning_rate=0.5).step()
        self.assertAlmostEqual(a.value, 9.5)
        self.assertAlmostEqual(b.value, -3.0)

    def test_step_does_not_reset_gradient(self):
        x = Variable(1.0, "x")
        x.gradient = 0.7
        optimizer = StochasticGradientDescent([x], learning_rate=0.1)
        optimizer.step()
        self.assertEqual(x.gradient, 0.7)


class TestStochasticGradientDescentZeroGradients(unittest.TestCase):
    def test_zero_gradients_clears_all_parameters(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        a.gradient = 5.0
        b.gradient = -3.0
        StochasticGradientDescent([a, b], learning_rate=0.1).zero_gradients()
        self.assertEqual(a.gradient, 0.0)
        self.assertEqual(b.gradient, 0.0)


class TestStochasticGradientDescentEndToEnd(unittest.TestCase):
    def test_one_step_reduces_quadratic_loss(self):
        x = Variable(0.0, "x")
        loss = (x - Constant(3.0)) ** Constant(2.0)
        optimizer = StochasticGradientDescent([x], learning_rate=0.1)

        loss_before = loss.forward()
        loss.backward(1.0)
        optimizer.step()
        optimizer.zero_gradients()
        loss_after = loss.forward()

        self.assertLess(loss_after, loss_before)

    def test_repeated_steps_converge_to_optimum(self):
        x = Variable(0.0, "x")
        loss = (x - Constant(3.0)) ** Constant(2.0)
        optimizer = StochasticGradientDescent([x], learning_rate=0.1)

        for _ in range(200):
            loss.forward()
            loss.backward(1.0)
            optimizer.step()
            optimizer.zero_gradients()

        self.assertAlmostEqual(x.value, 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
