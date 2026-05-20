import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.neural_network import NeuralNetwork
from source.loss_functions.mean_squared_error import MeanSquaredError
from source.optimizers.stochastic_gradient_descent import StochasticGradientDescent


class TestZeroAllGradients(unittest.TestCase):
    def test_zero_all_gradients_via_loss_resets_network_params(self):
        network = NeuralNetwork(layer_sizes=[2, 2, 1], activations=[None, None])
        predictions = network.forward(Constant(np.array([[1.0, 2.0]])))
        loss = MeanSquaredError().compute(predictions, np.array([[0.0]]))
        loss.forward()
        loss.backward(np.array(1.0))
        loss.zero_all_gradients()
        for parameter in network.parameters():
            np.testing.assert_array_equal(parameter.gradient, np.zeros_like(parameter.value))

    def test_collects_each_variable_only_once(self):
        shared = Variable(np.array([2.0]), "x")
        expression = shared * shared
        variables = expression.get_variables()
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], shared)


if __name__ == "__main__":
    unittest.main()
