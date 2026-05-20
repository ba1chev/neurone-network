import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.dense_layer import DenseLayer
from source.neuron_network.neural_network import NeuralNetwork
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


class TestNeuralNetworkConstruction(unittest.TestCase):
    def test_creates_correct_number_of_layers(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 4, 1])
        self.assertEqual(len(net._layers), 3)

    def test_layer_sizes_match_spec(self):
        net = NeuralNetwork(layer_sizes=[2, 5, 3], activations=[None, None])
        self.assertEqual(net._layers[0]._weights.value.shape, (2, 5))
        self.assertEqual(net._layers[1]._weights.value.shape, (5, 3))

    def test_layers_are_dense_layer_instances(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1])
        for layer in net._layers:
            self.assertIsInstance(layer, DenseLayer)

    def test_too_few_layer_sizes_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(layer_sizes=[3])

    def test_non_positive_layer_size_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(layer_sizes=[2, 0, 1])

    def test_activation_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(layer_sizes=[2, 3, 1], activations=["relu"])

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(layer_sizes=[2, 2, 1], activations=["banana", None])

    def test_default_activations_are_sigmoid(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1])
        outputs = net.forward(Constant(np.zeros((1, 1))))
        self.assertIsInstance(outputs, SigmoidUnaryExpression)


class TestNeuralNetworkForward(unittest.TestCase):
    def test_output_shape_matches_last_layer(self):
        net = NeuralNetwork(layer_sizes=[3, 4, 2], activations=[None, None])
        outputs = net.forward(Constant(np.ones((5, 3)))).forward()
        self.assertEqual(outputs.shape, (5, 2))

    def test_per_layer_activations_applied(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=["relu", "sigmoid"])
        outputs = net.forward(Constant(np.zeros((1, 1))))
        self.assertIsInstance(outputs, SigmoidUnaryExpression)

    def test_chains_layer_outputs_into_next_layer(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._weights.value = np.array([[2.0]])
        net._layers[0]._bias.value = np.array([1.0])
        net._layers[1]._weights.value = np.array([[3.0]])
        net._layers[1]._bias.value = np.array([0.0])
        outputs = net.forward(Constant(np.array([[4.0]]))).forward()
        # ((4*2)+1)*3 = 27
        np.testing.assert_array_equal(outputs, np.array([[27.0]]))

    def test_relu_hidden_layer_clamps_negatives(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=["relu", None])
        net._layers[0]._weights.value = np.array([[-1.0]])
        net._layers[0]._bias.value = np.array([0.0])
        net._layers[1]._weights.value = np.array([[5.0]])
        net._layers[1]._bias.value = np.array([2.0])
        outputs = net.forward(Constant(np.array([[3.0]]))).forward()
        # relu(-3) = 0, then 0*5 + 2 = 2
        np.testing.assert_array_equal(outputs, np.array([[2.0]]))


class TestNeuralNetworkParameters(unittest.TestCase):
    def test_parameter_count_matches_two_per_layer(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 1], activations=[None, None])
        self.assertEqual(len(net.parameters()), 4)

    def test_parameters_are_variables(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1])
        for parameter in net.parameters():
            self.assertIsInstance(parameter, Variable)

    def test_parameters_are_unique(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 2])
        ids = [id(parameter) for parameter in net.parameters()]
        self.assertEqual(len(ids), len(set(ids)))


class TestNeuralNetworkGradients(unittest.TestCase):
    def test_backward_propagates_to_input_variable(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._weights.value = np.array([[2.0]])
        net._layers[0]._bias.value = np.array([0.0])
        net._layers[1]._weights.value = np.array([[3.0]])
        net._layers[1]._bias.value = np.array([0.0])
        x = Variable(np.array([[1.0]]), "x")
        outputs = net.forward(x)
        outputs.forward()
        outputs.backward(np.ones((1, 1)))
        np.testing.assert_array_equal(x.gradient, np.array([[6.0]]))

    def test_backward_updates_all_layer_parameters(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._weights.value = np.array([[1.0]])
        net._layers[0]._bias.value = np.array([0.0])
        net._layers[1]._weights.value = np.array([[1.0]])
        net._layers[1]._bias.value = np.array([0.0])
        outputs = net.forward(Constant(np.array([[2.0]])))
        outputs.forward()
        outputs.backward(np.ones((1, 1)))
        for parameter in net.parameters():
            self.assertTrue(np.any(parameter.gradient != 0.0))

    def test_zero_all_gradients_resets_network_params(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1], activations=[None, None])
        outputs = net.forward(Constant(np.array([[1.0, 2.0]])))
        outputs.forward()
        outputs.backward(np.ones((1, 1)))
        outputs.zero_all_gradients()
        for parameter in net.parameters():
            np.testing.assert_array_equal(parameter.gradient, np.zeros_like(parameter.value))


if __name__ == "__main__":
    unittest.main()
