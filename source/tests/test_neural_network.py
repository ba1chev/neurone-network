import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.neural_network import NeuralNetwork
from source.neuron_network.neuron_layer import NeuronLayer
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


class TestNeuralNetworkConstruction(unittest.TestCase):
    def test_creates_correct_number_of_layers(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 4, 1])
        self.assertEqual(len(net._layers), 3)

    def test_layer_input_output_sizes_match_spec(self):
        net = NeuralNetwork(layer_sizes=[2, 5, 3], activations=[None, None])
        self.assertEqual(net._layers[0]._num_inputs, 2)
        self.assertEqual(net._layers[0]._num_neurons, 5)
        self.assertEqual(net._layers[1]._num_inputs, 5)
        self.assertEqual(net._layers[1]._num_neurons, 3)

    def test_layers_are_neuron_layer_instances(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1])
        for layer in net._layers:
            self.assertIsInstance(layer, NeuronLayer)

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
        outputs = net.forward([Constant(0.0)])
        self.assertIsInstance(outputs[0], SigmoidUnaryExpression)


class TestNeuralNetworkForward(unittest.TestCase):
    def test_output_length_matches_last_layer(self):
        net = NeuralNetwork(layer_sizes=[3, 4, 2], activations=[None, None])
        outputs = net.forward([Constant(1.0), Constant(2.0), Constant(3.0)])
        self.assertEqual(len(outputs), 2)

    def test_input_count_mismatch_raises(self):
        net = NeuralNetwork(layer_sizes=[3, 2, 1], activations=[None, None])
        with self.assertRaises(ValueError):
            net.forward([Constant(1.0), Constant(2.0)])

    def test_per_layer_activations_applied(self):
        net = NeuralNetwork(
            layer_sizes=[1, 1, 1],
            activations=["relu", "sigmoid"]
        )
        outputs = net.forward([Constant(0.0)])
        self.assertIsInstance(outputs[0], SigmoidUnaryExpression)

    def test_chains_layer_outputs_into_next_layer(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._neurons[0]._weights[0].value = 2.0
        net._layers[0]._neurons[0]._bias.value = 1.0
        net._layers[1]._neurons[0]._weights[0].value = 3.0
        net._layers[1]._neurons[0]._bias.value = 0.0
        outputs = net.forward([Constant(4.0)])
        self.assertEqual(outputs[0].forward(), 27.0)

    def test_relu_hidden_layer_clamps_negatives(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=["relu", None])
        net._layers[0]._neurons[0]._weights[0].value = -1.0
        net._layers[0]._neurons[0]._bias.value = 0.0
        net._layers[1]._neurons[0]._weights[0].value = 5.0
        net._layers[1]._neurons[0]._bias.value = 2.0
        outputs = net.forward([Constant(3.0)])
        self.assertEqual(outputs[0].forward(), 2.0)

    def test_relu_first_layer_returns_relu_expressions(self):
        net = NeuralNetwork(layer_sizes=[1, 2], activations=["relu"])
        outputs = net.forward([Constant(1.0)])
        for out in outputs:
            self.assertIsInstance(out, ReluUnaryExpression)


class TestNeuralNetworkParameters(unittest.TestCase):
    def test_parameter_count_matches_sum_of_layer_params(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 1], activations=[None, None])
        expected = (2 + 1) * 3 + (3 + 1) * 1
        self.assertEqual(len(net.parameters()), expected)

    def test_parameters_are_variables(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1])
        for p in net.parameters():
            self.assertIsInstance(p, Variable)

    def test_parameters_are_unique(self):
        net = NeuralNetwork(layer_sizes=[2, 3, 2])
        ids = [id(p) for p in net.parameters()]
        self.assertEqual(len(ids), len(set(ids)))


class TestNeuralNetworkGradients(unittest.TestCase):
    def test_backward_propagates_to_input_variables(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._neurons[0]._weights[0].value = 2.0
        net._layers[0]._neurons[0]._bias.value = 0.0
        net._layers[1]._neurons[0]._weights[0].value = 3.0
        net._layers[1]._neurons[0]._bias.value = 0.0
        x = Variable(1.0, "x")
        outputs = net.forward([x])
        outputs[0].forward()
        outputs[0].backward(1.0)
        self.assertEqual(x.gradient, 6.0)

    def test_backward_updates_all_layer_parameters(self):
        net = NeuralNetwork(layer_sizes=[1, 1, 1], activations=[None, None])
        net._layers[0]._neurons[0]._weights[0].value = 1.0
        net._layers[0]._neurons[0]._bias.value = 0.0
        net._layers[1]._neurons[0]._weights[0].value = 1.0
        net._layers[1]._neurons[0]._bias.value = 0.0
        outputs = net.forward([Constant(2.0)])
        outputs[0].forward()
        outputs[0].backward(1.0)
        for p in net.parameters():
            self.assertNotEqual(p.gradient, 0.0)

    def test_zero_all_gradients_resets_network_params(self):
        net = NeuralNetwork(layer_sizes=[2, 2, 1], activations=[None, None])
        outputs = net.forward([Constant(1.0), Constant(2.0)])
        outputs[0].forward()
        outputs[0].backward(1.0)
        outputs[0].zero_all_gradients()
        for p in net.parameters():
            self.assertEqual(p.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
