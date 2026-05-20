import math
import random
import unittest

from source.neuron_network.neuron_layer import NeuronLayer
from source.neuron_network.neural_network import NeuralNetwork
from source.neuron_network.neuron import Neuron, make_weight_initializer
from source.constants import UNIFORM_INITIALIZATION, XAVIER_INITIALIZATION


class TestMakeWeightInitializer(unittest.TestCase):
    def test_unknown_scheme_rejected(self):
        with self.assertRaises(ValueError):
            make_weight_initializer("banana", fan_in=4, fan_out=4)

    def test_uniform_scheme_returns_values_in_minus_one_to_one(self):
        random.seed(0)
        initializer = make_weight_initializer(UNIFORM_INITIALIZATION, fan_in=4, fan_out=4)
        for _ in range(200):
            value = initializer()
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_xavier_scheme_respects_glorot_limit(self):
        random.seed(0)
        fan_in, fan_out = 64, 32
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = make_weight_initializer(XAVIER_INITIALIZATION, fan_in, fan_out)
        for _ in range(500):
            value = initializer()
            self.assertGreaterEqual(value, -limit)
            self.assertLessEqual(value, limit)

    def test_xavier_scale_shrinks_with_fan_size(self):
        random.seed(0)
        small_initializer = make_weight_initializer(XAVIER_INITIALIZATION, 4, 4)
        large_initializer = make_weight_initializer(XAVIER_INITIALIZATION, 256, 256)
        small_samples = [abs(small_initializer()) for _ in range(500)]
        large_samples = [abs(large_initializer()) for _ in range(500)]
        self.assertGreater(max(small_samples), max(large_samples))


class TestNeuronWithCustomInitializer(unittest.TestCase):
    def test_custom_initializer_used_for_each_weight(self):
        values = iter([0.1, 0.2, 0.3])
        neuron = Neuron(num_inputs=3, activation=None, weight_initializer=lambda: next(values))
        weight_values = [w.value for w in neuron.parameters()[:-1]]
        self.assertEqual(weight_values, [0.1, 0.2, 0.3])

    def test_default_initializer_falls_back_to_uniform(self):
        random.seed(0)
        neuron = Neuron(num_inputs=10)
        for w in neuron.parameters()[:-1]:
            self.assertGreaterEqual(w.value, -1.0)
            self.assertLessEqual(w.value, 1.0)


class TestNeuronLayerInitialization(unittest.TestCase):
    def test_unknown_initialization_rejected(self):
        with self.assertRaises(ValueError):
            NeuronLayer(num_inputs=3, num_neurons=2, initialization="banana")

    def test_xavier_layer_weights_within_glorot_limit(self):
        random.seed(0)
        fan_in, fan_out = 64, 16
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        layer = NeuronLayer(
            num_inputs=fan_in,
            num_neurons=fan_out,
            activation=None,
            initialization=XAVIER_INITIALIZATION
        )
        for parameter in layer.parameters()[:-fan_out]:
            self.assertGreaterEqual(parameter.value, -limit)
            self.assertLessEqual(parameter.value, limit)


class TestNeuralNetworkInitialization(unittest.TestCase):
    def test_unknown_initialization_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork(layer_sizes=[2, 3, 1], initialization="banana")

    def test_xavier_propagates_to_all_layers(self):
        random.seed(0)
        net = NeuralNetwork(
            layer_sizes=[8, 16, 4],
            activations=[None, None],
            initialization=XAVIER_INITIALIZATION
        )
        first_layer_limit = math.sqrt(6.0 / (8 + 16))
        second_layer_limit = math.sqrt(6.0 / (16 + 4))
        for neuron in net._layers[0]._neurons:
            for w in neuron._weights:
                self.assertLessEqual(abs(w.value), first_layer_limit + 1e-9)
        for neuron in net._layers[1]._neurons:
            for w in neuron._weights:
                self.assertLessEqual(abs(w.value), second_layer_limit + 1e-9)


if __name__ == "__main__":
    unittest.main()
