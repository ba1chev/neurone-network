import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.neuron import Neuron
from source.neuron_network.neuron_layer import NeuronLayer
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


class TestNeuronLayerConstruction(unittest.TestCase):
    def test_creates_correct_number_of_neurons(self):
        layer = NeuronLayer(num_inputs=3, num_neurons=4)
        self.assertEqual(len(layer._neurons), 4)

    def test_each_neuron_has_correct_input_size(self):
        layer = NeuronLayer(num_inputs=5, num_neurons=2)
        for neuron in layer._neurons:
            self.assertEqual(len(neuron.parameters()) - 1, 5)

    def test_neurons_are_neuron_instances(self):
        layer = NeuronLayer(num_inputs=2, num_neurons=3)
        for neuron in layer._neurons:
            self.assertIsInstance(neuron, Neuron)

    def test_zero_inputs_rejected(self):
        with self.assertRaises(ValueError):
            NeuronLayer(num_inputs=0, num_neurons=3)

    def test_zero_neurons_rejected(self):
        with self.assertRaises(ValueError):
            NeuronLayer(num_inputs=3, num_neurons=0)

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            NeuronLayer(num_inputs=2, num_neurons=2, activation="banana")


class TestNeuronLayerForward(unittest.TestCase):
    def test_output_length_matches_neuron_count(self):
        layer = NeuronLayer(num_inputs=3, num_neurons=4, activation=None)
        outputs = layer.forward([Constant(1.0), Constant(2.0), Constant(3.0)])
        self.assertEqual(len(outputs), 4)

    def test_input_count_mismatch_raises(self):
        layer = NeuronLayer(num_inputs=3, num_neurons=2, activation=None)
        with self.assertRaises(ValueError):
            layer.forward([Constant(1.0), Constant(2.0)])

    def test_default_activation_wraps_outputs_in_sigmoid(self):
        layer = NeuronLayer(num_inputs=1, num_neurons=3)
        outputs = layer.forward([Constant(0.0)])
        for out in outputs:
            self.assertIsInstance(out, SigmoidUnaryExpression)

    def test_relu_activation_wraps_outputs(self):
        layer = NeuronLayer(num_inputs=1, num_neurons=2, activation="relu")
        outputs = layer.forward([Constant(0.0)])
        for out in outputs:
            self.assertIsInstance(out, ReluUnaryExpression)

    def test_linear_layer_computes_per_neuron_dot_product(self):
        layer = NeuronLayer(num_inputs=2, num_neurons=2, activation=None)
        layer._neurons[0]._weights[0].value = 1.0
        layer._neurons[0]._weights[1].value = 2.0
        layer._neurons[0]._bias.value = 0.0
        layer._neurons[1]._weights[0].value = 3.0
        layer._neurons[1]._weights[1].value = 4.0
        layer._neurons[1]._bias.value = 1.0
        outputs = layer.forward([Constant(1.0), Constant(1.0)])
        self.assertEqual(outputs[0].forward(), 3.0)
        self.assertEqual(outputs[1].forward(), 8.0)


class TestNeuronLayerParameters(unittest.TestCase):
    def test_parameters_aggregate_all_neuron_params(self):
        layer = NeuronLayer(num_inputs=3, num_neurons=2, activation=None)
        params = layer.parameters()
        self.assertEqual(len(params), 2 * (3 + 1))

    def test_parameters_are_variables(self):
        layer = NeuronLayer(num_inputs=2, num_neurons=2)
        for p in layer.parameters():
            self.assertIsInstance(p, Variable)

    def test_parameters_are_unique_across_neurons(self):
        layer = NeuronLayer(num_inputs=2, num_neurons=3)
        ids = [id(p) for p in layer.parameters()]
        self.assertEqual(len(ids), len(set(ids)))


class TestNeuronLayerGradients(unittest.TestCase):
    def test_backward_through_layer_accumulates_into_shared_input(self):
        layer = NeuronLayer(num_inputs=1, num_neurons=2, activation=None)
        layer._neurons[0]._weights[0].value = 2.0
        layer._neurons[0]._bias.value = 0.0
        layer._neurons[1]._weights[0].value = 3.0
        layer._neurons[1]._bias.value = 0.0
        x = Variable(1.0, "x")
        outputs = layer.forward([x])
        for out in outputs:
            out.forward()
            out.backward(1.0)
        self.assertEqual(x.gradient, 5.0)

    def test_zero_all_gradients_via_layer_output_resets_params(self):
        layer = NeuronLayer(num_inputs=2, num_neurons=2, activation=None)
        outputs = layer.forward([Constant(1.0), Constant(2.0)])
        for out in outputs:
            out.forward()
            out.backward(1.0)
        outputs[0].zero_all_gradients()
        outputs[1].zero_all_gradients()
        for p in layer.parameters():
            self.assertEqual(p.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
