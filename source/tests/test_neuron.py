import random
import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.neuron import Neuron
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.tanh_unary_expression import TanhUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression


class TestNeuronConstruction(unittest.TestCase):
    def test_creates_correct_number_of_weights(self):
        neuron = Neuron(num_inputs=3)
        self.assertEqual(len(neuron.parameters()), 4)

    def test_weights_are_variables(self):
        neuron = Neuron(num_inputs=2)
        for param in neuron.parameters():
            self.assertIsInstance(param, Variable)

    def test_weights_within_initial_range(self):
        neuron = Neuron(num_inputs=10)
        for w in neuron.parameters()[:-1]:
            self.assertGreaterEqual(w.value, -1.0)
            self.assertLessEqual(w.value, 1.0)

    def test_bias_starts_at_zero(self):
        neuron = Neuron(num_inputs=3)
        self.assertEqual(neuron.parameters()[-1].value, 0.0)

    def test_zero_inputs_rejected(self):
        with self.assertRaises(ValueError):
            Neuron(num_inputs=0)

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            Neuron(num_inputs=2, activation="banana")

    def test_default_activation_is_sigmoid(self):
        random.seed(0)
        neuron = Neuron(num_inputs=1)
        out = neuron.forward([Constant(0.0)])
        self.assertIsInstance(out, SigmoidUnaryExpression)


class TestNeuronForward(unittest.TestCase):
    def test_forward_input_count_mismatch_raises(self):
        neuron = Neuron(num_inputs=3, activation=None)
        with self.assertRaises(ValueError):
            neuron.forward([Constant(1.0), Constant(2.0)])

    def test_linear_neuron_computes_dot_product_plus_bias(self):
        neuron = Neuron(num_inputs=2, activation=None)
        neuron._weights[0].value = 2.0
        neuron._weights[1].value = 3.0
        neuron._bias.value = 1.0
        out = neuron.forward([Constant(4.0), Constant(5.0)])
        self.assertEqual(out.forward(), 24.0)

    def test_sigmoid_activation_wraps_output(self):
        neuron = Neuron(num_inputs=1, activation="sigmoid")
        out = neuron.forward([Constant(0.0)])
        self.assertIsInstance(out, SigmoidUnaryExpression)

    def test_relu_activation_wraps_output(self):
        neuron = Neuron(num_inputs=1, activation="relu")
        out = neuron.forward([Constant(0.0)])
        self.assertIsInstance(out, ReluUnaryExpression)

    def test_tanh_activation_wraps_output(self):
        neuron = Neuron(num_inputs=1, activation="tanh")
        out = neuron.forward([Constant(0.0)])
        self.assertIsInstance(out, TanhUnaryExpression)

    def test_sigmoid_at_zero_pre_activation(self):
        neuron = Neuron(num_inputs=2, activation="sigmoid")
        for w in neuron._weights:
            w.value = 0.0
        out = neuron.forward([Constant(5.0), Constant(-3.0)])
        self.assertAlmostEqual(out.forward(), 0.5)

    def test_relu_clamps_negative_pre_activation(self):
        neuron = Neuron(num_inputs=1, activation="relu")
        neuron._weights[0].value = -2.0
        neuron._bias.value = 0.0
        out = neuron.forward([Constant(3.0)])
        self.assertEqual(out.forward(), 0.0)


class TestNeuronGradients(unittest.TestCase):
    def test_backward_updates_weight_gradients(self):
        neuron = Neuron(num_inputs=2, activation=None)
        neuron._weights[0].value = 1.0
        neuron._weights[1].value = 1.0
        out = neuron.forward([Constant(3.0), Constant(4.0)])
        out.forward()
        out.backward(1.0)
        self.assertEqual(neuron._weights[0].gradient, 3.0)
        self.assertEqual(neuron._weights[1].gradient, 4.0)
        self.assertEqual(neuron._bias.gradient, 1.0)

    def test_backward_through_sigmoid(self):
        neuron = Neuron(num_inputs=1, activation="sigmoid")
        neuron._weights[0].value = 0.0
        neuron._bias.value = 0.0
        out = neuron.forward([Constant(2.0)])
        out.forward()
        out.backward(1.0)
        self.assertAlmostEqual(neuron._weights[0].gradient, 0.5)
        self.assertAlmostEqual(neuron._bias.gradient, 0.25)

    def test_inputs_can_be_variables_and_receive_gradient(self):
        neuron = Neuron(num_inputs=2, activation=None)
        neuron._weights[0].value = 2.0
        neuron._weights[1].value = 3.0
        x1 = Variable(1.0, "x1")
        x2 = Variable(1.0, "x2")
        out = neuron.forward([x1, x2])
        out.forward()
        out.backward(1.0)
        self.assertEqual(x1.gradient, 2.0)
        self.assertEqual(x2.gradient, 3.0)

    def test_zero_all_gradients_resets_neuron_params(self):
        neuron = Neuron(num_inputs=2, activation=None)
        out = neuron.forward([Constant(1.0), Constant(2.0)])
        out.forward()
        out.backward(1.0)
        for p in neuron.parameters():
            self.assertNotEqual(p.gradient, 0.0) if p.value != 0.0 else None
        out.zero_all_gradients()
        for p in neuron.parameters():
            self.assertEqual(p.gradient, 0.0)


class TestNeuronParameters(unittest.TestCase):
    def test_parameters_returns_weights_then_bias(self):
        neuron = Neuron(num_inputs=3, activation=None)
        params = neuron.parameters()
        self.assertIs(params[-1], neuron._bias)
        for i in range(3):
            self.assertIs(params[i], neuron._weights[i])

    def test_parameters_unique_per_neuron(self):
        n1 = Neuron(num_inputs=2)
        n2 = Neuron(num_inputs=2)
        ids1 = {id(p) for p in n1.parameters()}
        ids2 = {id(p) for p in n2.parameters()}
        self.assertEqual(ids1 & ids2, set())


if __name__ == "__main__":
    unittest.main()
