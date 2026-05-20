import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.dense_layer import DenseLayer
from source.expressions.unary_expressions.relu_unary_expression import ReluUnaryExpression
from source.expressions.unary_expressions.sigmoid_unary_expression import SigmoidUnaryExpression
from source.constants import XAVIER_INITIALIZATION


class TestDenseLayerConstruction(unittest.TestCase):
    def test_weights_shape_is_in_by_out(self):
        layer = DenseLayer(num_inputs=4, num_neurons=3, activation=None)
        self.assertEqual(layer._weights.value.shape, (4, 3))

    def test_bias_shape_is_out(self):
        layer = DenseLayer(num_inputs=4, num_neurons=3, activation=None)
        self.assertEqual(layer._bias.value.shape, (3,))

    def test_bias_initialized_to_zero(self):
        layer = DenseLayer(num_inputs=2, num_neurons=2, activation=None)
        np.testing.assert_array_equal(layer._bias.value, np.zeros(2))

    def test_zero_inputs_rejected(self):
        with self.assertRaises(ValueError):
            DenseLayer(num_inputs=0, num_neurons=2)

    def test_zero_neurons_rejected(self):
        with self.assertRaises(ValueError):
            DenseLayer(num_inputs=2, num_neurons=0)

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            DenseLayer(num_inputs=2, num_neurons=2, activation="banana")

    def test_unknown_initialization_rejected(self):
        with self.assertRaises(ValueError):
            DenseLayer(num_inputs=2, num_neurons=2, initialization="lecun")

    def test_xavier_weights_within_limit(self):
        layer = DenseLayer(num_inputs=10, num_neurons=20, activation=None, initialization=XAVIER_INITIALIZATION)
        limit = np.sqrt(6.0 / 30.0)
        self.assertTrue(np.all(np.abs(layer._weights.value) <= limit))


class TestDenseLayerForward(unittest.TestCase):
    def test_forward_shape_is_batch_by_out(self):
        layer = DenseLayer(num_inputs=3, num_neurons=2, activation=None)
        inputs = Constant(np.ones((4, 3)))
        result = layer.forward(inputs).forward()
        self.assertEqual(result.shape, (4, 2))

    def test_forward_with_known_weights_matches_manual(self):
        layer = DenseLayer(num_inputs=2, num_neurons=1, activation=None)
        layer._weights.value = np.array([[2.0], [3.0]])
        layer._bias.value = np.array([1.0])
        inputs = Constant(np.array([[1.0, 4.0]]))
        result = layer.forward(inputs).forward()
        # 1*2 + 4*3 + 1 = 15
        np.testing.assert_array_equal(result, np.array([[15.0]]))

    def test_no_activation_yields_pre_activation(self):
        layer = DenseLayer(num_inputs=2, num_neurons=2, activation=None)
        inputs = Constant(np.ones((1, 2)))
        result = layer.forward(inputs)
        self.assertNotIsInstance(result, SigmoidUnaryExpression)
        self.assertNotIsInstance(result, ReluUnaryExpression)

    def test_relu_activation_wraps_in_relu(self):
        layer = DenseLayer(num_inputs=2, num_neurons=2, activation="relu")
        inputs = Constant(np.ones((1, 2)))
        result = layer.forward(inputs)
        self.assertIsInstance(result, ReluUnaryExpression)


class TestDenseLayerBackward(unittest.TestCase):
    def test_backward_updates_weight_and_bias_gradients(self):
        layer = DenseLayer(num_inputs=2, num_neurons=1, activation=None)
        layer._weights.value = np.array([[1.0], [1.0]])
        layer._bias.value = np.array([0.0])
        inputs = Variable(np.array([[3.0, 4.0], [1.0, 2.0]]), "x")
        output = layer.forward(inputs)
        output.forward()
        output.backward(np.ones((2, 1)))
        # dW = X.T @ grad = [[3+1],[4+2]] = [[4],[6]]
        np.testing.assert_array_equal(layer._weights.gradient, np.array([[4.0], [6.0]]))
        # db = sum over batch = 2
        np.testing.assert_array_equal(layer._bias.gradient, np.array([2.0]))
        # dX = grad @ W.T = [[1,1],[1,1]]
        np.testing.assert_array_equal(inputs.gradient, np.array([[1.0, 1.0], [1.0, 1.0]]))


class TestDenseLayerParameters(unittest.TestCase):
    def test_parameters_returns_weight_and_bias(self):
        layer = DenseLayer(num_inputs=2, num_neurons=3, activation=None)
        params = layer.parameters()
        self.assertEqual(len(params), 2)
        self.assertIs(params[0], layer._weights)
        self.assertIs(params[1], layer._bias)


if __name__ == "__main__":
    unittest.main()
