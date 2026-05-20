import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.neuron_network.neural_network import NeuralNetwork
from source.loss_functions.mean_squared_error import MeanSquaredError
from source.loss_functions.binary_cross_entropy import BinaryCrossEntropy
from source.loss_functions.categorical_cross_entropy import CategoricalCrossEntropy
from source.optimizers.stochastic_gradient_descent import StochasticGradientDescent


class TestRegressionTrainingLoop(unittest.TestCase):
    def test_loss_decreases_when_fitting_a_simple_function(self):
        np.random.seed(0)
        inputs = np.random.uniform(-1.0, 1.0, size=(64, 2))
        targets = (inputs[:, 0:1] * 0.5 + inputs[:, 1:2] * -0.3 + 0.1)

        network = NeuralNetwork(layer_sizes=[2, 4, 1], activations=["relu", None])
        optimizer = StochasticGradientDescent(network.parameters(), learning_rate=0.05)
        loss_fn = MeanSquaredError()

        predictions = network.forward(Constant(inputs))
        loss = loss_fn.compute(predictions, targets)
        initial_loss = float(loss.forward())

        for _ in range(50):
            predictions = network.forward(Constant(inputs))
            loss = loss_fn.compute(predictions, targets)
            loss.forward()
            optimizer.zero_gradients()
            loss.backward(np.array(1.0))
            optimizer.step()

        final_loss = float(loss.forward())
        self.assertLess(final_loss, initial_loss * 0.5)


class TestMulticlassTrainingLoop(unittest.TestCase):
    def test_loss_decreases_on_three_class_separation(self):
        np.random.seed(1)
        # three clusters: target class is the index of the active feature
        inputs = np.eye(3) + np.random.normal(0.0, 0.05, size=(3, 3))
        targets = np.array([0, 1, 2])

        network = NeuralNetwork(layer_sizes=[3, 8, 3], activations=["relu", None])
        optimizer = StochasticGradientDescent(network.parameters(), learning_rate=0.1)
        loss_fn = CategoricalCrossEntropy()

        loss = loss_fn.compute(network.forward(Constant(inputs)), targets)
        initial_loss = float(loss.forward())

        for _ in range(100):
            loss = loss_fn.compute(network.forward(Constant(inputs)), targets)
            loss.forward()
            optimizer.zero_gradients()
            loss.backward(np.array(1.0))
            optimizer.step()

        final_loss = float(loss.forward())
        self.assertLess(final_loss, initial_loss * 0.5)


class TestBinaryClassificationLoop(unittest.TestCase):
    def test_loss_decreases_on_two_class_separation(self):
        np.random.seed(2)
        positive = np.random.normal(loc=1.0, scale=0.2, size=(8, 2))
        negative = np.random.normal(loc=-1.0, scale=0.2, size=(8, 2))
        inputs = np.concatenate([positive, negative], axis=0)
        targets = np.concatenate([np.ones((8, 1)), np.zeros((8, 1))], axis=0)

        network = NeuralNetwork(layer_sizes=[2, 4, 1], activations=["relu", "sigmoid"])
        optimizer = StochasticGradientDescent(network.parameters(), learning_rate=0.1)
        loss_fn = BinaryCrossEntropy()

        loss = loss_fn.compute(network.forward(Constant(inputs)), targets)
        initial_loss = float(loss.forward())

        for _ in range(100):
            loss = loss_fn.compute(network.forward(Constant(inputs)), targets)
            loss.forward()
            optimizer.zero_gradients()
            loss.backward(np.array(1.0))
            optimizer.step()

        final_loss = float(loss.forward())
        self.assertLess(final_loss, initial_loss * 0.5)


if __name__ == "__main__":
    unittest.main()
