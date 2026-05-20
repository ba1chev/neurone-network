import unittest
import numpy as np

from source.training.trainer import train
from source.neuron_network.neural_network import NeuralNetwork
from source.loss_functions.categorical_cross_entropy import CategoricalCrossEntropy
from source.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from source.constants import XAVIER_INITIALIZATION


def _make_synthetic_classification_dataset(rng: np.random.Generator):
    centroids = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    samples_per_class = 60
    inputs = np.vstack([centroid + rng.normal(scale=0.5, size=(samples_per_class, 2)) for centroid in centroids])
    targets = np.repeat(np.arange(3), samples_per_class)
    permutation = rng.permutation(inputs.shape[0])
    return inputs[permutation], targets[permutation]


class TestTrainerSmoke(unittest.TestCase):
    def test_loss_decreases_and_history_shape(self):
        rng = np.random.default_rng(0)
        x, y = _make_synthetic_classification_dataset(rng)
        network = NeuralNetwork(layer_sizes=[2, 8, 3], activations=["relu", None], initialization=XAVIER_INITIALIZATION)
        optimizer = StochasticGradientDescent(network.parameters(), learning_rate=0.1)

        history = train(
            network=network,
            loss_function=CategoricalCrossEntropy(),
            optimizer=optimizer,
            x_train=x,
            y_train=y,
            epochs=5,
            batch_size=20,
            verbose=False,
            rng=rng
        )

        self.assertEqual(len(history["train_loss"]), 5)
        self.assertEqual(len(history["train_accuracy"]), 5)
        self.assertEqual(len(history["epoch_seconds"]), 5)
        self.assertEqual(history["validation_loss"], [])
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])
        self.assertGreater(history["train_accuracy"][-1], 0.7)

    def test_validation_history_populated(self):
        rng = np.random.default_rng(1)
        x, y = _make_synthetic_classification_dataset(rng)
        x_train, x_validation = x[:120], x[120:]
        y_train, y_validation = y[:120], y[120:]
        network = NeuralNetwork(layer_sizes=[2, 6, 3], activations=["relu", None], initialization=XAVIER_INITIALIZATION)
        optimizer = StochasticGradientDescent(network.parameters(), learning_rate=0.1)

        history = train(
            network=network,
            loss_function=CategoricalCrossEntropy(),
            optimizer=optimizer,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            epochs=3,
            batch_size=20,
            verbose=False,
            rng=rng
        )

        self.assertEqual(len(history["validation_loss"]), 3)
        self.assertEqual(len(history["validation_accuracy"]), 3)
