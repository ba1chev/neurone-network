import time
import numpy as np
from typing import Optional

from source.expressions.constant import Constant
from source.optimizers.optimizer import Optimizer
from source.loss_functions.loss_function import LossFunction
from source.neuron_network.neural_network import NeuralNetwork


def _accuracy(network: NeuralNetwork, inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> float:
    correct = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        logits = network.forward(Constant(batch)).forward()
        correct += int(np.sum(np.argmax(logits, axis=1) == targets[start : start + batch_size]))
    return correct / inputs.shape[0]


def _epoch_loss(network: NeuralNetwork, loss_function: LossFunction, inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> float:
    weighted_sum = 0.0
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        predictions = network.forward(Constant(batch_inputs))
        batch_loss = float(loss_function.compute(predictions, batch_targets).forward())
        weighted_sum += batch_loss * batch_inputs.shape[0]
    return weighted_sum / inputs.shape[0]


def train(network: NeuralNetwork, loss_function: LossFunction, optimizer: Optimizer, x_train: np.ndarray,
    y_train: np.ndarray, x_validation: Optional[np.ndarray] = None, y_validation: Optional[np.ndarray] = None, epochs: int = 10,
    batch_size: int = 64, shuffle: bool = True, verbose: bool = True, rng: Optional[np.random.Generator] = None) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "epoch_seconds": []
    }

    indices = np.arange(x_train.shape[0])
    has_validation = x_validation is not None and y_validation is not None

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        if shuffle:
            rng.shuffle(indices)

        for start in range(0, indices.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = x_train[batch_indices]
            batch_targets = y_train[batch_indices]

            predictions = network.forward(Constant(batch_inputs))
            loss_expression = loss_function.compute(predictions, batch_targets)
            loss_expression.zero_all_gradients()
            loss_expression.forward()
            loss_expression.backward(np.array(1.0))
            optimizer.step()

        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = _epoch_loss(network, loss_function, x_train, y_train, batch_size)
        train_accuracy = _accuracy(network, x_train, y_train, batch_size)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["epoch_seconds"].append(epoch_seconds)

        if has_validation:
            validation_loss = _epoch_loss(network, loss_function, x_validation, y_validation, batch_size)
            validation_accuracy = _accuracy(network, x_validation, y_validation, batch_size)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)

        if verbose:
            line = f"epoch {epoch:>3}/{epochs}  loss={train_loss:.4f}  acc={train_accuracy:.4f}  time={epoch_seconds:.2f}s"
            if has_validation:
                line += f"  val_loss={history['validation_loss'][-1]:.4f}  val_acc={history['validation_accuracy'][-1]:.4f}"
            print(line, flush=True)

    return history
