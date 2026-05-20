import unittest
import numpy as np

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.loss_functions.mean_squared_error import MeanSquaredError
from source.loss_functions.mean_absolute_error import MeanAbsoluteError
from source.loss_functions.binary_cross_entropy import BinaryCrossEntropy
from source.loss_functions.categorical_cross_entropy import CategoricalCrossEntropy


class TestMeanSquaredError(unittest.TestCase):
    def test_zero_loss_for_perfect_predictions(self):
        predictions = Constant(np.array([1.0, 2.0, 3.0]))
        targets = np.array([1.0, 2.0, 3.0])
        loss = MeanSquaredError().compute(predictions, targets)
        np.testing.assert_allclose(loss.forward(), 0.0)

    def test_known_value(self):
        predictions = Constant(np.array([1.0, 2.0]))
        targets = np.array([2.0, 4.0])
        loss = MeanSquaredError().compute(predictions, targets)
        # ((1-2)^2 + (2-4)^2) / 2 = 5/2 = 2.5
        np.testing.assert_allclose(loss.forward(), 2.5)

    def test_gradient_is_two_over_n_times_diff(self):
        predictions = Variable(np.array([1.0, 2.0]), "p")
        targets = np.array([2.0, 4.0])
        loss = MeanSquaredError().compute(predictions, targets)
        loss.forward()
        loss.backward(np.array(1.0))
        # d/dp = 2*(p - t)/n = [-1.0, -2.0]
        np.testing.assert_allclose(predictions.gradient, np.array([-1.0, -2.0]))

    def test_empty_targets_rejected(self):
        with self.assertRaises(ValueError):
            MeanSquaredError().compute(Constant(np.array([])), np.array([]))


class TestMeanAbsoluteError(unittest.TestCase):
    def test_zero_loss_for_perfect_predictions(self):
        predictions = Constant(np.array([1.0, 2.0]))
        targets = np.array([1.0, 2.0])
        loss = MeanAbsoluteError().compute(predictions, targets)
        np.testing.assert_allclose(loss.forward(), 0.0)

    def test_known_value(self):
        predictions = Constant(np.array([1.0, 2.0]))
        targets = np.array([3.0, 5.0])
        loss = MeanAbsoluteError().compute(predictions, targets)
        # (|1-3| + |2-5|) / 2 = 5/2 = 2.5
        np.testing.assert_allclose(loss.forward(), 2.5)

    def test_gradient_is_sign_over_n(self):
        predictions = Variable(np.array([1.0, 5.0]), "p")
        targets = np.array([3.0, 2.0])
        loss = MeanAbsoluteError().compute(predictions, targets)
        loss.forward()
        loss.backward(np.array(1.0))
        # d/dp = sign(p - t) / n = [-0.5, 0.5]
        np.testing.assert_allclose(predictions.gradient, np.array([-0.5, 0.5]))


class TestBinaryCrossEntropy(unittest.TestCase):
    def test_zero_loss_for_perfect_certainty(self):
        predictions = Constant(np.array([0.999999, 0.000001]))
        targets = np.array([1.0, 0.0])
        loss = BinaryCrossEntropy().compute(predictions, targets)
        self.assertLess(float(loss.forward()), 1e-5)

    def test_known_value(self):
        predictions = Constant(np.array([0.5, 0.5]))
        targets = np.array([1.0, 0.0])
        loss = BinaryCrossEntropy().compute(predictions, targets)
        # -mean( log(0.5) + log(0.5) ) = -log(0.5) = log(2)
        np.testing.assert_allclose(loss.forward(), np.log(2.0))

    def test_gradient_pushes_prediction_toward_target(self):
        predictions = Variable(np.array([0.3, 0.7]), "p")
        targets = np.array([1.0, 0.0])
        loss = BinaryCrossEntropy().compute(predictions, targets)
        loss.forward()
        loss.backward(np.array(1.0))
        # gradient[0] should be negative (push toward 1), gradient[1] positive
        self.assertLess(predictions.gradient[0], 0.0)
        self.assertGreater(predictions.gradient[1], 0.0)


class TestCategoricalCrossEntropy(unittest.TestCase):
    def test_uniform_logits_yield_log_num_classes(self):
        predictions = Constant(np.zeros((1, 3)))
        targets = np.array([1])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        np.testing.assert_allclose(loss.forward(), np.log(3.0))

    def test_value_matches_manual_softmax_nll(self):
        logits_array = np.array([[2.0, 1.0, 0.5]])
        target_index = 0
        predictions = Constant(logits_array)
        targets = np.array([target_index])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        max_logit = logits_array.max()
        exps = np.exp(logits_array - max_logit)
        softmax = exps / exps.sum()
        expected = -np.log(softmax[0, target_index])
        np.testing.assert_allclose(loss.forward(), expected)

    def test_stable_for_large_logits(self):
        predictions = Constant(np.array([[1000.0, 999.0, 998.0]]))
        targets = np.array([0])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        result = float(loss.forward())
        self.assertFalse(np.isinf(result))
        # softmax(0) = 1 / (1 + e^-1 + e^-2), nll = -log of that
        expected = np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))
        self.assertAlmostEqual(result, expected, places=10)

    def test_gradient_equals_softmax_minus_one_hot_over_batch(self):
        logits = Variable(np.array([[2.0, 1.0, 0.5]]), "z")
        targets = np.array([1])
        loss = CategoricalCrossEntropy().compute(logits, targets)
        loss.forward()
        loss.backward(np.array(1.0))
        exps = np.exp(np.array([[2.0, 1.0, 0.5]]) - 2.0)
        softmax = exps / exps.sum(axis=1, keepdims=True)
        one_hot = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(logits.gradient, softmax - one_hot)

    def test_gradient_averages_across_batch(self):
        logits = Variable(np.array([[1.0, 0.0], [0.0, 1.0]]), "z")
        targets = np.array([0, 1])
        loss = CategoricalCrossEntropy().compute(logits, targets)
        loss.forward()
        loss.backward(np.array(1.0))
        # both rows have softmax = [e/(e+1), 1/(e+1)] and [1/(e+1), e/(e+1)]
        sm_high = np.exp(1.0) / (np.exp(1.0) + 1.0)
        sm_low = 1.0 / (np.exp(1.0) + 1.0)
        expected = np.array([
            [(sm_high - 1.0) / 2.0, sm_low / 2.0],
            [sm_low / 2.0, (sm_high - 1.0) / 2.0],
        ])
        np.testing.assert_allclose(logits.gradient, expected)

    def test_target_out_of_range_rejected(self):
        predictions = Constant(np.array([[1.0, 2.0]]))
        targets = np.array([5])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        with self.assertRaises(ValueError):
            loss.forward()

    def test_negative_target_rejected(self):
        predictions = Constant(np.array([[1.0, 2.0]]))
        targets = np.array([-1])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        with self.assertRaises(ValueError):
            loss.forward()

    def test_length_mismatch_rejected(self):
        predictions = Constant(np.array([[1.0, 2.0]]))
        targets = np.array([0, 1])
        loss = CategoricalCrossEntropy().compute(predictions, targets)
        with self.assertRaises(ValueError):
            loss.forward()


if __name__ == "__main__":
    unittest.main()
