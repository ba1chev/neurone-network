import math
import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.loss_functions.categorical_cross_entropy import CategoricalCrossEntropy


def _manual_softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    denominator = sum(exps)
    return [value / denominator for value in exps]


class TestCategoricalCrossEntropyValidation(unittest.TestCase):
    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute([], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute(
                [[Constant(0.0), Constant(1.0)]], [0, 1]
            )

    def test_empty_logits_per_sample_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute([[]], [0])

    def test_inconsistent_class_count_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute(
                [[Constant(0.0), Constant(1.0)], [Constant(0.0)]], [0, 0]
            )

    def test_target_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute(
                [[Constant(0.0), Constant(1.0)]], [5]
            )

    def test_negative_target_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalCrossEntropy().compute(
                [[Constant(0.0), Constant(1.0)]], [-1]
            )


class TestCategoricalCrossEntropyForward(unittest.TestCase):
    def test_uniform_logits_yield_log_num_classes(self):
        loss = CategoricalCrossEntropy().compute(
            [[Constant(0.0), Constant(0.0), Constant(0.0)]], [1]
        )
        self.assertAlmostEqual(loss.forward(), math.log(3.0))

    def test_value_matches_manual_softmax_nll(self):
        logits = [2.0, 1.0, 0.5]
        target = 0
        softmax = _manual_softmax(logits)
        expected = -math.log(softmax[target])
        loss = CategoricalCrossEntropy().compute(
            [[Constant(value) for value in logits]], [target]
        )
        self.assertAlmostEqual(loss.forward(), expected)

    def test_averaged_over_batch(self):
        loss_a = CategoricalCrossEntropy().compute(
            [[Constant(0.0), Constant(0.0)]], [0]
        )
        loss_b = CategoricalCrossEntropy().compute(
            [[Constant(0.0), Constant(0.0)], [Constant(0.0), Constant(0.0)]],
            [0, 1],
        )
        self.assertAlmostEqual(loss_a.forward(), math.log(2.0))
        self.assertAlmostEqual(loss_b.forward(), math.log(2.0))

    def test_log_sum_exp_stability_for_large_logits(self):
        logits = [1000.0, 999.0, 998.0]
        target = 0
        softmax = _manual_softmax(logits)
        expected = -math.log(softmax[target])
        loss = CategoricalCrossEntropy().compute(
            [[Constant(value) for value in logits]], [target]
        )
        self.assertAlmostEqual(loss.forward(), expected)

    def test_log_sum_exp_stability_for_very_negative_logits(self):
        logits = [-1000.0, -999.0, -1001.0]
        target = 1
        softmax = _manual_softmax(logits)
        expected = -math.log(softmax[target])
        loss = CategoricalCrossEntropy().compute(
            [[Constant(value) for value in logits]], [target]
        )
        self.assertAlmostEqual(loss.forward(), expected)


class TestCategoricalCrossEntropyBackward(unittest.TestCase):
    def test_gradient_equals_softmax_minus_one_hot(self):
        logits = [Variable(2.0, "z0"), Variable(1.0, "z1"), Variable(0.5, "z2")]
        target = 1
        loss = CategoricalCrossEntropy().compute([logits], [target])
        loss.forward()
        loss.backward(1.0)
        softmax = _manual_softmax([z.value for z in logits])
        for index, logit in enumerate(logits):
            indicator = 1.0 if index == target else 0.0
            self.assertAlmostEqual(logit.gradient, softmax[index] - indicator)

    def test_gradient_averages_across_batch(self):
        z0a = Variable(1.0, "z0a")
        z1a = Variable(0.0, "z1a")
        z0b = Variable(0.0, "z0b")
        z1b = Variable(1.0, "z1b")
        loss = CategoricalCrossEntropy().compute(
            [[z0a, z1a], [z0b, z1b]], [0, 1]
        )
        loss.forward()
        loss.backward(1.0)
        softmax_a = _manual_softmax([z0a.value, z1a.value])
        softmax_b = _manual_softmax([z0b.value, z1b.value])
        self.assertAlmostEqual(z0a.gradient, (softmax_a[0] - 1.0) / 2.0)
        self.assertAlmostEqual(z1a.gradient, (softmax_a[1] - 0.0) / 2.0)
        self.assertAlmostEqual(z0b.gradient, (softmax_b[0] - 0.0) / 2.0)
        self.assertAlmostEqual(z1b.gradient, (softmax_b[1] - 1.0) / 2.0)

    def test_gradient_zero_for_perfect_prediction(self):
        logits = [Variable(0.0, "z0"), Variable(50.0, "z1"), Variable(0.0, "z2")]
        target = 1
        loss = CategoricalCrossEntropy().compute([logits], [target])
        loss.forward()
        loss.backward(1.0)
        self.assertAlmostEqual(logits[1].gradient, 0.0, places=10)

    def test_collect_variables_reaches_logits(self):
        logits = [Variable(0.0, "z0"), Variable(0.0, "z1")]
        loss = CategoricalCrossEntropy().compute([logits], [0])
        variables = loss.get_variables()
        self.assertIn(logits[0], variables)
        self.assertIn(logits[1], variables)


if __name__ == "__main__":
    unittest.main()
