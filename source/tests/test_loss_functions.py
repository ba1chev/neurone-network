import math
import unittest

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.loss_functions.mean_squared_error import MeanSquaredError
from source.loss_functions.mean_absolute_error import MeanAbsoluteError
from source.loss_functions.binary_cross_entropy import BinaryCrossEntropy


class TestMeanSquaredError(unittest.TestCase):
    def test_zero_when_predictions_match_targets(self):
        loss = MeanSquaredError()
        out = loss.compute([Constant(1.0), Constant(2.0)], [1.0, 2.0])
        self.assertEqual(out.forward(), 0.0)

    def test_value_matches_manual_average(self):
        loss = MeanSquaredError()
        out = loss.compute([Constant(3.0), Constant(0.0)], [1.0, 0.0])
        self.assertEqual(out.forward(), 2.0)

    def test_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            MeanSquaredError().compute([], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            MeanSquaredError().compute([Constant(1.0)], [1.0, 2.0])

    def test_gradient_wrt_prediction(self):
        x = Variable(3.0, "x")
        out = MeanSquaredError().compute([x], [1.0])
        out.forward()
        out.backward(1.0)
        self.assertEqual(x.gradient, 4.0)

    def test_gradient_averages_across_predictions(self):
        x1 = Variable(2.0, "x1")
        x2 = Variable(4.0, "x2")
        out = MeanSquaredError().compute([x1, x2], [0.0, 0.0])
        out.forward()
        out.backward(1.0)
        self.assertEqual(x1.gradient, 2.0)
        self.assertEqual(x2.gradient, 4.0)


class TestMeanAbsoluteError(unittest.TestCase):
    def test_zero_when_predictions_match_targets(self):
        loss = MeanAbsoluteError()
        out = loss.compute([Constant(1.0), Constant(2.0)], [1.0, 2.0])
        self.assertEqual(out.forward(), 0.0)

    def test_value_matches_manual_average(self):
        loss = MeanAbsoluteError()
        out = loss.compute([Constant(3.0), Constant(-1.0)], [1.0, 1.0])
        self.assertEqual(out.forward(), 2.0)

    def test_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            MeanAbsoluteError().compute([], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            MeanAbsoluteError().compute([Constant(1.0)], [1.0, 2.0])

    def test_gradient_sign_for_positive_residual(self):
        x = Variable(3.0, "x")
        out = MeanAbsoluteError().compute([x], [1.0])
        out.forward()
        out.backward(1.0)
        self.assertEqual(x.gradient, 1.0)

    def test_gradient_sign_for_negative_residual(self):
        x = Variable(-2.0, "x")
        out = MeanAbsoluteError().compute([x], [1.0])
        out.forward()
        out.backward(1.0)
        self.assertEqual(x.gradient, -1.0)


class TestBinaryCrossEntropy(unittest.TestCase):
    def test_value_matches_manual_for_perfect_prediction(self):
        loss = BinaryCrossEntropy()
        out = loss.compute([Constant(0.9999)], [1.0])
        self.assertAlmostEqual(out.forward(), -math.log(0.9999), places=6)

    def test_symmetric_value_for_complementary_target(self):
        loss = BinaryCrossEntropy()
        out_pos = loss.compute([Constant(0.7)], [1.0])
        out_neg = loss.compute([Constant(0.3)], [0.0])
        self.assertAlmostEqual(out_pos.forward(), out_neg.forward())

    def test_averages_across_predictions(self):
        loss = BinaryCrossEntropy()
        out = loss.compute([Constant(0.5), Constant(0.5)], [1.0, 0.0])
        self.assertAlmostEqual(out.forward(), -math.log(0.5))

    def test_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            BinaryCrossEntropy().compute([], [])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            BinaryCrossEntropy().compute([Constant(0.5)], [1.0, 0.0])

    def test_invalid_probability_raises_via_log(self):
        with self.assertRaises(RuntimeError):
            BinaryCrossEntropy().compute([Constant(0.0)], [1.0]).forward()

    def test_gradient_for_positive_target(self):
        x = Variable(0.5, "x")
        out = BinaryCrossEntropy().compute([x], [1.0])
        out.forward()
        out.backward(1.0)
        self.assertAlmostEqual(x.gradient, -2.0)

    def test_gradient_for_negative_target(self):
        x = Variable(0.5, "x")
        out = BinaryCrossEntropy().compute([x], [0.0])
        out.forward()
        out.backward(1.0)
        self.assertAlmostEqual(x.gradient, 2.0)


if __name__ == "__main__":
    unittest.main()
