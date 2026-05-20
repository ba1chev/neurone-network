import unittest
import numpy as np

from source.expressions.broadcasting import unbroadcast


class TestUnbroadcastNoOp(unittest.TestCase):
    def test_same_shape_passes_through(self):
        gradient = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = unbroadcast(gradient, (2, 2))
        np.testing.assert_array_equal(result, gradient)

    def test_scalar_target_from_scalar_gradient(self):
        gradient = np.array(5.0)
        result = unbroadcast(gradient, ())
        np.testing.assert_array_equal(result, gradient)


class TestUnbroadcastLeadingAxes(unittest.TestCase):
    def test_extra_leading_axis_summed(self):
        gradient = np.ones((4, 3))
        result = unbroadcast(gradient, (3,))
        np.testing.assert_array_equal(result, np.full((3,), 4.0))

    def test_two_extra_leading_axes_summed(self):
        gradient = np.ones((2, 3, 4))
        result = unbroadcast(gradient, (4,))
        np.testing.assert_array_equal(result, np.full((4,), 6.0))

    def test_scalar_target_sums_everything(self):
        gradient = np.ones((2, 3))
        result = unbroadcast(gradient, ())
        np.testing.assert_array_equal(result, np.array(6.0))


class TestUnbroadcastSizeOneAxes(unittest.TestCase):
    def test_size_one_axis_summed_keeping_dim(self):
        gradient = np.ones((4, 3))
        result = unbroadcast(gradient, (1, 3))
        np.testing.assert_array_equal(result, np.full((1, 3), 4.0))

    def test_inner_size_one_axis_summed(self):
        gradient = np.ones((2, 4, 3))
        result = unbroadcast(gradient, (2, 1, 3))
        np.testing.assert_array_equal(result, np.full((2, 1, 3), 4.0))

    def test_multiple_size_one_axes_summed(self):
        gradient = np.ones((4, 5, 3))
        result = unbroadcast(gradient, (1, 1, 3))
        np.testing.assert_array_equal(result, np.full((1, 1, 3), 20.0))


class TestUnbroadcastCombined(unittest.TestCase):
    def test_leading_axis_and_size_one(self):
        gradient = np.ones((2, 3, 4))
        result = unbroadcast(gradient, (1, 4))
        np.testing.assert_array_equal(result, np.full((1, 4), 6.0))

    def test_bias_pattern_batch_to_features(self):
        # input (3,) broadcast to output (10, 3); gradient (10, 3) -> (3,)
        gradient = np.ones((10, 3))
        result = unbroadcast(gradient, (3,))
        np.testing.assert_array_equal(result, np.full((3,), 10.0))


if __name__ == "__main__":
    unittest.main()
