import unittest
import numpy as np

from source.expressions.variable import Variable


class TestVariableBasics(unittest.TestCase):
    def test_value_stored_as_float64(self):
        variable = Variable(np.array([1, 2, 3]), "x")
        self.assertEqual(variable.value.dtype, np.float64)

    def test_forward_returns_value(self):
        variable = Variable(np.array([1.0, 2.0]), "x")
        np.testing.assert_array_equal(variable.forward(), np.array([1.0, 2.0]))

    def test_initial_gradient_is_zeros_like_value(self):
        variable = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "x")
        np.testing.assert_array_equal(variable.gradient, np.zeros((2, 2)))

    def test_repr_contains_name_and_shape(self):
        variable = Variable(np.array([1.0, 2.0]), "weight_one")
        text = repr(variable)
        self.assertIn("weight_one", text)
        self.assertIn("(2,)", text)


class TestVariableGradientAccumulation(unittest.TestCase):
    def test_backward_accumulates_gradient(self):
        variable = Variable(np.array([1.0, 1.0]), "x")
        variable.backward(np.array([0.5, 0.5]))
        variable.backward(np.array([0.25, 0.25]))
        np.testing.assert_array_almost_equal(variable.gradient, np.array([0.75, 0.75]))

    def test_zero_gradient_resets_to_zeros(self):
        variable = Variable(np.array([1.0, 1.0]), "x")
        variable.backward(np.array([5.0, 5.0]))
        variable.zero_gradient()
        np.testing.assert_array_equal(variable.gradient, np.zeros(2))

    def test_value_setter_recasts_to_float64(self):
        variable = Variable(np.array([1.0]), "x")
        variable.value = np.array([42], dtype=np.int32)
        self.assertEqual(variable.value.dtype, np.float64)

    def test_collect_variables_dedups_by_id(self):
        variable = Variable(np.array([1.0]), "x")
        out: list = []
        seen: set = set()
        variable._collect_variables(out, seen)
        variable._collect_variables(out, seen)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
