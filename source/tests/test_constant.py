import unittest
import numpy as np

from source.expressions.constant import Constant


class TestConstant(unittest.TestCase):
    def test_value_stored_as_float64(self):
        constant = Constant(np.array([1, 2, 3]))
        self.assertEqual(constant._value.dtype, np.float64)

    def test_forward_returns_value(self):
        constant = Constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(constant.forward(), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_backward_is_no_op(self):
        constant = Constant(np.array([1.0]))
        self.assertIsNone(constant.backward(np.array([5.0])))

    def test_collect_variables_yields_nothing(self):
        constant = Constant(np.array([1.0, 2.0]))
        out: list = []
        seen: set = set()
        constant._collect_variables(out, seen)
        self.assertEqual(out, [])

    def test_repr_contains_shape(self):
        constant = Constant(np.array([[1.0, 2.0]]))
        self.assertIn("(1, 2)", repr(constant))


if __name__ == "__main__":
    unittest.main()
