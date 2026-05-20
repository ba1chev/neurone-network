import unittest

from source.expressions.constant import Constant
from source.expressions.expression import Expression


class TestConstant(unittest.TestCase):
    def test_is_expression(self):
        self.assertIsInstance(Constant(1.0), Expression)

    def test_forward_returns_value(self):
        self.assertEqual(Constant(3.5).forward(), 3.5)

    def test_forward_with_zero(self):
        self.assertEqual(Constant(0.0).forward(), 0.0)

    def test_forward_with_negative(self):
        self.assertEqual(Constant(-7.25).forward(), -7.25)

    def test_forward_is_idempotent(self):
        c = Constant(2.0)
        self.assertEqual(c.forward(), 2.0)
        self.assertEqual(c.forward(), 2.0)

    def test_backward_is_noop_and_returns_none(self):
        c = Constant(4.0)
        self.assertIsNone(c.backward(1.0))
        self.assertIsNone(c.backward(-99.0))
        self.assertEqual(c.forward(), 4.0)

    def test_repr(self):
        self.assertEqual(repr(Constant(2.0)), "Const(value=2.0)")


if __name__ == "__main__":
    unittest.main()
