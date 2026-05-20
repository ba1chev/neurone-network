import unittest

from source.expressions.variable import Variable
from source.expressions.expression import Expression


class TestVariable(unittest.TestCase):
    def test_is_expression(self):
        self.assertIsInstance(Variable(1.0, "x"), Expression)

    def test_forward_returns_value(self):
        self.assertEqual(Variable(2.5, "x").forward(), 2.5)

    def test_initial_gradient_is_zero(self):
        self.assertEqual(Variable(1.0, "x").gradient, 0.0)

    def test_value_property(self):
        v = Variable(3.0, "x")
        self.assertEqual(v.value, 3.0)

    def test_value_setter_updates_forward(self):
        v = Variable(1.0, "x")
        v.value = 5.0
        self.assertEqual(v.value, 5.0)
        self.assertEqual(v.forward(), 5.0)

    def test_backward_accumulates_gradient(self):
        v = Variable(1.0, "x")
        v.backward(2.0)
        v.backward(3.0)
        self.assertEqual(v.gradient, 5.0)

    def test_backward_with_negative_gradient(self):
        v = Variable(1.0, "x")
        v.backward(4.0)
        v.backward(-1.5)
        self.assertEqual(v.gradient, 2.5)

    def test_gradient_setter(self):
        v = Variable(1.0, "x")
        v.gradient = 7.0
        self.assertEqual(v.gradient, 7.0)

    def test_zero_gradient_resets(self):
        v = Variable(1.0, "x")
        v.backward(10.0)
        self.assertEqual(v.gradient, 10.0)
        v.zero_gradient()
        self.assertEqual(v.gradient, 0.0)

    def test_zero_gradient_after_setter(self):
        v = Variable(1.0, "x")
        v.gradient = 4.0
        v.zero_gradient()
        self.assertEqual(v.gradient, 0.0)

    def test_forward_does_not_change_gradient(self):
        v = Variable(2.0, "x")
        v.forward()
        v.forward()
        self.assertEqual(v.gradient, 0.0)

    def test_repr(self):
        self.assertEqual(repr(Variable(2.0, "x")), "Var(value=2.0, name=x)")


if __name__ == "__main__":
    unittest.main()
