import unittest

import source.expressions.operators  # registers operators on Expression
from source.expressions.constant import Constant
from source.expressions.variable import Variable


class TestVariableCollection(unittest.TestCase):
    def test_single_variable(self):
        x = Variable(1.0, "x")
        self.assertEqual(x.get_variables(), [x])

    def test_constant_has_no_variables(self):
        self.assertEqual(Constant(3.0).get_variables(), [])

    def test_binary_collects_both_sides(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = a + b
        result = expr.get_variables()
        self.assertEqual(len(result), 2)
        self.assertIn(a, result)
        self.assertIn(b, result)

    def test_constants_filtered_out(self):
        a = Variable(1.0, "a")
        expr = a * 5 + 2
        result = expr.get_variables()
        self.assertEqual(result, [a])

    def test_unary_descends(self):
        x = Variable(1.0, "x")
        expr = -x
        self.assertEqual(expr.get_variables(), [x])

    def test_deep_tree(self):
        a, b, c = Variable(1.0, "a"), Variable(2.0, "b"), Variable(3.0, "c")
        expr = (a + b) * c - a / b
        result = expr.get_variables()
        self.assertEqual(len(result), 3)
        for v in (a, b, c):
            self.assertIn(v, result)

    def test_shared_variable_deduplicated(self):
        x = Variable(2.0, "x")
        expr = x * x + x
        result = expr.get_variables()
        self.assertEqual(result, [x])


class TestZeroAllGradients(unittest.TestCase):
    def test_zeros_single_variable(self):
        x = Variable(3.0, "x")
        expr = x * x
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, 6.0)
        expr.zero_all_gradients()
        self.assertEqual(x.gradient, 0.0)

    def test_zeros_multiple_variables(self):
        a = Variable(2.0, "a")
        b = Variable(3.0, "b")
        c = Variable(4.0, "c")
        expr = a * b + c
        expr.forward()
        expr.backward(1.0)
        self.assertNotEqual(a.gradient, 0.0)
        self.assertNotEqual(b.gradient, 0.0)
        self.assertNotEqual(c.gradient, 0.0)
        expr.zero_all_gradients()
        self.assertEqual(a.gradient, 0.0)
        self.assertEqual(b.gradient, 0.0)
        self.assertEqual(c.gradient, 0.0)

    def test_does_not_touch_unrelated_variables(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        b.gradient = 7.0
        expr = a + 1
        expr.zero_all_gradients()
        self.assertEqual(a.gradient, 0.0)
        self.assertEqual(b.gradient, 7.0)

    def test_idempotent_when_already_zero(self):
        x = Variable(1.0, "x")
        expr = x * 2
        expr.zero_all_gradients()
        self.assertEqual(x.gradient, 0.0)
        expr.zero_all_gradients()
        self.assertEqual(x.gradient, 0.0)

    def test_supports_repeated_train_step_pattern(self):
        x = Variable(2.0, "x")
        expr = x ** 2
        for _ in range(3):
            expr.zero_all_gradients()
            expr.forward()
            expr.backward(1.0)
            self.assertAlmostEqual(x.gradient, 4.0)

    def test_handles_shared_variable_correctly(self):
        x = Variable(3.0, "x")
        expr = x * x
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(x.gradient, 6.0)
        expr.zero_all_gradients()
        self.assertEqual(x.gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
