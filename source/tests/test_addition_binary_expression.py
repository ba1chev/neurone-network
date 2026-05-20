import unittest

from source.expressions.constant import Constant
from source.expressions.variable import Variable
from source.expressions.binary_expressions.addition_binary_expression import AdditionBinaryExpression


class TestAdditionBinaryExpression(unittest.TestCase):
    def test_forward_two_constants(self):
        expr = AdditionBinaryExpression(Constant(2.0), Constant(3.0))
        self.assertEqual(expr.forward(), 5.0)

    def test_forward_with_negative(self):
        expr = AdditionBinaryExpression(Constant(-4.0), Constant(1.5))
        self.assertEqual(expr.forward(), -2.5)

    def test_forward_two_variables(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = AdditionBinaryExpression(a, b)
        self.assertEqual(expr.forward(), 3.0)

    def test_backward_propagates_same_gradient_to_both(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = AdditionBinaryExpression(a, b)
        expr.forward()
        expr.backward(1.0)
        self.assertEqual(a.gradient, 1.0)
        self.assertEqual(b.gradient, 1.0)

    def test_backward_with_arbitrary_gradient(self):
        a = Variable(1.0, "a")
        b = Variable(2.0, "b")
        expr = AdditionBinaryExpression(a, b)
        expr.forward()
        expr.backward(2.5)
        self.assertEqual(a.gradient, 2.5)
        self.assertEqual(b.gradient, 2.5)

    def test_backward_constant_operand_is_noop(self):
        a = Variable(1.0, "a")
        expr = AdditionBinaryExpression(a, Constant(5.0))
        expr.forward()
        expr.backward(3.0)
        self.assertEqual(a.gradient, 3.0)

    def test_backward_same_variable_on_both_sides_accumulates(self):
        x = Variable(2.0, "x")
        expr = AdditionBinaryExpression(x, x)
        self.assertEqual(expr.forward(), 4.0)
        expr.backward(1.0)
        self.assertEqual(x.gradient, 2.0)


if __name__ == "__main__":
    unittest.main()
