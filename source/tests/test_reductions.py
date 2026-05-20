import unittest
import numpy as np

from source.expressions.variable import Variable
from source.expressions.reduction_expressions.sum_reduction_expression import SumReductionExpression
from source.expressions.reduction_expressions.mean_reduction_expression import MeanReductionExpression
from source.expressions.reduction_expressions.log_sum_exp_reduction_expression import LogSumExpReductionExpression


class TestSumReductionExpression(unittest.TestCase):
    def test_forward_full_reduction(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "x")
        result = SumReductionExpression(x).forward()
        np.testing.assert_array_equal(result, np.array(10.0))

    def test_forward_axis_0(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "x")
        result = SumReductionExpression(x, axis=0).forward()
        np.testing.assert_array_equal(result, np.array([4.0, 6.0]))

    def test_backward_full_reduction_broadcasts_scalar(self):
        x = Variable(np.zeros((2, 3)), "x")
        expression = SumReductionExpression(x)
        expression.forward()
        expression.backward(np.array(5.0))
        np.testing.assert_array_equal(x.gradient, np.full((2, 3), 5.0))

    def test_backward_axis_reduction_broadcasts_along_axis(self):
        x = Variable(np.zeros((2, 3)), "x")
        expression = SumReductionExpression(x, axis=0)
        expression.forward()
        expression.backward(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(x.gradient, np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))


class TestMeanReductionExpression(unittest.TestCase):
    def test_forward_full_reduction(self):
        x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]), "x")
        result = MeanReductionExpression(x).forward()
        np.testing.assert_allclose(result, np.array(2.5))

    def test_forward_axis_1(self):
        x = Variable(np.array([[1.0, 3.0], [2.0, 4.0]]), "x")
        result = MeanReductionExpression(x, axis=1).forward()
        np.testing.assert_allclose(result, np.array([2.0, 3.0]))

    def test_backward_full_reduction_divides_by_count(self):
        x = Variable(np.zeros((2, 3)), "x")
        expression = MeanReductionExpression(x)
        expression.forward()
        expression.backward(np.array(6.0))
        np.testing.assert_allclose(x.gradient, np.full((2, 3), 1.0))

    def test_backward_axis_reduction_divides_by_axis_size(self):
        x = Variable(np.zeros((4, 3)), "x")
        expression = MeanReductionExpression(x, axis=0)
        expression.forward()
        expression.backward(np.ones(3))
        np.testing.assert_allclose(x.gradient, np.full((4, 3), 0.25))


class TestLogSumExpReductionExpression(unittest.TestCase):
    def test_forward_matches_naive_for_small_values(self):
        x = Variable(np.array([[1.0, 2.0, 3.0]]), "x")
        result = LogSumExpReductionExpression(x, axis=1).forward()
        expected = np.log(np.exp(np.array([1.0, 2.0, 3.0])).sum())
        np.testing.assert_allclose(result, np.array([expected]))

    def test_forward_stable_for_large_logits(self):
        x = Variable(np.array([[1000.0, 999.0, 998.0]]), "x")
        result = LogSumExpReductionExpression(x, axis=1).forward()
        # naive np.log(np.exp(...).sum()) overflows; ours should give ~ 1000.4076
        self.assertFalse(np.isinf(result).any())
        self.assertAlmostEqual(float(result[0]), 1000.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-2.0)), places=10)

    def test_backward_equals_softmax(self):
        x = Variable(np.array([1.0, 2.0, 3.0]), "x")
        expression = LogSumExpReductionExpression(x)
        expression.forward()
        expression.backward(np.array(1.0))
        exps = np.exp(np.array([1.0, 2.0, 3.0]))
        softmax = exps / exps.sum()
        np.testing.assert_allclose(x.gradient, softmax)

    def test_backward_axis_reduction_yields_per_row_softmax(self):
        x = Variable(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), "x")
        expression = LogSumExpReductionExpression(x, axis=1)
        expression.forward()
        expression.backward(np.ones(2))
        row_zero = np.exp(np.array([1.0, 2.0, 3.0]))
        row_zero = row_zero / row_zero.sum()
        np.testing.assert_allclose(x.gradient[0], row_zero)
        np.testing.assert_allclose(x.gradient[1], np.full(3, 1.0 / 3.0))


class TestReductionCollectVariables(unittest.TestCase):
    def test_propagates_to_child(self):
        x = Variable(np.array([1.0, 2.0]), "x")
        self.assertIn(x, SumReductionExpression(x).get_variables())
        self.assertIn(x, MeanReductionExpression(x).get_variables())
        self.assertIn(x, LogSumExpReductionExpression(x).get_variables())


if __name__ == "__main__":
    unittest.main()
