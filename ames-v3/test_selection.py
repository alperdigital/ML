import unittest

import numpy as np
from sklearn.model_selection import KFold

from run import BLEND_SPLIT_SEED, choose_model, crossfit_blend


class SelectionTests(unittest.TestCase):
    def costs(self, blend):
        return dict(v1_xgb=.129, v2_cat=.13, best_new=.131, v2=.1295,
                    equal_blend=.134, weighted_blend=blend)

    def test_simple_model_within_registered_tolerance(self):
        self.assertEqual(choose_model(self.costs(.1286)), "v1_xgb")

    def test_blend_with_clear_inner_gain(self):
        self.assertEqual(choose_model(self.costs(.1284)), "weighted_blend")

    def test_weight_scoring_cannot_fit_its_own_prices(self):
        prices = np.full(24, 100.)
        matrix = np.tile([100., 900.], (len(prices), 1))
        splits = list(KFold(3).split(prices))
        original, audit = crossfit_blend(prices, matrix, splits)
        self.assertEqual(sum(record["scored_rows"] for record in audit), len(prices))
        first_fold = splits[0][1]
        opposite, held_out = next(KFold(2, shuffle=True, random_state=BLEND_SPLIT_SEED).split(first_fold))
        prices[first_fold[held_out]] = 900.
        after, _ = crossfit_blend(prices, matrix, splits)
        np.testing.assert_allclose(after[first_fold[held_out]], original[first_fold[held_out]])
        np.testing.assert_allclose(after[splits[0][0]], original[splits[0][0]])
        np.testing.assert_allclose(after[first_fold[opposite]], 900., atol=.01)
