import unittest
from run import choose_model


class SelectionTests(unittest.TestCase):
    def costs(self, blend):
        return dict(v1_xgb=.129, v2_cat=.13, best_new=.131, v2=.1295,
                    equal_blend=.134, weighted_blend=blend)

    def test_simple_model_within_registered_tolerance(self):
        self.assertEqual(choose_model(self.costs(.1286)), "v1_xgb")

    def test_blend_with_clear_inner_gain(self):
        self.assertEqual(choose_model(self.costs(.1284)), "weighted_blend")
