# test_stats_centroid.py

import os
import pickle
import unittest

import numpy as np
import pandas as pd

from context import (fly, centroid)


class TestCentroidFunctions(unittest.TestCase):
    def setUp(self):
        filename = 'data/test-fly-centroid.csv'
        self.test_fly = fly.Fly().from_csv(filename)

    def test_angular_velocity_01(self):
        """Checks that orientations read from .csv test-fly are correctly 
        processed by centroid.angular_velocity()"""
        ang_vel = centroid.angular_velocity(self.test_fly)
        eq_to = np.ones(22)
        eq_to[0] = 0
        np.testing.assert_array_equal(
            ang_vel, eq_to
        )

    def test_angular_velocity_02(self):
        """Check with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(5)
        test_fly.body.orientation = np.array([1, 1.25, 1.3, 1.6, 1.65])
        test_fly.timestamps = np.array([1, 1.5, 1.75, 2, 2.5])
        # not sure why these arrays aren't equal when using
        # assert_array_equal?!
        np.testing.assert_allclose(
            centroid.angular_velocity(test_fly),
            np.array([0, 0.5, 0.2, 1.2, 0.1])
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)