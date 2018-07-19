# test_stats_spatial.py

import os
import pickle
import unittest

import numpy as np
import pandas as pd

from context import (
    fly, ts, centroid, spatial, transforms
)

class TestSpatialFunctions(unittest.TestCase):
    def setUp(self):
        self.test_male = fly.Fly().from_csv('data/test-male-spatial-01.csv')
        self.test_female = fly.Fly().from_csv('data/test-female-spatial-01.csv')
        self.expected = pd.read_csv('data/expected-spatial-01.csv')

    def test_nearest_neighbor_centroid_01(self):
        """Tests nearest_neighbor_centroid() with normal data."""
        np.testing.assert_allclose(
            spatial.nearest_neighbor_centroid(self.test_male, self.test_female),
            self.expected['r'],
            atol=1e-3
        )

    def test_relative_position_01(self):
        """Tests relative_position() with normal data."""
        theta, r = spatial.relative_position2(
            self.test_male, self.test_female
        )
        np.testing.assert_allclose(
            theta,
            self.expected['theta'],
            atol=0.01
        )
        np.testing.assert_allclose(
            r,
            self.expected['r'],
            atol=0.01
        )

    def test_relative_position2_01(self):
        """Tests relative_position2() with normal data."""
        theta, r = spatial.relative_position2(
            self.test_male, self.test_female
        )
        np.testing.assert_allclose(
            theta,
            self.expected['theta'],
            atol=0.01
        )
        np.testing.assert_allclose(
            r,
            self.expected['r'],
            atol=0.01
        )

    def test_relative_orientation(self):
        spatial.relative_orientation(
            self.test_male,
            self.test_female
        )

    def test_abs_reative_orientation(self):
        spatial.abs_relative_orientation(
            self.test_male,
            self.test_female
        )

    def test_nose_and_tail_to_ellipse(self):
        spatial.nose_and_tail_to_ellipse(
            self.test_male,
            self.test_female
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)

