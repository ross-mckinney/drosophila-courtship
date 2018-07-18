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
        """Checks angular_velocity() with normal data."""
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

    def test_abs_angular_velocity_01(self):
        """Checks abs_angular_velocity() with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(5)
        test_fly.body.orientation = np.array([1, 2, 1.5, 2.5, 2])
        test_fly.timestamps = np.arange(5)

        ang_vel = centroid.angular_velocity(test_fly)
        abs_ang_vel = centroid.abs_angular_velocity(test_fly)
        expect_arr = np.array([0, 1, -0.5, 1, -0.5])
        np.testing.assert_array_equal(
            ang_vel,
            expect_arr
        )

        np.testing.assert_array_equal(
            abs_ang_vel,
            np.abs(expect_arr)
        )

    def test_centroid_to_arena_edge(self):
        """Checks that centroid_to_arena_edge() works with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.centroid.row = np.array([3, 3])
        test_fly.body.centroid.col = np.array([2 ,3])

        arena_center = (0, 0)
        arena_radius = 5

        dists = centroid.centroid_to_arena_edge(
            test_fly,
            arena_center,
            arena_radius)
        np.testing.assert_array_equal(
            dists,
            arena_radius - np.sqrt([13, 18])
        )


if __name__ == '__main__':
    # unittest.main(verbosity=2)
    with open('data/test-fly-tracked-01.fcts', 'rb') as f:
        ts = pickle.load(f)

    test_fly1 = fly.Fly()
    test_fly1.init_params(5)
    test_fly1.body.centroid.row = ts.male.body.centroid.row[:5]
    test_fly1.body.centroid.col = ts.male.body.centroid.col[:5]
    test_fly1.body.head.row = ts.male.body.head.row[:5]
    test_fly1.body.head.col = ts.male.body.head.col[:5]
    test_fly1.body.rear.row = ts.male.body.rear.row[:5]
    test_fly1.body.rear.col = ts.male.body.rear.col[:5]
    test_fly1.body.rotation_angle = ts.male.body.rotation_angle[:5]

    test_fly2 = fly.Fly()
    test_fly2.init_params(5)
    test_fly2.body.centroid.row = ts.male.body.centroid.col[:5]
    test_fly2.body.centroid.col = ts.male.body.centroid.row[:5]
    test_fly2.body.head.row = ts.male.body.head.col[:5]
    test_fly2.body.head.col = ts.male.body.head.row[:5]
    test_fly2.body.rear.row = ts.male.body.rear.col[:5]
    test_fly2.body.rear.col = ts.male.body.rear.row[:5]
    test_fly2.body.rotation_angle = ts.male.body.rotation_angle[:5]

    vt1, vp1 = centroid.component_velocities(test_fly1)
    vt2, vp2 = centroid.component_velocities(test_fly2)

    print 'test_fly1:\n\tvt: ',
    print vt1
    print '\tvp: ',
    print vp1 

    print 'test_fly2:\n\tvt: ',
    print vt2
    print '\tvp: ',
    print vp2 