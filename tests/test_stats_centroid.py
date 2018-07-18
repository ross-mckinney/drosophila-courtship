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

    def test_component_velocity_01(self):
        """Checks component_velocity() with (all) movement along parallel axis."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.centroid.row = np.array([0, -1])
        test_fly.body.centroid.col = np.array([0, 1])
        test_fly.body.head.row = np.array([-1, -2])
        test_fly.body.head.col = np.array([1, 2])
        test_fly.body.rear.row = np.array([1, 0])
        test_fly.body.rear.col = np.array([-1, 0])
        test_fly.body.rotation_angle = np.array([45, 45]) * np.pi/180
        vt, vp = centroid.component_velocities(test_fly)
        np.testing.assert_allclose(
            vt,
            np.array([0, 0]),
            rtol=1e-5,
            atol=1e-7
        )
        np.testing.assert_allclose(
            vp,
            np.array([np.sqrt(2), np.sqrt(2)]),
            rtol=1e-5,
            atol=1e-7
        )

    def test_component_velocity_02(self):
        """Checks component_velocity() with (all) movement along tangential axis."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.centroid.row = np.array([0, 1])
        test_fly.body.centroid.col = np.array([0, 1])
        test_fly.body.head.row = np.array([-1, 0])
        test_fly.body.head.col = np.array([1, 2])
        test_fly.body.rear.row = np.array([1, 2])
        test_fly.body.rear.col = np.array([-1, 0])
        test_fly.body.rotation_angle = np.array([45, 45]) * np.pi/180
        vt, vp = centroid.component_velocities(test_fly)
        np.testing.assert_allclose(
            vp,
            np.array([0, 0]),
            rtol=1e-5,
            atol=1e-7
        )
        np.testing.assert_allclose(
            np.abs(vt),
            np.array([np.sqrt(2), np.sqrt(2)]),
            rtol=1e-5,
            atol=1e-7
        )

    def test_component_velocity_03(self):
        """Checks component_velocity() with equal movement along both axes."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.centroid.row = np.array([0, 0])
        test_fly.body.centroid.col = np.array([0, 1])
        test_fly.body.head.row = np.array([-1, -1])
        test_fly.body.head.col = np.array([1, 2])
        test_fly.body.rear.row = np.array([1, 1])
        test_fly.body.rear.col = np.array([-1, 0])
        test_fly.body.rotation_angle = np.array([45, 45]) * np.pi/180
        vt, vp = centroid.component_velocities(test_fly)
        np.testing.assert_allclose(
            vt, vp
        )

    def test_dmaj_axis_01(self):
        """Tests that change_in_major_axis_length() works with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.major_axis_length = np.array([0, 1])
        test_fly.timestamps = np.array([1, 2])
        np.testing.assert_array_equal(
            centroid.change_in_major_axis_length(test_fly),
            np.array([1])
        )

    def test_dmin_axis_01(self):
        """Tests that change_in_minor_axis_length() works with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.minor_axis_length = np.array([0, 1])
        test_fly.timestamps = np.array([1, 2])
        np.testing.assert_array_equal(
            centroid.change_in_minor_axis_length(test_fly),
            np.array([1])
        )

    def test_darea_01(self):
        """Tests that change_in_area() works with normal data."""
        test_fly = fly.Fly()
        test_fly.init_params(2)
        test_fly.body.major_axis_length = np.array([1, 2])
        test_fly.body.minor_axis_length = np.array([1, 1])
        test_fly.timestamps = np.array([1,2])

        a0 = np.pi * (0.5*1) * (0.5*1)
        a1 = np.pi * (0.5*2) * (0.5*1)
        da = a1 - a0
        np.testing.assert_array_equal(
            centroid.change_in_area(test_fly),
            [da]
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
