# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import pandas as pd

from context import fly
from context import behavior


class TestPoint(unittest.TestCase):
    """Tests functions held within Point class."""
    def test_coords(self):
        """Assure call to Point.coords_rc() returns correct, [N,2] array."""
        test_point = fly.Point()

        test_point.row = np.arange(5)         # [0, 1, 2, 3, 4]
        test_point.col = np.arange(5)[::-1]   # [4, 3, 2, 1, 0]

        assert_eq_arr = np.array([
            [0, 4],
            [1, 3],
            [2, 2],
            [3, 1],
            [4, 0]
        ])

        test_coords = test_point.coords_rc()
        np.testing.assert_array_equal(test_coords, assert_eq_arr)


class TestEllipseInit(unittest.TestCase):
    """Tests functions assocaited with initialization of Ellipse objects."""
    def test_init_params(self):
        """Assure call to Ellipse.init_params() allocates appropriate space
        for all parameters."""
        size = 10
        test_ellipse = fly.Ellipse()
        test_ellipse.init_params(size)

        self.assertEqual(test_ellipse.centroid.row.size, size)
        self.assertEqual(test_ellipse.centroid.col.size, size)
        self.assertEqual(test_ellipse.major_axis_length.size, size)
        self.assertEqual(test_ellipse.minor_axis_length.size, size)
        self.assertEqual(test_ellipse.orientation.size, size)

class TestBodyInit(unittest.TestCase):
    """Tests functions associated with initialization of Body objects."""
    def test_init_params(self):
        """Assure call to Body.init_params() allocates appropriate space
        for all parameters"""
        size = 10
        fly_body = fly.Body()
        fly_body.init_params(size)

        self.assertEqual(fly_body.centroid.row.size, size)
        self.assertEqual(fly_body.centroid.col.size, size)
        self.assertEqual(fly_body.major_axis_length.size, size)
        self.assertEqual(fly_body.minor_axis_length.size, size)
        self.assertEqual(fly_body.orientation.size, size)
        self.assertEqual(fly_body.rotation_angle.size, size)
        self.assertEqual(fly_body.head.row.size, size)
        self.assertEqual(fly_body.head.col.size, size)
        self.assertEqual(fly_body.rear.row.size, size)
        self.assertEqual(fly_body.rear.col.size, size)


class TestFlyInit(unittest.TestCase):
    """Tests functions associated with initialization of Fly objects."""
    def test_init_params(self):
        """Assure call to Fly.init_params() allocates appropriate space for
        all parameters."""
        size = 10
        test_fly = fly.Fly()
        test_fly.init_params(size)

        attributes = ['body', 'left_wing', 'right_wing']
        for attribute in attributes:
            attr = getattr(test_fly, attribute)

            self.assertEqual(attr.centroid.row.size, size)
            self.assertEqual(attr.centroid.col.size, size)
            self.assertEqual(attr.major_axis_length.size, size)
            self.assertEqual(attr.minor_axis_length.size, size)
            self.assertEqual(attr.orientation.size, size)

            if attribute == 'body':
                self.assertEqual(attr.rotation_angle.size, size)
                self.assertEqual(attr.head.row.size, size)
                self.assertEqual(attr.head.col.size, size)
                self.assertEqual(attr.rear.row.size, size)
                self.assertEqual(attr.rear.col.size, size)


class TestFlyIO(unittest.TestCase):
    """Tests functions associated with loading and saving Fly objects."""
    def setUp(self):
        """Define col_names to be used in subsequent tests."""
        self.col_names = [
            'body_centroid_col',
            'body_centroid_row',
            'body_head_col',
            'body_head_row',
            'body_major_axis_length',
            'body_minor_axis_length',
            'body_orientation',
            'body_rear_col',
            'body_rear_row',
            'body_rotation_angle',
            'left_centroid_col',
            'left_centroid_row',
            'left_major_axis_length',
            'left_minor_axis_length',
            'left_orientation',
            'right_centroid_col',
            'right_centroid_row',
            'right_major_axis_length',
            'right_minor_axis_length',
            'right_orientation',
            'timestamps'
        ]

    def test_to_csv_no_behaviors(self):
        """Assure Fly.to_csv() works when no behaviors are associated with
        a fly."""
        size = 10
        test_fly = fly.Fly()
        test_fly.init_params(size)

        # save fly as a .csv file.
        savename = '_temp_test_fly.csv'
        test_fly.to_csv(savename)

        # re-load this file, and check that the column names are as
        # expected.
        fly_df = pd.read_csv(savename)
        df_col_names = fly_df.columns.values
        np.testing.assert_array_equal(self.col_names, df_col_names)

        # also make sure the saved df is the expected shape
        self.assertEqual(size * len(self.col_names), fly_df.size)

        # finally, get rid of the file we just created.
        os.remove(savename)

    def test_to_csv_behaviors(self):
        """Assure Fly.to_csv() works when behaviors are associated with
        a fly."""
        size = 10
        test_fly = fly.Fly()
        test_fly.init_params(size)

        behavior_1 = behavior.Behavior('b1', size, [2, 8], [4, 9])
        behavior_2 = behavior.Behavior('b2', size, [0, 5], [3, 8])

        test_fly.behaviors = [behavior_1, behavior_2]

        col_names = self.col_names +  ['behavior_b1', 'behavior_b2']

        savename = '_temp_test_fly.csv'
        test_fly.to_csv(savename)

        fly_df = pd.read_csv(savename)
        df_col_names = fly_df.columns.values
        np.testing.assert_array_equal(col_names, df_col_names)

        self.assertEqual(size * len(col_names), fly_df.size)
        os.remove(savename)

    def test_from_csv_no_behaviors(self):
        """Assure that Fly.from_csv() works when loading Fly data with no
        associated behaviors."""
        # this file contains a fly initilized to contain 20 frames (rows).
        # the first 20x20 rowsxcolumns are an eye-matrix with shape.
        # the last column are fake timestamps.
        filename = 'data/test-fly-no-behaviors-01.csv'
        test_fly = fly.Fly().from_csv(filename)
        test_fly_df = test_fly.to_df()

        np.testing.assert_array_equal(test_fly_df.values[:, :20], np.eye(20))
        np.testing.assert_array_equal(
            test_fly.timestamps, test_fly_df.values[:, -1]
        )

    def test_from_csv_behaviors(self):
        """Assure that Fly.from_csv() works when loading Fly data with
        associated behaviors."""
        # this file contains a fly initialized to contain 22 frames (rows).
        # it contains two behaviors called 'b1' and 'b2'; and the
        # data in the file is an (almost) eye-matrix with shape [22, 23];
        # column 20 contains fake timestamps from 1..22.
        filename = 'data/test-fly-behaviors-01.csv'
        test_fly = fly.Fly().from_csv(filename)
        test_fly_df = test_fly.to_df()

        i_matrix = np.eye(22)
        i_matrix = np.insert(i_matrix, 20, np.arange(1, 23), axis=1)
        np.testing.assert_array_equal(test_fly_df.values, i_matrix)

        b1_expected = np.zeros(22)
        b2_expected = np.zeros(22)
        b1_expected[-2] = 1
        b2_expected[-1] = 1
        np.testing.assert_array_equal(
            b1_expected, test_fly.behaviors[0].as_array())
        np.testing.assert_array_equal(
            b2_expected, test_fly.behaviors[1].as_array())

        np.testing.assert_array_equal(
            test_fly.timestamps, np.arange(1,23)
        )


class TestFlyBehaviorFunctions(unittest.TestCase):
    """Tests functions related to getting behaviors from Fly."""
    def test_get_behavior1(self):
        """Assures that fly.get_behavior() works with valid behavior name."""
        filename = 'data/test-fly-behaviors-01.csv'
        test_fly = fly.Fly().from_csv(filename)

        behav1 = test_fly.get_behavior('b1')
        behav2 = test_fly.get_behavior('b2')
        self.assertEqual(behav1.name, 'b1')
        self.assertEqual(behav2.name, 'b2')

    def test_get_behavior2(self):
        """Assures that fly.get_behavior() raises Error with invalid name."""
        filename = 'data/test-fly-behaviors-01.csv'
        test_fly = fly.Fly().from_csv(filename)

        with self.assertRaises(AttributeError):
            test_fly.get_behavior('wrongkey')

    def test_list_behaviors(self):
        """Assures that fly.list_behaviors() works with valid behaviors."""
        filename = 'data/test-fly-behaviors-01.csv'
        test_fly = fly.Fly().from_csv(filename)

        np.testing.assert_array_equal(test_fly.list_behaviors(), ['b1', 'b2'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
