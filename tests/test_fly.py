# -*- coding: utf-8 -*-
import os
import sys
import logging
import unittest
import numpy as np
import pandas as pd
from context import fly


class TestPoint(unittest.TestCase):
    def test_coords(self):
        p = fly.Point()

        p.row = np.arange(5)         # [0, 1, 2, 3, 4]
        p.col = np.arange(5)[::-1]   # [4, 3, 2, 1, 0]

        assert_eq_arr = np.array([
            [0, 4],
            [1, 3],
            [2, 2],
            [3, 1],
            [4, 0]
        ])

        c = p.coords()
        np.testing.assert_array_equal(c, assert_eq_arr)


class TestEllipseInit(unittest.TestCase):
    def test_init_params(self):
        size = 10
        e = fly.Ellipse()
        e.init_params(size)

        self.assertEqual(e.centroid.row.size, size)
        self.assertEqual(e.centroid.col.size, size)
        self.assertEqual(e.major_axis_length.size, size)
        self.assertEqual(e.minor_axis_length.size, size)
        self.assertEqual(e.orientation.size, size)

class TestBodyInit(unittest.TestCase):
    def test_init_params(self):
        size = 10
        b = fly.Body()
        b.init_params(size)

        self.assertEqual(b.centroid.row.size, size)
        self.assertEqual(b.centroid.col.size, size)
        self.assertEqual(b.major_axis_length.size, size)
        self.assertEqual(b.minor_axis_length.size, size)
        self.assertEqual(b.orientation.size, size)
        self.assertEqual(b.rotation_angle.size, size)
        self.assertEqual(b.head.row.size, size)
        self.assertEqual(b.head.col.size, size)
        self.assertEqual(b.rear.row.size, size)
        self.assertEqual(b.rear.col.size, size)


class TestFlyInit(unittest.TestCase):
    def test_init_params(self):
        size = 10
        f = fly.Fly()
        f.init_params(size)

        attributes = ['body', 'left_wing', 'right_wing']
        for attribute in attributes:
            attr = getattr(f, attribute)

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
    def setUp(self):
        self.size = 10
        self.fly = fly.Fly()
        self.fly.init_params(self.size)

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
           'right_orientation'
        ]
    
    def test_to_csv(self):
        # save fly as a .csv file.
        savename = '_temp_test_fly.csv'
        self.fly.to_csv(savename)

        # re-load this file, and check that the column names are as
        # expected.
        fly_df = pd.read_csv(savename)
        df_col_names = fly_df.columns.values
        np.testing.assert_array_equal(self.col_names, df_col_names)

        # also make sure the saved df is the expected shape
        self.assertEqual(self.size * len(self.col_names), fly_df.size)

        # finally, get rid of the file we just created.
        os.remove(savename)

    def test_from_csv(self):
        filename = 'data/test-fly-01.csv'
        test_fly = fly.Fly().from_csv(filename)

if __name__ == '__main__':
    unittest.main(verbosity=2)