# test tracking.tracking

import unittest

import numpy as np
import matplotlib.pyplot as plt
import motmot.FlyMovieFormat.FlyMovieFormat as FMF

from context import (
    trk_track,
    trk_female,
    trk_arena,
    fly
)

class TestThresholding(unittest.TestCase):
    """Tests thresholding functions."""
    def test_high_pass_binary(self):
        """Assures High Pass filter (binary) working."""
        img1 = np.zeros(shape=(10, 10))
        img1[0, :] = 10
        img1[1, :] = 5

        # all indices less than 6 should be set to 0.
        bin_img1 = trk_track.high_pass_threshold_binary(
            img1, 6
        )
        self.assertEqual(np.sum(bin_img1), 10)
        arr_equal_to = np.zeros(shape=(10, 10))
        arr_equal_to[0, :] = 1
        np.testing.assert_array_equal(bin_img1, arr_equal_to)

        img2 = np.zeros(shape=(10, 10))
        img2[5, 5] = 1
        img2[5, 6] = 2

        # all indices less than 1 should be set to 0.
        bin_img2 = trk_track.high_pass_threshold_binary(
            img2, 1
        )
        arr_equal_to = np.zeros(shape=(10, 10))
        arr_equal_to[5, 6] = 1
        np.testing.assert_array_equal(bin_img2, arr_equal_to)

    def test_high_pass(self):
        """Assures High Pass filter working."""
        img1 = np.zeros(shape=(10, 10))
        img1[0, :] = 10
        img1[1, :] = 5

        thresh_img1 = trk_track.high_pass_threshold(
            img1, 6
        )

        arr_equal_to = np.zeros(shape=(10, 10))
        arr_equal_to[0, :] = 10
        np.testing.assert_array_equal(thresh_img1, arr_equal_to)

    def test_low_pass_binary(self):
        """Assures Low Pass filter (binary) working."""
        # all indices greater than 6 should be set to 0.
        x = np.ones(shape=(10, 10)) * 10
        x[0, :] = 5

        bin_img1 = trk_track.low_pass_threshold_binary(
            x, 6
        )
        self.assertEqual(np.sum(bin_img1), 10)
        arr_equal_to = np.zeros(shape=(10, 10))
        arr_equal_to[0, :] = 1
        np.testing.assert_array_equal(bin_img1, arr_equal_to)

    def test_low_pass(self):
        """Assures Low Pass filter working."""
        x = np.ones(shape=(10, 10)) * 10
        x[0, :] = 5

        thresh_img1 = trk_track.low_pass_threshold(
            x, 6
        )
        arr_equal_to = np.zeros(shape=(10, 10))
        arr_equal_to[0, :] = 5
        np.testing.assert_array_equal(thresh_img1, arr_equal_to)

    def test_subtract_background(self):
        """Assures background subtraction works."""
        img1 = np.ones(shape=(10,10))
        img1[0, :] = 10

        bg_img = np.ones(shape=(10,10))
        sub_img = trk_track.subtract_background(img1, bg_img)

        arr_equal_to = np.zeros(shape=(10,10))
        arr_equal_to[0, :] = 9
        np.testing.assert_array_equal(arr_equal_to, sub_img)


class TestTracking(unittest.TestCase):
    def setUp(self):
        """Loads video, arena, and female for testing tracking functions."""
        self.video = FMF.FlyMovie('D:/test_data/test_01.fmf')
        
        self.arena = trk_arena.CircularArena(self.video)
        self.arena.calculate_background()
        self.arena.center = (self.arena.height / 2, self.arena.width / 2)
        self.arena.radius = (self.arena.height / 2)

        self.female = trk_female.Female(self.arena)
        self.female.center = (238, 320)
        self.female.head = (233, 318)
        self.female.rear = (256, 281)
        self.female.maj_ax_rad = 28
        self.female.min_ax_rad = 14
        self.female.orientation = -15

        self.test_frame = long(6059)

    def test_find_male(self):
        img = self.video.get_frame(self.test_frame)[0]
        male_props = trk_track.find_male(img, self.female, self.arena, 35)

        self.assertIsNotNone(male_props)
        self.assertGreater(
            male_props.major_axis_length, male_props.minor_axis_length)
        # orientation is the only property that will be close to the 
        # value returned during tracking (which is loose_male_props;
        # returned from a call to find_wings())
        np.testing.assert_almost_equal(-0.758652508899, male_props.orientation)

    def test_find_female(self):
        img = self.video.get_frame(self.test_frame)[0]
        fem_props, fem_head, fem_rear = trk_track.find_female(
            img, self.female, 35
        )
        # np.testing.assert_allclose([343., 246.], fem_head)


if __name__ == '__main__':
    unittest.main(verbosity=2)