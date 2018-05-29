# test arena objects used in tracking.

import os
import unittest

import matplotlib.pyplot as plt
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np

from context import trk_arena


class TestArena(unittest.TestCase):
    """Tests common Arena functions."""
    def setUp(self):
        """Opens a video for use in test-cases."""
        self.video = FMF.FlyMovie('D:/test_data/test_01.fmf')

    def test_arena_init(self):
        """Assures we can initialize an arena using a test video."""
        arena = trk_arena.CircularArena(self.video)
        self.assertEqual(arena.height, 480)
        self.assertEqual(arena.width, 640)
        self.assertEqual(arena.n_frames, 14399)

    def test_calculate_background(self):
        """Assures we can calculate a background image from an Arena."""
        arena = trk_arena.CircularArena(self.video)

        arena.calculate_background()
        np.testing.assert_array_equal(
            arena.background_image.shape, (arena.height, arena.width))


class TestCircularArena(unittest.TestCase):
    """Tests CircularArena functions."""
    def setUp(self):
        """Opens a video for use in test-cases."""
        self.video = FMF.FlyMovie('D:/test_data/test_01.fmf')
        self.arena = trk_arena.CircularArena(self.video)
        self.arena.calculate_background()

        # set estimated arena center/radius
        self.arena.center = (self.arena.height / 2, self.arena.width / 2)
        self.arena.radius = (self.arena.height / 2)

    def test_draw_arena(self):
        """Makes sure that we can draw the outline of a CircularArena."""
        img = self.arena.draw_arena(color=(255, 0, 0), thickness=2)
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.savefig('figures/test_draw_arena.png')

    def test_get_arena_mask(self):
        """Assures we can generate a mask image of a CircularArena."""
        mask_img = self.arena.get_arena_mask()
        fig, ax = plt.subplots()
        ax.imshow(mask_img)
        fig.savefig('figures/test_arena_mask.png')

        np.testing.assert_array_equal(
            mask_img.shape,
            self.arena.background_image.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
