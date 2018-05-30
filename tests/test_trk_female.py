# test tracking.female

import unittest

import numpy as np
import matplotlib.pyplot as plt
import motmot.FlyMovieFormat.FlyMovieFormat as FMF

from context import (
    trk_arena,
    trk_female
)


class TestFemale(unittest.TestCase):
    """Tests functions in courtship.tracking.female.py"""
    def setUp(self):
        """Opens a video & sets an arena for use in test-cases."""
        self.video = FMF.FlyMovie('D:/test_data/test_01.fmf')
        self.arena = trk_arena.CircularArena(self.video)
        self.arena.calculate_background()

        # set estimated arena center/radius
        self.arena.center = (self.arena.height / 2, self.arena.width / 2)
        self.arena.radius = (self.arena.height / 2)

    def _get_female(self):
        female = trk_female.Female(self.arena)

        # "fake" female coordinates/rotation
        female.center = (238, 320)
        female.head = (233, 318)
        female.rear = (256, 281)
        female.maj_ax_rad = 28
        female.min_ax_rad = 14
        female.orientation = -15

        return female

    def test_init_female(self):
        """Assures initialization of Female works."""
        female = trk_female.Female(self.arena)
        self.assertEqual(female.settings_valid(), False)

    def test_get_female_mask(self):
        """Assures generation of female mask works."""
        female = self._get_female()

        mask = female.get_female_mask()

        fig, ax = plt.subplots()
        ax.imshow(mask)
        fig.savefig('figures/test_female_mask.png')

        np.testing.assert_array_equal(
            mask.shape,
            female.arena.background_image.shape
        )

    def test_draw_female(self):
        """Assures drawing ellipse fitted to female works."""
        female = self._get_female()

        female_img = female.draw_female()

        fig, ax = plt.subplots()
        ax.imshow(female_img)
        fig.savefig('figures/test_draw_female.png')

        np.testing.assert_array_equal(
            female_img.shape[:-1],
            female.arena.background_image.shape
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
