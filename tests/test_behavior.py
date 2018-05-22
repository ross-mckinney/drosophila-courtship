# -*- coding: utf-8 -*-
import unittest

import numpy as np

from context import behavior

class TestBehavior(unittest.TestCase):
    """Tests to make sure Behavior functions work properly."""
    def test_init1(self):
        """Assures attributes are set given an normal initialization."""
        behav1 = behavior.Behavior('behavior_1', 100, [50, 70], [60, 80])

        np.testing.assert_array_equal(behav1.start_ixs, np.array([50, 70]))
        np.testing.assert_array_equal(behav1.stop_ixs, np.array([60, 80]))
        self.assertEqual(behav1.name, 'behavior_1')
        self.assertEqual(behav1.length, 100)

    def test_init2(self):
        """Assures that stop indices longer than `length` are truncated."""
        behav = behavior.Behavior('b1', 100, [50, 80], [60, 120])
        np.testing.assert_array_equal(
            behav.start_ixs, [50, 80]
        )
        np.testing.assert_array_equal(
            behav.stop_ixs, [60, 100]
        )

    def test_init3(self):
        """Assures that start indices longer than `length` are truncated."""
        behav = behavior.Behavior('b1', 100, [50, 80, 120], [60, 100, 140])
        np.testing.assert_array_equal(
            behav.start_ixs, [50, 80]
        )
        np.testing.assert_array_equal(
            behav.stop_ixs, [60, 100]
        )

    def test_from_array1(self):
        """Assures a Behavior can be generated from a passed binary array."""
        arr = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        behav = behavior.Behavior.from_array('behavior1', arr)

        np.testing.assert_array_equal(behav.start_ixs, [3])
        np.testing.assert_array_equal(behav.stop_ixs, [6])
        self.assertEqual(behav.name, 'behavior1')
        self.assertEqual(behav.length, arr.size)

    def test_from_array2(self):
        """Assures a Behavior can be generated from an array of zeros."""
        arr = np.zeros(100)
        behav = behavior.Behavior.from_array('behavior2', arr)

        np.testing.assert_array_equal(behav.start_ixs, [])
        np.testing.assert_array_equal(behav.stop_ixs, [])
        self.assertEqual(behav.name, 'behavior2')
        self.assertEqual(behav.length, 100)

    def test_from_array3(self):
        """Assures a Behavior can be generated from an array with edge bouts."""
        arr1 = np.zeros(100)
        arr1[0] = 1
        behav1 = behavior.Behavior.from_array('b1', arr1)

        np.testing.assert_array_equal(
            behav1.ixs(),
            np.array([[0, 1]])
        )

        arr2 = np.zeros(100)
        arr2[-1] = 1
        behav2 = behavior.Behavior.from_array('b2', arr2)

        np.testing.assert_array_equal(
            behav2.ixs(),
            np.array([[99, 100]])
        )

    def test_ixs1(self):
        """Assures Behavior.ixs() returns expected array given start/stop
        indices upon initialization."""
        start_ixs = [20, 40, 60]
        stop_ixs = [30, 50, 70]
        behav = behavior.Behavior('behavior1', 100, start_ixs, stop_ixs)

        np.testing.assert_array_equal(
            behav.ixs(),
            np.array([start_ixs, stop_ixs]).T
        )

    def test_ixs2(self):
        """Assures Behavior.ixs() returns expected array given empty start/stop
        indices upon initialization."""
        behav = behavior.Behavior('b1', 100, [], [])
        self.assertEqual(behav.ixs().size, 0)
        np.testing.assert_array_equal(behav.ixs().shape, (0, 2))

    def test_as_array1(self):
        """Assures Behavior.as_array() works with non-empty behavior."""
        behav = behavior.Behavior('b1', 10, [2, 8], [4, 10])
        np.testing.assert_array_equal(
            behav.as_array(),
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
        )

    def test_as_array2(self):
        """Assures Behavior.as_array() works with empty behavior."""
        behav = behavior.Behavior('b1', 100, [], [])
        np.testing.assert_array_equal(
            behav.as_array(),
            np.zeros(100)
        )

    def test_index1(self):
        """Assures Behavior.index() `mode`s 'all' and 'condensed' works with
        bouts present."""
        behav1 = behavior.Behavior('b1', 100, [20, 80], [40, 100])
        self.assertEqual(behav1.index(mode='all'), 0.4)
        self.assertEqual(behav1.index(mode='condensed'), 0.5)

        behav2 = behavior.Behavior('b2', 10, [4, 8], [5, 10])
        self.assertEqual(behav2.index(mode='all'), 0.3)
        self.assertEqual(behav2.index(mode='condensed'), 0.5)

        behav3 = behavior.Behavior('b3', 10, [5], [10])
        self.assertEqual(behav3.index(mode='all'), 0.5)
        self.assertEqual(behav3.index(mode='condensed'), 1)

    def test_index2(self):
        """Assures Behavior.index() `mode`s 'all' and 'condensed' works with
        no bouts present."""
        behav = behavior.Behavior('b1', 10, [], [])
        self.assertEqual(behav.index(mode='all'), 0)
        self.assertEqual(behav.index(mode='condensed'), 0)

    def test_latency1(self):
        """Assures that Behavior.latency() works with bouts."""
        behav = behavior.Behavior('b', 10, [5], [10])
        self.assertEqual(behav.latency(), 5)

    def test_latency2(self):
        """Assures that Behavior.latency() works with no bouts."""
        behav = behavior.Behavior('b', 10, [], [])
        self.assertEqual(behav.latency(), 10)

    def test_bout_num1(self):
        """Assures that Behavior.bout_num() works with bouts."""
        behav1 = behavior.Behavior('b1', 10, [5], [10])
        self.assertEqual(behav1.bout_num(), 1)

        behav2 = behavior.Behavior('b2', 10, [2, 6, 8], [3, 7, 10])
        self.assertEqual(behav2.bout_num(), 3)

    def test_bout_num2(self):
        """Assures that Behavior.bout_num() works with no bouts."""
        behav1 = behavior.Behavior('b1', 10, [], [])
        self.assertEqual(behav1.bout_num(), 0)

    def test_bout_durations1(self):
        """Assures that Behavior.bout_durations() works with bouts."""
        behav1 = behavior.Behavior('b1', 10, [5], [10])
        np.testing.assert_array_equal(behav1.bout_durations(), [5])

        behav2 = behavior.Behavior('b2', 10, [2, 6, 8], [3, 7, 10])
        np.testing.assert_array_equal(behav2.bout_durations(), [1, 1, 2])

    def test_bout_durations2(self):
        """Assures that Behavior.bout_durations() works with no bouts."""
        behav1 = behavior.Behavior('b1', 10, [], [])
        np.testing.assert_array_equal(behav1.bout_durations(), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
