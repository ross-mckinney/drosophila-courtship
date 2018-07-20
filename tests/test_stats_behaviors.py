# test_stats_behaviors.py

import os
import pickle
import unittest

import numpy as np
import pandas as pd

from context import (
    fly, centroid, behavior, behaviors, ts
    )


class TestBehavioralStats(unittest.TestCase):
    def test_behavioral_index_01(self):
        """Tests behavioral_index() with normal data."""
        arr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        self.assertEqual(
            behaviors.behavioral_index(arr),
            5./10
        )
        self.assertEqual(
            behaviors.behavioral_index(arr, method='condensed'),
            5./6
        )

    def test_behavioral_latency_01(self):
        """Tests behavioral_latency() with normal data."""
        arr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        self.assertEqual(
            behaviors.behavioral_latency(arr),
            4
        )

    def test_bout_boundaries_01(self):
        """Tests bout_boundaries() with normal data."""
        arr = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(
            behaviors.bout_boundaries(arr),
            [[2, 7]]
        )

        arr = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(
            behaviors.bout_boundaries(arr),
            [[2, 6],
             [8, 11]]
        )

    def test_bout_boundaries_02(self):
        """Tests bout_boundaries() with single-frame bout."""
        arr = np.array([0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(
            behaviors.bout_boundaries(arr),
            [[3, 4]]
        )

        arr2 = np.zeros(arr.size)
        b_ixs = behaviors.bout_boundaries(arr)
        arr2[b_ixs[0,0]:b_ixs[0,1]] = 1
        np.testing.assert_array_equal(
            arr,
            arr2
        )

    def test_bout_boundaries_ts_01(self):
        """Tests bout_boundaries_ts() with normal data."""
        male = fly.Fly(10)
        male.add_behavior_from_array(
            'behavior_01',
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        )
        female = fly.Fly()
        tracking_sum = ts.FixedCourtshipTrackingSummary()
        tracking_sum.male = male
        tracking_sum.female = female

        b_ixs = behaviors.bout_boundaries_ts(tracking_sum, 'behavior_01')
        np.testing.assert_array_equal(
            b_ixs,
            [[2, 3]]
        )

    def test_bout_durations_01(self):
        """Tests bout_durations() with normal data."""
        arr = np.array([0, 1, 1, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(
            behaviors.bout_durations(arr),
            [3, 3]
        )

    def test_bout_frequency_01(self):
        """Tests bout_frequency() with normal data."""
        arr = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        self.assertEqual(
            behaviors.bout_frequency(arr),
            2/8.
        )

    def test_n_bouts_01(self):
        """Tests n_bouts() with normal data."""
        arr = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.assertEqual(
            behaviors.n_bouts(arr),
            3
        )

    def test_n_pauses(self):
        """Tests n_pauses() with normal data."""
        arr = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.assertEqual(
            behaviors.n_pauses(arr),
            2
        )

    def test_pause_durations(self):
        """Tests pause_durations() with normal data."""
        arr = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(
            behaviors.pause_durations(arr),
            [2, 1]
        )

    def test_pause_frequency(self):
        """Tests pause_frequency() with normal data."""
        arr = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        np.testing.assert_equal(
            behaviors.pause_frequency(arr),
            2./8
        )

    def test_get_transitional_tracks(self):
        """Not Implemented Yet."""
        pass

    def test_align_transitions(self):
        """Not Implemented Yet."""
        pass

    def test_exclude_behaviors_01(self):
        """Tests exclude_behaviors() with normal data."""
        focal = behavior.Behavior.from_array(
            'focal', [0, 0, 1, 1, 0, 0, 1, 1, 1])
        b1 = behavior.Behavior.from_array(
            'b1', [0, 0, 1, 0, 0, 0, 0, 0, 0]
        )
        b2 = behavior.Behavior.from_array(
            'b2', [0, 0, 0, 0, 0, 0, 1, 0, 0]
        )
        focal_excl = behaviors.exclude_behaviors(focal, [b1, b2])
        np.testing.assert_array_equal(
            focal_excl.as_array(),
            [0, 0, 0, 1, 0, 0, 0, 1, 1]
        )
        self.assertEqual(
            focal_excl.name,
            'focal -- excluding: b1, b2.'
        )

    def test_exclude_behavior_from_courtship_ts(self):
        """Tests exclude_behavior_from_courtship_ts() with normal data."""
        male = fly.Fly(12)
        female = fly.Fly(12)

        male.add_behavior_from_array(
            'courtship_gt', 
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            )
        male.add_behavior_from_array(
            'attempted-copulation',
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        )

        tracking_sum = ts.FixedCourtshipTrackingSummary()
        tracking_sum.male = male
        tracking_sum.female = female

        tracking_sum, bname = behaviors.exclude_behavior_from_courtship_ts(
            tracking_sum
        )

        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior(bname).as_array(),
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        )
        self.assertEqual(
            bname,
            'courtship_gt -- excluding: attempted-copulation.'
        )

    def test_exclude_behavior_from_courtship_exp(self):
        """Not Implemented Yet."""
        pass

    def test_hierarchize(self):
        """Tests hierarchize() with normal data."""
        b1 = behavior.Behavior.from_array(
            'b1', [0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        b2 = behavior.Behavior.from_array(
            'b2', [0, 0, 1, 1, 1, 1, 1, 0, 0]
        )
        b3 = behavior.Behavior.from_array(
            'b3', [0, 1, 1, 1, 1, 1, 1, 1, 0]
        )
        b4 = behavior.Behavior.from_array(
            'b4', np.ones(9)
        )

        h = behaviors.hierarchize([b1, b2, b3, b4])

        self.assertEqual(
            h[0].name,
            'b1 -- excluding:  (hiearchy)'
        )
        np.testing.assert_array_equal(
            h[0].as_array(),
            b1.as_array()
        )

        self.assertEqual(
            h[1].name,
            'b2 -- excluding: b1. (hiearchy)'
        )
        np.testing.assert_array_equal(
            h[1].as_array(),
            [0, 0, 1, 0, 0, 0, 1, 0, 0]
        )

        self.assertEqual(
            h[2].name,
            'b3 -- excluding: b1, b2. (hiearchy)'
        )
        np.testing.assert_array_equal(
            h[2].as_array(),
            [0, 1, 0, 0, 0, 0, 0, 1, 0]
        )

        self.assertEqual(
            h[3].name,
            'b4 -- excluding: b1, b2, b3. (hiearchy)'
        )
        np.testing.assert_array_equal(
            h[3].as_array(),
            [1, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_hiearchize_ts(self):
        """Tests hiearchize_ts() with normal data."""
        male = fly.Fly(10)
        female = fly.Fly(10)

        b1 = behavior.Behavior.from_array(
            'b1', [0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        b2 = behavior.Behavior.from_array(
            'b2', [0, 0, 1, 1, 1, 1, 1, 0, 0]
        )
        b3 = behavior.Behavior.from_array(
            'b3', [0, 1, 1, 1, 1, 1, 1, 1, 0]
        )
        b4 = behavior.Behavior.from_array(
            'b4', np.ones(9)
        )

        bs = [b1, b2, b3, b4]
        for b in bs:
            male.add_behavior(b)

        tracking_sum = ts.FixedCourtshipTrackingSummary()
        tracking_sum.male = male
        tracking_sum.female = female

        tracking_sum = behaviors.hierarchize_ts(
            tracking_sum,
            ['b1', 'b2', 'b3', 'b4']
            )

        # make sure that all of the original behaviors remain unchanged.
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior('b1').as_array(),
            [0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior('b2').as_array(),
            [0, 0, 1, 1, 1, 1, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior('b3').as_array(),
            [0, 1, 1, 1, 1, 1, 1, 1, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior('b4').as_array(),
            np.ones(9)
        )

        # make sure that all of the new behaviors are as expected
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior(
                'b1 -- excluding:  (hiearchy)').as_array(),
            [0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior(
                'b2 -- excluding: b1. (hiearchy)').as_array(),
            [0, 0, 1, 0, 0, 0, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior(
                'b3 -- excluding: b1, b2. (hiearchy)').as_array(),
            [0, 1, 0, 0, 0, 0, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            tracking_sum.male.get_behavior(
                'b4 -- excluding: b1, b2, b3. (hiearchy)').as_array(),
            [1, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_hiearchize_exp_01(self):
        """Not Implemented Yet."""
        pass

    def test_behavior_as_fraction_of_courtship(self):
        """Not Implemented Yet."""
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)