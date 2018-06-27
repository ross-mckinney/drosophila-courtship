# tests for TrackingSummary

import os, pickle
import unittest

import numpy as np
import pandas as pd

from context import ts

class TestTrackingMeta(unittest.TestCase):
    """Tests how meta data (from Meta) is handled."""
    def setUp(self):
        """Instantiates a couple of sample TrackingSummary objects for use in
        tests."""
        self.tracking_sum = ts.TrackingSummary()


if __name__ == '__main__':
    with open('D:/test_data2/fcts2/test_01.fcts', 'rb') as f:
        ts = pickle.load(f)