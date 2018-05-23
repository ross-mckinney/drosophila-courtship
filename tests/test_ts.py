# tests for TrackingSummary

import os
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
        