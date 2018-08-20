# tests for TrackingSummary

import os, pickle
import unittest

import numpy as np
import pandas as pd

from context import ts

## Note that the following two classes cannot be run at the same
## time due to read/write interference.
##
## Therefore, to run a specific class/test case from this file, use:
##    python test_ts.py TestTrackingSummaryRead
##                    -OR-
##    python test_ts.py TestTrackingSummarySave

class TestTrackingSummarySave(unittest.TestCase):
    """Tests that TrackingSummary IO Save functions work as expected."""
    def setUp(self):
        """Load data."""
        with open('D:/test_data2/fcts2/test_01.fcts', 'rb') as f:
            self.summary_01 = pickle.load(f)
    
    def test_to_xlsx(self):
        """Tests that saving to .xslx works."""
        self.summary_01.to_xlsx('D:/test_data2/test_01.xlsx')


class TestTrackingSummaryRead(unittest.TestCase):
    """Tests that TrackingSummary IO Read functions work as expected."""
    def test_from_xlsx(self):
        """Tests that from_xlsx works."""
        test_summary = ts.FixedCourtshipTrackingSummary.from_xlsx(
            'D:/test_data2/test_01.xlsx'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)