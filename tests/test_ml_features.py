# test_ml_features.py

import os
import pickle
import unittest

import numpy as np
import pandas as pd

from context import (
    fly, features, classifiers
    )


if __name__ == '__main__':
    test_dir = 'D:/test_data2'
    # with open(os.path.join(test_dir, 'fcts2', 'test_01.fcts'), 'rb') as f:
    #     ts = pickle.load(f)

    # sci_fmat = features.ScissoringFeatureMatrix(
    #     ts.male, 
    #     ts.female, 
    #     (ts.arena.center_pixel_rr, ts.arena.center_pixel_cc),
    #     ts.arena.radius_mm,  # should this be in mm or pixels?
    #     pixels_per_mm=float(ts.video.pixels_per_mm)
    # )

    # X = sci_fmat.to_df().to_csv('D:/test_data2/fcts2_csv/test_01.csv', index=False)