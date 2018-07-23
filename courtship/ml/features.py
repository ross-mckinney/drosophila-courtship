# -*- coding: utf-8 -*-

"""
.. module:: ml
   :synopsis: Contains class for generating feature matrices from Fly data.

.. moduleauthor:: Ross McKinney
"""
import pickle

import numpy as np
import pandas as pd

from scipy.ndimage.filters import generic_filter

from errors import *
from courtship.stats import spatial
from courtship.stats import wing
from courtship.stats import centroid
from courtship.stats import _signal as signal


class FeatureMatrix(object):
    """Base class for feature matrices.

    Parameters
    ----------
    pixels_per_mm : float (default = 1.)
        Conversion factor to normalize distance parameters between
        feature matrices. This is equivalent to the number of pixels
        in 1 mm of absolute distance. If 1., no conversion occurs.
    """

    reserved_words = ['fly', 'timestamps', 'pixels_per_mm']

    def __init__(self, pixels_per_mm=1.0):
        self.pixels_per_mm = pixels_per_mm

    def _clean_X(self, X):
        """Cleans a feature matrix by removing all NaNs and infs.

        Both NaNs and infs are replaced with 0s.

        Parameters
        ----------
        X : np.ndarray of shape [N, M]
            Matrix to clean.
        """
        X[np.where(np.isnan(X))] = 0
        X[np.where(np.isinf(X))] = 0

    def calculate_first_derivatives(self):
        """Takes the first derivative of all class attributes
        sets it as a new class attribute.

        .. warning:: Calls to first_derivative() will calculate the
                     first derivative of all non-reserved attributes.

        Parameters
        ----------
        window_size : int
            Window size to use for median filter.
            This must be odd.
        """
        for k, v in self.get_params().iteritems():
            setattr(
                self,
                'd1_' + k,
                signal.derivative(v, self.timestamps)
            )

    def calculate_second_derivatives(self):
        """Takes the second derivative of all class attributes and sets
        it as a new class attribute.

        .. warning:: Calls to second_derivative() will calculate the
                     derivative of all first derivative attributes.
        """
        for k, v in self.get_params().iteritems():
            if 'd1_' in k:
                setattr(
                    self,
                    'd2_' + k.split('d1_')[-1],
                    signal.derivative(v, self.timestamps)
                )

    def calculate_windowed_statistics(
        self,
        window_sizes=[3, 11, 21],
        sfxn={
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'median': np.median
            }
    ):
        """Calculates windowed statistics for all class attributes.

        Parameters
        ----------
        window_sizes : list of int (default=[3, 11, 21])
            Defines sliding window size(s).

        sfxn : dict of functions (
            default=dict(
                mean=np.mean,
                max=np.max,
                min=np.min,
                median=np.median)
            )
            Statistical functions to calculate for windows.
        """
        frame_diff_fxns = {'median': np.median, 'max': np.max, 'min': np.min}
        params = self.get_params()

        for i, feature in enumerate(params.keys()):
            for window_size in window_sizes:
                frame_locations = [0, window_size / 2, window_size - 1]
                for frame_location in frame_locations:
                    # calculate frame differences
                    for key, fxn in frame_diff_fxns.iteritems():
                        _diff = signal.diff(ix=frame_location, stat=fxn)
                        ID = (
                            feature +
                            '_' +
                            str(key) +
                            '_diff_ws{}_'.format(window_size) +
                            str(frame_location)
                        )
                        window = generic_filter(
                            params[feature],
                            function=_diff,
                            size=window_size
                        )
                        setattr(self, ID, window)

                    # calcualte z-score
                    z_score_fxn = signal.z_score(ix=frame_location)
                    scoreID = (
                        feature +
                        '_zscore_ws{}_'.format(window_size) +
                        str(frame_location)
                    )
                    score = generic_filter(
                        params[feature],
                        function=z_score_fxn,
                        size=window_size,
                        )
                    setattr(self, scoreID, score)

                # calculate specified sliding window statistics
                for key, fxn in sfxn.iteritems():
                    ID = (
                        feature +
                        '_' +
                        str(key) +
                        '_ws{}_'.format(window_size) +
                        str(frame_location)
                    )
                    window = generic_filter(
                        params[feature],
                        function=fxn,
                        size=window_size)
                    setattr(self, ID, window)

                ID = (
                    feature +
                    '_mean_change_ws{}'.format(window_size)
                )
                setattr(
                    self,
                    ID,
                    generic_filter(
                        params[feature],
                        function=signal.window_differences,
                        size=window_size
                    )
                )

    def get_params(self):
        """Returns a dictionary of all parameters/attributes.

        .. warning:: This will exclude any attribute that are present
                     in the reserved_words global.

        Returns
        -------
        params : dictionary
            Dictionary of parameters/attributes.
        """
        params = {}
        for k, v in self.__dict__.iteritems():
            if k in self.reserved_words:
                continue
            params[k] = v
        return params

    def get_X(self):
        """Returns a feature matrix containing all parameters.

        .. note:: All np.nan and np.inf values are replaced with 0.

        Returns
        -------
        X : np.ndarray
            Feature matrix. Each row is one data point, and each
            column is a distinct feature. All features will be
            sorted alphabetically.
        """
        X = self.to_df().values
        self._clean_X(X)
        return X

    def set_params(self, param_dict):
        """Sets paramaters given a dictionary.

        Keys should be parameter names, values should be param vals.

        Parameters
        ----------
        param_dict : dictionary
            Dictionary containing attributes/parameters to set for this
            FeatureMatrix.
        """
        for k, v in param_dict.iteritems():
            if isinstance(v, np.ndarray):
                setattr(self, k, v)
            elif isinstance(v, list):
                setattr(self, k, np.ndarray(v))
            else:
                setattr(self, k, v)

    def to_df(self):
        """Return a pandas.DataFrame containing all parameters.

        .. note:: a call to from_dict() automatically sorts columns in
                  ascending order.

        Returns
        -------
        df : pandas.DataFrame object
        """
        return pd.DataFrame.from_dict(self.get_params())

    def pickle(self, savename):
        """Saves a FeatureMatrix as a pickled object.

        Parameters
        ----------
        savename : string
            Path to save location.
        """
        with open(savename, 'wb') as f:
            pickle.dump(self, f)


class WingFeatures(FeatureMatrix):
    """Feature matrix to store features about wings.

    Parameters
    ----------
    fly : courtanal.objects.Fly object
        Fly whose wings we are interested in calculating features.

    Atributes
    ---------
    full_wing_angle : np.ndarray
        Angle made between left wing centroid > body centroid >
        right wing centroid.

    left_wing_angle : np.ndarray
        Positive angle made between left wing centroid > body centroid >
        x-axis.

    left_wing_area : np.ndarray
        Area occupied by left wing (units: square mm).

    right_wing_angle : np.ndarray
        Positive angle made between right wing centroid > body centroid >
        x-axis.

    right_wing_area : np.ndarray
        Area occupied by right wing (units: square mm).

    wing_distances : np.ndarray
        Sum of distances between left wing centroid, body centroid,
        and right wing centroid (units: mm).
    """
    def __init__(self, fly=None, pixels_per_mm=1.0):
        FeatureMatrix.__init__(self, pixels_per_mm)

        try:
            self.left_wing_angle, self.right_wing_angle = \
                wing.individual_wing_angles(fly)
            self.wing_distances = wing.wing_distances(fly) / self.pixels_per_mm
            self.full_wing_angle = wing.full_wing_angle(fly)
            self.left_wing_area = fly.left_wing.area() / self.pixels_per_mm
            self.right_wing_area = fly.right_wing.area() / self.pixels_per_mm
        except (AttributeError, TypeError, NameError) as e:
            raise e


class SocialFeatures(FeatureMatrix):
    """Feature matrix to store features about an interaction between
    two flies.

    Parameters
    ----------
    male : courtanal.objects.Fly object
        Male fly.

    female : courtanal.objects.Fly object
        Female fly.

    Attributes
    ----------
    c2c : np.ndarray
        Centroid-to-centroid distance between male and female fly (units: mm).

    n2e : np.ndarray
        Nose-to-ellipse distance between male nose and any point on an ellipse
        fitted to the female fly (units: mm).

    nte_diff : np.ndarray
        Difference between n2e and t2e (n2e - t2e).

    t2e : np.ndarray
        Tail-to-ellipse distance between male rear and any point on an ellipse
        fitted to the female fly (units: mm).

    rel_ori : np.ndarray
        Orientation of the male w.r.t. the female fly
        (units: radians, -np.pi/2 to np.pi/2).

    abs_ori : np.ndarray
        Absolute value of the orientation of the male w.r.t. the female fly
        (units: radians, 0 to np.pi/2).
    """
    def __init__(self, male=None, female=None, pixels_per_mm=1.0):
        FeatureMatrix.__init__(self, pixels_per_mm)

        try:
            self.c2c = spatial.nearest_neighbor_centroid(
                male, female, normalized=False) / self.pixels_per_mm
            self.n2e, self.t2e = spatial.nose_and_tail_to_ellipse(
                male, female, normalized=False)
            self.n2e /= self.pixels_per_mm
            self.t2e /= self.pixels_per_mm
            self.rel_ori = spatial.relative_orientation(male, female)
            self.abs_ori = spatial.abs_relative_orientation(male, female)
            self.nte_diff = self.n2e - self.t2e
        except (AttributeError, TypeError, NameError) as e:
            raise e


class GeneralFeatures(FeatureMatrix):
    """General features about movement of a fly and/or position
    and posture within an arena.

    Parameters
    ----------
    fly : courtanal.objects.Fly object
        Fly to calculate features for

    arena_center : tuple (int, int)
        Center of arena (rr, cc) in pixel coordinates.

    arena_radius : int
        Radius of arena in pixels.

    Attributes
    ----------
    angular_velocity : np.ndarray
        Angular velocity of fly.

    abs_angular_velocity : np.ndarray
        Absolute value of angular velocity of fly.

    centroid_velocity : np.ndarray
        Velocity of fly centroid.

    maj_ax_len : np.ndarray
        Major axis length of fly body.

    min_ax_len : np.ndarray
        Minor axis length of fly body.

    area : np.ndarray
        Area of fly body.

    orientation : np.ndarray
        Orientation of fly body.

    c2edge : np.ndarray
        Distance of centroid of fly body to arena edge.
    """
    def __init__(
        self,
        fly=None,
        arena_center=None,
        arena_radius=None,
        pixels_per_mm=1.0
    ):
        FeatureMatrix.__init__(self, pixels_per_mm)

        try:
            self.angular_velocity = centroid.angular_velocity(fly)
            self.abs_angular_velocity = centroid.abs_angular_velocity(fly)
            self.centroid_velocity = centroid.centroid_velocity(
                fly, normalized=False) / self.pixels_per_mm
            self.maj_ax_len = fly.body.major_axis_length / self.pixels_per_mm
            self.min_ax_len = fly.body.minor_axis_length / self.pixels_per_mm
            self.area = fly.body.area() / self.pixels_per_mm
            self.orientation = fly.body.orientation
            self.c2edge = centroid.centroid_to_arena_edge(
                fly, arena_center, arena_radius) / self.pixels_per_mm
        except (AttributeError, TypeError, NameError) as e:
            raise e


class ScissoringFeatureMatrix(GeneralFeatures, WingFeatures, SocialFeatures):
    """Feature Matrix for classifying scissoring/wing extension behaviors
    between fly pairs.
    """
    def __init__(
        self,
        male=None,
        female=None,
        arena_center=None,
        arena_radius=None,
        pixels_per_mm=1.0
    ):
        GeneralFeatures.__init__(
            self,
            male,
            arena_center,
            arena_radius,
            pixels_per_mm)
        WingFeatures.__init__(self, male, pixels_per_mm)
        SocialFeatures.__init__(self, male, female, pixels_per_mm)

        try:
            self.timestamps = male.timestamps
        except (AttributeError, TypeError, NameError):
            pass

        self.calculate_windowed_statistics()
        self.calculate_first_derivatives()
        self.calculate_second_derivatives()


class TappingFeatureMatrix(GeneralFeatures, SocialFeatures):
    """Feature matrix for classifying touching/tapping behaviors
    between fly pairs.
    """
    def __init__(
        self,
        male=None,
        female=None,
        arena_center=None,
        arena_radius=None,
        pixels_per_mm=1.
    ):
        GeneralFeatures.__init__(
            self,
            male,
            arena_center,
            arena_radius,
            pixels_per_mm)
        SocialFeatures.__init__(self, male, female, pixels_per_mm)

        try:
            self.timestamps = male.timestamps
        except (AttributeError, TypeError, NameError):
            pass

        self.calculate_windowed_statistics()
        self.calculate_first_derivatives()
        self.calculate_second_derivatives()
