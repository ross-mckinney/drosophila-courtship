# experiment.py

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pycircstat.descriptive as pysum
import pycircstat.tests as pytest

from courtship.stats import (
    behaviors,
    centroid,
    markov,
    spatial
)


class FixedCourtshipTrackingExperiment(object):
    """Class for handling experimental data.

    This data is in the form of multiple groups of .fcts files.

    Parameters
    ----------
    """
    def __init__(self, order=None, fps=24., duration_seconds=600., **kwargs):
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.group_names = []
        if order is None:
            self.order = sorted(kwargs.keys())
        else:
            self.order = order

        for group_name, group in kwargs.iteritems():
            if not isinstance(group, list):
                raise AttributeError('All passed **kwargs must contain ' +
                'lists.')
            setattr(self, group_name, group)
            self.group_names.append(group_name)

    @classmethod
    def load_from_fcts(cls, data_dirs, groups=None, order=None, fps=24.,
        duration_seconds=600.):
        """Loads a FixedCourtshipTrackingExperiment from .fcts files.

        Parameters
        ----------
        data_dirs : list of string
            Valid path(s) to directory(ies) containing .fcts files.

        groups : dict or None, optional (default=None)
            Should load in all .fcts (ts) files found in the passed dir. If None,
            then groups will be assigned based on the ts.group keys. If a dict,
            keys should be group names to load into the experiment, and values
            should be lists of possible group names.

        order : list or None, optional (default=None)
            Ordered list of string. Each item should be a valid key name (group
            name) in groups dictionary.

        fps : float, optional (default=24.)
            Video frame-rate (frames-per-second).

        duration_seconds : float, optional (default=600.)
            Total duration of video recordings.

        Returns
        -------
        FixedCourtshipTrackingExperiment :

        Examples
        --------
        >>> # loads in .fcts files found in '/path/to/dir1';
        >>> # groups in returned experiment will be ordered according to
        >>> # unique groups found among all ts.group attributes.
        >>> exp = load_experiment(data_dirs=['/path/to/dir1'])
        """
        if type(data_dirs) is not list:
            raise AttributeError('`data_dirs` must be of type list.')
        if type(groups) is not dict:
            if groups is not None:
                raise AttributeError('`groups` must be of type dict or None.')

        fcts_files = []
        for i, dir in enumerate(data_dirs):
            if not os.path.isdir(dir):
                raise AttributeError('`data_dir` item at index {} '.format(i) +
                    'is not a valid directory path.')
            fcts_files += sorted([os.path.join(dir, f) for f in os.listdir(dir)])

        # load all FixedCourtshipTrackingSummary files
        summaries, group_names = [], []
        for fname in fcts_files:
            with open(fname, 'rb') as f:
                ts = pickle.load(f)
            summaries.append(ts)
            group_names.append(ts.group)

        if groups is not None:  # then groups is a dict
            for k, v in groups.iteritems():
                if type(v) is not list:
                    raise AttributeError(
                        '`groups` values must be of type list.')
        else:                   # then groups is None
            group_names = set(group_names)
            groups = {g:[g] for g in group_names}

        experimental_groups = {g: [] for g in groups.keys()}
        num_ts_loaded = 0
        for i, ts in enumerate(summaries):
            for group_name, group_name_list in groups.iteritems():
                for gn in group_name_list:
                    if ts.group == gn:
                        experimental_groups[group_name].append(ts)
                        num_ts_loaded += 1

        if num_ts_loaded != len(summaries):
            warnings.warn(
                'Not all .fcts files were loaded. Check that items in ' + '`groups` dict are valid.\n' +
                'N files expected: {}'.format(len(summaries)) +
                'N files loaded:   {}'.format(num_ts_loaded),
                UserWarning
                )

        return cls(
            order=order,
            fps=fps,
            duration_seconds=duration_seconds,
            **experimental_groups
        )

    def itergroups(self):
        """Iterates over each TrackingSummary contained within this Experiment.

        Yields
        ------
        group_name : string
            Name of group to which current TrackingSummary belongs.

        tracking_summary : FixedCourtshipTrackingSummary
            Current TrackingSummary.
        """
        for group_name in self.group_names:
            group = getattr(self, group_name)
            for tracking_summary in group:
                yield group_name, tracking_summary

    def clean_behavior(
        self,
        focal_behavior_name,
        falls_within_behavior_name
        ):
        """Cleans a focal behavior such that all behavioral bouts within the
        focal behavior fall within all behavioral bouts present in the
        `falls_within` behavior.

        All males within each group of this Experiment will contain a
        new behavior in their `behaviors` list attribute that contains the
        cleaned behavior.

        Parameters
        ----------
        focal_behavior_name : string
            Name of focal behavior to clean.

        falls_within_behavior_name : string
            Name of behavior which focal behavior must fall within.

        Returns
        -------
        clean_behavior_name : string
            Name of new behavior added to in each male fly within this
            Experiment. This will string will be formatted as follows:
            '{}-falling-within-{}'.format(
                focal_behavior_name,
                falls_within_behavior_name
                )

        Examples
        --------
        This will generate a new behavior for each male fly such that only
        bouts of 'scissoring' that fall within bouts of 'courtship_gt' will be
        retained.

        >>> exp.clean_behavior(
                focal_behavior='scissoring',
                falls_within_behavior='courtship_gt'
                )
            'scissoring-falling-within-courtship_gt'
        """
        new_behavior_name = '{}-falling-within-{}'.format(
            focal_behavior_name,
            falls_within_behavior_name
        )

        for _, tracking_summary in self.itergroups():
            male = tracking_summary.male
            focal_behavior = male.get_behavior(focal_behavior_name)
            falls_within_behavior = male.get_behavior(
                falls_within_behavior_name
                )

            focal_behavior_arr = focal_behavior.as_array()
            falls_within_arr = falls_within_behavior.as_array()

            clean_behavior = focal_behavior_arr * falls_within_arr
            male.add_behavior_from_array(new_behavior_name, clean_behavior)

        return new_behavior_name

    def get_ang_location_summary(self,
        behavior_name,
        frm=0.0,
        to=1.0,
        statistic=pysum.mean,
        clean=True
        ):
        """Calculates a summary statistic on behavioral locations
        (angular values) for each individual within all control/experimental
        groups.

        Parameters
        ----------
        behavior_name : string
            Name of behavior to get all angular locations for.

        frm : float (ranges from [0.0 to 1.0]) (default=0.0)
            Start fraction of courtship trial for which to calculate
            angular locations.

        to : float (ranges from (0.0 to 1.0]) (default=1.0)
            End fraction of courtship trial for which to calculate
            angular locations.

        statistic : function (default=pysum.mean)
            Should be a function that can calculate a summary statistic on
            angular data.

        clean : bool (default = True)
            Removes np.nans from summary if True.

        Returns
        -------
        locations : dict of np.ndarray
            Dictionary of np.ndarrays of angular locations.
        """
        locations = {}
        for group_name in self.group_names:
            group = getattr(self, group_name)
            locs = spatial.summarize_behavior(
                group,
                behavior_name=behavior_name,
                frm=frm,
                to=to,
                statistic=statistic
            )

            if clean:
                locations[group_name] = locs[~np.isnan(locs)]
            else:
                locations[group_name] = locs

        return locations

    def get_ang_rvals(
        self,
        behavior_name,
        frm=0.0,
        to=1.0,
        statistic=pysum.mean
        ):
        """Calculates R-values for a summary statistic on behavioral locations
        (angular values) for each individual within all control/experimental
        groups.

        Parameters
        ----------
        behavior_name : string
            Name of behavior to get all angular locations for.

        frm : float (ranges from [0.0 to 1.0]) (default = 0.0)
            Start fraction of courtship trial for which to calculate
            angular locations.

        to : float (ranges from (0.0 to 1.0]) (default = 1.0)
            End fraction of courtship trial for which to calculate
            angular locations.

        statistic : function (default = pysum.mean)
            Should be a function that can calculate a summary statistic on
            angular data.

        Returns
        -------
        r_vals : dictionary
            Each key is the same as a group name, and each value
        """
        locations = self.get_ang_location_summary(
            behavior_name=behavior_name,
            frm=frm,
            to=to,
            statistic=statistic
        )

        r_vals = {}
        for k, group in locations.iteritems():
            r_vals[k] = pysum.resultant_vector_length(group)
            print '{}, R-val: {}'.format(k, r_vals[k])

        return r_vals

    def get_ang_rayleighvals(
        self,
        behavior_name,
        frm=0.0,
        to=1.0,
        statistic=pysum.mean
        ):
        """Performs a rayleigh test on behavioral locations (angular values) for each
        control/experimental group.

        Parameters
        ----------
        behavior_name : string
            Name of behavior to get all angular locations for.

        frm : float (ranges from [0.0 to 1.0]) (default = 0.0)
            Start fraction of courtship trial for which to calculate
            angular locations.

        to : float (ranges from (0.0 to 1.0]) (default = 1.0)
            End fraction of courtship trial for which to calculate
            angular locations.

        statistic : function (default = pysum.mean)
            Should be a function that can calculate a summary statistic on
            angular data.

        Returns
        -------
        p_vals : dictionary
            Dictionary of p-values (2-tailed) from rayleigh test.

        z_stats : dictionary
            Dictionary of z-statistics from rayleigh test.
        """
        locations = self.get_ang_location_summary(
            behavior_name=behavior_name,
            frm=frm,
            to=to,
            statistic=statistic
        )

        p_vals = {}
        z_stats = {}

        for k, group in locations.iteritems():
            p, z = pytest.rayleigh(group)
            p_vals[k] = p
            z_stats[k] = z
            print '{}, Rayleigh p-val: {}\n\tRayleigh z-statistic: {}'.format(
                k, p, z)

        return p_vals, z_stats

    def get_ang_watsonwilliams(
        self,
        behavior_name,
        frm=0.0,
        to=1.0,
        statistic=pysum.mean
        ):
        """Runs a Watson-Williams test on all groups.

        Parameters
        ----------
        behavior_name : string
            Name of behavior to get all angular locations for.

        frm : float (ranges from [0.0 to 1.0]) (default = 0.0)
            Start fraction of courtship trial for which to calculate
            angular locations.

        to : float (ranges from (0.0 to 1.0]) (default = 1.0)
            End fraction of courtship trial for which to calculate
            angular locations.

        statistic : function (default = pysum.mean)
            Should be a function that can calculate a summary statistic
            on angular data.

        Returns
        -------
        p_val : float
            p-value from Watson-Williams test.

        table : pandas DataFrame
            Table containing parameters calculated via the Watson-Williams
            test.
        """
        locations = self.get_ang_location_summary(
            behavior_name=behavior_name,
            frm=frm,
            to=to,
            statistic=statistic
        )

        p_val, table = pytest.watson_williams(*locations.values())
        print table

        return p_val, table

    def get_behavioral_distances(self,
        behavior_name,
        metric='centroid-to-centroid',
        nbins=50
        ):
        """Gets the average male-to-female distances for a specified behavior
        across all possible angular positions of the male.

        Parameters
        ----------
        behavior_name : string
            This should be a key within the TrackingSummary.behaviors
            attribute.

        metric : string (default='centroid-to-centroid')
            Which distance metric should be returned. Options are as follows.
            1. 'centroid-to-centroid' -- distance between male and female
                    centroids.
            2. 'head-to-ellipse' -- distance between male's head and any point
                    on an ellipse fitted to the female.
            3. 'rear-to-ellipse' -- distance between male's rear and any point
                    on an ellipse fitted to the female.

        nbins : int or None
            Number of bins to use for calculating the distances. Note that
            this will split the internal (-np.pi, np.pi] into nbins.
            Distances will be calculated for each bin. If None, the average
            across all bins will be taken.

        Returns
        -------
        distances : dict of np.ndarray | shape [N] or [N, nbins]
            The average male-to-female distance (in mm) during a specified
            behavior for each group and courting pair held within this
            experiment.
        """
        distances = {}
        for group_name in self.group_names:
            rs = []
            group = getattr(self, group_name)
            for tracking_summary in group:
                b_ixs = tracking_summary             \
                        .male                        \
                        .get_behavior(behavior_name) \
                        .as_array()

                if np.sum(b_ixs) == 0:
                    continue
                theta, _ = spatial.relative_position2(
                    tracking_summary.male, 
                    tracking_summary.female
                    )

                if metric == 'centroid-to-centroid':
                    r = spatial.nearest_neighbor_centroid(
                            tracking_summary.male,
                            tracking_summary.female,
                            normalized=False
                        ) / tracking_summary.video.pixels_per_mm
                elif metric == 'head-to-ellipse':
                    r, _ = spatial.nose_and_tail_to_ellipse(
                        tracking_summary.male,
                        tracking_summary.female,
                        normalized=False
                    )
                    r = r / tracking_summary.video.pixels_per_mm
                elif metric == 'rear-to-ellipse':
                    _, r = spatial.nose_and_tail_to_ellipse(
                        tracking_summary.male,
                        tracking_summary.female,
                        normalized=False
                    )
                    r = r / tracking_summary.video.pixels_per_mm
                else:
                    message = (
                        '{} '.format(metric) +
                        'is not a valid option for distance parameter.\n' +
                        'Valid options are as follows:\n' +
                        '\t1. \'centroid-to-centroid\' (default)\n' +
                        '\t2. \'head-to-ellipse\'\n' +
                        '\t3. \'rear-to-ellipse\''
                        )
                    raise AttributeError(message)

                theta = theta[np.flatnonzero(b_ixs)]
                r = r[np.flatnonzero(b_ixs)]

                if nbins is None:
                    rs.append(np.nanmean(r))
                    continue

                edges = np.linspace(-np.pi, np.pi, nbins)
                r_bins = [[] for i in xrange(nbins)]

                for i in xrange(theta.size):
                    rix = np.searchsorted(edges, theta[i])
                    r_bins[rix].append(r[i])

                r_bins = [np.nanmean(rb) for rb in r_bins]
                rs.append(r_bins)

            distances[group_name] = np.asarray(rs)
        return distances