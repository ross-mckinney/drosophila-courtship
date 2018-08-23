# experiment.py

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pycircstat.descriptive as pysum
import pycircstat.tests as pytest

from courtship.behavior import Behavior
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
    def __init__(self,
        order=None,
        video_fps=24.,
        video_duration_seconds=600.,
        video_duration_frames=14400,
        **kwargs):
        self.video_fps = video_fps
        self.video_duration_seconds = video_duration_seconds
        self.video_duration_frames = video_duration_frames

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
    def load_from_fcts(cls, data_dirs, groups=None, order=None, video_fps=24.,
        video_duration_seconds=600., video_duration_frames=14400):
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

        video_fps : float, optional (default=24.)
            Video frame-rate (frames-per-second).

        video_duration_seconds : float, optional (default=600.)
            Total duration of video recordings (in seconds).

        video_duration_frames : int, optional (default=14400)
            Total number of frames expected in each video recording.

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
                'Not all .fcts files were loaded. Check that items in ' +
                '`groups` dict are valid.\n' +
                'N files expected: {}'.format(len(summaries)) +
                'N files loaded:   {}'.format(num_ts_loaded),
                UserWarning
                )

        return cls(
            order=order,
            video_fps=video_fps,
            video_duration_seconds=video_duration_seconds,
            video_duration_frames=video_duration_frames,
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
        num_bins=50,
        include_nonbehavors=False
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

        num_bins : int or None
            Number of bins to use for calculating the distances. Note that
            this will split the internal (-np.pi, np.pi] into num_bins.
            Distances will be calculated for each bin. If None, the average
            across all bins will be taken.

        Returns
        -------
        distances : dict of np.ndarray | shape [N] or [N, num_bins]
            The average male-to-female distance (in mm) during a specified
            behavior for each group and courting pair held within this
            experiment.
        """
        if metric not in ['centroid-to-centroid', 'head-to-ellipse',
            'rear-to-ellipse']:
            message = (
                        '{} '.format(metric) +
                        'is not a valid option for distance parameter.\n' +
                        'Valid options are as follows:\n' +
                        '\t1. \'centroid-to-centroid\' (default)\n' +
                        '\t2. \'head-to-ellipse\'\n' +
                        '\t3. \'rear-to-ellipse\''
                        )
            raise AttributeError(message)

        distances = {}
        for group_name in self.order:
            rs = []
            group = getattr(self, group_name)
            for tracking_summary in group:
                b_ixs = tracking_summary             \
                        .male                        \
                        .get_behavior(behavior_name) \
                        .as_array()

                if np.sum(b_ixs) == 0:
                    if include_nonbehavors:
                        if num_bins is None:
                            rs.append(np.nan)
                        else:
                            rs.append(np.repeat(np.nan, num_bins))
                    continue

                if metric == 'centroid-to-centroid':
                    rs.append(
                        spatial.binned_centroid_to_centroid(
                            tracking_summary, behavior_name=behavior_name,
                            bins=num_bins
                            )
                        )
                elif metric == 'head-to-ellipse':
                    rs.append(
                        spatial.binned_head_to_ellipse(
                            tracking_summary, behavior_name=behavior_name,
                            bins=num_bins
                            )
                        )
                else:
                    rs.append(
                        spatial.binned_rear_to_ellipse(
                            tracking_summary, behavior_name=behavior_name,
                            bins=num_bins
                            )
                        )

            distances[group_name] = np.asarray(rs)
        return distances

    def get_behavioral_distances_peak_ratio(self,
        behavior_name,
        metric='centroid-to-centroid',
        num_bins=50
        ):
        """Gets the ratio of the mean male-to-female distance when the male is
        on the front half of the female versus the rear half of the female.

        Parameters
        ----------
        behavior_name : string
            This should be a valid behavior in each male in this Experiment.

        metric : string (default='centroid-to-centroid')
            Which distance metric should be returned. Options are as follows.
            1. 'centroid-to-centroid' -- distance between male and female
                    centroids.
            2. 'head-to-ellipse' -- distance between male's head and any point
                    on an ellipse fitted to the female.
            3. 'rear-to-ellipse' -- distance between male's rear and any point
                    on an ellipse fitted to the female.

        num_bins : int or None
            Number of bins to use for calculating the distances. Note that
            this will split the internal (-np.pi, np.pi] into num_bins.
            Distances will be calculated for each bin. If None, the average
            across all bins will be taken.

        Returns
        -------
        peak_distance_ratios : dict of np.array
            Dictionary containing the peak distance ratios for each individual.
            Keys are group names. Values are arrays, where each item in the
            array represents the peak distance ratio for a single individual.
        """
        dists = self.get_behavioral_distances(
            behavior_name=behavior_name,
            metric=metric,
            num_bins=num_bins
        )

        # get the indices that represent the front half of the female.
        thetas = np.linspace(-np.pi, np.pi, num_bins)
        theta_high = thetas >= -np.pi/2
        theta_low = thetas <= np.pi/2
        thetas_front = theta_high * theta_low

        ratios = {group_name: [] for group_name in self.order}
        for group_name in self.order:
            for ind in dists[group_name]:
                peak_front = np.nanmax(ind[thetas_front])
                peak_rear = np.nanmax(ind[~thetas_front])
                peak_ratio = peak_front / peak_rear
                if not np.isnan(peak_ratio):
                    ratios[group_name].append(peak_front / peak_rear)

        return {name: np.asarray(vals) for name, vals in ratios.iteritems()}

    def get_behavioral_distances_peak(self,
        behavior_name,
        peak='front',
        metric='centroid-to-centroid',
        num_bins=50
        ):
        dists = self.get_behavioral_distances(
            behavior_name=behavior_name,
            metric=metric,
            num_bins=num_bins,
            include_nonbehavors=True
            )
        # get the indices that represent the front/rear half of the female.
        thetas = np.linspace(-np.pi, np.pi, num_bins)
        theta_high = thetas >= -np.pi/2
        theta_low = thetas <= np.pi/2

        thetas_front = theta_high * theta_low
        thetas_rear = ~thetas_front

        if peak == 'front':
            thetas_ixs = thetas_front
        elif peak == 'rear':
            thetas_ixs = thetas_rear
        else:
            raise AttributeError('`peak` must be either \'front\' or \'rear\'')

        peak_dists = {group_name: [] for group_name in self.order}
        for group_name in self.order:
            for ind in dists[group_name]:
                peak_dists[group_name].append(np.nanmax(ind[thetas_ixs]))

        return {name: np.asarray(vals) for name, vals in peak_dists.iteritems()}

    def get_behavioral_matrices(self,
        behavior_name,
        sort='descending',
        return_sort_ixs=False
        ):
        """Gets a matrix for each group where rows represent individuals, and
        columns represent binary behavioral state over time.

        Parameters
        ----------
        behavior_name : string
            This should be a valid key in each individual's behavior
            dictionary.

        sort : string (default='descending', other options are 'ascending'
            or None).
            If 'descending', rows will be sorted such that individuals with
            earlier behavioral examples are stored in rows with lower indeces.
            If 'ascending', the returned matrix will be flipped. If None, each
            row corresponds to the individual stored at it's respective
            position in its group.

        return_sort_ixs : bool (optional, default=False)
            Whether or not to return the sort order for the matrices.

        Returns
        -------
        matrices : dict of np.narrays
            Each key is the group name, each value is an 2D np.array containing
            binary behavioral states.
        """
        matrices = dict()
        latencies = dict()

        total_duration = self.video_duration_frames

        for group_name in self.order:
            group = getattr(self, group_name)
            matrix = np.zeros(shape=(len(group), total_duration))
            latency = []
            for i, summary in enumerate(group):
                behavioral_arr = summary.male.get_behavior(behavior_name) \
                                             .as_array()

                # check that the size of the behavioral array is the same
                # as the number of columns we've allocated to the matrix.
                b_size = behavioral_arr.size
                if b_size > total_duration:
                    matrix[i, :] = behavioral_arr[:total_duration]
                elif b_size < total_duration:
                    matrix[i, :b_size] = behavioral_arr
                    matrix[i, b_size:] = np.nan
                else:
                    matrix[i, :] = behavioral_arr

                # get the latency to behave; make sure there is at least one
                # non-zero value, otherwise set this value to the last
                # column-index of the matrix.
                l = np.flatnonzero(behavioral_arr)
                if l.size == 0:
                    l = total_duration - 1
                else:
                    l = l[0]

                latency.append(l)
            matrices[group_name] = matrix
            latencies[group_name] = latency

        if sort is None:
            return matrices

        sorted_matrices = dict()
        sort_ixs = dict()
        for group_name, matrix in matrices.iteritems():
            latency = latencies[group_name]
            ixs = np.argsort(latency)
            if sort == 'ascending':
                ixs = ixs[::-1]

            sort_ixs[group_name] = ixs
            sorted_matrices[group_name] = matrix[ixs, :]
        if return_sort_ixs:
            return sorted_matrices, sort_ixs
        return sorted_matrices

    def add_behavior_from_csv(self,
        behavior_name,
        csv_filename,
        video_header='video',
        group_header='group',
        start_ixs_header='start',
        stop_ixs_header='stop',
        behaving_header='behaving'
        ):
        """Adds a behavior -- contained in a .csv file -- to each male fly
        within this Experiment.

        The .csv file should have at least 5 columns with the following headers:
            1. `video_header`
            2. `group_header`
            3. `start_ixs_header`
            4. `stop_ixs_header`
            5. `behaving_header`

        Parameters
        ----------
        behavior_name : string
            Name of behavior to add to each individual.

        csv_filename : string
            Path to .csv file containing behavioral info.

        video_header : string
            Name of column header containing video file path info. This is
            the basename of the video from which behavioral data was taken.

        group_header : string
            Name of column header containing group info.

        start_ixs_header : string
            Name of column header containing starting indices of behavioral
            bouts.

        stop_ixs_header : string
            Name of column header containing stopping indices of behavioral
            bouts.

        behaving_header : string
            Name of column header containing behavioral state of fly during
            start_ixs and stop_ixs. This should either be 0 or 1 for each
            behavioral bout.
        """
        behavioral_df = pd.read_csv(csv_filename)
        unique_videos = behavioral_df[video_header].unique()

        for video in unique_videos:
            video_df = behavioral_df[behavioral_df[video_header] == video]
            video_basename = os.path.basename(video)
            group_name = video_df[group_header].iloc[0]

            group = getattr(self, group_name)
            for tracking_summary in group:
                if os.path.basename(tracking_summary.video.filename) != \
                    video_basename:
                    continue

                start_ixs = video_df[start_ixs_header].values
                stop_ixs = video_df[stop_ixs_header].values
                behaving = video_df[behaving_header].values
                n_frames = tracking_summary.video.duration_frames

                classification_arr = np.zeros(n_frames)
                for i in xrange(start_ixs.size):
                    start = start_ixs[i]
                    stop = stop_ixs[i]
                    classification_arr[start:stop] = behaving[i]

                tracking_summary.male.add_behavior_from_array(
                    behavior_name,
                    classification_arr
                )
                break

    def plot_behavioral_distances_cartesian(
        self,
        behavior_name,
        num_bins=50,
        colors=None,
        linewidth=2,
        alpha=0.5,
        **kwargs
        ):
        """Generates a plot of mean male-female distances for a
        specified behavior from -np.pi to np.pi.

        .. note: To generate a plot in polar coordinates, use
        plot_behavioral_distances_polar().

        Parameters
        ----------
        behavior_name : string
            Name of behavior to plot. This should be a valid key
            in the TrackingSummary.behaviors attribute.

        num_bins : int (default=50)
            Number of bins to use to calculate distances across.

        colors : list of any valid matplotlib color or None (default=None)
            Color of mean line for each group. If None, random colors will be
            chosen.

        linewidth : int (default=2)
            Linewidth of mean line.

        alpha : float (optional, default=0.5)
            Alpha for fill_between().

        **kwargs :
            Keyword arguments to be passed to matplotlib.axes.plot().
            These arguments will affect the mean line.

        Returns
        -------
        fig, ax : matplotlib figure & axes handle.
        """
        if colors is None:
            colors = ['C{}'.format(i % 10) for i in xrange(len(self.order))]
        else:
            if len(colors) != len(self.order):
                raise AttributeError(
                    'color list must contain same number of items as' +
                    'groups.'
                )

        rs = self.get_behavioral_distances(behavior_name, num_bins=num_bins)
        thetas = np.linspace(-np.pi, np.pi, num_bins)

        fig, ax = plt.subplots()

        for i, group in enumerate(self.order):
            mean_rs = np.nanmean(rs[group], axis=0)
            sem_rs = np.nanstd(rs[group], axis=0) / np.sqrt(rs[group].shape[0])

            ax.plot(
                thetas,
                mean_rs,
                color=colors[i],
                linewidth=linewidth,
                zorder=1,
                **kwargs
            )

            ax.fill_between(
                thetas, mean_rs - sem_rs,
                mean_rs + sem_rs,
                color=colors[i],
                alpha=alpha,
                zorder=0
            )

        return fig, ax

    def hierarchize_behaviors(self, behavior_names):
        """Create a behavioral hierarchy for each male fly in this Experiment.

        Behaviors in the hierarchy are organized such that behaviors with more
        precedence will never overlap behaviors with less precedence. And if
        a behavior with less precedence has overlapping bouts with a behavior
        of higher precedence, the bouts in the lesser behavior are simply
        removed.

        Parameters
        ----------
        behavior_names : list of string
            List of behaviors. Behaviors appearring earlier in the list take
            precedence over behaviors appearing later in the list. Thus
            b0 > b1 > b2 > etc.

        Returns
        -------
        hierarchized_behavior_names : list of string
            Each item is a behavioral key that has been added to each male in
            this Experiment. The list is ordered such similarly to
            `behavior_names`, with items appearing earlier in this list taking
            greater precedence than those appearing later.
        """
        for group_name, tracking_summary in self.itergroups():
            new_ts = behaviors.hierarchize_ts(
                tracking_summary, behavior_names
                )

        hierarchy_names = [
            name for name in new_ts.male.list_behaviors() \
            if '(hierarchy)' in name
            ]
        order = np.argsort([len(n) for n in hierarchy_names])
        return [hierarchy_names[i] for i in order]

    def get_behavioral_indices(self,
        behavior_name,
        include_nonbehavors=False,
        method='condensed'
        ):
        """Gets the behavioral indices for all males contained in this
        Experiment.

        The behavioral index is the fraction of time the male spends engaging in
        a specified behavior with a female. This can be either calculated from
        the beginning of the video recording (method='all') or from the time the
        male first engages in the behavior (method='condensed').

        Parameters
        ----------
        behavior_name : string
            Name of behavior. This should be a key within
            FixedCourtshipTrackingSummary.behaviors.

        include_nonbehavors : bool (optional, default=True)
            Whether or not non-behavors should be included in the analysis.

        method : string (optional, default='condensed')
            Whether to start calculating the index from the start of the
            behavioral trial ('all') or from the start of the specified
            behavior ('condensed').

        Returns
        -------
        behavioral_index : dictionary
            Keys are group names, values are np.array of behavioral indices for
            each male within its group.
        """
        indices = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            index = behaviors.behavioral_index(
                    summary.male.get_behavior(behavior_name).as_array(),
                    method=method
                    )

            if index == 0 and not include_nonbehavors:
                continue
            else:
                indices[group_name].append(index)

        return {name: np.asarray(vals) for name, vals in indices.iteritems()}

    def get_behavioral_latencies(self,
        behavior_name,
        include_nonbehavors=False,
        return_type='seconds'
        ):
        """Gets the behavioral latencies for all males contained in this
        Experiment.

        Parameters
        ----------
        behavior_name : string
            Name of behavior. This should be a key within
            FixedCourtshipTrackingSummary.behaviors.

        include_nonbehavors : bool (optional, default=False)
            Whether or not non-behavors should be included in the analysis.

        return_type : string (optional, default='seconds')
            Whether to return in the latencies in 'seconds' or 'frames'.

        Returns
        -------
        behavioral_latency : dictionary
            Keys are group names, values are np.array of behavioral latencies
            for each male within its group.
        """
        latencies = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            latency = behaviors.behavioral_latency(
                summary.male.get_behavior(behavior_name).as_array(),
                dt=1.
            )

            if int(latency) == int(summary.video.duration_frames) \
                and not include_nonbehavors:
                continue

            if return_type == 'seconds':
                latency /= (1. * summary.video.duration_frames /
                    summary.video.duration_seconds)
            elif return_type == 'frames':
                pass
            else:
                raise AttributeError(
                    '`return_type` must be either \'frames\''+
                    ' or \'seconds\''
                    )

            latencies[group_name].append(latency)

        return {name: np.asarray(vals) for name, vals in latencies.iteritems()}

    def get_behavioral_index_as_fraction_of_courtship(self,
        behavior_name,
        courtship_behavior_name,
        include_nonbehavors=False
        ):
        """Gets the fraction of time that a male spent engaging in a specified
        behavior as a fraction of courtship.

        Parameters
        ----------
        behavior_name : string
            Behavior to get index as fraction of courtship. Must be a valid
            behavior name in all males in this experiment.

        courtship_behavior_name : string
            Name of courtship behavior. Must be a valid behavior name in all
            males in this experiment.

        include_nonbehavors : bool (optional, default=False)
            Whether or not to include flies that did not engage in courtship in
            the returned arrays.

        Returns
        -------
        dictionary :
            Keys are group names, values are np.arrays of behavioral indices for
            each male in its group. If non-behavors are not excluded, then
            np.nans will be returned for those non-behavors.
        """
        indices = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            behavior_arr = summary.male.get_behavior(behavior_name).as_array()
            courtship_arr = summary.male.get_behavior(courtship_behavior_name) \
                                        .as_array()

            behavior_sum = np.sum(behavior_arr)
            courtship_sum = np.sum(courtship_arr)

            if courtship_sum == 0 and not include_nonbehavors:
                continue

            if courtship_sum == 0:
                index = np.nan
            else:
                index = 1. * behavior_sum / courtship_sum

            indices[group_name].append(index)

        return {name: np.asarray(vals) for name, vals in indices.iteritems()}

    def get_binned_forward_velocities(
        self,
        behavior_name=None,
        num_bins=50
        ):
        """Gets the forward velocity components for each fly in each group at
        all angular bins surround the female.

        Parameters
        ----------
        behavior_name : string or None, optional (default=None)
            This function will only calculate velocities at frames that are
            positively classified for the specified behavior. Note that
            `behavior_name` must be a valid behavioral key in each
            TrackingSummary in this object. If None, velocities will be
            calculated for all frames.

        num_bins : int, optional (default=50)
            How many bins to split the area around the female into.

        Returns
        -------
        forward_velocities : dict of np.ndarray
            Each key is one of the groups in this object. Each value is an
            np.ndarray of shape [n_flies, num_bins]. Each row represents the
            velocities for a single fly. Each column represents an angular bin
            around the female.
        """
        v_fwd = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            v_fwd[group_name].append(
                spatial.binned_forward_velocity(
                    summary, behavior_name, num_bins).tolist()
            )
        return {name: np.asarray(vals) for name, vals in v_fwd.iteritems()}

    def get_binned_abs_sideways_velocities(
        self,
        behavior_name=None,
        num_bins=50
        ):
        """Gets the absolute sideways velocity components for each fly in each
        group at all angular bins surround the female.

        Parameters
        ----------
        behavior_name : string or None, optional (default=None)
            This function will only calculate velocities at frames that are
            positvely classified for the specified behavior. Note that
            `behavior_name` must be a valid behavioral key in each
            TrackingSummary in this object. If None, velocities will be
            calculated for all frames.

        num_bins : int, optional (default=50)
            How many bins to split the area around the female into.

        Returns
        -------
        abs_sideways_velocities : dict of np.ndarray
            Each key is one of the groups in this object. Each value is an
            np.ndarray of shape [n_flies, num_bins]. Each row represents the
            velocities for a single fly. Each column represents an angular bin
            around the female.
        """
        v_abs_sideways = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            v_abs_sideways[group_name].append(
                spatial.binned_abs_sideways_velocity(
                    summary, behavior_name, num_bins).tolist()
            )
        return {name: np.asarray(vals) for name, vals in v_abs_sideways.iteritems()}

    def get_binned_sideways_velocities(
        self,
        behavior_name=None,
        num_bins=50
        ):
        """Gets the sideways velocity components for each fly in each group at
        all angular bins surround the female.

        Parameters
        ----------
        behavior_name : string or None, optional (default=None)
            This function will only calculate velocities at frames that are
            positvely classified for the specified behavior. Note that
            `behavior_name` must be a valid behavioral key in each
            TrackingSummary in this object. If None, velocities will be
            calculated for all frames.

        num_bins : int, optional (default=50)
            How many bins to split the area around the female into.

        Returns
        -------
        sideways_velocities : dict of np.ndarray
            Each key is one of the groups in this object. Each value is an
            np.ndarray of shape [n_flies, num_bins]. Each row represents the
            velocities for a single fly. Each column represents an angular bin
            around the female.
        """
        v_sideways = {group_name: [] for group_name in self.order}
        for group_name, summary in self.itergroups():
            v_sideways[group_name].append(
                spatial.binned_sideways_velocity(
                    summary, behavior_name, num_bins).tolist()
            )
        return {name: np.asarray(vals) for name, vals in v_sideways.iteritems()}

    def save_behavioral_data(self,
        savename,
        behavior_names,
        courtship_behavior_name
        ):
        """Saves behavioral data from this experiment to the specified file.

        Behavioral data includes the following for each of the specified
        behaviors:
            1. Mean angular positions of male w.r.t. female across all
            behavioral bouts.
            2. Mean radial position of male w.r.t. female across all behavioral
                bouts.
            3. Behavioral index (as a fraction of courtship) for the male. This
                is the fraction of time that the male spends engaging in the
                specified behavior as a fraction of the time spent in courtship.
            4. Behavioral latency for the male. How long it took the male to
                start engaging in the specified behavior.

        The courtship index and latency will also be saved for each male, as
        will the mean courtship distance while the male is on the front half of
        the female and rear half of the female.

        Parameters
        ----------
        savename : string
            Where to save file. This should be a .csv file.

        behavior_names: dictionary
            Keys should be desired names to save in column headers. Values
            should be valid behavior names present in each male in this
            Experiment.

        courtship_behavior_name : string
            Name of courtship behavior to use for calculating behavioral
            indices.
        """
        summary_df = pd.DataFrame()

        group_names_arr = []
        for group_name in self.order:
            group_names_arr += [group_name] * len(getattr(self, group_name))
        summary_df['group'] = group_names_arr

        for save_name, behavior_name in behavior_names.iteritems():
            thetas = self.get_ang_location_summary(behavior_name, clean=False)
            indices = self.get_behavioral_index_as_fraction_of_courtship(
                behavior_name=behavior_name,
                courtship_behavior_name=courtship_behavior_name,
                include_nonbehavors=True
            )
            latencies = self.get_behavioral_latencies(behavior_name,
                include_nonbehavors=True
            )
            distances = self.get_behavioral_distances(
                behavior_name,
                num_bins=None,
                include_nonbehavors=True
                )

            thetas_list = []
            indices_list = []
            latencies_list = []
            distances_list = []
            for group_name in self.order:
                thetas_list += thetas[group_name].tolist()
                indices_list += indices[group_name].tolist()
                latencies_list += latencies[group_name].tolist()
                distances_list += distances[group_name].tolist()

            summary_count = np.sum(
                [len(getattr(self, group_name)) for group_name in self.order])
            assert summary_count == len(thetas_list), '{} != N, {} != {}'     \
                            .format('thetas', len(thetas_list), summary_count)
            assert summary_count == len(indices_list), '{} != N, {} != {}'    \
                            .format('indices', len(indices_list), summary_count)
            assert summary_count == len(latencies_list), '{} != N, {} != {}'  \
                            .format('latencies', len(latencies_list),
                            summary_count)
            assert summary_count == len(distances_list), '{} != N, {} != {}'  \
                            .format('distances', len(distances_list),
                            summary_count)

            summary_df[save_name + '-theta'] = thetas_list
            summary_df[save_name + '-distance'] = distances_list
            summary_df[save_name + '-index'] = indices_list
            summary_df[save_name + '-latency'] = latencies_list

        ci_all = self.get_behavioral_indices(
            courtship_behavior_name, include_nonbehavors=True,
            method='all')
        ci_cond = self.get_behavioral_indices(
            courtship_behavior_name, include_nonbehavors=True,
            method='condensed')
        cl = self.get_behavioral_latencies(
            courtship_behavior_name, include_nonbehavors=True
        )
        dist_front = self.get_behavioral_distances_peak(
            courtship_behavior_name,
            peak='front')
        dist_rear = self.get_behavioral_distances_peak(
            courtship_behavior_name,
            peak='rear'
        )

        ci_all_list = []
        ci_cond_list = []
        cl_list = []
        dist_front_list = []
        dist_rear_list = []
        for group_name in self.order:
            ci_all_list += ci_all[group_name].tolist()
            ci_cond_list += ci_cond[group_name].tolist()
            cl_list += cl[group_name].tolist()
            dist_front_list += dist_front[group_name].tolist()
            dist_rear_list += dist_rear[group_name].tolist()

        assert summary_count == len(ci_all_list), '{} != N, {} != {}'     \
                        .format('ci_all', len(ci_all_list), summary_count)
        assert summary_count == len(ci_cond_list), '{} != N, {} != {}'    \
                        .format('ci_cond', len(ci_cond_list), summary_count)
        assert summary_count == len(cl_list), '{} != N, {} != {}'  \
                        .format('cl', len(cl_list), summary_count)
        assert summary_count == len(dist_front_list), '{} != N, {} != {}'  \
                        .format('dist_front', len(dist_front_list),
                        summary_count)
        assert summary_count == len(dist_rear_list), '{} != N, {} != {}'  \
                        .format('dist_rear', len(dist_rear_list), summary_count)

        summary_df['courtship-index-all'] = ci_all_list
        summary_df['courtship-index-cond'] = ci_cond_list
        summary_df['courtship-latency'] = cl_list
        summary_df['courtship-dist-front'] = dist_front_list
        summary_df['courtship-dist-rear'] = dist_rear_list

        summary_df.to_csv(savename, na_rep='NA', index=False)

    def save_behavioral_matrices(self,
        savename,
        behavior_name,
        sort='ascending',
        downscale_bin_size=24
        ):
        """Saves behavioral matrices as a .csv file.

        Parameters
        ----------
        savename : string
            Where to save file. Must be .csv.

        behavior_name : string
            Which behavior should matrices be saved for? Needs to be a valid
            behavior name in all males in this Experiment.

        sort : string (optional, default='ascending')
            How to sort behavioral matrices. Can be either 'ascending' or
            'descending'. If None, no sorting will occur.

        downscale_bin_size : int
            Size of bin to downscale each row of matrix. If you want the
            resulting behavioral matrices to be in seconds, set this to
            video frames per second.
        """
        mats = self.get_behavioral_matrices(behavior_name, sort=sort)

        downscaled_mats = {}
        downscaled_rows = 0
        downscaled_cols = int(self.video_duration_frames / downscale_bin_size)

        for gn, mat in mats.iteritems():
            ds_mat = np.zeros(
                shape=(mat.shape[0], downscaled_cols)
                )
            downscaled_rows += mat.shape[0]
            for i in xrange(0, mat.shape[1], downscale_bin_size):
                if (i + downscale_bin_size) <= mat.shape[1] - 1:
                    end_ix = i + downscale_bin_size
                else:
                    end_ix = mat.shape[1] - 1
                ds_mat[:, i/downscale_bin_size] = \
                    np.median(mat[:, i:end_ix], axis=1).astype(np.int)

            downscaled_mats[gn] = ds_mat

        downscaled_mats_combined = np.zeros(
            shape=(downscaled_rows, downscaled_cols))
        current_row = 0
        groups = []
        for group_name in self.order:
            mat = downscaled_mats[group_name]
            num_rows = mat.shape[0]
            downscaled_mats_combined[current_row:current_row+num_rows, :] = mat
            groups += [group_name] * num_rows
            current_row += num_rows

        mat_df = pd.DataFrame(downscaled_mats_combined)
        mat_df['groups'] = groups
        mat_df.to_csv(savename, na_rep='NA', index=False)

    def save_behavioral_transitions(self,
        savename,
        behavior_names,
        behavior_order
        ):
        """
        Parameters
        ----------
        savename : string

        behavior_names : dictionary
            Keys should be display names in output .csv file. Values should be
            valid behavior names in each male in this Experiment.

        behavior_order : list of string
            How to order the behaviors listed in behavior_names. These should
            be valid key names in `behavior_names`.
        """
        bnames = [behavior_names[key] for key in behavior_order]
        for group_name in self.order:
            tm = markov.get_transition_matrix(self, group_name, bnames)