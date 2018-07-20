# experiment.py

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    
