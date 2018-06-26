# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Functions for analyzing and scoring behaviors.

.. moduleauthor:: Ross McKinney
"""
import numpy as np

import markov
import spatial

def behavioral_index(arr, method='all'):
    """Finds the fraction of time that an individual was engaging in a behavior.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavior of interest.

    method : string (default = 'all')
        How to score the index. If 'all', the index is calculated simply as
        the sum of the binary array divided by the size of the binary array.
        If 'condensed', the index is calculated from the first index in the
        binary array containing a positive value.

    Returns
    -------
    index : float
        If method is 'all', index will always be a positive floating value
        within the range [0.0, 1.0].

    Examples
    --------
    >>> arr1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    >>> behavioral_index(arr1, method = 'all')  #returns 5. / 10. = 0.5
    >>> behavioral_index(arr2, method = 'condensed')  #returns 5. / 6. = 0.83
    """
    arr = np.asarray(arr)

    if method == 'all':
        return np.sum(arr) * 1. / arr.size

    pos_ix = np.flatnonzero(arr)
    if pos_ix.size == 0:
        return 0

    return np.sum(arr[pos_ix[0]:]) * 1. / arr[pos_ix[0]:].size


def behavioral_latency(arr, dt=1.):
    """Finds the duration of time taken for an individual to start behaving.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavior of interest.

    dt : float (default = 1.)
        Sampling frequency of binary array (in seconds).
        A value of 1. means that each  index in the binary array represents
        1 second. A value of 24. mean that each index represents 1/24th of
        a second.

    Returns
    -------
    latency : float
        Seconds taken to engage in behavior.

    Examples
    --------
    >>> arr1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    >>> behavioral_latency(arr1, dt = 1.) #returns 5.
    >>> behavioral_latency(arr1, dt = 2.) #returns 5. / 2. = 2.5
    """
    arr = np.asarray(arr)

    pos_ix = np.flatnonzero(arr)

    if pos_ix.size == 0:
        return arr.size * 1. / dt

    return pos_ix[0] * 1. / dt


def bout_boundaries(arr):
    """Finds the bounding indices defining behavioral events.

    Parameters
    ----------
    arr : 1d array-like
        Binary array, specifying behavioral events, in which to find
        bounding indices.

    Returns
    -------
    bounding_ixs : np.ndarray of shape [N, 2]
        First column represents index of start of events,
        second column represents index of stop of events.

    Examples
    --------
    >>> arr = np.ndarray([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    >>> boundaries(arr) #returns np.ndarray([2, 6])
    """
    arr = np.asarray(arr)

    arr = np.hstack((0, arr, 0))  # pad arr with zeros to ensure below works.

    start = np.flatnonzero(np.diff(arr) == 1)
    stop = np.flatnonzero(np.diff(arr) == -1) - 1

    return np.vstack((start, stop)).T


def bout_boundaries_ts(ts, bname):
    """Gets the bout boundaries for a specific behavior from a TrackingSummary.

    Parameters
    ----------
    ts : FixedCourtshipTrackingSummary
        Should contain ts.male and ts.female attributes.

    bname : string
        Behavior to calculate bout boundaries from.

    Returns
    -------
    bout_boundaries : arr of shape [N, 2]
        First column of returned array is the start of individual bouts of
        specified behaviors. Second column is stop of individual bouts of
        behaviors.
    """
    arr = ts.behaviors[bname]
    boundaries = bout_boundaries(arr)
    return boundaries


def bout_durations(arr):
    """Finds the duration of each behavioral bout.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    Returns
    -------
    durations : np.ndarray of shape [n_behavioral_events]
        Durations of each bout.
    """
    return np.diff(bout_boundaries(arr), axis=1).flatten()


def bout_frequency(arr, method='all'):
    """Finds the frequency of behavioral bouts.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    method : string (default = 'all')
        Method to calculate frequency. If 'all' the frequency is calculated
        as the number of bouts divided by the total behavioral duration.
        If 'condensed', the frequency is calculated as the number of bouts
        divided by the duration of time starting from the initiation of the
        specified behavior.

    Returns
    -------
    bout_freq : float
        Frequency as described above.
    """
    bouts = n_bouts(arr)

    if method == 'all':
        return bouts * 1. / arr.size
    elif bouts == 0:
        return 0
    return bouts * 1. / (arr.size - bout_boundaries(arr).flatten()[0])


def n_bouts(arr):
    """Finds the number of behavioral events in a given array.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    Returns
    -------
    n_events : int
        Number of behavioral events.
    """
    return bout_boundaries(arr).shape[0]


def n_pauses(arr):
    """Finds the number of pauses between bouts of behavior.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    Returns
    -------
    n_pauses : int
        Number of pauses between bouts.
    """
    return pause_durations(arr).size


def pause_durations(arr):
    """Finds the duration of pauses (this is equivalent to the inter-
    bout interval).

    The inter-bout interval is defined as the
    durations of time between two consecutive bouts
    of a specified behavior.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    Returns
    -------
    ibi_intervals : np.ndarray of shape [n_bouts - 1]
        Duration of time between consecutive behavioral bouts.
    """
    flat_boundaries = bout_boundaries(arr).flatten()
    return np.diff(flat_boundaries)[1:flat_boundaries.size - 1:2]


def pause_frequency(arr, method='all'):
    """Finds the frequency of behavioral pauses.

    Parameters
    ----------
    arr : 1d array-like
        Binary array containing behavioral events.

    method : string (default = 'all')
        Method to calculate frequency. If 'all' the frequency is calculated
        as the number of pauses divided by the total behavioral duration.
        If 'condensed', the frequency is calculated as the number of pauses
        divided by the duration of time starting from the initiation of the
        specified behavior.

    Returns
    -------
    pause_freq : float
        Frequency as described above.
    """
    pauses = n_pauses(arr)
    if method == 'all':
        return pauses * 1./arr.size
    elif pauses == 0:
        return 0
    return pauses * 1. / (arr.size - bout_boundaries(arr).flatten()[0])


def get_transitional_tracks(
    ts,
    from_behavior,
    to_behavior,
    min_bout_length=1,
    start_padding=0):
    """Gets the (rotated) courtship tracks that occur between the
    stop of one behavior and the start of another.

    Parameters
    ----------
    ts : FixedCourtshipTrackingSummary
        TrackingSummary containing flies and behaviors to get the tracks from.

    from_behavior : string
        Name of behavior fly is transitioning from.

    to_behavior : string
        Name of behavior fly is transitioning to.

    min_bout_length : int, optional (default=1)
        Minimum number of frames containing transitional tracks to return.
        Having this be equal to 1 assures that there are at least some tracks
        to return. If this is 0 it means that there could be instances where
        the stop of from_behavior and the start of to_behavior are equal, which
        will return an empty array.

    start_padding : int, optional (default=0)
        The number of frames prior to the stop of from_behavior to include
        in returned tracks.

    Returns
    -------
    x, y : list, list
        Each item in the list is an np.ndarry containing the x- and y-
        coordinates of the transitional tracks. Note that these are
        cartesian coordinates and represent male-to-female centroid-to-centroid
        distances in mm.
    """
    from_ixs = bout_boundaries_ts(ts, from_behavior)
    to_ixs = bout_boundaries_ts(ts, to_behavior)

    from_ixs_mean = np.mean(from_ixs, axis=1)
    to_ixs_mean = np.mean(to_ixs, axis=1)

    # from behavior will be labeled as 1
    # to behavior will be labeled as 2
    labeled_arr, sorted_ixs = markov.sort_and_merge(
        from_ixs_mean,
        to_ixs_mean,
        np.ones(from_ixs_mean.size),
        2)

    transition_ixs = np.where(np.diff(sorted_ixs) == 1)[0]

    all_thetas, all_rs = spatial.relative_position2(ts.male, ts.female)
    all_rs /= ts.pixels_per_mm

    # convert to cartesian coordinates
    all_xs = all_rs * np.cos(all_thetas)
    all_ys = all_rs * np.sin(all_thetas)

    xs, ys = [], []
    for ix in transition_ixs:
        from_count = np.where(sorted_ixs[:ix] == 1)[0].size
        to_count = np.where(sorted_ixs[:ix] == 2)[0].size

        start = from_ixs[from_count, 1]
        stop = to_ixs[to_count, 0]

        if (stop - start) < min_bout_length:
            continue

        xs.append(all_xs[start-start_padding:stop])
        ys.append(all_ys[start-start_padding:stop])

    return xs, ys


def align_transitions(xs, ys, quadrant=3):
    """Aligns transitional tracks by centering them at the origin.
    Only tracks that start within the specified quadrant will be
    included in the returned coordinates.

    Parameters
    ----------
    xs : list of np.ndarray of length [N].
        Each item in the list are the x-coordinates of an individual bout of
        behavior.
    ys : list of np.ndarray of length [N].
        Each item in the list are the y-coordinates of an individual bout of
        behavior.

    quadrant : int, optional (default=3; options are 1,2,3,4).
        For which quadrant do you want to align transitional tracks?

    Returns
    -------
    cx : list of np.ndarray of shape [M].
        Centered x-coordinates of behavioral bouts occuring in quadrant around
        female.
    cy : list of np.ndarray of shape [M].
        Centered y-coordinates of behavioral bouts occuring in quadrant around
        female.
    """
    centered_xs, centered_ys = [], []
    for i in xrange(len(xs)):
        x0 = xs[i][0]
        y0 = ys[i][0]

        if x0 <= 0:
            if y0 <= 0:
                q = 3
            else:
                q = 2
        else:
            if y0 <= 0:
                q = 4
            else:
                q = 1

        if quadrant != q:
            continue

        cx = xs[i] - x0
        cy = ys[i] - y0

        centered_xs.append(cx)
        centered_ys.append(cy)
    return centered_xs, centered_ys


def exclude_behaviors(focal, exclude_lst):
    """Excludes behaviors (in exclude_lst) from a focal behavior.

    Parameters
    ----------
    focal : np.ndarray
        Binary.

    exclude_lst : list of np.ndarray
        All behaviors will be excluded from focal. All members
        of the list are binary arrays.

    Returns
    -------
    focal_exc : np.ndarray
        Binary.
    """
    focal = focal.copy()
    for b in exclude_lst:
        focal[np.where(b)] = 0
    return focal


def exclude_behavior_from_courtship_ts(
    ts,
    exclude_behavior='attempted-copulation',
    courtship_behavior='courtship_gt'
    ):
    """Excludes bouts of one behavior from another for a specified
    TrackingSummary.

    Parameters
    ----------
    ts : TrackingSummary

    exclude_behavior : string (optional, default='attempted-copulation')
        Must be a valid key in ts.behaviors dictionary.

    courtship_behavior : string (optional, default='courtship_gt')
        Name of courtship behavior. Must be a valid key in ts.behaviors
        dictionary.

    Returns
    -------
    ts : TrackingSummary
    new_behavior_name : string
        Name of behavior. This behavior & behavioral key is now present in the
        TrackingSummary.
    """
    courtship = ts.behaviors[courtship_behavior].copy()
    exclude = ts.behaviors[exclude_behavior]

    # where courtship and exclude behaviors overlap
    courtship[np.where(exclude)] = 0

    new_behavior_name = '{}-excluding-{}'.format(
        courtship_behavior, exclude_behavior)
    ts.behaviors[new_behavior_name] = courtship
    return ts, new_behavior_name


def exclude_behavior_from_courtship_exp(
    exp,
    exclude_behavior='attempted-copulation',
    courtship_behavior='courtship_gt'
    ):
    """Excludes bouts of one behavior from another for a specified
    FixedCourtshipExperiment.

    Parameters
    ----------
    exp : FixedCourtshipExperiment

    exclude_behavior : string (optional, default='attempted-copulation')
        Must be a valid key in ts.behaviors dictionary.

    courtship_behavior : string (optional, default='courtship_gt')
        Name of courtship behavior. Must be a valid key in ts.behaviors
        dictionary.

    Returns
    -------
    exp : FixedCourtshipExperiment
    excluded_bname : string
        Name of behavior. This behavior & behavioral key is now present in the
        TrackingSummary.
    """
    for group_name, group in exp.get_groups().iteritems():
        for ind in group:
            _, excluded_bname = exclude_behavior_from_courtship_ts(ind)
    return exp, excluded_bname


def hierarchize(behaviors):
    """Makes sure that each subsequent behavior does not contain any behavioral
    locations from any previous behavior.

    Parameters
    ----------
    behaviors : list of np.ndarray
        Each array should be binary (contain only zeros and ones). The list
        is hierarchized such that the first item takes the most precedent, and
        the last, the least.

    Returns
    -------
    hierarchized_behaviors : list of np.ndarray
        Each array is binary.
    """
    h_behav = []
    for i in xrange(len(behaviors)-1, -1, -1):
        h = exclude_behaviors(behaviors[i], behaviors[:i])
        h_behav.append(h)
    return h_behav[::-1]


def hierarchize_ts(ts, bnames):
    """Creates a behavioral hierarchy for each of the passed behavior names.

    Parameters
    ----------
    ts : TrackingSummary
    bnames : list of string
        These must be valid behavior.keys

    Returns
    -------
    ts : TrackingSummary
        Now contains behaviors that have been heirarchized. The hierarchized
        behaviors have bname + '_hierarchized' as their behavioral key.
    """
    for b in bnames:
        if b not in ts.behaviors.keys():
            raise AttributeError('{} not found in '.format(b) +
                'TrackingSummary.behaviors dictionary.')

    behaviors = [ts.behaviors[b] for b in bnames]
    hierarchized_behaviors = hierarchize(behaviors)
    for i in xrange(len(bnames)):
        ts.behaviors[bnames[i] + '_hierarchized'] = hierarchized_behaviors[i]
    return ts


def hierarchize_exp(exp, bnames):
    """For each trackingsummary in the experiment, hierarchize the specified
    behaviors.

    Parameters
    ----------
    exp : FixedCourtshipExperiment
    bnames : list of string

    Returns
    -------
    exp : FixedCourtshipExperiment
    hierarchized_bnames : list of string
        Keys used to find hierarchized behaviors in each TrackingSummary's
        behaviors dictionary.
    """
    for group_name, group in exp.get_groups().iteritems():
        for ind in group:
            _ = hierarchize_ts(ind, bnames)

    hierarchized_bnames = [b + '_hierarchized' for b in bnames]
    return exp, hierarchized_bnames


def get_fraction_behaving(exp, bname, courtship_key='courtship_gt'):
    """Finds the fraction of courtship that each animal in each group was
    engaging in a specified behavior."""
    fracs = {g: [] for g in exp.order}
    for group_name, group in exp.get_groups().iteritems():
        for ts in group:
            fracs[group_name].append(
                np.nansum(ts.behaviors[bname])/np.nansum(
                    ts.behaviors[courtship_key]
                )
            )

    return fracs
