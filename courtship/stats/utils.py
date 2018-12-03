
from copy import deepcopy

import numpy as np
from scipy import stats


def center_angles(angles):
    """Assures all angles fall between -pi and +pi.

    Parameters
    ----------
    angles : np.ndarray of shape [N]
        Angular values.

    Returns
    -------
    centered_angles : np.ndarray of shape [N]
        All angles are guarenteed to fall between +/-pi.
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def clean_paired_data(arr1, arr2):
    """Drops any values from indexes with np.nan values in arrays containing
    paired data."""
    drop_from1 = np.isnan(arr1).astype(np.uint8)
    drop_from2 = np.isnan(arr2).astype(np.uint8)
    drop = drop_from1 + drop_from2
    return arr1[~drop.astype(np.bool)], arr2[~drop.astype(np.bool)]


def clean_dataset(data):
    """Removes any nans from the passed dataset.

    Parameters
    ----------
    data : dict
        Dictionary containing lists of numeric values. ie:
        {'g1': [], 'g2': [], ...}

    Returns
    -------
    data : dict
        Dictionary with np.nans removed from each list.
    """
    clean_data = {g: [] for g in data.keys()}
    for group_name, lst in data.iteritems():
        for val in lst:
            if np.isnan(val):
                continue
            clean_data[group_name].append(val)
    return clean_data


def remove_outliers(d):
    """Removes outliers from a dataset.

    An outlier is defined as a value that falls outside of the range:

    [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]

    Where Q1 is the first quartile, Q3 is the third quartile, and IQR is the
    inter-quartile range.

    Parameters
    ----------
    d : dictionary
        Each value in the dict should be numeric, array-like.

    Returns
    -------
    d : dictionary
        Dataset with outliers removed.
    """
    clean_d = {g: [] for g in d.keys()}
    for group_name, group_vals in d.iteritems():
        q1 = np.percentile(group_vals, 25)
        q3 = np.percentile(group_vals, 75)
        iqr = stats.iqr(group_vals)
        assert iqr == (q3 - q1), '{} (q3-q1) != {} (iqr)'.format(q3-q1, iqr)
        for val in group_vals:
            if val < (q1 - 1.5*iqr) or val > (q3 + 1.5*iqr):
                continue
            clean_d[group_name].append(val)
    return clean_d


def is_numeric(val):
    """Checks whether a value is of a numeric type."""
    return type(val) in [int, float, long, complex]


def is_array_like(val):
    """Checks whether a value is array-like (either numpy.ndarray or list)."""
    return type(val) in [list, np.ndarray]


def pairwise_kruskal(data, sig=0.05, clean=True, print_results=True):
    """Runs pairwise Kruskal tests on a given data set.

    Parameters
    ----------
    data : dict
        Each value should be a numpy array.

    sig : float (optional, default=0.05)
        p-value required for significance. This will be adjusted using
        a Bonferonni correction.

    clean : bool (optional, default=0.05)
        Whether or not to clean the dataset.

    Returns
    -------
    p-vals : dict
        Dictionary where keys are group comparisons, and values are pairwise
        p-values determined via scipy.kruskal.
    """
    data = deepcopy(data)

    if clean:
        data = clean_dataset(data)

    for key in data.keys():
        data[key] = np.asarray(data[key])

    group_names = sorted(data.keys())
    num_groups = len(group_names)
    num_comparisons = num_groups * (num_groups-1)/2

    msg = (
        'Total number of comparisons: {}'.format(num_comparisons) +
        ' | sig p < {}'.format(float(sig)/num_comparisons) + '\n'
    )

    p_vals = {}
    for i in xrange(len(group_names)):
        for j in xrange(i):
            p = stats.kruskal(data[group_names[i]], data[group_names[j]]).pvalue
            p_vals['{} v {}'.format(group_names[i], group_names[j])] = p
            msg += '{} v {}: p = {}'.format(group_names[i], group_names[j], p)
            if p < float(sig)/num_comparisons:
                msg += '*'
            msg += '\n'

    if print_results:
        print msg

    return p_vals