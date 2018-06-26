
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
