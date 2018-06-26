# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Functions for general signal processing tasks.

.. moduleauthor:: Ross McKinney
"""
import numpy as np
import _wavelet as wavelet


def derivative(arr, timestamps):
    """Calculates the derivative of an array given timestamps.

    Parameters
    ----------
    arr : np.ndarray of shape [N]
        Array to calculate derivate over

    timestamps : np.ndarray of shape [N]
        Array containing timestamps at which values stored in arr
        were calculated.

    Returns
    -------
    derivative : np.ndarray of shape [N]
        Zero-padded array (on the front) containing d_arr/d_t values.
    """
    dt = np.diff(timestamps)
    darr_dt = np.gradient(arr, np.hstack((np.mean(dt), dt)))
    return darr_dt


def normalize(arr):
    """Normalizes an array to its mean values.

    Parameters
    ----------
    arr : array-like | shape = [N]
        The array to normalize.

    Returns
    -------
    normalized_array : np.ndarray | shape = [arr.shape]
    """
    return arr / np.mean(arr)


def pad(arr, padding, pad_with_zeros=False):
    """Pads an array with values.

    The front of the array will be padded with the value
    at index 0 of the array, and the end of the array will
    be padded with the value at index -1 of the array.

    Sliding windows passed across a padded array yeild
    final arrays that are not zero-padded.

    Parameters
    ----------
    arr : array-like | shape = [N]
        Array to pad.

    padding : int
        The number of values to pad on each side of the array.
        This is one-sided. For example, setting padding = 20
        would introduce 20 values onto the front of arr and
        20 values onto the end of the arr.

    pad_with_zeros : boolean (default = False)
        Whether or not to pad the array with zeros rather than
        with values at indices 0 and -1.

    Returns
    -------
    padded_arr : np.ndarray | shape = [N + 2 * padding]
    """
    if pad_with_zeros:
        start_val = 0
        end_val = 0
    else:
        start_val = arr[0]
        end_val = arr[-1]

    return np.hstack((
        np.repeat(start_val, padding),
        arr,
        np.repeat(end_val, padding)
        ))


def get_frequencies(
    signal,
    periods=5,
    sampling_freq=30,
    min_freq=1,
    max_freq=25,
    omega0=5
):
    """Calculates frequency data (harmonics) for a signal using the
    Morlet wavelet transform.

    Parameters
    ----------
    signal : np.ndarray | shape = [N]
        The signal that you want to transform via the Morlet wavelet transform.

    periods : int (default = 5)
        The number of frequency bands to decompose the signal into.

    sampling_freq : int (default = 30)
        How frequently the signal was sampled (in Hertz).

    min_freq : int (default = 1)
        Minimum frequency band to examine.

    max_freq : int (default = 25)
        Maximum frequency band to examine.

    omega0 : int (default = 5)
        Intrinsic parameter required for Morlet transform.

    Returns
    -------
    signal_frequencies : np.ndarray | shape = [periods, signal.size]
        Transformed signal.

    References
    ----------
    .. [1] Berman GJ et al., J R Soc Interface 2014.
       https://github.com/gordonberman/MotionMapper/blob/master/
       wavelet/fastWavelet_morlet_convolution_parallel.m
    """
    dt = 1. / sampling_freq
    minT = 1. / max_freq
    maxT = 1. / min_freq
    Ts = minT * 2**((np.arange(periods)) *
                    np.log(maxT/minT)/(np.log(2)*(periods)))
    f = 1. / Ts
    f = f[::-1]

    morlet = wavelet.Morlet(dt, f, omega0)

    # we need to pad the signal if there are an even number of elements
    if signal.size % 2 == 0:
        signal = np.append(signal, 0)
    signal_frequencies = morlet.transform(signal)
    return signal_frequencies


def _get_windowed_ixs(window_size, frame_position):
    """Finds indices for (1) a specific position within a window and
    (2) all other positions within the same window.

    DEPRECIATED

    Parameters
    ----------
    window_size : int
        Number of elements within a particular window.

    frame_position : string
        The specific frame to pull out of the window.
        Allowed values are 'start', 'middle', and 'stop'.

    Returns
    -------
    frame_ix : int
        The index of the element at frame_position.

    surrounding_ix : np.ndarray | shape = [window_size - 1]
        The indices of all elements within a window,
        excluding frame_ix.
    """
    if frame_position == 'start':
        frame_ix = 0
        surrounding_ix = np.arange(1, window_size)
    elif frame_position == 'middle':
        frame_ix = window_size / 2
        surrounding_ix = np.hstack((
            np.arange(0, window_size/2),
            np.arange(window_size/2 + 1, window_size)
            ))
    else:
        frame_ix = window_size - 1
        surrounding_ix = np.arange(0, window_size - 1)

    return frame_ix, surrounding_ix


def window_function(arr, statistic=np.mean, window_size=3):
    """Calculates statistics on an array using a sliding window.

    DEPRECIATED: use scipy.ndimage.filters.generic_filter instead.

    Parameters
    ----------
    arr : np.ndarray | shape = [N]
        The signal for which you want to calculate a windowed statistic.

    statistic : function (default = numpy.mean)
        Statistical function to use on each window.

    window_size : int (default = 3)
        The size of the window used for binning.

    Returns
    -------
    sliding_stat : np.ndarray | shape = arr.shape
    """
    if window_size > arr.size:
        window_size = arr.size

    sliding_stat = np.zeros(arr.size + window_size)

    # pad ends of array with values found at first and last index
    padded_arr = pad(arr, window_size/2)

    for i in xrange(arr.size):
        window = padded_arr[i:(i + window_size)]
        sliding_stat[i+window_size/2] = statistic(window)

    return sliding_stat[window_size/2:sliding_stat.size - window_size/2]


def diff_neighbor_function(
    arr,
    statistic=np.mean,
    window_size=3,
    frame_position='middle'
):
    """Calculates the difference between a specified frame's value and a
    statistic taken on a window surrounding the specified frame.

    DEPRECIATED: use diff class instead.

    Parameters
    ----------
    arr : np.ndarray | shape = [N]
        The signal for which you want to calculate a windowed difference.

    statistic : function (default = numpy.mean)
        Statistical function to use on each window.

    window_size : int (default = 3)
        The size of the window used for binning.

    frame_position : string (default = 'middle')
        The specified frame within a sliding window from which a difference
        with the surrounding values in the window is calculated.
        Allowed values are ('start', 'middle', and 'end').

    Returns
    -------
    sliding_diff : np.ndarray | shape = arr.shape
    """
    sliding_diff = np.zeros(arr.size + window_size)

    # pad ends of array with values found at first and last index
    padded_arr = pad(arr, window_size/2)

    for i in xrange(arr.size):
        window = padded_arr[i:(i + window_size)]

        frame_ix, surrounding_ix = _get_windowed_ixs(
            window_size, frame_position)

        frame_value = window[frame_ix]
        surrounding_window_stat = statistic(window[surrounding_ix])
        sliding_diff[i+window_size/2] = frame_value - surrounding_window_stat

    return sliding_diff[window_size/2:sliding_diff.size - window_size/2]


def z_score_window(arr, window_size=3, frame_position='middle'):
    """Calculates a local z-score for every element within an array
    using a sliding window.

    DEPRECIATED: use z_score class instead.

    Parameters
    ----------
    arr : np.ndarray | shape = [N]
        The signal for which you want to calculate local z-scores.

    window_size : int (default = 3)
        The size of the window used for binning.

    frame_position : string (default = 'middle')
        The specified frame within a sliding window for which a
        z-score is calculated. Other frames within the window,
        not including the specified frame, are used to calculate
        the local z-score. Allowed values are ('start', 'middle',
        and 'end').

    Returns
    -------
    local_z_scores : np.ndarray | shape = [N]
    """
    local_z_scores = np.zeros(arr.size + window_size)

    # pad ends of array with values found at first and last index
    padded_arr = pad(arr, window_size/2)

    for i in xrange(arr.size):
        window = padded_arr[i:(i + window_size)]

        frame_ix, surrounding_ix = _get_windowed_ixs(
            window_size, frame_position)
        frame_value = window[frame_ix]

        if np.std(window[surrounding_ix]) == 0:
            local_z_scores[i+window_size/2] = 0
        else:
            local_z_scores[i+window_size/2] = (
                frame_value - np.mean(
                    window[surrounding_ix])) / np.std(window[surrounding_ix])

    return local_z_scores[window_size/2:local_z_scores.size - window_size/2]


def window_differences(arr):
    """Calculates the difference in mean values at the start
    and end of an array.

    Parameters
    ----------
    arr : np.ndarray | shape = [N]
        The signal for which you want to calculate local z-scores.

    Returns
    -------
    window_diff : np.ndarray | shape = [N]
        Difference in mean of first and second halves of array.
    """
    start_ix = np.arange(arr.size / 2)
    end_ix = np.arange(arr.size / 2, arr.size)
    return np.mean(arr[start_ix]) - np.mean(arr[end_ix])


class diff(object):
    """Class used for custom window difference function generation.

    Parameters
    ----------
    ix : int
        Index within an array whose difference to calculate from a
        statistic taken across the remainder of the array.

    stat : function
        Statistic to calcuate on an array, excluding the value held at ix.
    """
    def __init__(self, ix=0, stat=np.mean):
        self.ix = ix
        self.stat = stat

    def __call__(self, arr):
        """Calculates the difference between a value at a specified position
        in an array and the value of a statistic of all other values within
        the same array.

        Parameters
        ----------
        arr : 1D np.ndarray
            Array

        Returns
        -------
        diff : float
            Difference between value at position ix and statistic calculated
            on remaining array. ie: arr[ix] - stat(arr[not_ix])
        """
        not_ix = np.where(np.arange(arr.size) != self.ix)[0]
        return arr[self.ix] - self.stat(arr[not_ix])


class z_score(object):
    """Class used for custom z-score function generation.

    Parameters
    ----------
    ix : int
        Index within array whose z-score to calculate.
    """
    def __init__(self, ix=0):
        self.ix = ix

    def __call__(self, arr):
        """Standard score of specified element in array.

        .. note:: If the standard deviation of arr is zero,
                  0 is returned to avoid a divide-by-zero result.

        Parameters
        ----------
        arr : 1D np.ndarray
            Array.

        Returns
        -------
        z-score : float
            Z-score of arr[ix].
        """
        try:
            return (arr[self.ix] - np.mean(arr)) / np.std(arr)
        except ZeroDivisionError:
            return 0


if __name__ == '__main__':
    pass
