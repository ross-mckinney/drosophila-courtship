
import numpy as np


class Behavior(object):
    """Behavior container.

    Parameters
    ----------
    name : string
        Name of behavior.

    length : int
        Total length of the video in frames.

    start_ixs : 1D, array-like
        Array containing starting indices of behavior.

    stop_ixs : 1D, array-like
        Array containing stopping indices of behavior.
    """
    def __init__(self, name, length, start_ixs, stop_ixs):
        self.name = name
        self.length = length
        self.start_ixs, self.stop_ixs = self._format_ixs(start_ixs, stop_ixs)

    @classmethod
    def from_array(cls, behavior_name, behavioral_arr):
        """Generates an Behavioral instance from a binary array.

        Parameters
        ----------
        behavior_name : string
            Name of behavior.

        behavioral_arr : 1D, array-like
            Array to convert to Behavior.

        Returns
        -------
        Behavior :
            Instance of Behavior class.
        """
        behavioral_arr = np.asarray(behavioral_arr)
        ixs = bout_boundaries(behavioral_arr)
        return cls(behavior_name, behavioral_arr.size, ixs[:, 0], ixs[:, 1])

    def _format_ixs(self, start, stop):
        """Makes sure that start and stop ixs fall within length.

        Parameters
        ----------
        start : list of int
        stop : list of int

        Returns
        -------
        start : list of int
            Assured that start <= length.
        stop : list of int
            Assured that stop <= length.
        """
        start = np.asarray(start).flatten()
        stop = np.asarray(stop).flatten()

        if start.size != stop.size:
            raise AttributeError(
                'start_ixs and stop_ixs must be the same size.'
                )

        new_start = []
        new_stop = []
        for i in xrange(start.size):
            if start[i] >= self.length:
                break

            new_start.append(start[i])
            if stop[i] <= self.length:
                new_stop.append(stop[i])
            else:
                new_stop.append(int(self.length))

        assert len(new_start) == len(new_stop), 'Length of start != stop'
        return np.asarray(new_start), np.asarray(new_stop)

    def as_array(self):
        """Converts this behavior into a binary array.

        Returns
        -------
        arr : np.ndarray of shape [length]
            Binary array (1 = behaving, 0 = not behaving).

        Examples
        --------
        >>> behav = Behavior('b1', 10, [2, 8], [4, 10])
        >>> behav.as_array()
        np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        """
        arr = np.zeros(self.length)
        for i in xrange(self.start_ixs.size):
            arr[self.start_ixs[i]:self.stop_ixs[i]] = 1
        return arr

    def index(self, mode='all'):
        """The index is defined as the fraction of time spent behaving.

        Parameters
        ----------
        mode : string, optional (default='all'; 'all' or 'condensed')
            How to calculate the behavioral index. If 'all', the index
            will be calculated as the fraction of time spent behaving from the
            beginning of the trial (time/frame=0) until the end of the trial
            (time/frame=Behavior.length). If 'condensed', the index will be
            calculated as the fraction of time spent behaving from the first
            occurrence of the behavior until the end of the trial.

        Returns
        -------
        index : float
            Fraction of time spent engaging in this behavior.

        Examples
        --------
        >>> behav = Behavior('b1', 100, [20, 80], [40, 100])
        >>> behav.index(mode='all')
        0.4
        >>> behav.index(mode='condensed')
        0.5

        >>> behav = Behavior('b2', 100, [50], [100])
        >>> behav.index(mode='all')
        0.5
        >>> behav.index(mode='condensed')
        1.0
        """
        if mode not in ['all', 'condensed']:
            raise AttributeError("`mode` must be either 'all' or 'condensed'")

        ixs = self.ixs()
        if ixs.size == 0:
            return 0

        if mode == 'all':
            return 1. * np.sum(np.diff(ixs, axis=1)) / self.length

        return 1. * np.sum(np.diff(ixs, axis=1)) / (
            self.length - self.start_ixs[0])

    def latency(self):
        """The latency is defined as the first observation of a behavior from
        the beginning of a trial.

        Returns
        -------
        int :
            Latency to start behaving. If there were no observed behaviors,
            then the behavior's length (Behavior.length) is returned. This is
            zero-indexed.

        Examples
        --------
        >>> behav = Behavior('b1', 100, [20, 80], [40, 100])
        >>> behav.latency()
        20

        >>> behav = Behavior('b2', 100, [], [])
        >>> behav.latency()
        100
        """
        if self.start_ixs.size == 0:
            return self.length
        return self.start_ixs[0]

    def ixs(self):
        """Gets the the starting and stopping indices of all bouts of this
        behavior.

        Returns
        -------
        ixs : np.ndarray of shape [N,2].
            First column (ixs[:, 0]) is equivalent to self.start_ixs.
            Second column (ixs[:, 1]) is equivalent to self.stop_ixs.

        Examples
        --------
        >>> behav = Behavior('b1', 100, [20, 80], [40, 100])
        >>> behav.ixs()    # np.array([[20, 40], [80, 100]])
        """
        return np.vstack((self.start_ixs, self.stop_ixs)).T


def array_to_behavior(behavior_name, behavior_arr):
    """Converts a binary array to a Behavior.

    Parameters
    ----------
    behavior_name : string
        Name of beahvior.

    behavior_arr : array-like
       Binary array to convert to Behavior.

    Returns
    -------
    new_behavior : Behavior
        New behavior with given name.
    """
    behavior_arr = np.asarray(behavior_arr)
    boundaries = bout_boundaries(behavior_arr)
    return Behavior(
        behavior_name,
        behavior_arr.size,
        boundaries[:, 0],
        boundaries[:, 1])


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
    >>> boundaries(arr) #returns np.ndarray([2, 7])
    """
    arr = np.asarray(arr)
    arr = np.hstack((0, arr, 0))  # pad arr with zeros to ensure below works.

    diff_arr = np.diff(arr)
    start = np.flatnonzero(diff_arr == 1)
    stop = np.flatnonzero(diff_arr == -1)

    return np.vstack((start, stop)).T
