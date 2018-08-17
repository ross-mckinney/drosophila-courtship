# courtship.plots.utils.py

import numpy as np
import pandas as pd


def jitter(at, size, spread=0.25, distribution=np.random.rand):
    """Gets jittered values around a given point.

    Parameters
    ----------
    at : int or float
        Where to center jittered values.

    size : int
        Number of jittered values to return.

    spread : float
        Magnitude of jitter.

    distribution : function (optional, default=np.random.rand)
        Distribution used to generate jitter.

    Returns
    -------
    np.ndarray :
        Jittered values.
    """
    return np.repeat(at, size) - spread/2 + distribution(size) * spread


def bin(vals, at, num_bins, align='center', spread=0.25):
    """Histograms & bins values for easy plotting.
    
    Parameters
    ----------
    vals : 1D array-like
        Values to bin.
    
    at : int or float
        Where to center values for plotting.

    num_bins : int
        How many bins to split data into.

    align : string (optional, default='center')
        If 'center', values will be aligned centrally at `at`; when plotted,
        these values will look similar to a violin plot. If 'left', values will
        be aligned such that the first value in each bin appears at `at`; 
        when plotted, these values will look similar to a histogram.

    spread : float (optional, default=0.25)
        How much width should the resulting values take up on a plot.

    Returns
    -------
    xx : np.array
        Binned x-locations for plotting.

    vals : np.array
        Transformed vals. Each value within vals is rounded to its nearest
        central bin value.

    Examples
    --------
    >>> vals = np.array([1, 1.2, 1.23, 1.22, 2.22, 2.32, 2.25, 5, 5, 6])
    >>> plot_utils.bin(vals, at=1, num_bins=5, align='center', spread=0.1)

        (array([0.92 , 0.973, 1.027, 1.08 , 0.94 , 1.   , 1.06 , 0.94 , 1.   ,
        1.06 ]), array([1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 5.5, 5.5, 5.5]))
    """
    vals = np.asarray(vals)

    min_val = np.min(vals)
    max_val = np.max(vals)

    bins = np.linspace(min_val, max_val, num_bins + 1)

    hist_count, edges = np.histogram(vals, bins)
    total_count = np.sum(hist_count)

    mids = np.diff(edges) / 2 + edges[:-1]

    x, y = [], []
    for j, mid in enumerate(mids):
        count = hist_count[j]
        if count == 0:
            continue

        y += np.repeat(mid, count).tolist()
        if align == 'center':
            start = at - 2*spread*count/total_count
            stop = at + 2*spread*count/total_count
        elif align == 'left':
            start = at
            stop = at + 4*spread*count/total_count
        else:
            raise AttributeError('`align` must be either "center" or "left".')

        x += np.linspace(
            start,
            stop,
            count
        ).tolist()

    return np.asarray(x), np.asarray(y)
