# _data.py

import numpy as np
import pandas as pd


class Data(object):
    """Class to organize data for plotting.

    Parameters
    ----------
    data : dict
        Data to plot. Keys should be group names (as strings) and values should
        be 1D data distributions (passed as either lists or np.arrays).

    labels : dict or None (optional, default=None)
        Alternative labels for group names (in `data`), which will be shown on
        plots. Keys should be group names (as strings) and values should be
        alternative labels (as strings). If None, `labels` are the same as
        group names in `data`.

    order : list of string or None (optional, default=None)
        Each item in `order` should be a group name (in `data`). The order of
        this list determines the order (from left to right) that groups will be
        displayed along the x-axis of a plot (or r-axis [from inner to outer]
        for polar plots). If None, `data` will be displayed in alphabetical
        order.

    at : list of float or None (optional, default=None)
        Positions to plot distributions at along the x-axis.

    colors : list of valid matplotlib colors or None (optional, default=None).
        Color of each group. Colors should be organized as in `order`.
    """
    def __init__(
        self,
        data,
        labels=None,
        order=None,
        at=None,
        colors=None
        ):
        # find the min and max of all the data
        self.min = np.min([np.min(vals) for vals in data.itervalues()])
        self.max = np.max([np.max(vals) for vals in data.itervalues()])

        # make sure all data.values are np.arrays
        self.data = {k: np.asarray(vals) for k, vals in data.iteritems()}

        # make sure all other attributes are formatted correctly
        if labels is None:
            labels = {k: k for k in data.keys()}
        if order is None:
            order = sorted(self.labels.keys())
        if at is None:
            at = np.arange(1, len(data) + 1)
        if colors is None:
            colors = ['C{}'.format(i % 10) for i in xrange(len(data))]

        # and make sure all other attributes have the appropriate shapes
        attrs = {'labels': labels, 'order': order, 'at': at, 'colors': colors}
        for attr_name, attr_vals in attrs.iteritems():
            if len(attr_vals) != len(data):
                raise AttributeError(
                    'len(`{}`) != len(`data`)'.format(attr_name)
                )
            setattr(self, attr_name, attr_vals)

    def iter_xy_jitter(self, jitter=0):
        """Each distribution (y-values) is yielded with a corresponding pair of
        jittered x-values. This is useful for creating stripcharts.

        Parameters
        ----------
        jitter : float (optional, default=0)
            How much jitter (spread) to add to returned x-values. If 0, x-values
            will simply be a 1D array of repeated `at` values.

        Yields
        ------
        x, y : np.array, np.array
            X-values and y-values for plotting.

        Examples
        --------
        >>> d = Data(
            {'g1': [1, 1.5, 1.25, 1.25], 'g2': [2, 2.5, 2.25, 2.25]},
            labels={'g1': 'group1', 'g2': 'group2'},
            order=['g1', 'g2'],
            at=[1, 2],
            colors=['red', 'green']
            )
        >>> for x, y in data.iter_xy_jitter(0):
                print x, y

            [1, 1, 1, 1], [1, 1.5, 1.25, 1.25]
            [2, 2, 2, 2], [2, 2.5, 2.25, 2.25]
        """
        for i, key in enumerate(self.order):
            num_vals = self.data[key].size

            x = np.repeat(self.at[i], num_vals) - jitter/2
            x += np.random.rand(num_vals) * jitter
            y = self.data[key]

            yield x, y

    def iter_xy_bin(self, nbins, spread=0.2):
        """Each distribution (y-values) is yielded with a corresponding pair
        of x-values, such that when plotted as on a stripchart, points will
        appear to be binned and stacked on top of each other.

        Parameters
        ----------
        nbins : int
            Number of bins to split data into.

        spread : float (optional, default=0.2)
            How much spacing should be given to the points in each bin. If more
            points are going to appear in each bin, increase the spread so that
            you can visually see all (or more) of the points.

        Yields
        ------
        x, y : np.array, np.array
        """
        for i, group_name in enumerate(self.order):
            for group_vals in self.data.itervalues():
                min_val = np.min(group_vals)
                max_val = np.max(group_vals)

                bins = np.linspace(min_val, max_val, nbins)

                hist_count, edges = np.histogram(group_vals, bins)
                total_count = np.sum(hist_count)

                mids = np.diff(edges) / 2 + edges[:-1]

                x, y = [], []
                for i, mid in enumerate(mids):
                    count = hist_count[i]
                    if count == 0:
                        continue

                    y += np.repeat(mid, count).tolist()
                    x += np.linspace(
                        self.at[i] - spread*count/total_count,
                        self.at[i] + spread*count/total_count,
                        count
                    ).tolist()

                yield x, y

    def get_ordered_values(self):
        """Returns a list of all values in PFrame as a list of lists."""

        vals = []
        for k in self.order:
            vals.append(self.data[k])
        return vals
