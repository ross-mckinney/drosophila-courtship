# _circular.py

import numpy as np
import matplotlib.pyplot as plt

import pycircstat.descriptive as pysum
import pycircstat.tests as pytest

from courtship.stats import (
    spatial
)

def _binned_dot(
    ax,
    data,
    descriptor,
    ci,
    bootstrap_iter,
    n_bins,
    descriptor_color,
    arrow_color,
    dot_color,
    rayleigh
    ):
    """Helper function to plot individual dot plots.

    All parameters not listed in 'Parameters' are described
    in binned_dot.

    Parameters
    ----------
    ax : matplotlib axis handle
        Axis handle on which plot will be generated.

    data : np.ndarray | shape = [N]
        Data to plot. Values should be in radians.
    """
    assert isinstance(data, np.ndarray), "Input must be of type numpy.ndarray."

    # clean data
    if np.sum(np.isnan(data).astype(np.int)) >= 1:
        print 'NaNs found in data. These have been removed.'
        cleaned_data = data[~np.isnan(data)]
    else:
        cleaned_data = data

    data = cleaned_data

    # test to see if we should plot the resultant vector
    plot_resultant_vector = False

    if rayleigh:
        pval, zstat = pytest.rayleigh(data)
        if pval < 0.05:
            plot_resultant_vector = True

    # get descriptor parameters
    if descriptor == pysum.median:
            val_descriptors = descriptor(
                data,
                ci=ci,
                bootstrap_iter=bootstrap_iter
            )
    else:
        val_descriptors = descriptor(data, ci=ci)

    mean_direction = pysum.mean(data)
    center = val_descriptors[0]
    upper = val_descriptors[1][1]
    lower = val_descriptors[1][0]

    # find median and +/- 95% confidence intervals around the median
    if center < lower or center > upper:
        if lower < center:
            # this means that the upper is lower than the center,
            # so add np.pi*2 to upper
            upper += np.pi * 2
        else:
            # this means that lower is greater than center,
            # so subtract np.pi*2 to lower
            lower -= np.pi * 2

    # move data into bins
    bins = np.linspace(0, np.pi * 2, n_bins)
    hist = np.histogram(data, bins)[0]
    possible_thetas = bins + (np.pi / 2) / n_bins
    possible_thetas = possible_thetas[:-1]

    thetas = []
    rs = []
    for j, theta in enumerate(possible_thetas):
        if hist[j] > 0:
            for k in xrange(hist[j]):
                thetas.append(theta)
                rs.append((k + 1) * 0.1 + 1)

    binned_data = np.array(thetas)
    r_vals = np.array(rs)

    if plot_resultant_vector:
        # plot confidence interval
        ax.plot(
            np.linspace(lower, upper, 100),
            np.ones(100) * 0.95,
            '-',
            color=descriptor_color
        )

        # plot confidence interval end-caps
        ax.plot([lower, lower], [0.90, 1.00], '-', color=descriptor_color)
        ax.plot([upper, upper], [0.90, 1.00], '-', color=descriptor_color)

        # plot descriptor along the descriptor lines
        ax.plot([center, center], [0, 0.95], '-', color=descriptor_color)
        ax.plot(center, 0.95, 'o', color=descriptor_color)

        # plot R vector
        vector_length = pysum.resultant_vector_length(data) * 0.95
        ax.annotate(
            "",
            xy=(mean_direction, vector_length),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=arrow_color, linewidth=2)
        )

    r_max = np.max(r_vals) + 0.4

    # make custom grid (along x and y planes).
    ax.plot(
        [np.pi, 0],
        [r_max - 0.4, r_max - 0.4],
        color=(0.5, 0.5, 0.5, 0.5)
    )
    ax.plot(
        [np.pi/2, 3*np.pi/2],
        [r_max - 0.4, r_max - 0.4],
        color=(0.5, 0.5, 0.5, 0.5)
    )

    # make outer circle
    ax.plot(
        np.linspace(0, np.pi * 2, 360),
        np.ones(360) + 0.1,
        'k-',
        linewidth=2
    )

    # make inner circle
    ax.plot(
        np.linspace(0, 2 * np.pi, 360),
        np.ones(360) * 0.95,
        color=(0.5, 0.5, 0.5, 0.5),
        linestyle='--'
    )

    # and plot the binned data
    ax.plot(
        binned_data,
        r_vals,
        marker='o',
        color=dot_color,
        linestyle='None'
    )

    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_rmax(r_max)


def binned_dot(
    data,
    order=None,
    shape=None,
    descriptor=pysum.median,
    ci=0.95,
    bootstrap_iter=1000,
    n_bins=80,
    descriptor_color=(0.5, 0.5, 0.5, 0.5),
    arrow_color='#06A0D4',
    dot_color='black',
    rayleigh=True
    ):
    """Generates a circular, binned dot plot for angular data.

    Each data point will be plotted around the outside of a two-
    ringed circle. The inner circle is plotted at r = 1, and defines the
    maximal value of the center arrow, whose length is equal to the R-value
    of the collection of data. Confidence intervals are also calculated and
    shown using gray lines.

    Parameters
    ----------
    data : np.ndarray or dictionary of np.ndarrays.
        Each key should be a group name, and each value should be an np.ndarray
        of shape [N]. All keys within the dictionary will have their own
        subplot within the plotted (or returned) figure.

    order : list of string (default = None)
        The order to plot each of the key-value pairs within data. If None,
        plots will be ordered alphabetically by key.

    shape (nrows, ncols) : tuple of int (default = None)
        The shape of the individal plots contained within the overall figure.
        If None, all subplots will be shown on a single row. Note that
        nrows * ncols must be equal to len(data) if data is a dictionary.

    descriptor : function (default = pycircstat.descriptive.median)
        Function to specify how confidence intervals are drawn. This function
        should be specific to circular data.

    ci : float (default = 0.95)
        Confidence interval to plot (around median of data).

    bootstrap_iter : int (default = 1000)
        Number of bootstraps to use to calculate the confidence interval around
        the median of the data.

    n_bins : int (default = 80)
        Number of bins to split circular data into.

    descriptor_color : string or RGBA tuple (default = (0.5, 0.5, 0.5, 0.5))
        Color of descriptor lines emanating from center of circle.

    arrow_color : string or RGBA tuple (default = '#06A0D4')
        Color of central arrow.

    dot_color : string or RGBA tuple (default = 'black')
        Color of points/dots.

    rayleigh : bool (default = True)
        Whether or not to perform a Rayleigh test on the given data.
        If True, an arrow will only be drawn in a specific plot if the data
        within that plot is NOT uniformly distributed around the circle
        (as determined by use of the function: pycircstat.tests.rayleigh).
        If False, an arrow will always be drawn.

    Returns
    -------
    fig : matplotlib figure

    ax : matplotlib axis handle

    Examples
    --------
    >>> import courtship.plots as cplt
    >>> import numpy as np
    >>> x = np.random.randn(3, 40) #get random normal mock data
    >>> # make sure all data is positive
    >>> x[np.where(x < 0)] = x[np.where(x < 0)] + 2 * np.pi
    >>> # place data into a dictionary of np.ndarrays
    >>> data = {'x': x[0, :], 'y': x[1, :], 'z': x[2, :]}
    >>> cplt.binned_dot(data) #call function
    """
    if isinstance(data, np.ndarray):
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        _binned_dot(
            ax,
            data,
            descriptor,
            ci,
            bootstrap_iter,
            n_bins,
            descriptor_color,
            arrow_color,
            dot_color,
            rayleigh)
        return fig, ax
    elif isinstance(data, dict) and len(data) == 1:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        _binned_dot(
            ax,
            data[data.keys()[0]],
            descriptor,
            ci,
            bootstrap_iter,
            n_bins,
            descriptor_color,
            arrow_color,
            dot_color,
            rayleigh)
        ax.set_title(data.keys()[0])
        return fig, ax
    else:
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary or numpy.ndarray.")

    if order is None:
        order = sorted(data.keys())
    else:
        if len(data) != len(order):
            raise AttributeError(
                'Number of keys in data does not' +
                'match number of keys given in order.'
            )

    if shape is not None:
        if not isinstance(shape, tuple):
            raise AttributeError('<shape> must be a tuple of length 2.')

        if len(shape) != 2:
            raise AttributeError('<shape> must be a tuple of length 2.')

        if shape[0] * shape[1] < len(data):
            raise AttributeError(
                '<shape>[0] * <shape>[1] must be greater than number' +
                'of keys in <data>.'
            )

        nrows = shape[0]
        ncols = shape[1]
    else:
        nrows = 1
        ncols = len(data)

    counter = 0
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw=dict(polar=True)
    )
    for rr in xrange(nrows):
        for cc in xrange(ncols):
            if counter < len(data):
                if nrows > 1 and ncols > 1:
                    ax_handle = ax[rr, cc]
                else:
                    ax_handle = ax[counter]

                _binned_dot(
                    ax_handle,
                    data[order[counter]],
                    descriptor,
                    ci,
                    bootstrap_iter,
                    n_bins,
                    descriptor_color,
                    arrow_color,
                    dot_color,
                    rayleigh
                    )
                ax_handle.set_title(order[counter])
            else:
                return fig, ax
            counter += 1

    return fig, ax


def dot(
    exp,
    behavior_name,
    ax=None,
    colors=None,                  #FIXME!
    plot_points=True,
    plot_medians=True,
    error_push_out=0.2,
    error_linewidth=2,
    error_linestyle='dashed',
    arrow_linewidth=5,
    **kwargs
):
    """Plots means, medians, and 95% confidence intervals around
    the median for circular data.

    Means are represented by arrows, where the resultant
    vector length (1 - variance) is represented by the length of the
    arrow. Medians are represented as colored lines going from the
    center of the circle to colored points near the periphery.
    Confidence intervals are represented by colored arcs near the
    periphery, containing the median.

    Parameters
    ----------
    exp : FixedCourtshipExperiment
        Object containing all flies/groups to be plotted.

    behavior_name : string
        Must be a valid key in each fly's behavior attribute.

    ax : matplotlib axes handle (optional, default=None)

    colors : list of valid matplotlib colors (optional, default=None)
        This should contain the same number of items as groups
        contained within exp.

    plot_points : bool (optional, default=True)
        Whether or not to show individual data points on plot.

    plot_medians : bool (optional, default=True)
        Whether or not to plot a point at the median value of the
        data for each group.

    error_push_out : float (optional, default=0.2)
        How much to push error line away from the center of the plot.

    error_linewidth : int (optional, default=2)
        Linewidth of median line & error line.

    error_linestyle : string (optional, default='dashed')
        Valid matplotlib linestyle for the error line.

    arrow_linewidth : int (optional, default=3)
        Linewidth of the mean arrow.

    **kwargs :
        Values to pass to ax.scatter

    Returns
    -------
    fig, ax : matplotlib figure and axes handles

    Examples
    --------
    >>> fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        subplot_kw=dict(polar=True)
    )
    >>> arrow_plot(
            exp,
            cleaned_scissoring,
            ax=ax[0],
            plot_points=False
        )
    >>> arrow_plot(
            exp,
            cleaned_scissoring,
            ax=ax[1],
            plot_points=True
        )
    """

    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        return_fig = True

    thetas = exp.get_ang_location_summary(behavior_name)
    rvls = {}
    medians = {}

    if colors is None:
        colors = [
            np.random.random_sample(3) for i in xrange(len(exp.order))
        ]
    else:
        if len(colors) != len(exp.order):
            raise AttributeError(
                'len(colors) must be equal to number of groups in Experiment.'
            )

    if plot_points:
        median_position_spread = 0.2
        median_visible = False
        ymax = 1.8
    else:
        median_position_spread = 0.1
        median_visible = True
        ymax = 1.5

    median_rs = np.arange(
        1.1,
        1.1 + median_position_spread*len(exp.order),
        median_position_spread
    )

    for i, group in enumerate(exp.order):
        t = np.asarray(thetas[group])

        mean_theta = pysum.mean(t)
        resultant_vector_length = pysum.resultant_vector_length(t)
        try:
            ci = pysum.mean_ci_limits(
                    t,
                    ci=0.95
                )
        except UserWarning as e:
            print e

        rvls[group] = resultant_vector_length
        medians[group] = pysum.median(
                            t,
                            ci=0.95,
                            bootstrap_iter=500
                        )

        ax.annotate(
            "",
            xy=(mean_theta, resultant_vector_length),
            xytext=(0, 0),
            arrowprops=dict(
                        arrowstyle='-|>',
                        linewidth=arrow_linewidth,
                        color=colors[i]
                        )
        )

        if plot_points:
            ax.scatter(
                t,
                (np.random.rand(t.size) * 0.05 +
                 np.ones(t.size) * (median_rs[i] - 0.05)),
                color=colors[i],
                **kwargs
            )

    if plot_medians:
        for i, group in enumerate(exp.order):
            ax.plot(
                medians[group][0],
                median_rs[i] + error_push_out,
                color=colors[i],
                marker='.',
                alpha=0.5,
                visible=median_visible
            )
            ax.plot(
                [0, medians[group][0]],
                [0, median_rs[i] + error_push_out],
                color=colors[i],
                alpha=0.5,
                linestyle=error_linestyle,
                linewidth=error_linewidth
            )

            lower = medians[group][1][0]
            upper = medians[group][1][1]

            if lower > upper:
                lower -= 2*np.pi

            thetas = np.linspace(lower, upper, 50)
            rs = np.ones(50) * median_rs[i]
            ax.plot(
                thetas,
                rs + error_push_out,
                color=colors[i],
                alpha=0.5,
                linestyle=error_linestyle,
                linewidth=error_linewidth)

    ax.plot([
        np.pi, 0],
        [1, 1],
        color='lightgray',
        linestyle='-'
    )
    ax.plot(
        [np.pi/2, 3*np.pi/2],
        [1, 1],
        color='lightgray',
        linestyle='-'
    )

    ax.plot(
        np.linspace(0, 2*np.pi, 360),
        np.ones(360),
        color='lightgray',
        linestyle='-'
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim(0, ymax)
    ax.axis('off')

    if return_fig:
        return fig, ax
    return ax


def _format_axes(
    ax,
    color='lightgray',
    linestyle='-',
    r_max=1,
    r_lim=1.5
    ):
    """Formats polar axes.

    Draws a single vertical and horizontal grid line within a polar axis.

    Parameters
    ----------
    ax : matplotlib.Axes handle
        Axes handle to format.

    color : valid matplotlib color (optional, default='lightgray')
        Color of grid lines.

    linestyle : valid matplotlib linestyle (optional, default='-')
        Style of grid lines.

    r_max : float (optional, default=1)
        Maximal r-value to draw circle at.

    r_lim : float (optional, default=1.5)
        Axes limits  (this is the same as ax.set_ylim(`r_lim`)).

    Returns
    -------
    ax : matplotlib.Axes handle
        Formatted Axes.
    """
    # horizontal axis
    ax.plot(
        [np.pi, 0],
        [r_max, r_max],
        color=color,
        linestyle=linestyle
    )

    # vertical axis
    ax.plot(
        [np.pi/2, 3*np.pi/2],
        [r_max, r_max],
        color=color,
        linestyle=linestyle
    )

    # circle
    ax.plot(
        np.linspace(0, 2*np.pi, 360),
        np.ones(360) * r_max,
        color=color,
        linestyle=linestyle
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim(0, r_lim)
    ax.axis('off')

    return ax


def arrow(
    data,
    order=None,
    colors=None,
    linewidth=2,
    ax=None
    ):
    """Plots an arrow representing Rayleigh R-values on a polar axes.
    
    Parameters
    ----------
    data : dict
        Each value should be a 1D-array like of angular values (thetas). The
        plotted arrow represents the Rayleigh R-value of the theta distribution.

    order : list of string or None (optional, default=None)
        Order to plot groups. These should correspond the the keys present in
        `data` dictionary. If None, then groups will be ordered
        alphabetically.

    colors : list of valid matplotlib colors or None (optional, default=None)
        Colors of arrows to plot.

    linewidth : int (optional, default=2)
        Linewidth of arrows.

    ax : matplotlib.pyplot.Axes handle or None (optional, default=None)
        Axes handle to plot arrows onto.

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure & Axes handles with plotted data.
    """
    if order is None:
        order = sorted(data.keys())

    if colors is None:
        colors = ['C{}'.format(i % 10) for i in xrange(len(data))]

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        # get the current figure from pyplot
        fig = plt.gcf()

    if len(order) != len(data):
        raise AttributeError('len(data) != len(order)')
    if len(colors) != len(data):
        raise AttributeError('len(data) != len(colors)')

    for i, group_name in enumerate(order):
        thetas = np.asarray(data[group_name])

        mean_theta = pysum.mean(thetas)
        resultant_vector_length = pysum.resultant_vector_length(thetas)

        ax.annotate(
            "",
            xy=(mean_theta, resultant_vector_length),
            xytext=(0, 0),
            arrowprops=dict(
                        arrowstyle='-|>',
                        linewidth=linewidth,
                        color=colors[i]
                        )
            )

    return fig, _format_axes(ax)


if __name__ == '__main__':
    pass