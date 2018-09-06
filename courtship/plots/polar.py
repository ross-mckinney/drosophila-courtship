# courtship.plots._polar.py

import numpy as np
import matplotlib.pyplot as plt

import pycircstat.descriptive as pysum

import utils


def polar_dot(
    thetas,
    at=1,
    color='C0',
    jitter=0,
    ax=None,
    **kwargs
    ):
    """Plots points onto the outside of a polar Axes.

    Parameters
    ----------
    thetas : 1D array-like
        Angular values to plot.

    at : int or float (optional, default=1)
        Position (r-value, along radial axis) to plot points.

    color : valid matplotlib color (optional, default='C0')
        Color of dots.

    jitter : float (optional, default=0)
        How much jitter should be applid to r-value for plotting dots?

    ax : matplotlib Axes handle (optional, default=None)
        Axes to plot dots. If `ax` is passed, make sure that it is already
        formatted in polar coordinates!

    **kwargs : keyword arguments to be passed to ax.scatter()

    Returns
    -------
    fig, ax : matplotlib Figure & Axes handle.
    """
    thetas = np.asarray(thetas)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    rs = utils.jitter(at=at, size=thetas.size, spread=jitter)

    ax.scatter(
        thetas,
        rs,
        color=color,
        **kwargs
    )

    return fig, ax


def polar_arrow(
    thetas,
    ax=None,
    direction_function=pysum.mean,
    **arrow_args
    ):
    """Plots an arrow representing Rayleigh R-values on a polar axes.

    Parameters
    ----------
    thetas : 1D array-like
        Distribution of angles to plot arrow representing Rayleigh statistic.

    ax : matplotlib.pyplot.Axes handle or None (optional, default=None)
        Axes handle to plot arrows onto.

    direction_function : function (optional, default=pycircstat.mean)
        Which direction should the arrow be pointing. Other possibilities 
        include pycircstat.median. In any case, this function should return
        a singular, numeric value.

    **arrow_args : arrowprop dictionary to be passed to ax.annotate()
        These can include: width, headwidth, frac, shrink, etc. (see: 
        https://matplotlib.org/users/annotations_intro.html for more info on
        what arguments can be passed here).

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure & Axes handles with plotted data.
    """
    thetas = np.asarray(thetas)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    mean_theta = direction_function(thetas)
    resultant_vector_length = pysum.resultant_vector_length(thetas)

    ax.annotate(
            "",
            xy=(mean_theta, resultant_vector_length),
            xytext=(0, 0),
            arrowprops=arrow_args
        )

    return fig, ax


def polar_median_error(
    thetas,
    at=1,
    color='C0',
    ax=None,
    median_args=dict(
        ci=0.95,
        bootstrap_iter=1000
    ),
    **kwargs
    ):
    """Plots error bars onto a polar axis.

    Parameters
    ----------
    thetas : 1D array-like
        Angular values to plot error bars for.

    at : int or float (optional, default=1)
        Position (r-value, along radial axis) to plot error bars.

    color : valid matplotlib color (optional, default='C0')
        Color of error bars.

    ax : matplotlib Axes handle (optional, default=None)
        Where to plot error bars. If None, a new Axes handle will be generated.
        If `ax` is passed, make sure it is already in polar coordinates!

    median_args : dict (optional, default={'ci':0.95, 'bootstrap_iter':1000})
        Arguments to be passed to pycircstat.median. This function will be used
        to calculate the median and confidence intervals around the median. See:
        https://github.com/circstat/pycircstat/blob/master/pycircstat/descriptive.py
        for possible arguments to add here.

    **kwargs : keword arguments to be passed to ax.plot()
        Arguments to further refine look of error bars (which are just lines).
    
    Returns
    -------
    fig, ax : matplotlib Figure & Axes handle.
    """
    thetas = np.asarray(thetas)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    median, ci = pysum.median(thetas, **median_args)
    lower = ci[0]
    upper = ci[1]

    if lower > upper:
        lower -= 2*np.pi

    kwargs.update({'color': color})
    ax.plot(np.linspace(lower, upper, 50), np.repeat(at, 50), **kwargs)
    ax.plot([median, median], [0, at], **kwargs)

    return fig, ax


def polar_dot_binned(
    thetas,
    at=1,
    color='C0',
    spread=0.25,
    num_bins=50,
    ax=None,
    **kwargs
    ):
    """Plots histogrammed/binned points onto the outside of a polar Axes.

    Parameters
    ----------
    thetas : 1D array-like
        Angular values to plot.

    at : int or float (optional, default=1)
        Position (r-value, along radial axis) to plot points.

    color : valid matplotlib color (optional, default='C0')
        Color of dots.

    spread : float (optional, default=2.)
        How spread out should the dots within the histogram bin be?

    num_bins : int (optional, default=50)
        How many bins should be used for generating the histogram?

    ax : matplotlib Axes handle (optional, default=None)
        Axes to plot dots. If `ax` is passed, make sure that it is already
        formatted in polar coordinates!

    **kwargs : keyword arguments to be passed to ax.scatter()

    Returns
    -------
    fig, ax : matplotlib Figure & Axes handle.
    """
    thetas = np.asarray(thetas)

    rr, tt = utils.bin(thetas, at=at, num_bins=num_bins, align='left',
        spread=spread)
    
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()
    
    ax.scatter(tt, rr, color=color, **kwargs)

    return fig, ax


def format_polar_axes(
    ax,
    color='lightgray',
    linestyle='-',
    radius=1,
    **kwargs
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

    radius : float (optional, default=1)
        Radius of circle to draw for axes.

    **kwargs : kwargs to pass to ax.plot

    Returns
    -------
    ax : matplotlib.Axes handle
        Formatted Axes.
    """
    # horizontal axis
    ax.plot(
        [np.pi, 0],
        [radius, radius],
        color=color,
        linestyle=linestyle,
        **kwargs
    )

    # vertical axis
    ax.plot(
        [np.pi/2, 3*np.pi/2],
        [radius, radius],
        color=color,
        linestyle=linestyle,
        **kwargs
    )

    # circle
    ax.plot(
        np.linspace(0, 2*np.pi, 360),
        np.ones(360) * radius,
        color=color,
        linestyle=linestyle,
        **kwargs
    )

    ax.set_xticks([])
    ax.set_yticks([])

    r_lim = ax.get_ylim()[1]

    ax.set_ylim(0, r_lim)
    ax.axis('off')

    return ax


if __name__ == '__main__':
    pass