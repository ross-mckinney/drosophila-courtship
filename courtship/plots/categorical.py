# courtship.plots.categorical.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pycircstat.descriptive as pysum

import utils


def boxplot(
    data,
    order=None,
    colors=None,
    at=None,
    ax=None,
    showfliers=False,
    capprops=dict(linewidth=0),
    boxprops=dict(linewidth=1, linestyle='-'),
    medianprops=dict(linewidth=2, color='k'),
    whiskerprops=dict(linewidth=1, linestyle='-'),
    **box_args
    ):
    """Boxplot.

    Parameters
    ----------
    data : dictionary
        Data to plot. Keys should be string, vals should be 1D array-like.
    
    order : list of string or None
        Each string should be a valid key in `data`. This is the order to plot
        each group along the x-axis. If None, groups will be plotted
        alphabetically from left to right.

    colors : list of valid matplotlib colors or None
        Colors of each group to plot. Should be ordered as in `order`. If None,
        current matplotlib color cycle will be used to select colors.
    
    at : list of int or float or None
        Where to plot each group along the x-axis. Should be ordered as in 
        `order`.

    ax : matplotlib Axes handle or None
        Axes handle for plotting. If None, one will be generated for you.

    showfliers, capprops, boxprops, medianprops, whiskerprops : see **box_args

    **box_args : keyword arguments to be passed to ax.boxplot().
        See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
        for available args to pass.

    Returns
    -------
    fig, ax : matplotlib Figure & Axes handle.
    """
    # format all data as lists (as required by ax.boxplot())
    new_data = {}
    for group_name, group_vals in data.iteritems():
        new_data[group_name] = np.asarray(group_vals).tolist()
    data = new_data

    if order is None:
        order = sorted(data.keys())

    if colors is None:
        colors = ['C{}'.format(i % 10) for i in xrange(len(data))]

    if at is None:
        at = range(1, len(data) + 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    boxes = ax.boxplot(
        [data[group_name] for group_name in order],
        positions=at,
        patch_artist=True,
        showfliers=showfliers,
        capprops=capprops,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        **box_args
    )

    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(colors[i])

    return fig, ax


def striphist(
    data,
    order=None,
    markerfacecolors=None,
    markeredgecolors=None,
    at=None,
    num_bins=50,
    align='center',
    spread=0.25,
    ax=None,
    **kwargs
    ):
    """Strip chart with histogrammed point visualization.

    Parameters
    ----------
    data : dictionary
        Data to plot. Keys should be string, vals should be 1D array-like.
    
    order : list of string or None
        Each string should be a valid key in `data`. This is the order to plot
        each group along the x-axis. If None, groups will be plotted
        alphabetically from left to right.

    markerfacecolors, markeredgecolors : list of valid matplotlib colors or None
        Marker face and edge colors of each group to plot. Should be ordered as 
        in `order`. If None, current matplotlib color cycle will be used to 
        select colors.
    
    at : list of int or float or None
        Where to plot each group along the x-axis. Should be ordered as in 
        `order`.

    num_bins : int
        How many bins to use for histogramming/binning data.

    align : string
        How to display histogrammed points. Options are 'center' or 'left'.

    spread : float
        How spread out should the histogrammed points be?

    ax : matplotlib Axes handle or None
        Axes handle for plotting. If None, one will be generated for you.

    Parameters
    ----------
    **kwargs : keyword arguments to be passed to ax.scatter().

    Returns
    -------
    fig, ax : matplotlib Figure & Axes handle.
    """
    if order is None:
        order = sorted(data.keys())

    if markerfacecolors is None:
        markerfacecolors = ['C{}'.format(i % 10) for i in xrange(len(data))]
    if markeredgecolors is None:
        markeredgecolors = ['C{}'.format(i % 10) for i in xrange(len(data))]

    if at is None:
        at = range(1, len(data) + 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    for i, group_name in enumerate(order):
        xx, yy = utils.bin(data[group_name], at=at[i], num_bins=num_bins,
            align=align, spread=spread)

        kwargs.update(
            {
                'c': markerfacecolors[i],
                'edgecolors': markeredgecolors[i]
            }
        )

        ax.scatter(xx, yy, **kwargs)

    return fig, ax


def format_categorical_axes(
    ax,
    despine=['right', 'top']
    ):
    """Removes specified spines from plot on non-polar Axes.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes handle to format.
    
    despine : list of string (optional, default=['right', 'top'])
        Items should be valid spine strings, including ['left', 'right', 'top',
        and 'bottom'].

    Returns
    -------
    ax : matplotlib Axes handle
    """
    for spine in despine:
        ax.spines[spine].set_visible(False)
    return ax


if __name__ == '__main__':
    pass