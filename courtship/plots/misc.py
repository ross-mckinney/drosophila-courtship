
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import pandas as pd

import courtship.plots as cp
from courtship.stats import markov
from courtship.stats.utils import (
    center_angles,
    clean_paired_data,
    clean_dataset,
    remove_outliers
)


def get_paired_ixs(num_groups, num_features):
    """Gets x-axis indices for plotting paired data.

    Skips one integer value after every feature has been plotted.

    Parameters
    ----------
    num_groups : int
        Number of groups in each 'pair'.

    num_features : int
        Number of different features that will be present on plot.

    Returns
    -------
    at : np.array of shape [num_features, num_groups]
        x-values to plot data at.

    Examples
    --------
    >>> get_paired_ixs(2, 3)
    [[1, 2],
     [4, 5],
     [7, 8]]

    >>> get_paired_ixs(4, 3)
    [[1, 2, 3, 4],
     [6, 7, 8, 9],
     [11, 12, 13, 14]]
    """
    at = []
    for i in np.arange(1, (num_groups+1)*num_features):
        if i % (num_groups + 1) == 0:
            continue
        at.append(i)
    return np.array(at).reshape(num_features, num_groups)


def _get_avg_ellipse(exp, sex):
    """Gets the average major & minor axis lengths fit to either all male or
    female flies in an experiment.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to get average ellipses from.

    sex : string
        Which sex fly to calculate avg ellipse for: 'male' or 'female'.

    Returns
    -------
    maj, min : np.array
        Average major and minor axis lengths for every fly in the experiment.
    """
    if sex not in ['male', 'female']:
        raise AttributeError('`sex` must be either \'male\' or \'female\'')

    maj, min = [], []
    for _, summary in exp.itergroups():
        fly = getattr(summary, sex)
        maj.append(np.nanmean(fly.body.major_axis_length) * 1. /
            summary.video.pixels_per_mm)
        min.append(np.nanmean(fly.body.minor_axis_length) * 1. /
            summary.video.pixels_per_mm)

    return np.asarray(maj), np.asarray(min)


def get_avg_female_ellipse(exp):
    """Gets the average major and minor axis lengths of the ellipse
    fitted to the female for all females across all groups in an experiment.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to get average ellipses from.

    Returns
    -------
    maj, min : np.array
        Average major and minor axis lengths for every female in the experiment.
    """
    return _get_avg_ellipse(exp, sex='female')


def get_avg_male_ellipse(exp):
    """Gets the average major and minor axis lengths of the ellipse
    fitted to the male for all males across all groups in an experiment.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to get average ellipses from.

    Returns
    -------
    maj, min : np.array
        Average major and minor axis lengths for every male in the experiment.
    """
    return _get_avg_ellipse(exp, sex='male')


def get_expected_tapping_dists(exp, num_bins=50):
    """Gets the expected tapping distances for each bin surrounding a female 
    in an Experiment.

    The expected tapping distance is defined as the average female ellipse --
    across all flies in an experiment -- plus half of the average major axis
    length of all males in the experiment.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to get expected tapping distance from.

    num_bins : int (optional, default=50)
        How many angular bins to split the area surrounding the female into.
        Note that these bins will represent the range [-np.pi, np.pi] that has
        been split into `num_bins`.
    
    Returns
    -------
    t : np.array
        Edge of theta bin.

    dists : np.array
        Expected tapping distance for males across the experiment.
    """
    fem_maj, fem_min = get_avg_female_ellipse(exp)
    male_maj, male_min = get_avg_male_ellipse(exp)

    a = 0.5 * np.nanmean(fem_maj)
    b = 0.5 * np.nanmean(fem_min)

    t = np.linspace(-np.pi, np.pi, num_bins)
    r = (a*b) / np.sqrt((b**2 - a**2) * np.cos(t)**2 + a**2)

    # add on half of the major axis length of the male fly.
    r += np.nanmean(male_maj) / 2.

    return t, r


def plot_female_outline(exp, ax):
    """Draws the average female ellipse across all individuals within an
    FixedCourtshipTrackingExperiment onto a polar (or cartesian) Axes handle.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
    ax : matplotlib.Axes handle

    Returns
    -------
    ax : matplotlib.Axes handle
    """
    major, minor = get_avg_female_ellipse(exp)
    a = 0.5 * np.nanmean(major)
    b = 0.5 * np.nanmean(minor)

    t = np.linspace(-np.pi, np.pi, 360)
    r = (a*b) / np.sqrt((b**2 - a**2) * np.cos(t)**2 + a**2)

    ax.plot(t, r, linestyle='--', color='k', alpha=0.5)
    ax.plot([0,0], [0,a], color='k', alpha=0.5)
    return ax


def plot_expected_tapping_distance(exp, ax, num_bins=360):
    """Draws the expected male-female tapping distance onto a given Axes.

    The expected tapping distance is defined as the average female ellipse --
    across all flies in an experiment -- plus half of the average major axis
    length of all males in the experiment.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
    ax : matplotlib.Axes handle
    num_bins : int (optional, default=50)
        How many angular bins to create around the female.

    Returns
    -------
    ax : matplotlib.Axes handle
    """
    t, r = get_expected_tapping_dists(exp, num_bins)
    ax.plot(t, r, linestyle='--', color='k', alpha=0.5)
    return ax


def despine(ax):
    """Removes top and right spines from an Axes handle.

    Parameters
    ----------
    ax : matplotlib.Axes handle

    Returns
    -------
    ax : matplotlib.Axes handle
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_dists_cartesian(
    exp,
    bname='courtship_gt',
    colors=['k', 'steelblue'],
    ax=None,
    draw_expected_tapping_distance=True
    ):
    """Plots the average courtship distances for each group in an Experiment.
    
    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to plot avg distances for.

    bname : string (optional, default='courtship_gt')
        Name of behavior to plot avg distances for.

    colors : list (optional, default=['k', 'steelblue'])
        Each item in list must be a valid matplotlib color.

    ax : matplotlib.Axes handle
        Axes handle to plot avg dists.

    draw_expected_tapping_distance : bool (optional, default=True)
        Whether or not to plot the expected tapping distances.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # these distances are centroid-to-centroid
    dists = exp.get_behavioral_distances(behavior_name=bname)
    for i, group_name in enumerate(exp.order):
        # the avg male-female dist is equivalent in the first and last
        # bins (-pi == +pi), but the current values do not reflect this.
        dists[group_name][:, 0] = dists[group_name][:, -1]

        avg = np.nanmean(dists[group_name], axis=0)
        std = np.nanstd(dists[group_name], axis=0)
        sem = std/np.sqrt(dists[group_name].shape[0])

        xx = np.linspace(-np.pi, np.pi, 50)
        ax.plot(xx, avg, color=colors[i], linewidth=2)
        ax.fill_between(xx, avg-sem, avg+sem, color=colors[i], alpha=0.5)

    if draw_expected_tapping_distance:
        ax = plot_expected_tapping_distance(exp, ax)

    ax = despine(ax)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-pi', '-pi/2' , '0', 'pi/2', 'pi'])
    ax.set_xlim(-np.pi, np.pi)
    return fig, ax


def plot_residual_dists(
    exp,
    bname='courtship_gt',
    colors=['k', 'steelblue'],
    ax=None,
    draw_zero_line=True
    ):
    """Plots the difference between the average centroid-to-centroid distances
    and the expected tapping distance across all males in each group.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to plot residual distances.

    bname : string (optional, default='courtship_gt')
        Name of behavior to plot avg residual distances for.

    colors : list (optional, default=['k', 'steelblue'])
        Each item in list must be a valid matplotlib color.

    ax : matplotlib.Axes handle
        Axes handle to plot avg dists.

    draw_zero_line : bool (optional, default=True)
        Whether or not to draw horizontal line at 0. This line now represents
        the tapping distance.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # these distances are centroid-to-centroid
    dists = exp.get_behavioral_distances(behavior_name=bname)
    _, expected_tapping_dists = get_expected_tapping_dists(exp, num_bins=50)

    for i, group_name in enumerate(exp.order):
        # the avg male-female dist is equivalent in the first and last
        # bins (-pi == +pi), but the current values do not reflect this.
        dists[group_name][:, 0] = dists[group_name][:, -1]

        res_dists = dists[group_name] - expected_tapping_dists

        avg = np.nanmean(res_dists, axis=0)
        std = np.nanstd(res_dists, axis=0)
        sem = std/np.sqrt(res_dists.shape[0])

        xx = np.linspace(-np.pi, np.pi, 50)
        ax.plot(xx, avg, color=colors[i], linewidth=2)
        ax.fill_between(xx, avg-sem, avg+sem, color=colors[i], alpha=0.5)

    if draw_zero_line:
        ax.hlines(0, -np.pi, np.pi, linestyle='--', color='gray')

    ax = despine(ax)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-pi', '-pi/2' , '0', 'pi/2', 'pi'])
    ax.set_xlim(-np.pi, np.pi)
    return fig, ax


def plot_dists_polar(
    exp,
    bname='courtship_gt',
    colors=['k', 'steelblue'],
    draw_female_ellipse=False,
    ax=None
    ):
    """Plots the male-female, centroid-centroid courtship distance for each
    group in the passed exp on the same, polar, plot.
    
    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to plot.

    bname : string (optional, default='courtship_gt')
        Name of behavior to plot.

    colors : list (optional, default=['k', 'steelblue'])
        Each item in list must be a valid matplotlib color.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    dists = exp.get_behavioral_distances(bname, num_bins=50)
    for i, group_name in enumerate(exp.order):
        # the avg male-female dist is equivalent in the first and last
        # bins (-pi == +pi), but the current values do not reflect this.
        dists[group_name][:, 0] = dists[group_name][:, -1]

        avg = np.nanmean(dists[group_name], axis=0)
        std = np.nanstd(dists[group_name], axis=0)
        sem = std/np.sqrt(dists[group_name].shape[0])

        xx = np.linspace(-np.pi, np.pi, 50)
        ax.plot(xx, avg, color=colors[i], linewidth=2, zorder=1)
        ax.fill_between(xx, avg-sem, avg+sem, color=colors[i], alpha=0.5,
            zorder=0)

    # draw circular axes
    max_r = ax.get_rmax()
    max_r = (max_r + (max_r * .1))

    ax = cp.format_polar_axes(ax, radius=max_r, zorder=-1)

    if draw_female_ellipse:
        ax = plot_female_outline(exp, ax)

    return fig, ax


def plot_dists_peak_ratios(
    exp,
    bname='courtship_gt',
    colors=['k', 'steelblue'],
    show_outliers=True,
    ax=None,
    spread=0.25,
    s=20
    ):
    """Creates a boxplot showing the ratio of the average male-female
    distance (centroid-to-centroid) when the fly is on the front half of the
    female to when the fly is on the rear half of the female.

    $$ DPR = \frac{\tilde{d_{\text{front}}}}{\tilde{d_{\text{rear}}}} $$

    Parameters
    ----------
    exp : canal.objects.experiment.FixedCourtshipExperiment

    bname : string
        Must be a valid behavior name in each male in the Experiment.

    colors : list of valid matplotlib colors

    show_outliers : bool, optional (default=True)
        Whether or not to show outliers.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ratios = exp.get_behavioral_distances_peak_ratio(bname)
    if not show_outliers:
        ratios = remove_outliers(ratios)

    cp.boxplot(
        ratios,
        colors=['lightgray'] * len(exp.order),
        order=exp.order,
        ax=ax,
        zorder=0,
        showfliers=False,
        widths=0.5
    )

    cp.striphist(
        ratios,
        order=exp.order,
        ax=ax,
        zorder=1,
        spread=spread,
        marker='o',
        markerfacecolors=colors,
        markeredgecolors=colors,
        alpha=0.75,
        num_bins=30,
        s=s
    )

    ax.set_xticklabels([])

    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([])

    ax.hlines(1, 0, len(exp.order) + 1, linestyle='--', color='k', alpha=0.5)
    ax=despine(ax)
    fig.set_size_inches(3,4)
    return fig, ax


def plot_arrow_polar_01(
    exp,
    bname,
    group_name,
    dot_color='k',
    arrow_color='k',
    error_color='gray',
    show_arrow=True,
    show_error=True,
    ax=None,
    spread=0.25,
    num_bins=50,
    s=15
    ):
    """Plots mean angular positions during bouts of a given behavior for all
    flies in a given group in the passed experiment.

    This plot will display points as binned data.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment containing behavior to plot.

    bname : string
        Name of behavior to plot mean angular locations for. Must be a valid
        behavior name in each TrackingSummary in `exp`.
    
    group_name : string
        Name of group to plot.

    dot_color : valid matplotlib color (optional, default='k')
        Color of dots. Each dot represents the mean angular location of a given
        fly across all bouts of `bname`.
    
    arrow_color : valid matplotlib color (optional, default='k')
        Color of Rayleigh R-value arrow, which will be shown on plot (if 
        `show_arrow` is set to True).

    error_color : valid matplotlib color (optional, default='gray')
        Color of error bars (95% confidence intervals around the median) to show
        on plot (if 'show_error' is set to True).

    show_arrow : bool (optional, default=True)

    show_error : bool (optional, default=True)

    ax : matplotlib.Axes handle
        This should be in polar coordinates.

    spread : float (optional, default=0.25)
        How far apart dots should be place. Set this high enough so that 
        individual points can be visualized.

    num_bins : int (optional, default=50)
        How many bins to split the data into for plotting?

    s : int (optional, default=15)
        Size of dots. Size system is taken from matplotlib.scatter.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    thetas = exp.get_ang_location_summary(bname)[group_name]
    cp.polar_dot_binned(
        thetas,
        spread=spread,
        at=1.1,
        color=dot_color,
        ax=ax,
        s=s,
        num_bins=num_bins,
        clip_on=False
        )
    if show_arrow:
        cp.polar_arrow(thetas, ax=ax, color=arrow_color)
    if show_error:
        cp.polar_median_error(
            thetas,
            ax=ax,
            color=error_color,
            at=1.1,
            linewidth=3
            )
    cp.format_polar_axes(ax)
    fig.set_size_inches(4,4)

    return fig, ax


def plot_arrow_polar_02(
    exp,
    bname,
    group_name,
    dot_color='k',
    arrow_color='k',
    error_color='gray',
    show_arrow=True,
    show_error=True,
    ax=None,
    jitter=0.25,
    s=15
    ):
    """Plots mean angular positions during bouts of a given behavior for all
    flies in a given group in the passed experiment.

    This plot will display points as jittered (non-binned) data.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment containing behavior to plot.

    bname : string
        Name of behavior to plot mean angular locations for. Must be a valid
        behavior name in each TrackingSummary in `exp`.
    
    group_name : string
        Name of group to plot.

    dot_color : valid matplotlib color (optional, default='k')
        Color of dots. Each dot represents the mean angular location of a given
        fly across all bouts of `bname`.
    
    arrow_color : valid matplotlib color (optional, default='k')
        Color of Rayleigh R-value arrow, which will be shown on plot (if 
        `show_arrow` is set to True).

    error_color : valid matplotlib color (optional, default='gray')
        Color of error bars (95% confidence intervals around the median) to show
        on plot (if 'show_error' is set to True).

    show_arrow : bool (optional, default=True)

    show_error : bool (optional, default=True)

    ax : matplotlib.Axes handle
        This should be in polar coordinates.

    jitter : float (optional, default=0.25)
        How much spread to use when plotting dots. Set this high enough so that 
        individual points can be visualized.

    s : int (optional, default=15)
        Size of dots. Size system is taken from matplotlib.scatter.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    else:
        fig = plt.gcf()

    thetas = exp.get_ang_location_summary(bname)[group_name]
    cp.polar_dot(thetas, jitter=jitter, at=1.25, color=dot_color, ax=ax, s=s)
    if show_arrow:
        cp.polar_arrow(thetas, ax=ax, color=arrow_color)
    if show_error:
        cp.polar_median_error(
            thetas,
            ax=ax,
            color=error_color,
            at=1.1,
            linewidth=3,
            median_args={'ci': 0.95, 'bootstrap_iter': 200})
    cp.format_polar_axes(ax)
    fig.set_size_inches(4,4)

    return fig, ax


def plot_ci(
    exp,
    colors,
    bname='courtship_gt',
    include_nonbehavors=False,
    method='condensed',
    ax=None
    ):
    """Plots the courtship index as a boxplot overlayed with a binned strip
    chart.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to plot CI for.

    colors : list
        Each item should be a valid matplotlib color. Colors of dots in strip
        chart. These should be ordered as in `exp.order`.

    bname : string (optional, default='courtship_gt')
        Name of behavior to calculate index for. This name should represent
        courtship if you want to plot the courtship index.

    include_nonbehavors : bool (optional, default=False)
        Whether or not to include flies that did not engage in courtship in the
        calculated index.

    method : string (optional, default='condensed')
        Method to use for calculating courtship index.

    ax : matplotlib.Axes handle or None (optional, default=None)
        Axes handle to plot onto.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    ci = exp.get_behavioral_indices(
        bname,
        include_nonbehavors=include_nonbehavors,
        method=method
    )

    cp.boxplot(
        ci,
        exp.order,
        colors=['lightgray']*len(exp.order),
        ax=ax,
        zorder=0,
        showfliers=False,
        widths=0.25
    )

    cp.striphist(
        ci,
        order=exp.order,
        ax=ax,
        zorder=1,
        spread=0.25,
        marker='o',
        markerfacecolors=colors,
        markeredgecolors=colors,
        num_bins=30,
        clip_on=False,
        s=10
    )

    ax = despine(ax)
    ax.set_xticklabels([])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(0, 1)
    fig.set_size_inches(4,3)
    return fig, ax


def plot_cl(
    exp,
    colors,
    bname='courtship_gt',
    include_nonbehavors=False,
    ax=None
    ):
    """Plots the courtship latency as a boxplot overlayed with a binned strip
    chart.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment
        Experiment to plot CL for.

    colors : list
        Each item should be a valid matplotlib color. Colors of dots in strip
        chart. These should be ordered as in `exp.order`.

    bname : string (optional, default='courtship_gt')
        Name of behavior to calculate latency for. This name should represent
        courtship if you want to plot the courtship latency.

    include_nonbehavors : bool (optional, default=False)
        Whether or not to include flies that did not engage in courtship in the
        calculated latency.

    ax : matplotlib.Axes handle or None (optional, default=None)
        Axes handle to plot onto.

    Returns
    -------
    fig, ax : matplotlib.Figure and matplotlib.Axes handles.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    cl = exp.get_behavioral_latencies(
        bname,
        include_nonbehavors=include_nonbehavors
    )

    cp.boxplot(
        cl,
        exp.order,
        colors=['lightgray']*len(exp.order),
        ax=ax,
        zorder=0,
        showfliers=False,
        widths=0.25
    )

    cp.striphist(
        cl,
        order=exp.order,
        ax=ax,
        zorder=1,
        spread=0.25,
        marker='o',
        markerfacecolors=colors,
        markeredgecolors=colors,
        num_bins=30,
        clip_on=True,
        s=10
    )

    ax = despine(ax)
    ax.set_xticklabels([])
    ax.set_yticks([0, 300, 600])
    ax.set_ylim(0, 600)
    fig.set_size_inches(4,3)
    return fig, ax


def plot_behavioral_mats(
    exp,
    group,
    bnames,
    cmap,
    sort_bname='courtship_gt',
    ):
    """Plots behavioral matrices for a given group and set of behaviors.

    Parameters
    ----------
    exp : canal.objects.experiment.FixedCourtshipExperiment

    group : string
        Valid name of group in `exp`.

    bnames : list of string
        Behavioral matrices will be plotted for each behavioral name present in
        this list. For example, if `bnames=['tapping', 'courtship_gt']`, the
        returned ax handle will contain two subplots where the first is the
        behavioral matrix showing bouts of tapping and the second is the
        behavioral matrix showing bouts of courtship.

    cmap : matplotib.cm colormap

    Returns
    -------
    fig, ax : matplotlib figure and axes handle
    """
    fig, ax = plt.subplots(nrows=1, ncols=len(bnames), sharey=True)

    latency = exp.get_behavioral_latencies(sort_bname, include_nonbehavors=True)
    for i, b in enumerate(bnames):
        mat = exp.get_behavioral_matrices(b, sort=None)
        ixs = np.argsort(latency[group])
        ax[i].matshow(mat[group][ixs,:], aspect='auto', cmap=cmap)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_yticks([0, 15, 30])
        ax[i].set_xticks([0, 24*300, 24*600])
        despine(ax[i])

    fig.set_size_inches(12,3)

    return fig, ax


def plot_behavioral_mat_summary(exp, bnames, colors=['k', 'crimson']):
    """Generates a lineplot showing the average fraction of behaving
    individuals over time.

    Parameters
    ----------
    exp : canal.objects.experiment.FixedCourtshipExperiment

    bnames : list of string
        These should be valid behavior keys in each TrackingSummary within
        `exp`.

    colors : list of valid matplotlib colors of length [len(exp.order)]

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(nrows=1, ncols=len(bnames))

    for i, b in enumerate(bnames):
        # get the behavioral matrix and make sure all behaviors start
        # at t=0.
        mats = exp.get_behavioral_matrices(b)

        for j, group_name in enumerate(exp.order):
#             mat = push_courtship_forward(mats[group_name])
            mat = mats[group_name]
            # then find the fraction behaving of each behavior for plotting.
            frac = np.nansum(mat, axis=0) / mat.shape[0]
            ax[i].plot(frac, color=colors[j], linewidth=1)

        ax[i].set_ylim(0, 1)
        ax[i].set_xlim(0, 14400)
        ax[i].set_yticks([0, 0.5, 1])
        ax[i].set_yticklabels([])
        ax[i].set_xticks([0, 24*300, 24*600])
        ax[i].set_xticklabels([])

    fig.set_size_inches(12,3)
    return fig, ax


def plot_transition_probas_as_boxes_01(
    exp,
    bnames,
    colors,
    show_points=True
    ):
    """
    Parameters
    ----------
    bnames : list of string ['tap', 'ori', 'sci']
        These should be valid behavior names.

    Returns
    -------
    fig, ax : figure & axes handles
    transition_names : list of string
    transition_data : dict of dict
        Each key is a transition name, each value is a second dictionary where
        keys are group names and values are arrays of transitions probas for
        each individual in the group.
    """
    fig, ax = plt.subplots()
    transition_names = [
        'tap->tap', 'tap->ori', 'tap->sci',
        'ori->tap', 'ori->ori', 'ori->sci',
        'sci->tap', 'sci->ori', 'sci->sci'
    ]
    transition_data = {tn: [] for tn in transition_names}
    all_data = {tn: {} for tn in transition_names}

    num_transitions = len(transition_names)
    num_groups = len(exp.order)

    at = get_paired_ixs(num_groups, num_transitions).T.tolist()

    for i, group_name in enumerate(exp.order):
        trans_mat = markov.get_transition_matrix(exp, group_name, bnames)
        j = 0
        for rr in xrange(trans_mat.shape[0]):
            for cc in xrange(trans_mat.shape[1]):
                trans_arr = pd.Series(trans_mat[rr, cc, :])
                transition_data[transition_names[j]] = trans_arr.dropna().values
                all_data[transition_names[j]][group_name] = trans_arr.dropna().values
                j += 1

        cp.boxplot(
            data=transition_data,
            order=transition_names,
            colors=[colors[i]]*9,
            at=at[i],
            ax=ax,
            zorder=0
        )

        if show_points:
            cp.striphist(
                data=transition_data,
                order=transition_names,
                at=at[i],
                ax=ax,
                markerfacecolors=['k']*9,
                markeredgecolors=['k']*9,
                zorder=1,
                s=10,
                clip_on=False
            )

    ax = despine(ax)
    # ax.set_xlim(0, 27)
    ax.set_xticks(np.mean(at, axis=0))
    ax.set_xticklabels(transition_names)

    ax.set_ylim(-0.1, 1)
    ax.set_yticks([0, 0.5, 1])
    fig.set_size_inches(12, 3)
    return fig, ax, transition_names, all_data


def get_transition_data_as_dict(exp, bnames):
    """Gets all markov transition data for each individual in this Experiment.

    Parameters
    ----------
    exp : FixedCourtshipExperiment

    bnames : list of string
        These should represent tapping, orienting, and scissoring (in that order).

    Returns
    -------
    transition_data : dict of dict
        Each key is a transition name, each value is a second dictionary where
        keys are group names and values are arrays of transitions probas for
        each individual in the group.
    """
    transition_names = [
        'tap->tap', 'tap->ori', 'tap->sci',
        'ori->tap', 'ori->ori', 'ori->sci',
        'sci->tap', 'sci->ori', 'sci->sci'
    ]
    all_data = {tn: {} for tn in transition_names}

    for i, group_name in enumerate(exp.order):
        trans_mat = markov.get_transition_matrix(exp, group_name, bnames)
        j = 0
        for rr in xrange(trans_mat.shape[0]):
            for cc in xrange(trans_mat.shape[1]):
                trans_arr = pd.Series(trans_mat[rr, cc, :])
                all_data[transition_names[j]][group_name] = trans_arr.dropna().values
                j += 1

    return all_data


def plot_transition_probas(
    exp,
    bnames,
    colors,
    from_behavior='tap'
    ):
    data = get_transition_data_as_dict(exp, bnames)
    if from_behavior == 'tap':
        order = ['tap->tap', 'tap->ori', 'tap->sci']
    elif from_behavior == 'ori':
        order = ['ori->tap', 'ori->ori', 'ori->sci']
    elif from_behavior == 'sci':
        order = ['sci->tap', 'sci->ori', 'sci->sci']
    else:
        raise AttributeError('`from_behavior` must one of [\'tap\', \'ori\', ' +
            '\'sci\']')

    num_groups = len(exp.order)
    num_transitions = len(order)

    at = get_paired_ixs(num_groups, num_transitions).tolist()

    fig, ax = plt.subplots()
    for i, transition_name in enumerate(order):
        cp.boxplot(
            data[transition_name],
            order=exp.order,
            at=at[i],
            colors=colors,
            ax=ax,
            widths=0.5
        )

    # ax.set_xlim(0, 9)
    ax.set_ylim(-0.1, 1.1)

    ax.set_xticks(np.mean(at, axis=1))
    ax.set_xticklabels(order)
    ax.set_yticks([0, 0.5, 1])
    return fig, ax


def plot_behavioral_index_as_frac_of_courtship(
    exp,
    bnames,
    colors,
    show_points=True
    ):
    """
    Parameters
    ----------
    bnames : list of string
        Should always be in the following order ['tap', 'ori', 'sci']
    """
    fig, ax = plt.subplots()

    num_behaviors = len(bnames)
    num_groups = len(exp.order)

    at = get_paired_ixs(num_groups, num_behaviors).tolist()

    for i, name in enumerate(bnames):
        index = exp.get_behavioral_index_as_fraction_of_courtship(
            name,
            'courtship_gt'
            )

        cp.boxplot(index, order=exp.order, colors=colors, at=at[i], ax=ax,
            widths=0.5, zorder=0)
        if show_points:
            cp.striphist(
                index,
                order=exp.order,
                markerfacecolors=['k'] * num_groups,
                markeredgecolors=['k'] * num_groups,
                at=at[i],
                ax=ax,
                zorder=1,
                s=10
                )

    ax = despine(ax)
    # ax.set_xlim(0, 9)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.mean(at, axis=1))
    ax.set_xticklabels(['tap', 'ori', 'sci'])
    ax.set_yticks([0, 0.5, 1])

    fig.set_size_inches(4,3)
    return fig, ax


def plot_forward_velocities(
    exp,
    colors,
    bname='courtship_gt',
    num_bins=50,
    ax=None,
    ylim=2,
    **kwargs
    ):
    """Plots the avg forward velocities (+/- sem) of males at each angular bin
    around a female.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment

    colors : list of valid matplotlib colors

    bname : string (optional, default='courtship_gt')
        During which behavior should binned velocities be calculated?

    num_bins : (optional, default=50)
        How many bins to split 360 degrees surrounding female into.

    ax : (optional, default=None)
        Matplotlib axes handle to plot on to.

    **kwargs : keyword args to pass to ax.plot()
        These will affect the mean line.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    vels = exp.get_binned_forward_velocities(bname, num_bins=num_bins)
    for i, group_name in enumerate(exp.order):
        # drop any rows that are all np.nan
        v = pd.DataFrame(vels[group_name]).dropna(how='all').values

        avg = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        sem = std/np.sqrt(v.shape[0])

        thetas = np.linspace(-np.pi, np.pi, num_bins)
        ax.plot(thetas, avg, color=colors[i], zorder=1, linewidth=2, **kwargs)
        ax.fill_between(thetas, avg-sem, avg+sem, color=colors[i], alpha=0.5,
            zorder=0)

    left_rect = Rectangle((-3*np.pi/4, 0), width=np.pi/2, height=ylim,
        facecolor='C2', alpha=0.25, zorder=-1)
    right_rect = Rectangle((np.pi/4, 0), width=np.pi/2, height=ylim,
        facecolor='C2', alpha=0.25, zorder=-1)

    ax.add_patch(left_rect)
    ax.add_patch(right_rect)

    ax = cp.format_categorical_axes(ax)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-pi', '-pi/2', '0', 'pi/2', 'pi'])

    ax.set_ylim(0, 2)
    ax.set_yticks([0, 1, 2])
    return fig, ax


def plot_sideways_velocities(
    exp,
    colors,
    bname='courtship_gt',
    num_bins=50,
    ax=None,
    **kwargs
    ):
    """Plots the avg absolute sideways velocities (+/- sem) of males at each
    angular bin around a female.

    Parameters
    ----------
    exp : FixedCourtshipTrackingExperiment

    colors : list of valid matplotlib colors

    bname : string (optional, default='courtship_gt')
        During which behavior should binned velocities be calculated?

    num_bins : (optional, default=50)
        How many bins to split 360 degrees surrounding female into.

    ax : (optional, default=None)
        Matplotlib axes handle to plot on to.

    **kwargs : keyword args to pass to ax.plot()
        These will affect the mean line.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    vels = exp.get_binned_abs_sideways_velocities(bname, num_bins=num_bins)
    for i, group_name in enumerate(exp.order):
        # drop any rows that are all np.nan
        v = pd.DataFrame(vels[group_name]).dropna(how='all').values

        avg = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        sem = std/np.sqrt(v.shape[0])

        thetas = np.linspace(-np.pi, np.pi, num_bins)
        ax.plot(thetas, avg, color=colors[i], zorder=1, linewidth=2, **kwargs)
        ax.fill_between(thetas, avg-sem, avg+sem, color=colors[i], alpha=0.5,
        zorder=0)

    left_rect = Rectangle((-3*np.pi/4, 0), width=np.pi/2, height=8,
        facecolor='C2', alpha=0.25, zorder=-1)
    right_rect = Rectangle((np.pi/4, 0), width=np.pi/2, height=8,
        facecolor='C2', alpha=0.25, zorder=-1)

    ax.add_patch(left_rect)
    ax.add_patch(right_rect)

    ax = cp.format_categorical_axes(ax)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-pi', '-pi/2', '0', 'pi/2', 'pi'])

    ax.set_ylim(0, 8)
    ax.set_yticks([0, 4, 8])
    return fig, ax


def plot_sideways_velocity_ratios(
    exp,
    colors,
    behavior_name='courtship_gt',
    num_bins=50,
    ax=None,
    **kwargs
    ):
    """
    Returns
    -------
    fig :
    ax :
    v_side_to_front :
    v_side_to_rear :
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    vels = exp.get_binned_abs_sideways_velocities(
        behavior_name,
        num_bins=num_bins
        )

    thetas = np.linspace(-np.pi, np.pi, num_bins)
    rear_quadrant_ixs = (thetas > 3*np.pi/4) + (thetas < -3*np.pi/4)
    front_quadrant_ixs = (thetas < np.pi/4) * (thetas > -np.pi/4)
    left_quadrant_ixs = (thetas < -np.pi/4) * (thetas > -3*np.pi/4)
    right_quadrant_ixs = (thetas > np.pi/4) * (thetas < 3*np.pi/4)

    side_to_front = {}
    side_to_rear = {}
    for group_name in exp.order:
        v = vels[group_name]

        v_side = np.nanmean(v[:, left_quadrant_ixs + right_quadrant_ixs],
            axis=1)
        v_front = np.nanmean(v[:, front_quadrant_ixs], axis=1)
        v_rear = np.nanmean(v[:, rear_quadrant_ixs], axis=1)

        v_side_to_front = v_side/v_front
        v_side_to_rear = v_side/v_rear

        side_to_front[group_name] = v_side_to_front
        side_to_rear[group_name] = v_side_to_rear

    # plot
    cp.boxplot(
        clean_dataset(side_to_front),
        order=exp.order,
        colors=colors,
        ax=ax,
        at=[1,2],
        zorder=1,
        **kwargs
    )
    cp.boxplot(
        clean_dataset(side_to_rear),
        order=exp.order,
        colors=colors,
        ax=ax,
        at=[4,5],
        zorder=1,
        **kwargs
    )

    ax.hlines(1, 0, 6, linestyle='--', color='k', zorder=0)

    ax = cp.format_categorical_axes(ax)
    ax.set_xlim(0, 6)
    ax.set_xticks([1,2, 4,5])
    ax.set_xticklabels([])

    ax.set_ylim(0, 6)
    ax.set_yticks([0, 3, 6])

    return fig, ax, clean_dataset(side_to_front), clean_dataset(side_to_rear)


if __name__ == '__main__':
    pass