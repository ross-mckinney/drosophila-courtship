# -*- coding: utf-8 -*-
"""
.. module:: statistics
   :synopsis: Calculates statistics about relative spatial locations of
                 SocialBehaviors.

.. moduleauthor:: Ross McKinney
"""
import os, warnings

import numpy as np
import pycircstat.descriptive as pysum

from centroid import (
    centroid_velocity,
    angular_velocity,
    component_velocities
    )

import transforms
from _signal import normalize
from transforms import rotate_coordinates


def summarize_behavior(
    tracking_summaries,
    behavior_name,
    frm=0.0,
    to=1.0,
    statistic=pysum.mean
    ):
    """Calculates a summary statistic on behaviors held within a list of
    tracking summaries.

    Specifically, calculates a function on locations of behavioral postions.

    Parameters
    ----------
    tracking_summaries : list of FixedCourtshipTrackingSummary
        Behaviors to calculate summary statistic on.

    behavior_name : string
        Name of behavior to calculate statistics on. This behavior
        must be in each tracking summary's behavior dictionary.

    frm : float (0 to 1, default = 0.0)
        Percentage of frames to start calculating summary statistic on.
        This ranges from 0 to 1, where 0 means to start from the beginning,
        0.5 means to start from the middle, and 1 would mean to start from
        the end. start_pct must be less than stop_pct.

    to : float (0 to 1, default = 1.0)
        Percentage of frames to stop calculating summary statistic on. This
        ranges from 0 to 1, and must be greater than start_pct.

    statistic : function (default=pycircstat.descriptive.mean)
        Statistical function to calculate on behavioral locations.

    Returns
    -------
    locations : np.ndarray | shape = [len(behaviors)]
        Mean/median/etc. angular location (theta) for each member of behaviors.
        If a behavior does not contain any positive classifications, the
        statistic for that fly/behvior will be filled with a np.nan value.
    """
    if frm > to:
        print ("Error in function spatial.summarize_behaviors(): " +
            "start_pct must be less than stop_pct.")
        return

    stats = []
    for ts in tracking_summaries:
        classifications = ts.male.get_behavior(behavior_name).as_array()
        n_frames = classifications.size

        low_ix = int(n_frames * frm)
        high_ix = int(n_frames * to)

        theta, r = relative_position2(ts.male, ts.female)
        behaving_ix = np.where(classifications[low_ix:high_ix])[0]

        if theta.size == 0:
            stats.append(np.nan)
            continue

        stats.append(float(statistic(theta[behaving_ix])))
    return np.array(stats)


def compiled_locations(tracking_summaries, behavior_name, r_in_mm=True):
    """Compiles locations from behaviors held within a list of
    tracking summaries.

    Parameters
    ----------
    tracking_summaries : list of FixedCourtshipTrackingSummary
        TrackingSummary objects containing flies & behavior of
        interest for which to compile all locations.

    behavior_name : string
        Name of behavior to compile locations for.

    r_in_mm : bool (default = True)
        If true, the returned r-distance corresponds to absolute
        male-to-female distance in millimeters. If false, the returned
        r-distance will be a relative male-to-female distance that has
        been normalized to the mean area of the male within each
        TrackinSummary.

    Returns
    -------
    locations : np.ndarray | shape = [N, 2]
        Compiled locations for all behaviors. First column is either x or theta,
        second column is either y or r.
    """
    n_rows = np.sum(
        [np.where(ts.male.get_behavior(behavior_name).as_array())[0].size
        for ts in tracking_summaries]
        )
    locations = np.zeros(shape=(n_rows, 2))

    start_ix = 0
    for ts in tracking_summaries:
        behavior_arr = ts.male.get_behavior(behavior_name).as_array()
        n_rows = np.where(behavior_arr)[0].size
        theta, r = relative_position2(ts.male, ts.female)

        if r_in_mm:
            r = nearest_neighbor_centroid(
                ts.male, ts.female, normalized=False
                ) / ts.video.pixels_per_mm

        thetas = theta[np.where(behavior_arr)[0]]
        rs = r[np.where(behavior_arr)[0]]
        locations[start_ix:start_ix + n_rows, 0] = thetas
        locations[start_ix:start_ix + n_rows, 1] = rs
        start_ix += n_rows

    return locations


def _bin_circular(
    tracking_summary,
    statistic,
    behavior_name=None,
    bins=50,
    **kwargs
    ):
    """Calculates a passed statistic between two flies contained in a
    tracking summary across all possible angle bins at which the male
    is located w.r.t. the female.

    Parameters
    ----------
    tracking_summary : objects.FixedCourtshipTrackingSummary
        Must contain a male and female fly.

    statistic : statistical function
        This function must take fly1 (tracking_summary.male) as it's first
        argument and fly2 (tracking_summary.female) as it's second argument.
        This function should return an np.array which contains a single value
        which represents the statistic calculated for each frame contained in
        the specified tracking summary. **kwargs will be passed to this
        function.

    behavior_name : string or None (default=None)
        This should be a valid key in tracking_summary.behaviors. The
        returned statistic will only be calculated for positions where the
        male is behaving. If None, the statistic will be calculated across all
        frames.

    bins : int or None (default=50)
        How many bins to split the 360 degrees surrounding the female into.
        If None, the a mean will be returned for the full 360 degree space.

    **kwargs :
        Valid parameters to be passed to statistic.

    Returns
    -------
    binned_stat : np.ndarray of shape [bins]
        Mean statistic for each bin surrounding the female. Note that this will
        return all NaNs if there are no frames containing behavioral locations.
    """
    # Get the a binary array containing the frames to calculate the stat on.
    if behavior_name is None:
        b_ixs = np.ones(tracking_summary.male.n_frames)
    else:
        b_ixs = tracking_summary.male.get_behavior(behavior_name).as_array()

    # Make sure that there are at least some valid behavioral locations.
    # Otherwise, return an array of NaNs.
    if np.sum(b_ixs) == 0:
        warnings.warn(
            'No behavioral indices found for video:' +
            '{} '.format(os.path.basename(tracking_summary.video_file)) +
            'NaNs have been returned.'
            )
        return np.repeat(np.nan, bins)

    # Get an array of all of the angular positions of the male w.r.t. the
    # female.
    theta, _ = relative_position2(
        tracking_summary.male,
        tracking_summary.female
        )

    # Calculate the passed statistic
    stat = statistic(tracking_summary.male, tracking_summary.female, **kwargs)

    # Limit our 'view' of the statistic to only those frames containing
    # the passed behavior.
    theta = theta[np.flatnonzero(b_ixs)]
    stat = stat[np.flatnonzero(b_ixs)]

    if bins is None:
        return np.nanmean(stat)

    # Generate bin edges and corresponding bins to hold stat values.
    edges = np.linspace(-np.pi, np.pi, bins)
    stat_bins = [[] for i in xrange(bins)]

    # For each theta, locate the appropriate edge bin, and then append the
    # value of the statistic at this theta to it's corresponding bin.
    for i in xrange(theta.size):
        six = np.searchsorted(edges, theta[i])
        stat_bins[six].append(stat[i])

    # Return the mean value of the statistic for all bins.
    stat_bins = [np.nanmean(sb) for sb in stat_bins]
    return np.asarray(stat_bins)


def binned_centroid_to_centroid(ts, behavior_name=None, bins=50):
    """Calculates the average centroid-to-centroid distance between fly1 and
    fly2 across all angular positions surrounding fly2.

    Parameters
    ----------
    ts : objects.FixedCourtshipTrackingSummary
        Which fly pair to look at.

    behavior_name : string or None (default=None)
        Which behavior to look at for the specified ts.

    bins : int (default=50)
        How many bins to split up the 360 degrees surrounding the female.

    Returns
    -------
    c2c_dists : np.array of shape [bins]
        Distances are in mm.
    """

    return _bin_circular(
        ts,
        nearest_neighbor_centroid,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm


def binned_head_to_ellipse(ts, behavior_name=None, bins=50):
    """Calculates the average head-to-ellipse distance from fly1's head to
    any point on the ellipse fitted to fly2's body across all angular positions
    surrounding fly2.

    Parameters
    ----------
    ts : objects.FixedCourtshipTrackingSummary
        Which fly pair to look at.

    behavior_name : string or None (default=None)
        Which behavior to look at for the specified ts.

    bins : int (default=50)
        How many bins to split up the 360 degrees surrounding the female.

    Returns
    -------
    h2e_dists : np.array of shape [bins]
        Distances are in mm.
    """

    # We need to generate a wrapper function that returns a single value to
    # pass to _bin_circular.
    def head_to_ellipse(fly1, fly2):
        h2e, _ = nose_and_tail_to_ellipse(fly1, fly2, normalized=False)
        return h2e

    return _bin_circular(
        ts,
        head_to_ellipse,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm


def binned_forward_velocity(ts, behavior_name=None, bins=50):
    """Gets the average forward velocity of a male fly in a TrackingSummary 
    for all angular bins surrounding a female.

    Parameters
    ----------
    ts : FixedCourtshipTrackingSummary

    behavior_name : string (optional, default=None)
        This should be a valid behavioral key in the ts.behaviors dictionary.
        The forward velocity will only be calculated for positively-classified
        frames. If behavior_name is None, then the forward velocity will be
        calculated for all frames

    bins : int (optional, default=50)
        How many angular bins should be defined around the female?

    Returns
    -------
    forward_velocity : np.ndarry of shape [ts.male.n_frames]
        Forward velocity of the male fly in each bin surrounding female.
        Note that the returned velocity units are mm/second.
    """
    def forward_velocity(fly1, fly2):
        v_sideways, v_forward = component_velocities(fly1)
        return v_forward

    return _bin_circular(
        ts,
        forward_velocity,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm * ts.video.fps


def binned_sideways_velocity(ts, behavior_name=None, bins=50):
    """Gets the average sideways velocity of a male fly in a TrackingSummary 
    for all angular bins surrounding a female.

    Parameters
    ----------
    ts : FixedCourtshipTrackingSummary

    behavior_name : string (optional, default=None)
        This should be a valid behavioral key in the ts.behaviors dictionary.
        The sideways velocity will only be calculated for positively-classified
        frames. If behavior_name is None, then the sideways velocity will be
        calculated for all frames

    bins : int (optional, default=50)
        How many angular bins should be defined around the female?

    Returns
    -------
    sideways_velocity : np.ndarray of shape [ts.male.n_frames]
        Sideways velocity of the male fly in each bin surrounding female.
        Note that the returned velocity units are mm/second.
    """
    def sideways_velocity(fly1, fly2):
        v_sideways, v_forward = component_velocities(fly1)
        return v_sideways

    return _bin_circular(
        ts,
        sideways_velocity,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm * ts.video.fps


def binned_abs_sideways_velocity(ts, behavior_name=None, bins=50):
    """Gets the average sideways velocity of a male fly in a TrackingSummary 
    for all angular bins surrounding a female.

    Parameters
    ----------
    ts : FixedCourtshipTrackingSummary

    behavior_name : string (optional, default=None)
        This should be a valid behavioral key in the ts.behaviors dictionary.
        The sideways velocity will only be calculated for positively-classified
        frames. If behavior_name is None, then the sideways velocity will be
        calculated for all frames

    bins : int (optional, default=50)
        How many angular bins should be defined around the female?

    Returns
    -------
    sideways_velocity : np.ndarry of shape [ts.male.n_frames]
        Sideways velocity of the male fly in each bin surrounding female.
        Note that the returned velocity units are mm/second.
    """
    def sideways_velocity(fly1, fly2):
        v_sideways, v_forward = component_velocities(fly1)
        return np.abs(v_sideways)

    return _bin_circular(
        ts,
        sideways_velocity,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm * ts.video.fps


def binned_rear_to_ellipse(ts, behavior_name=None, bins=50):
    """Calculates the average rear-to-ellipse distance from fly1's rear to
    any point on the ellipse fitted to fly2's body across all angular positions
    surrounding fly2.

    Parameters
    ----------
    ts : objects.FixedCourtshipTrackingSummary
        Which fly pair to look at.

    behavior_name : string or None (default=None)
        Which behavior to look at for the specified ts.

    bins : int (default=50)
        How many bins to split up the 360 degrees surrounding the female.

    Returns
    -------
    r2e_dists : np.array of shape [bins]
        Distances are in mm.
    """

    # We need to generate a wrapper function that returns a single value to
    # pass to _bin_circular.
    def rear_to_ellipse(fly1, fly2):
        _, r2e = nose_and_tail_to_ellipse(fly1, fly2, normalized=False)
        return r2e

    return _bin_circular(
        ts,
        rear_to_ellipse,
        behavior_name=behavior_name,
        bins=bins
    ) / ts.video.pixels_per_mm


def binned_relative_heading(ts, behavior_name=None, bins=50):
    """Get the average relative heading of fly1 w.r.t. fly2 for all angles
    surrounding fly2.

    Parameters
    ----------
    ts : objects.FixedCourtshipTrackingSummary
        Which fly pair to look at.

    behavior_name : string or None (default=None)
        Which behavior to look at for the specified ts.

    bins : int (default=50)
        How many bins to split up the 360 degrees surrounding the female.

    Returns
    -------
    relative_headings : np.array of shape [bins]
        Bins spans from -np.pi to +np.pi rads, with the female's head located
        at 0 rad.
    """
    def relative_angular_orientation(fly1, fly2):
        f1 = transforms.rotate_coordinates(fly1, fly2, direction='east')
        f1_headings = transforms._get_heading_vector(f1)
        oris = np.arctan2(
            f1_headings[:, 1], f1_headings[:, 0]
        )
        return oris

    return _bin_circular(
        ts,
        relative_angular_orientation,
        behavior_name=behavior_name,
        bins=bins
    )


def binned_angular_velocity(
    ts,
    behavior_name=None,
    bins=50,
    minimum_movement_speed=None
    ):
    """Calculates the average angular velocity of fly1 for all angular values
    surrounding fly2.

    Parameters
    ----------
    ts : objects.FixedCourtshipTrackingSummary
        Which fly pair to look at.

    behavior_name : string or None (default=None)
        Which behavior to look at for the specified ts.

    bins : int (default=50)
        How many bins to split up the 360 degrees surrounding the female.

    minimum_movement_speed : int or None (default=None)
        How fast, in pixels per second, must the fly be moving for a valid
        value to be taken? If None, a mimimum_movement_speed is automatically
        calculated by taking an otsu threshold of all movement speeds over the
        course of a courtship trial.

    Returns
    -------
    ang_vel : np.array of shape [bins]
        Velocity is in mm/sec.
    """
    pass


def nearest_neighbor_centroid(fly1, fly2, normalized=False):
    """Finds the centroid-to-centroid distance to a Fly's nearest neighbor.

    Flies should be tracked from within the same movie.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    normalized : boolean (default = True)
        Whether or not the returned array should be normalized to its
        mean.

    Returns
    -------
    distances : np.ndarray | shape = [fly1.n_frames]
        Centroid-to-centroid distances between fly1 and fly2.
    """
    fly1_centroids = fly1.body.centroid.coords_xy()
    fly2_centroids = fly2.body.centroid.coords_xy()

    distances = np.sqrt(
        (fly1_centroids[:,0] - fly2_centroids[:,0])**2 +
        (fly1_centroids[:,1] - fly2_centroids[:,1])**2
        )

    if normalized:
        return normalize(distances)
    return distances


def normalized_nearest_neighbor_centroid(fly1, fly2):
    """Calculates the centroid-to-centroid distance between two flies.

    The resulting array is normalized to the mean area of fly1. This
    helps to control for inter-video differences in pixel-to-distance
    conversion factors.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    Returns
    -------
    distances : np.ndarray | shape = [fly1.n_frames]
        Centroid-to-centroid distances between fly1 and fly2,
        normalized to mean area of fly1.
    """

    distances = nearest_neighbor_centroid(fly1, fly2, normalized=False)
    mean_area = np.mean(fly1.body.area())

    return distances.astype(np.float) / mean_area


def abs_relative_orientation(fly1, fly2):
    """Calculates the absolute value of the relative heading of fly1 w.r.t fly2.

    This returns values in the range 0 to np.pi/2, and is calculated
    based off the orientations contained in the Fly.orientation attribute.

    This should only be used for creating the feature matrix.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    Returns
    -------
    abs_relative_orientation : np.ndarray | shape = [fly1.n_frames]
        Absolute value of relative heading between fly1 and fly2.
    """
    return np.abs(relative_orientation(fly1, fly2))


def relative_orientation(fly1, fly2):
    """Calculates the relative heading of fly1 w.r.t fly2.

    This returns values in the range -np.pi/2 to np.pi/2, and is calculated
    based off the orientations contained in the Fly.orientation attribute.

    This should only be used for creating the feature matrix.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    Returns
    -------
    relative_orientation : np.ndarray | shape = [fly1.n_frames]
        Relative heading between fly1 and fly2.
    """
    fly1_orientations = fly1.body.orientation
    fly2_orientations = fly2.body.orientation

    return fly1_orientations - fly2_orientations


def nose_and_tail_to_ellipse(fly1, fly2, normalized = False):
    """Calculates nearest head-to-ellipse and tail-to-ellipse points
    between two flies.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    normalized : boolean (default = True)
        Whether or not to normalize returned arrays to their
        mean values.

    Returns
    -------
    n2e : np.ndarray | shape = [fly1.n_frames]
        Nearest distance from nose of fly1 to any point on the
        ellipse fitted to fly2.

    t2e : np.ndarray | shape = [fly1.n_frames]
        Nearest distance from tail (abdomen) of fly1 to any point
        on the ellipse fitted to fly2.
    """
    n2e = np.zeros(fly1.n_frames)
    t2e = np.zeros(fly1.n_frames)
    male_nose = fly1.body.head.coords_xy()
    male_tail = fly1.body.rear.coords_xy()

    female_centroid = fly2.body.centroid.coords_xy()[1,:]
    female_orientation = fly2.body.orientation[1]
    female_maj_axis = fly2.body.major_axis_length[1] / 2.
    female_min_axis = fly2.body.minor_axis_length[1] / 2.

    R = np.array([[np.cos(-female_orientation), -np.sin(-female_orientation)],
                  [np.sin(-female_orientation), np.cos(-female_orientation)]])

    # get 200 points that lie along the female ellipse
    t = np.linspace(0, 2 * np.pi, 200)
    cc = female_maj_axis * np.cos(t)
    rr = female_min_axis * np.sin(t)
    pts = np.vstack((cc, rr))
    rt_pts = np.dot(R, pts).T
    rt_pts += female_centroid[::-1]

    for i in range(male_nose.shape[0]):
        nose_dists = np.sqrt(
            np.sum((male_nose[i, :][::-1] - rt_pts)**2, axis = 1)
            )
        tail_dists = np.sqrt(
            np.sum((male_tail[i, :][::-1] - rt_pts)**2, axis = 1)
            )

        n2e_min_ix = np.argmin(nose_dists)
        t2e_min_ix = np.argmin(tail_dists)

        n2e[i] = nose_dists[n2e_min_ix]
        t2e[i] = tail_dists[t2e_min_ix]

    if normalized:
        stacked = np.vstack((n2e, t2e))
        normed_stack = stacked / np.mean(stacked)
        return normed_stack[0,:], normed_stack[1,:]

    return n2e, t2e


def relative_position(fly1, fly2):
    """Calculates the relative position of the position of
    fly1 to fly2.

    The relative positions of fly1 are rotated such that
    the abdomen-to-head vector of fly2 is pointing toward the
    right.

    Parameters
    ----------
    fly1 : Fly object
        First fly; should have n_frames == fly2.n_frames.

    fly2 : Fly object
        Second fly; should have n_frames == fly1.n_frames.

    Returns
    -------
    theta : np.ndarray | shape = [fly1.n_frames]
        Angle, in polar coordinates, that defines fly1's position
        in space relative to fly2.

    r : np.ndarray | shape = [fly1.n_frames]
        Distance from fly1 to fly2.
    """

    if np.sum(fly2.body.head.coords_xy()[:,0]) == 0:
        print "Female head & tail positions have not been defined. Aborting."
        return

    female_nose = fly2.body.head.coords_xy()[0,:][::-1]
    female_tail = fly2.body.rear.coords_xy()[0,:][::-1]

    female_centroids = fly2.body.centroid.coords_xy()
    female_heading_vector = female_nose - female_tail
    female_heading_vector = np.hstack(
        (female_heading_vector[0], -1 * female_heading_vector[1])
        )

    male_centroids = fly1.body.centroid.coords_xy() #note that this is (y,x);
    shape = male_centroids.shape
    male_centroids = male_centroids - female_centroids
    male_centroids = np.hstack((male_centroids[:,1],male_centroids[:,0]))
    male_centroids = male_centroids.reshape(shape, order ='F')

    female_heading_vector = female_nose - female_tail

    #rotate everything relative to the female heading vector
    rotation_angle = np.arctan2(
        female_heading_vector[1], female_heading_vector[0]
        )
    rotation_matrix = np.array(
        [[np.cos(rotation_angle), np.sin(rotation_angle)],
         [-np.sin(rotation_angle), np.cos(rotation_angle)]]
        )
    rotated_male_centroids = np.dot(
        rotation_matrix, np.transpose(male_centroids)
        )

    rotated_male_centroids = np.transpose(rotated_male_centroids)
    rotated_male_centroids = rotated_male_centroids

    theta = np.arctan2(rotated_male_centroids[:,1], rotated_male_centroids[:,0])
    r = normalized_nearest_neighbor_centroid(fly1, fly2)

    return theta, r


def relative_position2(fly1, fly2, direction='east'):
    """Calculates the relative position of fly1 w.r.t. fly2.

    .. note:: This function should be a replacement for relative_position
              following testing. The current difference between the two
              functions is that relative_position2 returns coordinates with an
              inverted y-axis w.r.t. relative_position.

    Parameters
    ----------
    fly1 : objects.Fly
        Focal fly.

    fly2 : objects.fly
        The position of fly1 will be calculated w.r.t. fly2.

    direction : string (default='east')
        Direction to rotate fly2. Options are 'east', 'west', 'north', and
        'south'.

    Returns
    -------
    theta, r : np.array of shape [N], np.array of shape [N]
        Centroid position (in polar coordinates) of fly1 w.r.t. fly2.
        Theta is in radians, and r is in pixels.
    """
    f1 = rotate_coordinates(fly1, fly2, direction=direction)

    theta = np.arctan2(
        f1.body.centroid.row, f1.body.centroid.col
    )
    r = nearest_neighbor_centroid(fly1, fly2)

    return theta, r


if __name__ == '__main__':
    pass
