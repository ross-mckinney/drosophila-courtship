# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Functions for determining summary statistics between
              two animals.

.. moduleauthor:: Ross McKinney
"""

import numpy as np

from _signal import normalize
from transforms import (rotate_coordinates)

def nearest_neighbor_centroid(fly1, fly2, normalized = False):
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
    fly1_centroids = fly1.body.centroid.coords()
    fly2_centroids = fly2.body.centroid.coords()

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

    distances = nearest_neighbor_centroid(fly1, fly2, normalized = False)
    mean_area = np.mean(fly1.body.area)

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
    male_nose = fly1.body.head.coords()
    male_tail = fly1.body.rear.coords()

    female_centroid = fly2.body.centroid.coords()[1,:]
    female_orientation = fly2.body.orientation[1]
    female_maj_axis = fly2.body.major_axis_length[1] / 2.
    female_min_axis = fly2.body.minor_axis_length[1] / 2.

    R = np.array([[np.cos(-female_orientation), -np.sin(-female_orientation)],
                  [np.sin(-female_orientation), np.cos(-female_orientation)]])

    #get 200 points that lie along the ellipse
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

    if np.sum(fly2.body.head.coords()[:,0]) == 0:
        print "Female head & tail positions have not been defined. Aborting."
        return

    female_nose = fly2.body.head.coords()[0,:][::-1]
    female_tail = fly2.body.rear.coords()[0,:][::-1]

    female_centroids = fly2.body.centroid.coords()
    female_heading_vector = female_nose - female_tail
    female_heading_vector = np.hstack(
        (female_heading_vector[0], -1 * female_heading_vector[1])
        )

    male_centroids = fly1.body.centroid.coords() #note that this is (y,x);
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
        f1.body.centroid.y, f1.body.centroid.x
    )
    r = nearest_neighbor_centroid(fly1, fly2)

    return theta, r

if __name__ == "__main__":
    pass
