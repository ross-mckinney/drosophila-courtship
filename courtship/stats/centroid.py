# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Functions for determining statistics about the
              positioning &/or movement of an individual animal.

.. moduleauthor:: Ross McKinney
"""

import numpy as np

from _signal import normalize
import transforms


def angular_velocity(fly):
    """Calculates the angular velocity of a fly.

    This is based off of the fly.orientations attribute and
    will be a signed value between -np.pi/2 and np.pi/2.

    Parameters
    ----------
    fly : Fly

    Returns
    -------
    angular_velocity : np.ndarray | shape = [fly.n_frames]
    """
    orientations = fly.body.orientation
    ang_vel = np.diff(orientations).astype(np.float)/np.diff(fly.timestamps)
    return np.hstack((0, ang_vel))


def abs_angular_velocity(fly):
    """Calculates the absolute value of the angular velocity of a fly.

    This is based off of the fly.orientations attribute and
    will be a positive float between 0 and np.pi/2.

    Parameters
    ----------
    fly : Fly object

    Returns
    -------
    angular_velocity : np.ndarray | shape = [fly.n_frames - 1]
    """
    return np.abs(angular_velocity(fly))


def centroid_to_arena_edge(fly, arena_center, arena_radius):
    """Calculates the distance from the fly to the edge of the arena.

    Parameters
    ----------
    fly : Fly

    arena_center : length-2 tuple (int, int)
        Coordinates specifying the center of the arena (rr, cc). These are pixel
        coordinates.

    arena_radius : int
        Radius of circular arena. This radius should be in pixels.

    Returns
    -------
    distance_from_edge : int
        Distance (in pixels) fly is from edge of circular arena.
    """
    b_center = fly.body.centroid.coords_xy()
    if isinstance(arena_center, tuple):
        arena_center = np.array(arena_center)
    else:
        raise AttributeError("arena_center must be a tuple.")

    dists = arena_radius - np.sqrt(
        np.sum((arena_center - b_center)**2, axis = 1)
        )
    return dists


def centroid_velocity(fly, normalized=False):
    """Calculates the velocity of a fly based of the movement of its centroid.

    Parameters
    ----------
    fly : Fly object

    normalized : boolean (default=False)
        Whether or not the velocity should be normalized to its mean.

    Returns
    -------
    centroid_velocity : np.ndarray | shape = [fly.n_frames]
    """
    centroids = fly.body.centroid.coords_xy()
    squared_differences = np.diff(centroids, axis=0) ** 2
    distance_traveled = np.sqrt(np.sum(squared_differences, axis=1))
    velocity = distance_traveled / np.diff(fly.timestamps).astype(np.float)
    velocity = np.hstack((0, velocity))

    if normalized:
         return normalize(velocity)

    return velocity


def component_velocities(fly):
    """Finds component velocities of the centroid w.r.t. the fly's heading vector.

    Parameters
    ----------
    fly : Fly

    Returns
    -------
    v_th : 1D np.array | shape = [fly.n_frames]
        Velocity - in pixel units - that is tangent to the heading vector.
        This is the side-to-side velocity of the fly.

    v_ph : 1D np.array | shape = [fly.n_frames]
        Velocity - in pixel units - that is parallel to the heading vector.
        This is the forward/reverse velocity of the fly.
    """
    body_coords = fly.body.centroid.coords_xy()
    head_coords = fly.body.head.coords_xy()
    rear_coords = fly.body.rear.coords_xy()

    # find the velocity vector
    v = np.gradient(body_coords)[0]

    # and the heading vector
    h = head_coords - rear_coords

    # find the angle between the two vectors using the cos formula
    # cos (theta) = (h . v) / (||h|| x ||v||)
    m_v = np.sqrt(np.sum(v**2, axis=1))  # magnitude of velocity
    m_h = np.sqrt(np.sum(h**2, axis=1))  # magnitude of heading

    # this method is much faster than just running
    # a for-loop and taking the pairwise dot-product
    # of each row of v and h.
    dot_hv = np.einsum('ij,ij->i', v, h)     # 'pairwise' dot product
    theta = np.arccos(dot_hv / (m_v * m_h))  # these range from 0 to pi

    # get all of the rotation angles of the heading vector
    # (this is the angle we need to rotate the heading vector
    #  [in a clockwise direction] to make the heading vector
    #  point East)
    rotation_angles = fly.body.rotation_angle

    # Now get the component velocities
    # v_ph == v_parallel_heading, forward-reverse velocity
    # v_th == v_tangent_heading, left-right velocity (we don't
    #   know the direction of this from the above data only)
    v_ph = np.zeros(v.shape[0])
    v_th = np.zeros(v.shape[0])
    for i, angle in enumerate(theta):
        v_mag = m_v[i]
        if angle <= np.pi/2:
            vt = v_mag * np.sin(angle)
            vp = v_mag * np.cos(angle)
        else:
            vt = v_mag * np.sin(np.pi - angle)
            vp = -v_mag * np.cos(np.pi - angle)

        # rotate the velocity vector w.r.t. the heading vector to figure
        # out if the tangential velocity is positive or negative.
        rotated_v = transforms._rotate(
            rotation_angles[i] * (np.pi/180),
            np.array([v[i, :]]))[0]
        if rotated_v[1] > 0:
            vt = -vt

        v_th[i] = vt
        v_ph[i] = vp

    return v_th, v_ph


def change_in_major_axis_length(fly, normalized=False):
    """Calculates the first derivative of the length
    of the major axis of the ellipse fitted to a fly.

    Parameters
    ----------
    fly : Fly

    normalized : boolean (default=False)
        Whether or not the first derivative should be normalized to its mean.

    Returns
    -------
    d_maj_ax_len : np.ndarray | shape = [fly.n_frames - 1]
    """
    maj_ax_len = fly.body.major_axis_length
    d_maj_ax_len = np.diff(maj_ax_len)/np.diff(fly.timestamps).astype(np.float)

    if normalized:
        return normalize(d_maj_ax_len)
    return d_maj_ax_len


def change_in_minor_axis_length(fly, normalized=False):
    """Calculates the first derivative of the length
    of the minor axis of the ellipse fitted to a fly.

    Parameters
    ----------
    fly : Fly object

    normalized : boolean (default=False)
        Whether or not the first derivative should be normalized to its mean.

    Returns
    -------
    d_min_ax_len : np.ndarray | shape = [fly.n_frames - 1]
    """
    min_ax_len = fly.body.minor_axis_length
    d_min_ax_len = np.diff(min_ax_len)/np.diff(fly.timestamps).astype(np.float)

    if normalized:
        return normalize(d_min_ax_len)
    return d_min_ax_len


def change_in_area(fly, normalized=False):
    """Calculates the first derivative of the area
    of the ellipse fitted to a fly.

    Parameters
    ----------
    fly : Fly object

    normalized : boolean (default=False)
        Whether or not the first derivative should be normalized to its mean.

    Returns
    -------
    d_area : np.ndarray | shape = [fly.n_frames - 1]
    """
    area = fly.body.area
    d_area = np.diff(area) / np.diff(fly.timestamps).astype(np.float)

    if normalized:
        return normalize(d_area)
    return d_area


if __name__ == '__main__':
    pass
