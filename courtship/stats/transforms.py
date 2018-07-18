## -*- coding: utf-8 -*-
"""
.. module:: statistics
   :synopsis: Useful functions for performing coordinate transformations.

.. moduleauthor:: Ross McKinney
"""

import copy
import functools

import numpy as np


def _get_heading_vector(fly):
    """Calculates the heading vectors for a specified fly across all frames.

    Parameters
    ----------
    fly : objects.Fly
        Fly to calculate heading vector for.

    Returns
    -------
    heading_vectors : np.array of shape [N, 2].
        Heading vector of fly in each frame.
    """
    return fly.body.head.coords_xy() - fly.body.rear.coords_xy()


def _rotate(angle, coords):
    """Rotates a given set of coordinates (x, y), around a specified angle.

    Parameters
    ----------
    angle : float
        Angle given in radians.

    coords : np.array of shape [N, 2].
        Coordinates to rotate.

    Returns
    -------
    transformed_coords : np.array of shape [N, 2].
        Coordinates (x, y) rotated about given angle.
    """
    # Get the rotation matrix based on the given angle..
    R = np.array(
        [
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ]
    )

    # Make sure the passed coords are of type np.array.
    coords = np.asarray(coords)

    # Make sure this is a column-based array.
    if coords.shape[1] != 2:
        raise AttributeError(
            'Parameter coords passed in shape [{},{}]. '.format(
                coords.shape[0], coords.shape[1]) +
            'Please make sure that coords are of shape [N, 2]'
            )

    return np.transpose(
            np.dot(
                    R,
                    np.transpose(coords)
                )
            )


_sentinel = object()
def _rgetattr(obj, attr, default=_sentinel):
    """Function to recursively get a named attribute.

    Parameters
    ----------
    obj : object
        Object from which to get attribute.

    attr : string
        Name of attribute to get. This could potentially contain a dotted
        attribute name. Example 'fly.body.centroid.x'.

    References
    ----------
    [1]: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    if default is _sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj]+attr.split('.'))


def _rsetattr(obj, attr, val):
    """Function to recursively set a named attribute.

    Parameters
    ----------
    obj : object
        Object to set attribute on.

    attr : string
        Name of attribute to set. This could potentially contain a dotted
        attribute name. Example 'fly.body.centroid.x'.

    val : value
        Value to assign to attribute.

    References
    ----------
    [1]: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def rotate_coordinates(fly1, fly2, direction='east'):
    """Rotates all the coordinates of fly1 w.r.t. fly2.

    .. note:: fly1 coordinates will be centered w.r.t. fly2's centroid.

    Parameters
    ----------
    fly1 : objects.Fly
        Fly containing coordinates to rotate.

    fly2 : objects.Fly
        fly1 will be rotated w.r.t. fly2's heading vector.

    direction : string (default='east')
        Which direction should fly2 be facing?
        Options are 'east', 'west', 'north', and 'south'.

    Returns
    -------
    fly1_copy : objects.Fly
        This is a deepcopy of the originally passed, fly1 object. All of the
        point properties within fly1 have been rotated w.r.t. fly2's heading
        vector.
    """

    # Get the heading of fly2.
    fly2_heading = _get_heading_vector(fly2)

    # Get the angle that the heading vector makes with the positive x-axis.
    theta = np.arctan2(
        fly2_heading[0, 1], fly2_heading[0, 0]
    )

    # Adjust theta based on the specified direction of rotation.
    if direction == 'west':
        theta += np.pi
    # Note that north and south are flipped because of the y-axis inversion,
    # which is inherent in image processing.
    elif direction == 'north':
        theta -= np.pi/2
    elif direction == 'south':
        theta += np.pi/2
    elif direction == 'east':
        pass
    else:
        raise AttributeError(
            'direction parameter must be one of the following:' +
            '\n\t1. {}'.format('east (default)') +
            '\n\t2. {}'.format('west') +
            '\n\t3. {}'.format('north') +
            '\n\t4. {}'.format('south')
        )

    # Generate a copy of fly1... we will insert the rotated coords into this
    # copy.
    fly1_copy = copy.deepcopy(fly1)

    # Go through each of the fly's body parts and rotate the appropriate coords.
    to_rotate = {
        'body': ['centroid', 'head', 'rear'],
        'left_wing': ['centroid'],
        'right_wing': ['centroid']
    }
    for part_name, point_list in to_rotate.iteritems():
        for point in point_list:
            coords = _rgetattr(
                    fly1_copy,'{}.{}'.format(part_name, point)
                ).coords_xy()

            # Make sure to center all coordinates w.r.t. the centroid of fly2.
            coords = coords - fly2.body.centroid.coords_xy()

            rotated_coords = _rotate(theta, coords)
            for i, xy in enumerate(['x', 'y']):
                _rsetattr(
                    fly1_copy,
                    '{}.{}.{}'.format(part_name, point, xy),
                    rotated_coords[:, i]
                )
    return fly1_copy
