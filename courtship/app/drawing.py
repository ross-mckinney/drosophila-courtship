
import numpy as np

from skimage.color import (
    gray2rgb,
    rgb2gray
)
from skimage.draw import (
    circle
)


def draw_tracked_wings(image, left_centroid, right_centroid, head_centroid, 
    tail_centroid, female_head=None, female_rear=None, left_color=(0, 0, 255),
    right_color=(255, 0, 0), tail_color=(0, 55, 125), head_color=(0, 55, 125),
    female_head_color=(0, 0, 0), female_rear_color=(0, 255, 0), radius=2):
    """Annotates an image given tracked wing positions.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to annotate.

    left_centroid : np.ndarray of size 2.
        Position of left wing (rr, cc).

    right_centroid : np.ndarray of size 2.
        Position of right wing (rr, cc).

    head_centroid : np.ndarray of size 2.
        Position of male head (rr, cc).

    tail_centroid : np.ndarray of size 2.
        Position of male rear (rr, cc).

    female_head : None or np.ndarray of size 2.
        Position of female head (rr, cc).

    female_rear : None or np.ndarray of size 2.
        Position of female rear (rr, cc).

    Returns
    -------
    image : 3D np.ndarray
        Annotated image.
    """
    if len(image.shape) != 2:
        raise AttributeError('`image` must be a 2d np.ndarray')

    image = gray2rgb(image.copy())

    # draw circles
    to_draw = [left_centroid, right_centroid, head_centroid, tail_centroid, 
        female_head, female_rear]
    to_draw_colors = [left_color, right_color, head_color, tail_color,
        female_head_color, female_rear_color]
    for i in xrange(len(to_draw)):
        if to_draw[i] is not None:
            rr, cc = circle(
                to_draw[i][0], to_draw[i][1], radius, shape=image.shape)
            image[rr, cc, :] = to_draw_colors[i]

    return image