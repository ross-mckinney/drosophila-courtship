
import numpy as np

from skimage.draw import (
    ellipse,
    ellipse_perimeter
)
from skimage.color import gray2rgb
from skimage.morphology import (
    dilation,
    disk
)

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from arena import CircularArena


class Female(QObject):
    """Container for fixed female object.

    Parameters
    ----------
    arena : Arena object
        Arena that contains the female.

    Attributes
    ----------
    center : tuple of int
        Coordinats of the center of the female's body (rr, cc).

    maj_ax_rad : int
        Radius of major axis of ellipse

    min_ax_rad : int
        Radius of minor axis of ellipse

    orientation : int
        Rotation of ellipse in degrees. This is the
        rotation of the major axis of the ellipse from 
        the x-axis

    head : tuple of int
        Coordinates of the female's head (rr, cc).

    rear : tuple of int
        Coordinates of the female's rear (rr, cc).
    """
    def __init__(self, arena, parent=None):
        super(Female, self).__init__(parent)
        self.arena = arena
        self.center = None
        self.maj_ax_rad = None # radius of first ellipse axis
        self.min_ax_rad = None # radius of second ellipse axis
        self.orientation = None
        self.head = None
        self.rear = None

    def settings_valid(self):
        """Checks to see whether all settings have been set.

        Returns
        -------
        settings_valid : bool
            True if arena, center, maj_ax_rad, min_ax_rad, 
            orientation, head, and rear are not None.
        """
        if self.arena is not None and \
           self.center is not None and \
           self.maj_ax_rad is not None and \
           self.min_ax_rad is not None and \
           self.orientation is not None and \
           self.head is not None and \
           self.rear is not None:
            return True
        return False

    def get_female_mask(self):
        """Gets a mask where the ellipse containing the female 
        is 1s and everything else is 0s.

        Returns 
        -------
        mask : np.ndarray of shape [self.arena.background_image.shape]
            Female ellipse is 1s everything else is 0s.
        """
        mask = np.zeros_like(self.arena.background_image)
        mask[self.get_female_coords()] = 1
        return mask

    def get_female_coords(self):
        """Gets the coordinates of the ellipse that contain the a female.

        Returns 
        -------
        rr : np.ndarray of shape [N]
            Rows containing female ellipsoid mask.

        cc : np.ndarray of shape [N]
            Columns containing female ellipsoid mask.
        """
        rr, cc = ellipse(
            self.center[0],
            self.center[1],
            self.min_ax_rad,
            self.maj_ax_rad,
            rotation=(self.orientation * np.pi/180),
            shape=self.arena.background_image.shape
        )
        return rr, cc

    def draw_female(self, color=(255, 0, 0), thickness=2):
        """Draws an ellipse perimeter around a female.

        Parameters
        ----------
        color : 3-tuple
            Color of ellipse outline.
        thickness : int
            Thickness of ellipse outline.

        Returns
        -------
        image : 3D np.ndarray of shape [arena.background_image.shape]
            Background image with female surrounded by ellipse.
        """
        image = self.arena.background_image.copy()

        # convert to color
        image = gray2rgb(image)
        assert image.dtype == np.uint8, "image is not of type uint8"

        rr, cc = ellipse_perimeter(
            self.center[0],
            self.center[1],
            self.min_ax_rad,
            self.maj_ax_rad,
            orientation=-(self.orientation * np.pi/180),
            shape=self.arena.background_image.shape
        )
        ellipse_outline_mask = np.zeros_like(self.arena.background_image)
        ellipse_outline_mask[rr, cc] = 1
        dilated_ellipse_outline = dilation(
            ellipse_outline_mask,
            disk(thickness)
        )
        rr, cc = np.where(dilated_ellipse_outline)
        image[rr, cc, :] = color
        return image
