

import numpy as np

from skimage.color import gray2rgb
from skimage.draw import (
    circle,
    circle_perimeter,
    circle_perimeter_aa
)
from skimage.morphology import (
    dilation,
    disk
)

from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Arena(QObject):
    """Base class for arena objects.

    Contains simple function to generate a background image for
    a video.

    Parameters
    ----------
    video : motmot.FlyMovieFormat.FlyMovie object
        Video to calculate a background image from.

    Attributes
    ----------
    height : int
        Video height in pixels.

    width : int
        Video width in pixels.

    n_frames : int
        Total number of frames contained within video.

    background_image : None or np.ndarray of shape [height, width] (default = None)
        Background image calculated as mean of a specified number of frames
        take from the video.

    arena_size : int
        Size of arena in mm.

    pixels_to_mm : int
        Number of pixels in 1 mm of arena.

    Signals
    -------
    background_frame_ix : int
        Contains processing information to keep track of which frame is being
        added to the background image.
    """
    background_frame_ix = pyqtSignal(int)

    def __init__(self, video, parent = None):
        super(Arena, self).__init__(parent)
        self.video = video

        self.height = video.get_height()
        self.width = video.get_width()
        self.n_frames = video.get_n_frames()

        self.arena_size = None
        self.pixels_to_mm = None

        self.background_image = None

    def calculate_background(self, n_background_images = 200):
        """Calculates a background image for a video.

        Parameters
        ----------
        n_background_images : int (default = 200)
            Number of image to use for calculting background.
        """
        background_image = np.zeros(
            shape=(self.height, self.width), dtype=np.float)

        if n_background_images > self.n_frames:
            raise AttributeError("number of images to use for background " +
                "calculation should be less than the total number of images " +
                "present in the video associated with this arena.")

        frame_ix = np.random.randint(0, self.n_frames, n_background_images)
        for i, ix in enumerate(frame_ix):
            # need to cast to long to avoid overflow errors.
            ix = long(ix)

            # debugging output; 
            # print '\nchunk_start: {}'.format(self.video.chunk_start)
            # print 'bytes_per_chunk: {}'.format(self.video.bytes_per_chunk)
            # print 'ix: {}'.format(ix)
            # print self.video.chunk_start+self.video.bytes_per_chunk*ix

            current_img = self.video.get_frame(ix)[0]
            background_image = background_image + current_img
            self.background_frame_ix.emit(i)

        background_image = background_image / frame_ix.size
        self.background_image = background_image.astype(np.uint8)


class CircularArena(Arena):
    """Circular Arena object.

    Attributes
    ----------
    center : tuple of length 2
        Center of circle that encloses arena as (rr, cc).

    radius : int
        Radius of circle that encloses arena.
    """

    def __init__(self, video, parent=None):
        super(CircularArena, self).__init__(video, parent)
        self.center = None
        self.radius = None

    def get_arena_mask(self):
        """Returns the mask of the arena based."""
        coords = self.get_arena_coordinates()
        mask = np.zeros_like(self.background_image)
        mask[coords] = 1
        return mask

    def get_arena_coordinates(self):
        """Returns image coordinates contained within CircularArena.

        Returns
        -------
        rr : np.ndarray | shape = [N]
            Rows containing circle coodinates.

        cc : np.ndarray | shape = [N]
            Columns containing circle coordinates.
        """
        return circle(self.center[0], self.center[1], self.radius,
            shape = self.background_image.shape)

    def settings_valid(self):
        """Checks to see that all necessary settings have been set."""
        if self.center is not None and \
        self.radius is not None and \
        self.background_image is not None and \
        self.arena_size is not None and \
        self.pixels_to_mm is not None:
            return True

        return False

    def draw_arena(self, color=(255, 0, 0), thickness=2):
        """Draws perimeter of circular arena.

        Parameters
        ----------
        color : tuple of length 3 (default = (255, 0, 0))
            Color of outline of arena (R, G, B).

        thickness : int
            How thick should the line used to define the edge of the
            arena be?

        Returns
        -------
        arena_image : np.ndarray | shape = [height, width, 3]
            Three-dimensional (color) image of arena.
        """
        # make a copy of the background image so that the actual
        # background image is not affected.
        arena_image = self.background_image.copy()

        # using skimage
        # =============
        arena_image = gray2rgb(arena_image)
        assert arena_image.dtype == np.uint8, "image not of type uint8"

        # get the coordinates defining the perimeter of this arena.
        rr, cc = circle_perimeter(
            self.center[0],
            self.center[1],
            self.radius,
            shape=self.background_image.shape
        )

        # expand the line width of the above circle coordinates.
        perimeter_mask = np.zeros_like(self.background_image)
        perimeter_mask[rr, cc] = 1
        dilated_perimeter_mask = dilation(
            perimeter_mask,
            disk(thickness)
        )
        rr, cc = np.where(dilated_perimeter_mask)
        arena_image[rr, cc, :] = color
        return arena_image

