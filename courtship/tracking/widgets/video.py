"""
video.py

Widgets for examining video data.
"""
import time, datetime

import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np
from skimage.draw import (
    line_aa,
    polygon_perimeter
)

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget

from ..drawing import draw_tracked_wings
from ..dialogs.videozoom import ZoomedVideo
from ..utils import (
    get_q_image,
    get_mouse_coords
)

class VideoControls(QWidget):
    """Container for standard video controls.

    This includes the play, stop, next, and previous
    buttons as well as the slider.
    """
    def __init__(self, parent=None):
        super(VideoControls, self).__init__(parent)


class BaseVideoPlayer(QWidget):
    """Base class for video players.

    This is just a single QLabel that can hold and update
    frames from a specified video.

    Attributes
    ----------
    video : motmot.FlyMovieFormat.FlyMovie
        Video to display in widget.

    is_playing : bool (default = False)
        Whether or not the video is playing.

    current_frame_ix : int
        Frame index of currently-displayed image in label.

    frame_rate : int (default = 24)
        How fast the video should be playing (in frames per second).

    frame_label : QLabel
        QLabel that will display the video data.

    video_file_name : string
        Path to video file currently being displayed.

    image_annotation : bool (default = False)
        Whether or not an annotated (tracked) image should be displayed.

    tracking_summary : FixedCourtshipTrackingSummary object or None (default=None)
        Containing flies to draw annotated images (if necessary)

    Signals
    -------
    frame_changed : int, str, int
        Signal holding the following values:

        1. frame number : int
            Current frame number being displayed.

        2. time : str
            Formatted as HH:MM:SS, based on the mean frame rate calculated
            upon loading video.

        3. frame rate : int
            Mean frame rate of video (rounded to floor).
    """
    frame_changed = pyqtSignal(int, str, int)

    def __init__(self, parent=None):
        super(BaseVideoPlayer, self).__init__(parent)
        self.video = None
        self.video_file_name = None
        self.is_playing = False
        self.current_frame_ix = 0
        self.playback_speed = 1
        self.frame_rate = 24
        self.frame_label = QLabel()
        self.image_annotation = False
        self.tracking_summary = None
        self.zoom_enabled = False
        self.zoom_coords = []

        img = np.zeros(shape=(480, 640), dtype=np.uint8)
        pixmap = QPixmap.fromImage(get_q_image(img))
        self.frame_label.setPixmap(pixmap)

    def _frame_to_time(self, ix):
        """Converts a given frame to a formatted time string (HH:MM:SS)"""
        total_seconds = int(ix) / int(self.frame_rate)
        hours = total_seconds / 3600
        minutes = (total_seconds - (hours * 3600)) / 60
        seconds = (total_seconds - (minutes * 60) - (hours * 3600))
        return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

    def previous(self):
        if self.current_frame_ix > 0:
            self.current_frame_ix -= 1
        self.update_label()

    def next(self):
        frame_increment = int(
                (self.frame_rate * self.playback_speed) / self.frame_rate
            )
        if self.current_frame_ix < (self.video.get_n_frames() - frame_increment):
            self.current_frame_ix += frame_increment
        else:
            self.is_playing = False
        self.update_label()

    def update_label(self, frame_number=None):
        """Sets label Pixmap based on self.current_frame_ix."""
        if frame_number is None:
            frame_number = self.current_frame_ix
        else:
            if frame_number <= self.video.get_n_frames():
                self.current_frame_ix = frame_number
            else:
                self.current_frame_ix = self.video.get_n_frames() - 1

        try:
            if not self.image_annotation:
                img = self.video.get_frame(self.current_frame_ix)[0]
            else:
                try:
                    img = draw_tracked_wings(
                            image=self.video.get_frame(self.current_frame_ix)[0],
                            left_centroid=self.tracking_summary.male.left_wing.centroid.coords()[self.current_frame_ix, :],
                            right_centroid=self.tracking_summary.male.right_wing.centroid.coords()[self.current_frame_ix, :],
                            head_centroid=self.tracking_summary.male.body.head.coords()[self.current_frame_ix, :],
                            tail_centroid=self.tracking_summary.male.body.rear.coords()[self.current_frame_ix, :]
                        )
                except:
                    img = self.video.get_frame(self.current_frame_ix)[0]

            if self.zoom_enabled and self.zoom_coords != []:
                self.zoom_coords = np.asarray(self.zoom_coords)

                min_rr = np.min(self.zoom_coords[:, 0])
                max_rr = np.max(self.zoom_coords[:, 0])
                min_cc = np.min(self.zoom_coords[:, 1])
                max_cc = np.max(self.zoom_coords[:, 1])

                line_rr, line_cc = polygon_perimeter(
                    [min_rr, max_rr, max_rr, min_rr],
                    [min_cc, min_cc, max_cc, max_cc]
                )

                img[line_rr, line_cc] = 0

            pixmap = QPixmap.fromImage(get_q_image(img))
            self.frame_label.setPixmap(pixmap)
            self.frame_changed.emit(
                self.current_frame_ix, 
                self._frame_to_time(self.current_frame_ix), 
                int(self.frame_rate)
                )
        except FMF.NoMoreFramesException:
            self.current_frame_ix = 0
            self.update_label()

    def play(self):
        while self.is_playing:
            start_time = time.time()

            self.next()
            QApplication.processEvents()

            end_time = time.time()
            elapsed_time = end_time - start_time
            if (1. / self.frame_rate - elapsed_time) > 0:
                time.sleep(1. / self.frame_rate - elapsed_time)

    def on(self):
        self.is_playing = True
        self.play()

    def off(self):
        self.is_playing = False

    def set_video(self, video_file):
        """Opens video to be played in VideoPlayer.

        Parameters
        ----------
        video_file : string
            Path to video file.
        """
        self.video_file_name = video_file
        self.video = FMF.FlyMovie(video_file)

        #sample a subset of frames for the
        sample_size = 20
        mean_frame_rates = []
        timestamp_start_ixs = np.random.random_integers(
            low=0,
            high=self.video.get_n_frames() - sample_size - 1,
            size=4
            )

        for i in timestamp_start_ixs:
            timestamp_subset = []
            for ix in xrange(i, i + sample_size):
                f, t = self.video.get_frame(ix)
                timestamp_subset.append(t)
            mean_frame_rates.append(1. / np.mean(np.diff(timestamp_subset)))
        self.frame_rate = np.mean(mean_frame_rates)

        if np.isnan(self.frame_rate):
            self.frame_rate = 24

        self.update_label()

    def slider_update_label(self, value):
        """Updates image label based on slider signals."""
        self.current_frame_ix = value
        self.update_label()

    def mousePressEvent(self, event):
        if not self.zoom_enabled:
            return

        rr, cc = get_mouse_coords(event)
        self.zoom_coords = [[rr, cc]]
        print self.zoom_coords[0]
        # print 'Mouse press @ ({},{})'.format(rr, cc)

    def mouseMoveEvent(self, event):
        if not self.zoom_enabled:
            return

        rr, cc = get_mouse_coords(event)
        current_frame = self.video.get_frame(self.current_frame_ix)[0]

        min_rr = np.min([rr, self.zoom_coords[0][0]])
        max_rr = np.max([rr, self.zoom_coords[0][0]])
        min_cc = np.min([cc, self.zoom_coords[0][1]])
        max_cc = np.max([cc, self.zoom_coords[0][1]])

        line_rr, line_cc = polygon_perimeter(
            [min_rr, max_rr, max_rr, min_rr],
            [min_cc, min_cc, max_cc, max_cc]
        )

        current_frame[line_rr, line_cc] = 0

        pixmap = QPixmap.fromImage(get_q_image(current_frame))
        self.frame_label.setPixmap(pixmap)

    def mouseReleaseEvent(self, event):
        if not self.zoom_enabled:
            return

        rr, cc = get_mouse_coords(event)
        self.zoom_coords.append([rr, cc])
        self.open_zoom_window()
        # print 'Mouse release @ ({},{})'.format(rr, cc)

    def open_zoom_window(self):
        self.videoZoomWindow = ZoomedVideo(video_player=self)
        self.videoZoomWindow.set_zoom(np.array(self.zoom_coords))
        self.frame_changed.connect(self.videoZoomWindow.updateFrame)
        self.update_label()
        self.videoZoomWindow.show()

    def close_zoom_window(self):
        self.zoom_enabled = False
        self.videoZoomWindow.accept()
        self.update_label()


class MainVideoPlayer(BaseVideoPlayer):
    """Class container for CentralWidget video player."""
    def __init__(self, parent = None):
        super(MainVideoPlayer, self).__init__(parent)

        layout = QGridLayout()
        layout.addWidget(self.frame_label, 0, 0)
        self.setLayout(layout)


class TrackedVideoPlayer(BaseVideoPlayer):
    """Class for playing tracked videos."""
    def __init__(self, parent = None):
        super(TrackedVideoPlayer, self).__init__(parent)
