"""
statproc.py

Dialog window showing zoomed area over mouse.
"""
import numpy as np

from skimage.transform import resize

from PyQt5.QtCore import *
from PyQt5.QtGui import *

# from ..widgets.video import *

from ..utils import get_q_image


class ZoomedVideo(QDialog):
    def __init__(self, video_player=None, parent=None):
        super(ZoomedVideo, self).__init__(parent)
        self.video = video_player.video
        self.zoom_to = None
        self.zoomed_image = None

        self.video_label = QLabel()

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)

        self.setLayout(self.layout)

    def set_zoom(self, coords):
        """sets coordinates to zoom to"""
        self.zoom_to = np.asarray(coords)
        self.min_rr = np.min(coords[:, 0])
        self.max_rr = np.max(coords[:, 0])

        self.min_cc = np.min(coords[:, 1])
        self.max_cc = np.max(coords[:, 1])

    @pyqtSlot(int, str, int)
    def updateFrame(self, frame_ix, time, frame_rate):
        frame, _ = self.video.get_frame(frame_ix)
        self.zoomed_image = frame[self.min_rr:self.max_rr, self.min_cc:self.max_cc]

        # zoom 4x
        self.zoomed_image = resize(
            self.zoomed_image,
            ((self.max_rr - self.min_rr) * 4, (self.max_cc - self.min_cc) * 4),
            preserve_range=True
            )

        pixmap = QPixmap.fromImage(
            get_q_image(self.zoomed_image.astype(np.uint8))
            )
        self.video_label.setPixmap(pixmap)
