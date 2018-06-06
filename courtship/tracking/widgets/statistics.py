"""
statistics.py

Widgets for examining statistical data.
"""
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ..utils import (
    get_q_image, 
    get_mouse_coords
)


class StatisticWindowWidget(QWidget):
    """This class allows for a selected statistic to be displayed along
    with a video.
    Parameters
    ----------
    statistics : dictionary
        Possible statistics to display. Keys will be displayed in stat_combobox,
        and values will be displayed in the stat_image_label.

    tracking_summary : FixedCourtshipTrackingSummary or None (default=None)
        TrackingSummary object from which to show selected statistics.

    window_size : int (default = 640)
        Number of frames to display in given window.

    window_height : int (default = 30)
        Height of statistic window (in pixels).

    Attributes
    ----------
    stat_image_label : QLabel
        Contains an image of the specified statistic.

    stat_spinbox : QSpinBox
        Contains a list of statistics that can be displayed in the statistic
        image label (stat_image_label).
    """
    def __init__(self, 
            tracking_summary=None, 
            statistics={}, 
            window_size=640,
            window_height=30,
            parent=None
        ):
        super(StatisticWindowWidget, self).__init__(parent)
        self.tracking_summary = tracking_summary
        self.raw_statistics = statistics
        self.statistics = statistics
        self.window_size = window_size
        self.window_height = window_height
        self.current_ix = 0

        layout = QHBoxLayout()
        self.stat_combobox = QComboBox()
        self.stat_combobox.addItems(sorted(self.statistics.keys()))
        self.stat_combobox.currentIndexChanged.connect(self.combo_sig_update)

        self.stat_image_label = QLabel()
        self.stat_image_label.setFixedWidth(self.window_size)
        self.stat_image_label.setFixedHeight(self.window_height)
        self.stat_image_label.setPixmap(
                QPixmap.fromImage(
                        get_q_image(
                            np.ones(
                                shape=(self.window_height, self.window_size), 
                                dtype=np.uint8
                                ) * 55
                            )
                    )
            )
        layout.addWidget(self.stat_image_label)
        layout.addWidget(self.stat_combobox)
        self.setLayout(layout)

    def _push_to_range(self, arr, max_val):
        """Makes an array fall within a certain range (0 to max_val).

        Parameters
        ----------
        arr : np.ndarray of shape [N]

        max_val : int
            This will be the new largest value within the returned array.

        Returns
        -------
        bound_arr : np.ndarray of shape [N]
            Array with all values bound between 0 and max_val.
        """
        if np.max(arr) == 0:
            return arr.astype(np.int)
            
        ones = arr / np.max(arr).astype(np.float)
        ones *= (max_val - 1)
        return ones.astype(np.int)

    @pyqtSlot(int)
    def combo_sig_update(self, ix):
        """Updates the statistic label when a user switches to a different
        statistic listed in the stat_combobox."""
        self.update_stat_label(self.current_ix)

    @pyqtSlot(int)
    def update_stat_label(self, ix):
        """Updates the displayed statistic.

        This is a slot for signals coming from a slider/other user input.
        """
        self.current_ix = ix
        try: 
            current_stat = self.statistics[
                str(self.stat_combobox.currentText())]
        except KeyError:
            return

        if ix < self.window_size / 2:
            stat_window = current_stat[:self.window_size]
            marker = ix
        elif ix > current_stat.size - self.window_size / 2:
            stat_window = current_stat[current_stat.size - self.window_size:]
            marker = self.window_size + ix - current_stat.size
        else:
            stat_window = current_stat[
                ix - self.window_size / 2:ix + self.window_size/2]
            marker = self.window_size / 2

        self.stat_image = np.zeros(
            shape=(self.window_height, self.window_size, 3), dtype=np.uint8)
        for col, val in enumerate(stat_window):
            self.stat_image[
                int(self.window_height - val):, col, :] = [141, 221, 247]
        self.stat_image[:, marker, :] = 100

        self.stat_image_label.setPixmap(
                QPixmap.fromImage(
                        get_q_image(self.stat_image)
                    )
            )

    def update_combobox(self):
        """Updates the items displayed in the combobox to whatever is 
        currently stored in self.statistics. Also makes sure that all
        statistics are within the appropriate ranges."""

        #Remove all items
        while self.stat_combobox.count() > 0:
            ix = self.stat_combobox.count() - 1
            self.stat_combobox.removeItem(ix)

        #add new items
        self.stat_combobox.addItems(sorted(self.statistics.keys()))

        pushed_stats = {}
        for k, v in self.statistics.iteritems():
            pushed_stats[k] = self._push_to_range(v, self.window_height - 1)

        self.statistics = pushed_stats


class SelectableWindow(StatisticWindowWidget):
    """Base class for any window that allows a user to interact with it."""
    scrolled_on_window = pyqtSignal(int)

    def __init__(self, 
        tracking_summary=None,
        statistics={},
        window_size=640,
        window_height=30,
        parent=None):
        super(SelectableWindow, self).__init__(
            tracking_summary=tracking_summary,
            statistics=statistics,
            window_size=window_size,
            window_height=window_height,
            parent=parent)

        self.window_set = False

        self.stat_image_label.mousePressEvent = self.mousePress 
        self.stat_image_label.mouseMoveEvent = self.mouseMove 
        self.stat_image_label.mouseReleaseEvent = self.mouseRelease
        self.stat_image_label.wheelEvent = self.wheelMove

    def set_tracking_summary(self, tracking_summary):
        """Makes sure to set both tracking summary file and
        behaviors held within the tracking_summary.

        Parameters
        ----------
        tracking_summary : FixedCourtshipTrackingSummary object
        """
        self.tracking_summary = tracking_summary
        self.statistics = self.tracking_summary.behaviors
        if len(self.statistics) > 0:
            self.n_frames = np.asarray(
                self.statistics[self.statistics.keys()[0]]).size
        else:
            self.n_frames = 0

    def _get_frame_ix_onclick(self, cc):
        """Returns the relative and absolute coordinates of a
        users clicks within the classification image label (
        stat_image_label).

        Parameters
        ----------
        cc : int
            User click position (column coordinate) from event.

        Returns
        -------
        rel_coord : int 
            Relative click position (column coordinate) within stat_image_label.
            This is bounded between 0 and self.window_size.

        abs_coord : int
            Absolute frame at which user clicked. This maps rel_coord from 
            being bounded by [0, 640) to [0, n_frames).
        """
        w_mid = self.window_size / 2
        rel_coord = cc
        if self.current_ix <= w_mid:
            abs_coord = cc
        elif self.current_ix > self.n_frames - w_mid:
            abs_coord = self.n_frames - (self.window_size - cc)
        else:
            abs_coord = (cc - w_mid) + self.current_ix

        if abs_coord >= self.n_frames - 1:
            abs_coord = self.n_frames - 1
        elif abs_coord < 0:
            abs_coord = 0

        if rel_coord >= self.window_size - 1:
            rel_coord = self.window_size - 1
        elif rel_coord < 0:
            rel_coord = 0

        return rel_coord, abs_coord

    def mouseMove(self, event):
        rr, cc = get_mouse_coords(event)
        self.end_ix_rel, self.end_ix_abs = self._get_frame_ix_onclick(cc)
        self.update_class_stat_label()
        # print self.end_ix_rel, self.end_ix_abs

    def mousePress(self, event):
        self.window_set = True
        rr, cc = get_mouse_coords(event)
        self.start_ix_rel, self.start_ix_abs = self._get_frame_ix_onclick(cc)
        # print self.start_ix_rel, self.start_ix_abs

    def mouseRelease(self, event):
        pass

    def wheelMove(self, event):
        degrees = event.delta() / 8
        steps = degrees / 15
        self.update_stat_label(self.current_ix + steps)
        self.scrolled_on_window.emit(self.current_ix + steps)

    def update_class_stat_label(self):
        img = self.stat_image.copy()
        img[:, self.start_ix_rel] = [255, 0, 0]
        img[:, self.end_ix_rel] = [255, 0, 0]
        if self.start_ix_rel <= self.end_ix_rel:
            img[self.window_height-1, self.start_ix_rel:self.end_ix_rel] = [255, 0, 0]
            img[0, self.start_ix_rel:self.end_ix_rel] = [255, 0, 0]
        else:
            img[self.window_height-1, self.end_ix_rel:self.start_ix_rel] = [255, 0, 0]
            img[0, self.end_ix_rel:self.start_ix_rel] = [255, 0, 0]
        self.stat_image_label.setPixmap(
                QPixmap.fromImage(
                        get_q_image(img)
                    )
            )


class StatProcessingWindowWidget(SelectableWindow):
    """Widget that allows the user to select a part of a statistic.

    This is useful so that a user may 'grab' pieces of a signal to 
    do analyses on them.
    """
    #signal that contains the raw statistic data.
    #Should emit the statistic/signal name, followed by the start
    #index, the stop index, then the raw statistic value.
    region_selected = pyqtSignal(str, int, int, list)

    def __init__(self,
        tracking_summary=None,
        statistics={},
        window_size=640,
        window_height=60,
        parent=None
        ):
        super(StatProcessingWindowWidget, self).__init__(
            tracking_summary=tracking_summary,
            statistics=statistics,
            window_size=window_size,
            window_height=window_height,
            parent=parent
            )

        self.select_region_action = QAction('Select Region', self)
        self.select_region_action.setShortcut(Qt.Key_Enter)
        self.select_region_action.triggered.connect(self.grab_signal_data)
        self.addAction(self.select_region_action)

    def grab_signal_data(self):
        current_stat = self.raw_statistics[
            str(self.stat_combobox.currentText())]

        if self.end_ix_abs <= self.start_ix_abs:
            start = self.end_ix_abs
            end = self.start_ix_abs
        else:
            start = self.start_ix_abs
            end = self.end_ix_abs

        self.region_selected.emit(
            str(self.stat_combobox.currentText()), 
                start,
                end, 
                current_stat[start:end].tolist()
            )
        

class ClassificationWindowWidget(SelectableWindow):
    """Sliding window to show any classifications that may be present 
    within a FCTS file.

    This class differs from StatisticsWindowWidget in that the
    user may click on the statistics window & change a classification.
    """

    #signal to send to main window when a user has changed a classification.
    classification_changed = pyqtSignal(str, list)

    def __init__(self, 
        tracking_summary=None,
        classifications={},
        window_size=640,
        window_height=30,
        parent=None):
        super(ClassificationWindowWidget, self).__init__(
            tracking_summary=tracking_summary,
            statistics=classifications,
            window_size=window_size,
            window_height=window_height,
            parent=parent)

        set_negative_action = QAction('Classify as negative examples.', self)
        set_negative_action.triggered.connect(
            self.update_classifications_negative)
        set_negative_action.setShortcut(QKeySequence('0'))
        self.addAction(set_negative_action)
        
        set_positive_action = QAction('Classify as positive examples.', self)
        set_positive_action.triggered.connect(
            self.update_classifications_positive)
        set_positive_action.setShortcut(QKeySequence('1'))
        self.addAction(set_positive_action)

    def update_classifications_negative(self):
        """Updates classification array so that specified frames contain
        negative examples of a behavior."""
        if not self.window_set:
            return

        try: 
            current_stat = self.statistics[
                str(self.stat_combobox.currentText())]
        except KeyError:
            return

        if self.start_ix_abs <= self.end_ix_abs:
            current_stat[self.start_ix_abs:self.end_ix_abs] = 0
        else:
            current_stat[self.end_ix_abs:self.start_ix_abs] = 0
        self.statistics[str(self.stat_combobox.currentText())] = current_stat
        self.update_stat_label(self.current_ix)

        self.window_set = False

    def update_classifications_positive(self):
        """Updates classification array so that specified frames contain
        positive exmples of a behavior."""
        if not self.window_set: 
            return

        try: 
            current_stat = self.statistics[
                str(self.stat_combobox.currentText())]
        except KeyError:
            return

        if self.start_ix_abs <= self.end_ix_abs:
            current_stat[self.start_ix_abs:self.end_ix_abs] = (
                self.window_height - 2)
        else:
            current_stat[self.end_ix_abs:self.start_ix_abs] = (
                self.window_height - 2)
        self.statistics[str(self.stat_combobox.currentText())] = current_stat
        self.update_stat_label(self.current_ix)

        self.window_set = False

