"""
Entry point into GUI.
"""
import os
import sys
import time
import pickle
import logging

import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt)
from PyQt5.QtGui import (QIcon, QKeySequence, QPixmap, QCursor)
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QSlider
)

from dialogs.batch import BatchTrackingDialog
from dialogs.statproc import StatProcessing
from widgets.video import MainVideoPlayer
from widgets.fileio import FileExplorer
from widgets.statistics import StatisticWindowWidget, ClassificationWindowWidget

from ..ts import FixedCourtshipTrackingSummary as FCTS
import courtship.stats.spatial as spatial_stats
import courtship.stats.wing as wing_stats
import courtship.stats.centroid as centroid_stats

DIR = os.path.dirname(__file__)
logging.basicConfig()

class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.root_folder = os.path.expanduser("~")
        self.logger = logging.getLogger(name='MainWindow Logger')

        self.set_main_widget()

        #create menus
        self.file_menu = self.menuBar().addMenu('&File')
        self.view_menu = self.menuBar().addMenu('View')
        self.video_menu = self.menuBar().addMenu('Video')
        self.tracking_menu = self.menuBar().addMenu('Tracking')
        self.analysis_menu = self.menuBar().addMenu('Analysis')

        #create toolbar for navigating through videos
        self.video_toolbar = self.addToolBar('Video')
        self.video_trackbar = self.addToolBar('Video Track Bar')

        self.set_file_menu()
        self.set_video_menu()
        self.set_tracking_menu()
        self.set_analysis_menu()

        self.set_slider()
        self.set_status()
        self.set_file_explorer()
        self.set_statistic_toolbar()
        self.set_classification_toolbar()

        self.set_view_menu()

    def set_main_widget(self):
        self.video_player = MainVideoPlayer()
        self.setCentralWidget(self.video_player)

    def add_menu_action(self, menu, name, connection, status_tip,
        shortcut=None, icon=None, tool_bar=None, return_action=False):
        """Helper function for adding a QAction to a menu.

        Parameters
        ----------
        menu : QMenu
            Which menu to add new QAction to.

        name : string
            Name of new QAction. This is displayed as the QAction
            text in the QMenu.

        connection : function
            Which function should the QAction be connected to.

        status_tip : string
            Status tip to display on MouseOver.

        shortcut : QKeySequence or None (default = None)
            Shortcut to execute action.

        icon : QIcon or None (default = None)
            Icon to attach to QAction.

        tool_bar : QToolBar or None
            QToolBar to attach QAction to (in addition to the specified QMenu).

        return_action : bool (default = False)
            Whether or not to return the action.
        """
        if icon is None:
            action = QAction(name, self)
        else:
            action = QAction(icon, name, self)

        if shortcut is not None:
            action.setShortcut(shortcut)

        action.setStatusTip(status_tip)
        action.triggered.connect(connection)
        menu.addAction(action)

        if tool_bar is not None:
            tool_bar.addAction(action)

        if return_action:
            return action

    def set_file_menu(self):
        self.add_menu_action(
            menu=self.file_menu,
            name='Open Directory',
            connection=self.open_file_directory,
            status_tip='Open a folder in the file explorer.',
            shortcut=QKeySequence.Open
            )
        self.add_menu_action(
            menu=self.file_menu,
            name='Open Video',
            connection=self.open_video,
            status_tip='Open an individual video.',
            shortcut=None
            )
        self.add_menu_action(
            menu=self.file_menu,
            name='Open Tracking Summary',
            connection=self.open_tracking_summary,
            status_tip='Open a tracked video file (.fcts, .xlsx, etc)'
            )

    def set_view_menu(self):
        """Sets up menu to allow users to select which DockWidgets/ToolBars
        are displayed."""
        #file explorer
        display_file_explorer_action = self.file_dock_widget.toggleViewAction()
        self.view_menu.addAction(display_file_explorer_action)

        #statistic widget
        display_statistic_widget_action = self.statistic_toolbar.toggleViewAction()
        self.view_menu.addAction(display_statistic_widget_action)

        #video toolbar
        display_video_toolbar_action = self.video_toolbar.toggleViewAction()
        self.view_menu.addAction(display_video_toolbar_action)

        #slider toolbar
        display_slider_toolbar_action = self.video_trackbar.toggleViewAction()
        self.view_menu.addAction(display_slider_toolbar_action)

    def set_video_menu(self):
        self.previous_action = self.add_menu_action(
            menu=self.video_menu,
            name='Previous Frame',
            connection=self.video_player.previous,
            status_tip='Go to previous frame.',
            icon=QIcon(os.path.join(DIR, 'icons', 'prev_icon.png')),
            tool_bar=self.video_toolbar,
            return_action=True
            )
        self.previous_action.setEnabled(False)
        self.previous_action.setShortcut(QKeySequence(','))

        self.play_action = self.add_menu_action(
            menu=self.video_menu,
            name='Play',
            connection=self.video_player.on,
            status_tip='Play Video.',
            icon=QIcon(os.path.join(DIR, 'icons', 'play_icon.png')),
            tool_bar=self.video_toolbar,
            return_action=True
            )
        self.play_action.setEnabled(False)
        self.play_action.setShortcut(QKeySequence(']'))

        self.stop_action = self.add_menu_action(
            menu=self.video_menu,
            name='Stop',
            connection=self.video_player.off,
            status_tip='Stop Video.',
            icon=QIcon(os.path.join(DIR, 'icons', 'stop_icon.png')),
            tool_bar=self.video_toolbar,
            return_action=True
            )
        self.stop_action.setEnabled(False)
        self.stop_action.setShortcut(QKeySequence('['))

        self.next_action = self.add_menu_action(
            menu=self.video_menu,
            name='Next Frame',
            connection=self.video_player.next,
            status_tip='Go to next frame.',
            icon=QIcon(os.path.join(DIR, 'icons', 'next_icon.png')),
            tool_bar=self.video_toolbar,
            return_action=True
            )
        self.next_action.setEnabled(False)
        self.next_action.setShortcut(QKeySequence('.'))

        self.stop_start_action = self.add_menu_action(
            menu=self.video_menu,
            name='Pause/Play',
            connection=self.stop_start,
            status_tip='Start playing, pause, or resume playing video.',
            return_action=True
            )
        self.stop_start_action.setEnabled(False)
        self.stop_start_action.setShortcut(QKeySequence(Qt.Key_Space))

        self.video_zoom_action = self.add_menu_action(
            menu=self.video_menu,
            name='Zoom',
            connection=self.zoom,
            status_tip='Zoom in on a segment of video.',
            return_action=True
        )
        self.video_zoom_enabled = False
        self.video_zoom_action.setEnabled(False)
        self.video_zoom_action.setShortcut(QKeySequence('Z'))

        self.video_menu.addSeparator()

        # Actions to control the speed of the video.
        self.videoMenuSpeedMenu = self.video_menu.addMenu('Playback Speed')
        speedFunctions = [
            self.adjustSpeed1x,
            self.adjustSpeed1p5x,
            self.adjustSpeed2x,
            self.adjustSpeed3x,
            self.adjustSpeed5x,
            self.adjustSpeed10x
        ]
        speedFunctionNames = [
            '1x', '1.5x', '2x', '3x', '5x', '10x'
        ]
        speedActionGroup = QActionGroup(self)
        for i in xrange(len(speedFunctionNames)):
            action = self.add_menu_action(
                menu=self.videoMenuSpeedMenu,
                name=speedFunctionNames[i],
                connection=speedFunctions[i],
                status_tip='Adjust the playback speed of the current video.',
                return_action=True
            )
            action.setCheckable(True)
            speedActionGroup.addAction(action)
            if i == 0:
                action.setChecked(True)

    def set_tracking_menu(self):
        self.add_menu_action(
            menu=self.tracking_menu,
            name='Batch Processing',
            connection=self.track_batch,
            status_tip='Open Batch Tracking Wizard.'
            )

    def set_analysis_menu(self):
        self.add_menu_action(
            menu=self.analysis_menu,
            name='Signal Processing',
            connection=self.signal_processing,
            status_tip='Open Signal Processing Dialog.'
            )

    def open_file_directory(self):
        file_dialog = QFileDialog()
        directory_name = str(file_dialog.getExistingDirectory(
            caption='Open Folder',
            options=QFileDialog.ShowDirsOnly))

        if not os.path.isdir(directory_name):
            self.logger.warning('Passed directory was not valid (in ' +
                'MainWindow.open_file_directory(), line ~282 of entry.py).')
            return

        self.file_explorer.set_path(directory_name)
        self.root_folder = directory_name

    def open_video(self, video_fname=None):
        """Opens an individual video for display in the central widget."""
        if not isinstance(video_fname, str):
            file_dialog = QFileDialog(self)
            video_fname = str(file_dialog.getOpenFileName(
                caption='Open Video File',
                filter='Video Files (*.fmf)',
                directory=self.root_folder
                )[0])
        
        if not os.path.isfile(video_fname):
            self.logger.warning('Passed `video_fname` was not valid (in ' +
                'MainWindow.open_video(), line 290 of entry.py).')
            return

        self.video_player.set_video(video_fname)
        self.video_player.image_annotation = False
        self.video_player.tracking_summary = None
        self.enable_video_controls()

    def enable_video_controls(self):
        """Enables all video controls and connects signals/slots if necessary."""
        #connect video_player signals
        self.video_player.frame_changed.connect(self.update_frame_label)

        #enable video controls
        self.previous_action.setEnabled(True)
        self.next_action.setEnabled(True)
        self.play_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.stop_start_action.setEnabled(True)
        self.video_zoom_action.setEnabled(True)

        #enable slider
        self.slider.setMaximum(self.video_player.video.get_n_frames() - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        self.frame_label.setText('Frame: 0')
        self.video_label.setText('Current Video: {}'.format(
            self.video_player.video_file_name))

    def open_tracking_summary(self, fcts_fname=None, video_fname=None):
        """Opens a tracking summary file (.fcts, .xlsx, etc) for display."""
        if not isinstance(fcts_fname, str):
            file_dialog = QFileDialog(self)
            fcts_fname=str(file_dialog.getOpenFileName(
                caption='Open Tracking Summary File',
                filter='Tracking Summary File (*.fcts *.xlsx)',
                directory=self.root_folder
                )[0])

        if not os.path.isfile(fcts_fname):
            self.logger.warning('Passed `fcts_fname` was not valid (in ' +
                'MainWindow.open_tracking_summary(), line 332 of entry.py).')
            return

        file_type = fcts_fname.split('.')[-1]
        if file_type == 'fcts':
            with open(fcts_fname, 'rb') as f:
                tracking_summary = pickle.load(f)
        elif file_type == 'xlsx':
            tracking_summary = FCTS().from_xlsx(fcts_fname) # TODO: Add from_xlsx() function
        else:
            return

        if os.path.exists(tracking_summary.video.filename):
            self.video_player.set_video(tracking_summary.video.filename)
        else:
            if not isinstance(video_fname, str):
                file_dialog = QFileDialog(self)
                video_fname = str(file_dialog.getOpenFileName(
                    caption='Select Associated Video File',
                    filter='Video File (*.fmf)',
                    directory=self.root_folder
                    )[0])
            self.video_player.set_video(video_fname)

        self.video_player.tracking_summary = tracking_summary
        self.video_player.image_annotation = True
        self.video_player.update_label()
        self.enable_video_controls()

        # setup statistic toolbar
        self.statistic_toolbar.setEnabled(True)
        # calculate statistics about fly held in current experiment.
        lwa, rwa = wing_stats.individual_wing_angles(tracking_summary.male)
        fwa = wing_stats.full_wing_angle(tracking_summary.male)
        wd = wing_stats.wing_distances(tracking_summary.male)
        nn = spatial_stats.nearest_neighbor_centroid(
            tracking_summary.male, tracking_summary.female)
        n2e, t2e = spatial_stats.nose_and_tail_to_ellipse(
            tracking_summary.male, tracking_summary.female)
        vel = centroid_stats.centroid_velocity(tracking_summary.male)
        self.statistics_widget.statistics = {
            'LeftWingAngle': lwa,
            'RightWingAngle': rwa,
            'FullWingAngle': fwa,
            'WingDistance': wd,
            'Centroid-to-Centroid': nn,
            'Nose-to-Ellipse': n2e,
            'Rear-to-Ellipse': t2e,
            'CentroidVelocity': vel
        }
        self.statistics_widget.update_combobox()
        self.statistics_widget.update_stat_label(0)
        self.statistic_toolbar.show()

        self.classification_toolbar.setEnabled(True)
        self.classification_widget.set_tracking_summary(tracking_summary)
        self.classification_widget.statistics = \
            tracking_summary.male.get_all_behaviors_as_dict()
        self.classification_widget.update_combobox()
        self.classification_widget.update_stat_label(0)
        self.classification_toolbar.show()

    def set_slider(self):
        """Places slider widget in video track bar menu."""
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.video_trackbar.addWidget(self.slider)
        self.slider.valueChanged[int].connect(self.video_player.slider_update_label)
        self.video_player.frame_changed.connect(self.update_slider)
        self.slider.setEnabled(False)

    def set_status(self):
        """Annotates video statistics at bottom of main window."""
        status = self.statusBar()
        status.setSizeGripEnabled(True)

        self.frame_label = QLabel()
        self.frame_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.frame_label.setText('Frame: NA (00:00:00)')

        self.video_label = QLabel()
        self.video_label.setText('Current Video: NA')

        status.addPermanentWidget(self.frame_label)
        status.addWidget(self.video_label)

    def set_file_explorer(self):
        """Initializes the file explorer as a DockWidget."""
        self.file_explorer = FileExplorer()
        self.file_dock_widget = QDockWidget('File Explorer', self)
        self.file_dock_widget.setWidget(self.file_explorer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock_widget)
        self.file_explorer.open_video.connect(self.load_video_from_file_exp)

    def set_statistic_toolbar(self):
        self.statistic_toolbar = QDockWidget('Tracking Statistics')
        self.statistics_widget = StatisticWindowWidget()
        self.statistic_toolbar.setWidget(self.statistics_widget)
        self.statistic_toolbar.setEnabled(False)
        self.statistic_toolbar.hide()
        self.video_player.frame_changed.connect(self.statistics_widget.update_stat_label)
        self.addDockWidget(Qt.TopDockWidgetArea, self.statistic_toolbar)

    def set_classification_toolbar(self):
        self.classification_toolbar = QDockWidget('Behavioral Classifications')
        self.classification_widget = ClassificationWindowWidget()
        self.classification_toolbar.setWidget(self.classification_widget)
        self.classification_toolbar.setEnabled(False)
        self.classification_toolbar.hide()
        self.video_player.frame_changed.connect(self.classification_widget.update_stat_label)
        self.addDockWidget(Qt.TopDockWidgetArea, self.classification_toolbar)
        self.tabifyDockWidget(self.statistic_toolbar, self.classification_toolbar)

    def track_batch(self):
        dialog = BatchTrackingDialog(self.root_folder)
        if dialog.exec_():
            return

    def signal_processing(self):
        dialog = StatProcessing(root_folder = self.root_folder, parent = self)
        dialog.show()

    @pyqtSlot(int, str, int)
    def update_frame_label(self, ix, hms, frame_rate):
        self.frame_label.setText(
            'Frame: {} ({}) | FPS: {} | Playback Speed: {}x'.format(
                ix, hms, frame_rate, self.video_player.playback_speed
                )
            )

    @pyqtSlot(str)
    def load_video_from_file_exp(self, fname):
        self.open_video(str(fname))

    @pyqtSlot(int)
    def update_slider(self, ix):
        self.slider.setValue(ix)

    @pyqtSlot()
    def stop_start(self):
        if self.video_player.is_playing:
            self.video_player.off()
        else:
            self.video_player.on()

    @pyqtSlot()
    def zoom(self):
        """Allows user to select a region of the video to zoom in on."""
        if self.video_zoom_enabled:
            self.video_zoom_enabled = False
            self.video_player.zoom_enabled = False
            non_zooming_cursor = QCursor(Qt.ArrowCursor)
            self.video_player.setCursor(non_zooming_cursor)
            self.video_player.close_zoom_window()
            return

        self.video_zoom_enabled = True
        zooming_cursor = QCursor(Qt.CrossCursor)
        self.video_player.setCursor(zooming_cursor)
        self.video_player.zoom_enabled = True


    # Playback speed adjustments
    @pyqtSlot()
    def adjustSpeed1x(self):
        self.video_player.playback_speed = 1

    @pyqtSlot()
    def adjustSpeed1p5x(self):
        self.video_player.playback_speed = 1.5

    @pyqtSlot()
    def adjustSpeed2x(self):
        self.video_player.playback_speed = 2

    @pyqtSlot()
    def adjustSpeed3x(self):
        self.video_player.playback_speed = 3

    @pyqtSlot()
    def adjustSpeed5x(self):
        self.video_player.playback_speed = 5

    @pyqtSlot()
    def adjustSpeed10x(self):
        self.video_player.playback_speed = 10



def main():
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet(pyside = False))
    app_icon = QIcon(QPixmap(os.path.join(DIR, 'icons', 'logo.png')))

    main_window = MainWindow()
    main_window.setWindowIcon(app_icon)
    main_window.setWindowTitle('courtship')

    main_window.show()
    app.exec_()

if __name__ == '__main__':
    main()
