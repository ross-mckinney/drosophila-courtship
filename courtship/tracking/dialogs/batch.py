"""
tracking.py

Main tracking dialog window.
"""
import os
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ..widgets.batch import *

class BatchTrackingDialog(QDialog):
    """Dialog for batch processing videos."""
    def __init__(self, root_folder, parent = None):
        super(TrackingDialog, self).__init__(parent)
        self.setWindowTitle('Batch Processing Wizard')
        self.step_number = 0
        self.video_settings = []
        self.root_folder = root_folder
        
        self.layout = QGridLayout()
        # move all steps into a stacked layout.
        # this allows each layout to be shown individually,
        # while providing a container for all layouts.
        self.widgets = QStackedWidget()
        file_widget = StackedStepWidget(
            0, BatchFileSelector(self.root_folder, 0))
        arena_widget = StackedStepWidget(
            1, BatchArenaSpecifier(1))
        try:
            arena_widget.settings_widget.background_frame_ix.connect(
                self.update_progress_bar
                )
        except RuntimeError:
            print ('Error while trying to connect the arena widget' +
                   'to the update_progress_bar fxn.')

        female_widget = StackedStepWidget(
            2, BatchFemaleSpecifier(2))
        tight_threshold_widget = StackedStepWidget(
            3, BatchTightThresholdSpecifier(3))
        tight_threshold_widget.settings_widget.image_calc_progress.connect(
                self.update_progress_bar
            )

        loose_threshold_widget = StackedStepWidget(
            4, BatchLooseThresholdSpecifier(4))
        loose_threshold_widget.settings_widget.image_calc_progress.connect(
                self.update_progress_bar
            )

        tracking_widget = StackedStepWidget(
            5, BatchTrackingWidget(5))
        tracking_widget.settings_widget.tracking_progress.connect(
                self.update_progress_bar
            )

        self.widgets.addWidget(file_widget)
        self.widgets.addWidget(arena_widget)
        self.widgets.addWidget(female_widget)
        self.widgets.addWidget(tight_threshold_widget)
        self.widgets.addWidget(loose_threshold_widget)
        self.widgets.addWidget(tracking_widget)
        
        # connect all widget signals to this class's update_settings
        # function.
        self.connect_widget_signals()
        
        self.layout.addWidget(self.widgets, 0, 0, 6, 6)

        # setup the next and previous buttons
        self.next_button = QPushButton('Next')
        self.next_button.setEnabled(False)
        previous_button = QPushButton('Previous')
        self.next_button.clicked.connect(self.step_forward)
        previous_button.clicked.connect(self.step_backward)

        self.layout.addWidget(previous_button, 6, 0, 1, 1)
        self.layout.addWidget(self.next_button, 6, 5, 1, 1)

        # setup the status bar interface
        self.set_status()
        self.layout.addWidget(self.status, 7, 0, 1, 6)
        self.setLayout(self.layout)
        self.resize(1000, 600)

    def connect_widget_signals(self):
        """Connects all widgets in the StackedLayout to self.update_settings."""
        for i in xrange(self.widgets.count() + 1):
            # this call returns 0 if widget at index is not present.
            if self.widgets.widget(i):
                self.widgets.widget(i).settings_widget.all_settings_valid.connect(
                    self.update_settings)

    def set_status(self):
        """Sets interface for status bar at bottom of dialog window."""
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(True)

        self.process_label = QLabel('Process')
        self.process_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.progress_bar = QProgressBar()

        self.status.addWidget(self.process_label)
        self.status.addPermanentWidget(self.progress_bar)

    def _step(self):
        """Upates layout currently being displayed in StackedLayout."""
        self.widgets.setCurrentIndex(self.step_number)
        if self.widgets.currentWidget().settings_widget.check_settings_valid():
            self.next_button.setEnabled(True)

    def step_forward(self):
        """Moves the StackedLayout forward by one index."""
        if self.step_number < self.widgets.count() - 1:
            self.step_number += 1
        self._step()

    def step_backward(self):
        """Moves the StackedLayout backward by one index."""
        if self.step_number > 0:
            self.step_number -= 1
        self._step()

    @pyqtSlot(bool, int, list)
    def update_settings(self, is_set, widget_ix, settings_list):
        """This function updates all of the video settings.

        It is a slot for the BatchSettingsWidget.all_settings_valid
        signal, and not only sets the video_settings of this 
        class (TrackingDialog), but updates all of the settings in
        each of the widgets in the StackedLayout.
        """
        if is_set:
            self.next_button.setEnabled(True)
            self.video_settings = settings_list
            for i in xrange(self.widgets.count() + 1):
                if self.widgets.widget(i):
                    self.widgets.widget(i).settings_widget.update_settings(
                        settings_list)
        else:
            self.next_button.setEnabled(False)

    @pyqtSlot(int, str)
    def update_progress_bar(self, progress, description):
        """Updates the progress bar based on actions happening
        in widgets within the StackedLayout."""
        self.process_label.setText('Process -- {}'.format(
            description))
        self.progress_bar.setValue(progress)

