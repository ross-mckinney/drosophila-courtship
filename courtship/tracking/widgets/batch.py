"""
batch.py

Widgets for batch processing videos.
"""
import os
import pickle
import time
import json

from datetime import datetime

import cv2
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from skimage import transform

from objects.settings import VideoSettings
from objects.arena import CircularArena
from objects.female import Female
from text import *
from tracking.thresholding import *
from tracking.centroid import *
from tracking.image_transforms import *
from tracking.image_annotations import *
from tracking.wing import *

from canal.objects.fly import Fly
from canal.objects.experiment import FixedCourtshipTrackingSummary

import utils

class BatchSettingsWidget(QWidget):
    """Base class for all widgets.

    This serves one main purpose: to be able to pass 
    along settings that have been specified for each
    video.

    Parameters
    ----------
    widget_ix : int
        Index of widget in StackedLayout.

    Attributes
    ----------
    video_settings : list of VideoSettings
        Video settings for all of the specified videos.

    Signals
    -------
    all_settings_valid : bool, list of VideoSettings
        The emitted signal should be implemented in derived 
        classes, since each step has different requirements.

    Slots
    -----
    update_settings : list of VideoSettings
        Updates the self.video_settings variable to account
        for changes across all BatchSettingWidgets.
    """
    all_settings_valid = pyqtSignal(bool, int, list)

    def __init__(self, widget_ix, parent = None):
        super(BatchSettingsWidget, self).__init__(parent)
        self.video_settings = []
        self.widget_ix = widget_ix

    def update_view(self):
        """Updates the layout in this widget based on user actions
        in other widgets.

        .. note:: This should be implemented in each widget that 
        inherits this class.
        """
        raise NotImplementedError(
            "update_view() needs to be implemented in derived class")

    @pyqtSlot(list)
    def update_settings(self, settings_list):
        self.video_settings = settings_list
        self.update_view()

    def check_settings_valid(self):
        """Implement in derived classes."""
        raise NotImplementedError(
            "check_settings_valid() needs to be implemented in derived class")


class StackedStepWidget(QWidget):
    """Wrapper to generate widgets for batch processing.

    This widget just combines the BatchStepWidget with
    whichever BatchSettingsWidget and sets up the layout
    so the above two widgets are arranged as two columns
    of appropriate sizes.
    """
    def __init__(self, step_number, widget, parent=None):
        super(StackedStepWidget, self).__init__(parent)

        self.layout = QGridLayout()
        stepList = BatchStepList(step_number)
        self.settings_widget = widget

        self.layout.addWidget(stepList, 0, 0, 1, 1)
        self.layout.addWidget(self.settings_widget, 0, 1, 1, 4)
        self.setLayout(self.layout)


class ImageArrayWidget(QWidget):
    """Generates a nice layout and convenience functions for displaying a series
    of images.

    The following are attributes which should be connected to/set by classes 
    that utilize this widget.

    Attributes
    ----------
    image_labels : list of QLabel
        Labels that should contain images.

    instruction_text_edit : QTextEdit
        TextEdit containing user instructions.

    randomize_button : QPushButton
        Connect randomize_button.clicked with a custom randomize function
        to update images.

    threshold_spinbox : QSpinBox
        SpinBox containing user threshold. Connect threshold_spinbox.valueChanged
        to custom function to set specific threshold &/or update images.
    """

    def __init__(self, parent = None):
        super(ImageArrayWidget, self).__init__(parent)
        self.layout = QGridLayout()
        self.image_width = 180
        self.image_height = 180
        
        self.set_image_array_gb()
        self.set_instruction_text_edit()
        self.set_threshold_gb()

        self.setLayout(self.layout)

    def set_image_array_gb(self):
        """Sets up the groupbox containing the image array."""
        gb = QGroupBox('Frame Subset')
        gb_layout = QGridLayout()

        #set up a 3 x 3 grid of image labels, and set their pixmaps to 0s.
        self.image_labels = [QLabel() for i in xrange(9)]
        for i in xrange(9):
            gb_layout.addWidget(self.image_labels[i], i / 3, i % 3)
            self.update_image_label(i, np.zeros(shape = (480 / 3, 640 / 3)))

        self.randomize_button = QPushButton('Randomize')
        gb_layout.addWidget(self.randomize_button, 4, 2, 1, 1)

        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 0, 0, 2, 2)

    def set_instruction_text_edit(self):
        """Sets up the instruction text edit."""
        gb = QGroupBox('Instructions')
        gb_layout = QHBoxLayout()

        self.instruction_text_edit = QTextEdit(
                'Set this text by calling ' +
                'ImageArrayWidget.instruction_text_edit.setText()'
            )
        self.instruction_text_edit.setReadOnly(True)

        gb_layout.addWidget(self.instruction_text_edit)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 2, 0, 1, 1)

    def set_threshold_gb(self):
        """Sets up the threshold spinbox and randomize frame labels button."""
        gb = QGroupBox('Threshold Settings')
        gb_layout = QHBoxLayout()

        spinbox_label = QLabel('Threshold')
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(255)

        gb_layout.addWidget(spinbox_label)
        gb_layout.addWidget(self.threshold_spinbox)
        gb.setLayout(gb_layout)

        self.layout.addWidget(gb, 2, 1, 1, 1)

    def update_image_label(self, label_ix, img):
        """Updates a single label within self.image_labels with a specified
        image.

        Parameters
        ----------
        label_ix : int
            Index of label (within self.image_labels) to update.

        img : np.ndarray of type np.uint8
            Image to place in specified label.
        """
        self.image_labels[label_ix].setPixmap(
                QPixmap.fromImage(
                        utils.get_q_image(img)
                    ).scaled(
                        QSize(self.image_width, self.image_height),
                        Qt.KeepAspectRatio)
            )


class BatchFileSelector(BatchSettingsWidget):
    """Allows specification of video files and save files for 
    batch processing.
    """
    def __init__(self, root_folder, parent = None):
        super(BatchFileSelector, self).__init__(parent)
        self.video_directory = None
        self.save_directory = None
        self.save_file_type = 'fcts'
        self.video_file_names = []
        self.save_file_names = []
        self.root_folder = root_folder

        self.set_ui()

    def set_ui(self):
        """Sets main user interface for widget."""
        self.set_select_files_groupbox()
        self.set_display_files_groupbox()
        self.set_statistics_groupbox()

        layout = QGridLayout()
        layout.addWidget(self.select_file_groupbox, 0, 0, 1, 2)
        layout.addWidget(self.statistics_groupbox, 0, 2, 1, 1)
        layout.addWidget(self.display_file_groupbox, 1, 0, 1, 3)
        self.setLayout(layout)

    def set_select_files_groupbox(self):
        """Groupbox to allow users to select a batch of videos to track."""
        self.select_file_groupbox = QGroupBox()
        select_file_groupbox_layout = QGridLayout()
        
        row_labels = ['Video Directory', 'Save Directory', 'Save File Type']
        for i, l in enumerate(row_labels):
            select_file_groupbox_layout.addWidget(QLabel(l), i, 0)

        self.video_dir_lineedit = QLineEdit('')
        self.save_dir_lineedit = QLineEdit('')
        self.video_dir_lineedit.setReadOnly(True)
        self.save_dir_lineedit.setReadOnly(True)

        self.save_type_combobox = QComboBox()
        self.save_type_combobox.addItems(['.fcts','.xlsx'])
        self.save_type_combobox.currentIndexChanged.connect(
            self.update_save_type)

        video_browse_button = QPushButton('Browse')
        save_browse_button = QPushButton('Browse')
        
        video_browse_button.clicked.connect(self.browse_video)
        save_browse_button.clicked.connect(self.browse_save)

        dialog_button_box = QDialogButtonBox(QDialogButtonBox.Reset)
        dialog_button_box.setOrientation(Qt.Horizontal)
        dialog_button_box.clicked.connect(self.reset)

        select_file_groupbox_layout.addWidget(
            self.video_dir_lineedit, 0, 1, 1, 2)
        select_file_groupbox_layout.addWidget(
            self.save_dir_lineedit, 1, 1, 1, 2)
        select_file_groupbox_layout.addWidget(
            self.save_type_combobox, 2, 1, 1, 1)
        select_file_groupbox_layout.addWidget(video_browse_button, 0, 3)
        select_file_groupbox_layout.addWidget(save_browse_button, 1, 3)
        select_file_groupbox_layout.addWidget(dialog_button_box, 3, 0, 1, 4)

        self.select_file_groupbox.setLayout(select_file_groupbox_layout)

    def set_display_files_groupbox(self):
        """Groupbox containing ListWidgets for displaying video/save files."""
        self.display_file_groupbox = QGroupBox()
        display_file_groupbox_layout = QGridLayout()

        self.video_files_list = QListWidget()
        self.save_files_list = QListWidget()

        # Make sure that if a user clicks on an item in either list, 
        # its associated file is also highlighted in the other list.
        self.video_files_list.currentRowChanged.connect(
            self.connect_video_to_save_list)
        self.save_files_list.currentRowChanged.connect(
            self.connect_save_to_video_list)

        col_labels = ['Video Files to Track:', 'Save Files:']
        for i, f in enumerate(col_labels):
            display_file_groupbox_layout.addWidget(QLabel(f), 0, i)

        display_file_groupbox_layout.addWidget(self.video_files_list, 1, 0)
        display_file_groupbox_layout.addWidget(self.save_files_list, 1, 1)

        self.display_file_groupbox.setLayout(display_file_groupbox_layout)

    def set_statistics_groupbox(self):
        """Groupbox containing summary statistics about selected files."""
        self.statistics_groupbox = QGroupBox()
        statistics_groupbox_layout = QGridLayout()

        statistic_labels = [
            'Number of Video Files: ', 
            'Number of Save Files: ', 
            'Video Directory: ',
            'Save Directory: ',
            'Save File Type: '
            ]
        for i, l in enumerate(statistic_labels):
            statistics_groupbox_layout.addWidget(QLabel(l), i, 0)

        self.stats_n_video_label = QLabel(str(len(self.video_file_names)))
        self.stats_n_save_label = QLabel(str(len(self.save_file_names)))
        self.video_dir_label = QLabel(str(self.video_directory))
        self.save_dir_label = QLabel(str(self.save_directory))
        self.save_type_label = QLabel(str(self.save_file_type))

        stat_values = [self.stats_n_video_label, self.stats_n_save_label,
            self.video_dir_label, self.save_dir_label, self.save_type_label]
        for i, v in enumerate(stat_values):
            statistics_groupbox_layout.addWidget(v, i, 1)

        self.statistics_groupbox.setLayout(statistics_groupbox_layout)

    def browse_video(self):
        """Opens a FileDialog to select videos to track."""
        file_dialog = QFileDialog()
        self.video_directory = str(file_dialog.getExistingDirectory(
            caption = 'Select Folder Containing Video Files',
            directory = self.root_folder,
            options = QFileDialog.ShowDirsOnly))

        if self.video_directory == '':
            self.video_directory = None
        else:
            self.video_file_names = sorted(
                [os.path.join(self.video_directory, f)
                    for f in os.listdir(self.video_directory)
                    if f.split('.')[-1] == 'fmf']
                )
            utils.clear_list(self.video_files_list)
            if len(self.video_file_names) == 0:
                self.video_directory = None
            else:
                self.video_files_list.addItems(self.video_file_names)
                self.video_dir_lineedit.setText(self.video_directory)

        self.update_statistics()
        self.update_video_settings()
        self.check_settings_valid()

    def browse_save(self):
        """Opens a FileDialog to select save folder."""
        file_dialog = QFileDialog()
        self.save_directory = str(file_dialog.getExistingDirectory(
            caption = 'Select Save Folder',
            directory = self.root_folder,
            options = QFileDialog.ShowDirsOnly))

        if self.save_directory == '':
            self.save_directory = None
        else:
            if self.video_directory is not None:
                self.save_file_names = [
                    os.path.join(
                            self.save_directory, 
                            '.'.join(
                                [os.path.basename(f).split('.')[0], 
                                self.save_file_type]
                            )
                        )
                        for f in self.video_file_names
                    ]
                utils.clear_list(self.save_files_list)
                self.save_files_list.addItems(self.save_file_names)

            self.save_dir_lineedit.setText(self.save_directory)

        self.update_statistics()
        self.update_video_settings()
        self.check_settings_valid()

    def update_statistics(self):
        """Updates label attributes within statistics groupbox."""
        self.stats_n_video_label.setText(str(len(self.video_file_names)))
        self.stats_n_save_label.setText(str(len(self.save_file_names)))
        if self.video_directory is not None:
            self.video_dir_label.setText(
                str('../' + os.path.split(self.video_directory)[1]))
        
        if self.save_directory is not None:
            self.save_dir_label.setText(
                str('../' + os.path.split(self.save_directory)[1]))
        self.save_type_label.setText(str(self.save_file_type))
        

    def update_video_settings(self):
        if self.video_directory is not None \
            and self.save_directory is not None:
            if len(self.video_settings) == 0:
                for i, video_fname in enumerate(self.video_file_names):
                    settings = VideoSettings()
                    settings.video_file = video_fname
                    settings.save_file = self.save_file_names[i]
                    self.video_settings.append(settings)
            else:
                for i, settings in enumerate(self.video_settings):
                    self.video_settings[i].video_file = self.video_file_names[i]
                    self.video_settings[i].save_file = self.save_file_names[i]

    def check_settings_valid(self):
        """Checks to see whether or not video settings have been set.

        If settings have been set, a signal is emitted to the main 
        dialog window containing this widget.
        """
        if len(self.video_settings) != 0:
            self.all_settings_valid.emit(
                True, self.widget_ix, self.video_settings)
            return True
        else:
            self.all_settings_valid.emit(
                False, self.widget_ix, self.video_settings)
            return False

    @pyqtSlot(int)
    def update_save_type(self, ix):
        """Slot connecting save_type_combobox to currentIndexChanged.
    
        When the user changes the save file type, the signal is sent here
        and all of the save files are appropriately updated.
        """
        self.save_file_type = str(self.save_type_combobox.currentText())[1:]

        if self.save_file_names is not None:
            new_names = [
                os.path.join(
                    self.save_directory, 
                    '.'.join(
                            [os.path.basename(f).split('.')[0], 
                            self.save_file_type]
                        )
                    )
                    for f in self.save_file_names
                ]

            utils.clear_list(self.save_files_list)
            self.save_file_names = new_names
            self.save_files_list.addItems(self.save_file_names)
         
        self.update_statistics()
        self.update_video_settings()
        self.check_settings_valid()
            
    @pyqtSlot(int)
    def connect_video_to_save_list(self, ix):
        """Interconnects video_files_list and save_files_list.

        Based on a users click within the video_files_list, the
        same item will become highlighted within the save_files_list.
        """
        if self.save_files_list.count() > 0:
            self.save_files_list.setCurrentRow(ix)

    @pyqtSlot(int)
    def connect_save_to_video_list(self, ix):
        """Interconnects save_files_list and video_files_list.

        Based on a users click within the save_files_list, the
        same item will become highlighted within the video_files_list.
        """
        if self.video_files_list.count() > 0:
            self.video_files_list.setCurrentRow(ix)

    def reset(self):
        """Clears all files and resets attributes to default values.

        This is a slot for dialog_button_box.rejected.
        """
        self.video_directory = None
        self.save_directory = None
        self.save_file_type = 'fcts'
        self.video_file_names = []
        self.save_file_names = []
        self.video_settings = []

        self.video_dir_lineedit.setText('')
        self.save_dir_lineedit.setText('')
        utils.clear_list(self.video_files_list)
        utils.clear_list(self.save_files_list)

        self.update_statistics()
        self.update_video_settings()
        self.check_settings_valid()

    def update_view(self):
        pass

class BatchArenaSpecifier(BatchSettingsWidget):
    """Allows the user to specify circular arenas for multiple videos."""
    background_frame_ix = pyqtSignal(int, str)

    def __init__(self, parent = None):
        super(BatchArenaSpecifier, self).__init__(parent)
        self.current_video_ix = 0
        self.center = None
        self.radius = None
        self.arena_size_mm = 22
        self.pixels_to_mm = self.pix_to_mm_conv(self.arena_size_mm)

        self.layout = QGridLayout()

        self.set_image_label()
        self.set_video_list_widget()
        self.set_instruction_text_edit()
        self.set_arena_gb()

        self.setLayout(self.layout)

    def set_image_label(self):
        """Generates a groupbox containing the current
        video's name, and the image from that video."""
        gb = QGroupBox('Current Video')
        gb_layout = QGridLayout()

        self.image_label = QLabel()

        # Init the image label to a blank 480 x 640 QImage
        image = np.zeros(shape = (480, 640), dtype = np.uint8)
        self.image_label.setPixmap(
            QPixmap.fromImage(utils.get_q_image(image))
            )
        self.image_label.mousePressEvent = self.image_click
        self.image_label.mouseMoveEvent = self.image_move
        self.image_label.mouseReleaseEvent = self.image_release

        self.current_video_label = QLabel('')

        gb_layout.addWidget(self.current_video_label, 0, 0, 1, 1)
        gb_layout.addWidget(self.image_label, 1, 0, 2, 2)
        gb.setLayout(gb_layout)

        self.layout.addWidget(gb, 0, 2, 2, 2)

    def set_video_list_widget(self):
        """Generates a groupbox containing a single label
        and a ListWidget to hold video file names."""
        gb = QGroupBox('Videos')
        gb_layout = QVBoxLayout()

        self.video_list_widget = QListWidget()
        self.video_list_widget.currentRowChanged.connect(self.update)

        gb_layout.addWidget(self.video_list_widget)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 0, 0, 3, 2)

    def set_instruction_text_edit(self):
        """Generates a groupbox containing a TextEdit and
        an Accept/Reject ButtonBox to display user instructions."""
        gb = QGroupBox('Instructions')
        gb_layout = QHBoxLayout()

        self.instruction_text_edit = QTextEdit(
            'Click and drag from one side of the arena to the other.'
            )
        self.instruction_text_edit.setReadOnly(True)

        gb_layout.addWidget(self.instruction_text_edit)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 2, 2, 1, 1)

    def set_arena_gb(self):
        """Sets up a groupbox containing a SpinBox to specify the size 
        (diameter) of the arena."""
        gb = QGroupBox('Arena Properties')
        gb_layout = QGridLayout()

        self.arena_size_spinbox = QSpinBox()
        self.arena_size_spinbox.setMinimum(1)
        self.arena_size_spinbox.setValue(self.arena_size_mm)
        self.arena_size_spinbox.valueChanged.connect(self.update_arena_mm)

        spinbox_label = QLabel('Diameter of Arena (mm)')

        #setup a vertical button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Reset
            )
        button_box.clicked.connect(self.reject_arena)
        button_box.setOrientation(Qt.Horizontal)

        gb_layout.addWidget(spinbox_label, 0, 0)
        gb_layout.addWidget(self.arena_size_spinbox, 0, 1)
        gb_layout.addWidget(button_box, 1, 0, 1, 2)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 2, 3, 1, 1)

    def update_view(self):
        """Updates each wiget with new video_settings passed by user
        from one of the other widgets within the StackedLayout."""

        #make sure that we are not appending to the list every time a 
        #user navigates away from this widget.
        utils.clear_list(self.video_list_widget)

        for settings in self.video_settings:
            base_name = os.path.basename(settings.video_file)
            self.video_list_widget.addItem(base_name)

        self.video_list_widget.setMinimumWidth(
            self.video_list_widget.sizeHintForColumn(0)
            )

        self.video_list_widget.setCurrentRow(self.current_video_ix)

    def draw_arena(self):
        """Draws arena based on user clicks."""
        self.get_circle_params()
        self.accept_arena()
        self.image_label.setPixmap(
                QPixmap.fromImage(
                    utils.get_q_image(
                        self.video_settings[
                            self.current_video_ix].arena.draw_arena()
                        )
                    )
                )

    def get_circle_params(self):
        """Finds the circle center and radius based on user clicks."""
        self.center = tuple((self.arena_start + self.arena_end) / 2)
        self.radius = int(np.sqrt(
                np.sum((self.arena_end - self.arena_start)**2)
            ) / 2)

    def image_click(self, event):
        """Records location of first user click on image label."""
        rr, cc = utils.get_mouse_coords(event)
        self.arena_start = np.array([rr, cc])

    def image_move(self, event):
        """Gets position of mouse move during click event on image label."""
        rr, cc = utils.get_mouse_coords(event)
        self.arena_end = np.array([rr, cc])
        self.draw_arena()

    def image_release(self, event):
        """Checks that all arenas have been set."""
        self.update_arena_mm(self.arena_size_mm)
        self.check_settings_valid()

    def pix_to_mm_conv(self, mm):
        """Calculates the number of pixels in the current diameter of the arena.

        Parameters
        ----------
        mm : int
            Diameter of arena in pixels.

        Returns
        -------
        pixels_to_mm : int or None
            Number of pixels per mm; or None if self.radius is not set.
        """
        try:
            return int((self.radius * 2.) / mm)
        except TypeError:
            return None

    @pyqtSlot(int)
    def update(self, index):
        """Updates each widget based on current selected video."""
        self.current_video_ix = index
        self.current_video_label.setText(
            'File path: {}'.format(self.video_settings[index].video_file)
            )

        # sets a video and an arena for each item in the video_settings
        # list if that item does not currently contain a video.
        if self.video_settings[index].video is None:
            video = FMF.FlyMovie(
                self.video_settings[index].video_file
            )
            self.video_settings[index].video = video
            self.video_settings[index].arena = CircularArena(video)

            # connects the background image calculation loop to 
            # calculate_background, and calculates a background image.
            self.video_settings[index].arena.background_frame_ix.connect(
                    self.calculate_background
                )
            self.video_settings[index].arena.calculate_background()

        # draws the arena image if the center and radius of the arena
        # have been set. otherwise, just draws the background.
        if self.video_settings[index].arena.center is not None \
        and self.video_settings[index].arena.radius is not None:
            self.image_label.setPixmap(
                QPixmap.fromImage(
                    utils.get_q_image(
                        self.video_settings[index].arena.draw_arena()
                        )
                    )
                )
        else:
            self.image_label.setPixmap(
                QPixmap.fromImage(
                    utils.get_q_image(
                        self.video_settings[index].arena.background_image
                        )
                    )
                )

    @pyqtSlot(int)
    def update_arena_mm(self, size):
        """Sets the size of the arena based on user input to arena_size_spinbox.

        Also sets the conversion factor pixels_to_mm.
        """
        self.arena_size_mm = size
        self.pixels_to_mm = self.pix_to_mm_conv(size)
        self.accept_arena()

    @pyqtSlot(int)
    def calculate_background(self, frame_ix):
        """Connects background image calculation to the progress bar
        within the main widget. 

        This acts as a slot for each arena's background_frame_ix attribute,
        and emits a signal that is detected by the main widget's
        update_progress_bar function.
        """
        try:
            description = 'Calculating background image for video: {}'.format(
                    str(self.video_list_widget.item(
                            self.video_list_widget.currentRow()
                        ).text())
                )
        except AttributeError:
            description = ''
            
        self.background_frame_ix.emit(frame_ix, description)

    def accept_arena(self):
        self.video_settings[self.current_video_ix].arena.center = self.center
        self.video_settings[self.current_video_ix].arena.radius = self.radius
        self.video_settings[self.current_video_ix].arena.arena_size = self.arena_size_mm
        self.video_settings[self.current_video_ix].arena.pixels_to_mm = self.pixels_to_mm
        
    def reject_arena(self):
        self.center = None
        self.radius = None
        self.pixels_to_mm = None
        
        self.accept_arena()
        
        self.image_label.setPixmap(
            QPixmap.fromImage(
                utils.get_q_image(
                    self.video_settings[
                        self.current_video_ix].arena.background_image
                    )
                )
            )
        self.check_settings_valid()

    def check_settings_valid(self):
        """Checks to see whether or not video settings have been set.

        If settings have been set, a signal is emitted to the main 
        dialog window containing this widget.
        """
        for video_setting in self.video_settings:
            if video_setting.arena is None:
                self.all_settings_valid.emit(
                    False, self.widget_ix, self.video_settings)
                return False
            if not video_setting.arena.settings_valid():
                self.all_settings_valid.emit(
                    False, self.widget_ix, self.video_settings)
                return False
        
        self.all_settings_valid.emit(True, self.widget_ix, self.video_settings)
        return True

class BatchFemaleSpecifier(BatchSettingsWidget):
    """Arena and Female Specifiers should inherit from a common
    widget to minimize code."""
    def __init__(self, parent = None):
        super(BatchFemaleSpecifier, self).__init__(parent)
        self.current_video_ix = 0
        self.center = None
        self.head = None
        self.rear = None
        self.maj_ax_rad = None
        self.min_ax_rad = None
        self.orientation = None
        self.ellipse_ratio = 2

        self.layout = QGridLayout()

        self.set_image_label()
        self.set_video_list_widget()
        self.set_instruction_text_edit()
        self.set_ellipse_gb()

        self.setLayout(self.layout)

    def set_image_label(self):
        """Generates a groupbox containing the current
        video's name, and the image from that video."""
        gb = QGroupBox('Current Video')
        gb_layout = QGridLayout()

        self.image_label = QLabel()

        # Init the image label to a blank 480 x 640 QImage
        image = np.zeros(shape = (480, 640), dtype = np.uint8)
        self.image_label.setPixmap(
            QPixmap.fromImage(utils.get_q_image(image))
            )
        self.image_label.mousePressEvent = self.image_click
        self.image_label.mouseMoveEvent = self.image_move
        self.image_label.mouseReleaseEvent = self.image_release

        self.current_video_label = QLabel('')

        gb_layout.addWidget(self.current_video_label, 0, 0, 1, 1)
        gb_layout.addWidget(self.image_label, 1, 0, 2, 2)
        gb.setLayout(gb_layout)

        self.layout.addWidget(gb, 0, 2, 2, 2)

    def set_video_list_widget(self):
        """Generates a groupbox containing a single label
        and a ListWidget to hold video file names."""
        gb = QGroupBox('Videos')
        gb_layout = QVBoxLayout()

        self.video_list_widget = QListWidget()
        self.video_list_widget.currentRowChanged.connect(self.update)

        gb_layout.addWidget(self.video_list_widget)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 0, 0, 3, 2)

    def set_instruction_text_edit(self):
        """Generates a groupbox containing a TextEdit and
        an Accept/Reject ButtonBox to display user instructions."""
        gb = QGroupBox('Instructions')
        gb_layout = QHBoxLayout()

        self.instruction_text_edit = QTextEdit(
            'Click and drag from the rear of the female to ' +
            'the head of the female.'
            )
        self.instruction_text_edit.setReadOnly(True)

        gb_layout.addWidget(self.instruction_text_edit)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 2, 2, 1, 1)

    def set_ellipse_gb(self):
        """Sets up a groupbox that allows a user to adjust the 
        ratio of the major-to-minor axis lengths."""
        gb = QGroupBox('Ellipse Settings')
        gb_layout = QGridLayout()

        self.ratio_spinbox = QSpinBox()
        # this would be a circle.
        self.ratio_spinbox.setMinimum(1)
        # major axis is 3-times as long as minor.
        self.ratio_spinbox.setValue(self.ellipse_ratio)
        self.ratio_spinbox.valueChanged.connect(self.update_ellipse_ratio)

        spinbox_label = QLabel('Major:Minor Axis Length')

        # setup a vertical button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.Reset
            )
        button_box.clicked.connect(self.reject_female)
        button_box.setOrientation(Qt.Horizontal)

        gb_layout.addWidget(spinbox_label, 0, 0)
        gb_layout.addWidget(self.ratio_spinbox, 0, 1)
        gb_layout.addWidget(button_box, 1, 0, 1, 2)
        gb.setLayout(gb_layout)
        self.layout.addWidget(gb, 2, 3, 1, 1)

    def update_view(self):
        """Updates each wiget with new video_settings passed by user
        from one of the other widgets within the StackedLayout."""

        # make sure that we are not appending to the list every time a 
        # user navigates away from this widget.
        utils.clear_list(self.video_list_widget)

        for settings in self.video_settings:
            base_name = os.path.basename(settings.video_file)
            self.video_list_widget.addItem(base_name)

        self.video_list_widget.setMinimumWidth(
            self.video_list_widget.sizeHintForColumn(0)
            )

        self.video_list_widget.setCurrentRow(self.current_video_ix)

    @pyqtSlot(int)
    def update(self, index):
        """Updates each widget based on current selected video."""
        self.current_video_ix = index
        self.current_video_label.setText(
            'File path: {}'.format(self.video_settings[index].video_file)
            )

        # adds a female object to video_settings entries that lack one
        if self.video_settings[index].female is None:
            arena = self.video_settings[index].arena
            self.video_settings[index].female = Female(arena)

        # draws an image of the background image with an ellipse surrounding
        # the female if all female settings have been set, otherwise just
        # display a background image.
        if self.video_settings[index].female.settings_valid():
            self.image_label.setPixmap(
                QPixmap.fromImage(
                    utils.get_q_image(
                        self.video_settings[index].female.draw_female()
                        )
                    )
                )
        else:
            self.image_label.setPixmap(
                QPixmap.fromImage(
                    utils.get_q_image(
                        self.video_settings[index].arena.background_image
                        )
                    )
                )

    @pyqtSlot(int)
    def update_ellipse_ratio(self, ratio):
        """Updates the major-to-minor axis ratio of the ellipse based on
        changing input from the ratio_spinbox.

        This function will also update the image if there is already an
        ellipse present.
        """
        self.ellipse_ratio = ratio
        self.draw_ellipse()
        

    def accept_female(self):
        """Sets the female settings for the current video being examined."""
        self.video_settings[self.current_video_ix].female.center = self.center
        self.video_settings[self.current_video_ix].female.head = self.head
        self.video_settings[self.current_video_ix].female.rear = self.rear
        self.video_settings[self.current_video_ix].female.orientation = self.orientation
        self.video_settings[self.current_video_ix].female.maj_ax_rad = self.maj_ax_rad
        self.video_settings[self.current_video_ix].female.min_ax_rad = self.min_ax_rad

    def reject_female(self):
        """Clears the female settings for the current video being examined."""
        self.center = None
        self.head = None
        self.rear = None
        self.orientation = None
        self.maj_ax_rad = None
        self.min_ax_rad = None

        self.accept_female()
        self.image_label.setPixmap(
            QPixmap.fromImage(
                utils.get_q_image(
                    self.video_settings[
                        self.current_video_ix].arena.background_image
                    )
                )
            )
        self.check_settings_valid()

    def get_ellipse_params(self):
        """Gets all of the parameters that define an ellipse based on user clicks."""
        try:
            self.center = tuple((self.head + self.rear) / 2)
            self.maj_ax_rad = int(
                np.sqrt(np.sum((self.head - self.rear)**2)) / 2
                )
            self.min_ax_rad = int(1. * self.maj_ax_rad / self.ellipse_ratio)
    
            # get relative coordinates of head and rear, and find angle 
            # between these two along the x-axis.
            rc = (self.head - self.rear)
            self.orientation = int(np.arctan2(rc[0], rc[1]) * 180. / np.pi)

            # add the parameters to the current video_setting.female attribute
            self.accept_female()
        except TypeError:
            print 'No female set yet. Not drawing ellipse.'

    def draw_ellipse(self):
        """Draws an ellipse around a female based on user clicks."""
        self.get_ellipse_params()
        self.image_label.setPixmap(
            QPixmap.fromImage(
                utils.get_q_image(
                    self.video_settings[
                        self.current_video_ix].female.draw_female()
                    )
                )
            )
        
    def image_click(self, event):
        """Sets the rear of the female fly based on the first user click."""
        rr, cc = utils.get_mouse_coords(event)
        self.rear = np.array([rr, cc])

    def image_move(self, event):
        """Sets the head of the female fly based on the most recent mouse move."""
        rr, cc = utils.get_mouse_coords(event)
        self.head = np.array([rr, cc])
        self.draw_ellipse()

    def image_release(self, event):
        self.check_settings_valid()
        self.video_list_widget.setCurrentRow(self.current_video_ix)

    def check_settings_valid(self):
        """Checks to see whether or not video settings have been set.

        If settings have been set, a signal is emitted to the main 
        dialog window containing this widget.
        """
        for video_setting in self.video_settings:
            if video_setting.female is None:
                self.all_settings_valid.emit(
                    False, self.widget_ix, self.video_settings)
                return False
            if not video_setting.female.settings_valid():
                self.all_settings_valid.emit(
                    False, self.widget_ix, self.video_settings)
                return False

        self.all_settings_valid.emit(True, self.widget_ix, self.video_settings)
        return True

class BatchTightThresholdSpecifier(BatchSettingsWidget):
    """Allows the user to """
    # slot for loop to generate images for display in 
    # image_array_widget 
    image_calc_progress = pyqtSignal(int, str) 

    def __init__(self, parent = None):
        super(BatchTightThresholdSpecifier, self).__init__(parent)

        self.current_video_ix = 0
        self.threshold = 35
        self.frame_ixs = []

        self.layout = QGridLayout()

        self.set_image_array_widget()
        self.set_video_list_widget()

        self.setLayout(self.layout)

    def check_settings_valid(self):
        self.all_settings_valid.emit(True, self.widget_ix, self.video_settings)
        return True

    def randomize_frames(self):
        """Randomizes the frames displayed in the image array widget.

        This is a slot for the image_array_widget.randomize_button.clicked
        signal.
        """
        self.frame_ixs = np.random.randint(
            low = 0, 
            high = self.video_settings[self.current_video_ix].video.get_n_frames(),
            size = 9)
        self.update_frames()

    def set_image_array_widget(self):
        """Sets up the image array."""
        help_text = ('Please define a "tight" threshold for each video. ' + 
            'To do this, use the spinbox to the right to adjust the threshold ' +
            'such that the images displayed in the "Frame Subset" group box ' +
            'remove as much of each fly as possible. Try to make sure the wings are ' +
            'excluded from the male fly.')

        self.image_array_widget = ImageArrayWidget()
        self.image_array_widget.image_width = 180
        self.image_array_widget.image_height = 180
        self.image_array_widget.instruction_text_edit.setText(help_text)
        self.image_array_widget.randomize_button.clicked.connect(self.randomize_frames)
        self.image_array_widget.threshold_spinbox.setValue(self.threshold)
        self.image_array_widget.threshold_spinbox.valueChanged.connect(self.update_threshold)

        self.layout.addWidget(self.image_array_widget, 0, 1, 3, 3)

    def set_video_list_widget(self):
        """Sets up the video list widget to the left of the screen."""
        gb = QGroupBox('Videos')
        gb_layout = QVBoxLayout()

        self.video_list_widget = QListWidget()
        self.video_list_widget.currentRowChanged.connect(self.update_current_video)

        gb_layout.addWidget(self.video_list_widget)
        gb.setLayout(gb_layout)

        self.layout.addWidget(gb, 0, 0, 3, 1)

    @pyqtSlot(int)
    def update_current_video(self, index):
        """Updates the main layout based on the current video that has been
        selected from the video_list_widget.

        This is a slot for video_list_widget.currentRowChanged.
        """
        self.current_video_ix = index
        try:
            self.frame_ixs = np.random.randint(
                low = 0, 
                high = self.video_settings[index].video.get_n_frames(),
                size = 9)
            self.image_array_widget.threshold_spinbox.setValue(
                    self.video_settings[index].tight_threshold
                )
        except TypeError:
            self.video_settings[index].tight_threshold = self.image_array_widget.threshold_spinbox.value()
        except (AttributeError, NameError):
            pass
        self.update_frames()

    def update_frames(self):
        """Upates all of the images contained within the image_array_widget 
        based on the current threshold.

        The first row should just be a binary image showing the threshold of
        the entire arena. The second row should be a close up image of the
        male fly rotated based on its orientation RegionProp calculated using
        an absolute threshold. And the third row should be a close up image
        of the male, highlighted with pixels that are below the user-defined
        tight threshold.
        """
        if not self.video_settings[self.current_video_ix].female.settings_valid() \
        or not self.video_settings[self.current_video_ix].arena.settings_valid():
            return

        for i, ix in enumerate(self.frame_ixs):
            frame = self.video_settings[self.current_video_ix].video.get_frame(ix)[0]
            try:
                male_props_abs = find_male(
                        frame, 
                        female = self.video_settings[self.current_video_ix].female,
                        arena = self.video_settings[self.current_video_ix].arena,
                        lp_threshold = self.threshold)
            except NoPropsDetected as NPD:
                image = np.zeros(shape = (100, 100), dtype = np.uint8)
                self.image_array_widget.update_image_label(i, image)

                percent_complete = (i + 1.) / self.frame_ixs.size * 100
                self.image_calc_progress.emit(
                    percent_complete, 'Caculating background threshold images.')
                continue

            # first three rows in image array
            if i < 3:
                image = low_pass_threshold_binary(frame, self.threshold) * 255
                image[np.where(self.video_settings[self.current_video_ix].arena.get_arena_mask() == 0)] = 55

            # second three rows in image array		
            elif i < 6:	
                image = center_image_in_frame(
                        frame,
                        centroid = male_props_abs.centroid,
                        size = (200, 200)
                    )
                image = transform.rotate(image, -male_props_abs.orientation * 180 / np.pi, preserve_range = True).astype(np.uint8)

                # rotate image based on orientation, and draw a line through the middle.
                image = trim_image2d(image, size = (100, 100))
                image = cv2.cvtColor(image, cv2.cv.CV_GRAY2BGR)
                cv2.line(image, (0, 50), (100, 50), (255, 0, 0), 1, cv2.cv.CV_AA)

            # final three rows in image array
            else:
                threshold_img = low_pass_threshold_binary(frame, self.threshold)
                # get coordinates of all regions in image that are below user threshold
                b_rr, b_cc = np.where(threshold_img)
                # and get coordinates for female mask
                f_rr, f_cc = np.where(
                    self.video_settings[self.current_video_ix].female.get_female_mask()
                    )
                image = cv2.cvtColor(frame, cv2.cv.CV_GRAY2BGR)
                image[b_rr, b_cc, 0] = 200
                image[f_rr, f_cc, :] = 255
                image = center_image_in_frame3d(
                        image, 
                        centroid = male_props_abs.centroid,
                        size = (100, 100)
                    )

            self.image_array_widget.update_image_label(i, image)
            percent_complete = (i + 1.) / self.frame_ixs.size * 100
            self.image_calc_progress.emit(percent_complete, 'Caculating background threshold images.')

    @pyqtSlot(int)
    def update_threshold(self, threshold):
        """Sets theshold for current video.

        This is a slot for the threshold_spinbox contained within the
        image_array_widget.
        """
        self.threshold = threshold
        self.video_settings[self.current_video_ix].tight_threshold = threshold
        self.update_frames()

    def update_view(self):
        """Updates each wiget with new video_settings passed by user
        from one of the other widgets within the StackedLayout."""

        # make sure that we are not appending to the list every time a 
        # user navigates away from this widget.
        utils.clear_list(self.video_list_widget)

        for settings in self.video_settings:
            base_name = os.path.basename(settings.video_file)
            self.video_list_widget.addItem(base_name)

        self.video_list_widget.setMinimumWidth(
            self.video_list_widget.sizeHintForColumn(0)
            )

        self.video_list_widget.setCurrentRow(self.current_video_ix)

class BatchLooseThresholdSpecifier(BatchSettingsWidget):
    """Tight and LooseThresholdSpecifier should inherit from a 
    common widget to minimize code."""

    # signal to send to main widget's progress bar.
    image_calc_progress = pyqtSignal(int, str)

    def __init__(self, parent = None):
        super(BatchLooseThresholdSpecifier, self).__init__(parent)
        
        self.current_video_ix = 0
        self.threshold = 5
        self.frame_ixs = []

        self.layout = QGridLayout()

        self.set_image_array_widget()
        self.set_video_list_widget()

        self.setLayout(self.layout)

    def check_settings_valid(self):
        self.all_settings_valid.emit(True, self.widget_ix, self.video_settings)
        return True

    def randomize_frames(self):
        """Randomizes the frames displayed in the image array widget.

        This is a slot for the image_array_widget.randomize_button.clicked
        signal.
        """
        self.frame_ixs = np.random.randint(
            low = 0, 
            high = self.video_settings[self.current_video_ix].video.get_n_frames(),
            size = 9)
        self.update_frames()

    def set_image_array_widget(self):
        """Sets up the image array."""
        help_text = ('Please define a "loose" threshold for each video. ' + 
            'To do this, use the spinbox to the right to adjust the threshold ' +
            'such that the images displayed in the "Frame Subset" group box ' +
            'contain as much of each fly as possible. Try to make sure the entirety ' +
            'of the wings are included from the male fly.')

        self.image_array_widget = ImageArrayWidget()
        self.image_array_widget.image_width = 180
        self.image_array_widget.image_height = 180
        self.image_array_widget.instruction_text_edit.setText(help_text)
        self.image_array_widget.randomize_button.clicked.connect(self.randomize_frames)
        self.image_array_widget.threshold_spinbox.setValue(self.threshold)
        self.image_array_widget.threshold_spinbox.valueChanged.connect(self.update_threshold)

        self.layout.addWidget(self.image_array_widget, 0, 1, 3, 3)

    def set_video_list_widget(self):
        """Sets up the video list widget to the left of the screen."""
        gb = QGroupBox('Videos')
        gb_layout = QVBoxLayout()

        self.video_list_widget = QListWidget()
        self.video_list_widget.currentRowChanged.connect(self.update_current_video)

        gb_layout.addWidget(self.video_list_widget)
        gb.setLayout(gb_layout)

        self.layout.addWidget(gb, 0, 0, 3, 1)

    @pyqtSlot(int)
    def update_current_video(self, index):
        """Updates widget layout based on user click from video_list_widget.

        This is a slot for self.video_list_widget.

        Parameters
        ----------
        index : int
            Row index of item contained in self.video_list_widget.
        """
        self.current_video_ix = index

        try:
            self.frame_ixs = np.random.randint(
                low = 0, 
                high = self.video_settings[index].video.get_n_frames(),
                size = 9)
            self.image_array_widget.threshold_spinbox.setValue(
                    self.video_settings[index].loose_threshold
                ) * 255
        except TypeError:
            self.video_settings[index].loose_threshold = self.image_array_widget.threshold_spinbox.value()
        except (AttributeError, NameError):
            pass

        self.update_frames()

    def update_frames(self):
        """Updates frames displayed in image_array_widget based on 
        the current loose threshold.

        First row shows a binary image of just the male's body. 
        Second row shows a binary image of the male's body including the wings.
        Third row shows a binary image of the male's body subtracted away from
        the male's body including the wings.
        """
        if not self.video_settings[self.current_video_ix].female.settings_valid() \
        or not self.video_settings[self.current_video_ix].arena.settings_valid() \
        or self.video_settings[self.current_video_ix].tight_threshold is None:
            return

        for i, ix in enumerate(self.frame_ixs):
            frame = self.video_settings[self.current_video_ix].video.get_frame(ix)[0]

            try:
                male_props_abs = find_male(
                    image = frame, 
                    female = self.video_settings[self.current_video_ix].female,
                    arena = self.video_settings[self.current_video_ix].arena,
                    lp_threshold = self.video_settings[self.current_video_ix].tight_threshold)
            except NoPropsDetected:
                image = np.zeros(shape = (100, 100), dtype = np.uint8)

                self.image_array_widget.update_image_label(i, image)
                percent_complete = (i + 1.) / self.frame_ixs.size * 100
                self.image_calc_progress.emit(
                    percent_complete, 'Caculating background threshold images.')
                continue

            if i < 3:
                image = get_body_image(
                        in_shape = frame.shape,
                        male_props = male_props_abs,
                        rotation = -male_props_abs.orientation * 180/np.pi,
                        out_shape = (100, 100)
                    ) * 255
            elif i < 6:
                image, r = get_wing_image(
                        image = frame,
                        female = self.video_settings[self.current_video_ix].female,
                        arena = self.video_settings[self.current_video_ix].arena, 
                        male_props = male_props_abs,
                        loose_threshold = self.threshold,
                        shape = (100, 100),
                        head = 'Right',
                        subtract_body = False
                    )
                image *= 255
            else:
                image, r = get_wing_image(
                        image = frame,
                        arena = self.video_settings[self.current_video_ix].arena,
                        female = self.video_settings[self.current_video_ix].female,
                        male_props = male_props_abs,
                        loose_threshold = self.threshold,
                        head = 'Right',
                        subtract_body = True
                    )
                image *= 255
            
            image = cv2.cvtColor(image, cv2.cv.CV_GRAY2BGR)
    
            self.image_array_widget.update_image_label(i, image)
            percent_complete = (i + 1.) / self.frame_ixs.size * 100
            self.image_calc_progress.emit(
                percent_complete, 'Caculating background threshold images.')

    @pyqtSlot(int)
    def update_threshold(self, threshold):
        """Updates the 'loose' threshold for the current video.

        This is a slot for self.threshold_spinbox.

        Parameters
        ----------
        threshold : int
            Loose threshold to set for current video.
        """
        self.threshold = threshold
        self.video_settings[self.current_video_ix].loose_threshold = threshold
        self.update_frames()

    def update_view(self):
        """Updates each wiget with new video_settings passed by user
        from one of the other widgets within the StackedLayout."""

        #make sure that we are not appending to the list every time a 
        #user navigates away from this widget.
        utils.clear_list(self.video_list_widget)

        for settings in self.video_settings:
            base_name = os.path.basename(settings.video_file)
            self.video_list_widget.addItem(base_name)

        self.video_list_widget.setMinimumWidth(
            self.video_list_widget.sizeHintForColumn(0)
            )

        self.video_list_widget.setCurrentRow(self.current_video_ix)

class BatchTrackingWidget(BatchSettingsWidget):

    tracking_progress = pyqtSignal(int, str)

    def __init__(self, parent = None):
        super(BatchTrackingWidget, self).__init__(parent)

        self.layout = QGridLayout()
        self.headers = [
            'Video File', 
            'Save File', 
            'Save Type', 
            'Tight Threshold',
            'Loose Threshold',
            'Group']

        self.set_settings_table()
        self.table_widget.cellChanged.connect(self.update_group)
        self.set_progress_log()

        track_button = QPushButton('Track')
        track_button.clicked.connect(self.track)
        self.layout.addWidget(track_button, 2, 4, 1, 1)

        self.setLayout(self.layout)

    def check_settings_valid(self):
        """Checks to see whether all settings are valid in the current widget.

        The settings are always going to be valid if a user has made it this far.
        However, since we want to disable the 'Next' button, we will emit a signal
        telling the main widget that the settings are not valid.
        """
        self.all_settings_valid.emit(False, self.widget_ix, self.video_settings)

    def set_progress_log(self):
        """Adds a QTextEdit to the layout, which will inform the user
        of tracking progress.
        """
        self.progress_log = QTextEdit('Tracking Progress')
        self.progress_log.setReadOnly(True)
        self.layout.addWidget(self.progress_log, 1, 0, 1, 5)

    def set_settings_table(self):
        """Sets up a QTableWidget on the layout to show all of the settings
        the user has specified.
        """
        self.table_widget = QTableWidget()
        self.layout.addWidget(self.table_widget, 0, 0, 1, 5)

    def track(self):
        """Main function to track all flies, and save as appropriate file type."""
        for ix in xrange(len(self.video_settings)):
            start_time = time.time()

            male = Fly()
            female = Fly()

            tracking_summary = FixedCourtshipTrackingSummary()
            tracking_settings = {
                'video_file': self.video_settings[ix].video_file,
                'arena_type': 'circular',
                'pixels_per_mm': self.video_settings[ix].arena.pixels_to_mm,
                'arena_center_rr': self.video_settings[ix].arena.center[0],
                'arena_center_cc': self.video_settings[ix].arena.center[1],
                'arena_radius': self.video_settings[ix].arena.radius,
                'group': self.video_settings[ix].group,
                'tight_threshold': self.video_settings[ix].tight_threshold,
                'loose_threshold': self.video_settings[ix].loose_threshold,
                'date_tracked': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                'tracking_software': 'CourTrack v{}'.format(__version__),
                'arena_size_mm': self.video_settings[ix].arena.arena_size
            }

            male.video_file = tracking_settings['video_file']
            female.video_file = tracking_settings['video_file']

            self.progress_log.append('Tracking started for video: {} \nStart Time: {}'.format(
                    tracking_settings['video_file'],
                    time.strftime('%H:%M:%S', time.localtime(start_time))
                ))

            male.allocate_param_space(self.video_settings[ix].video.get_n_frames())
            female.allocate_param_space(self.video_settings[ix].video.get_n_frames())

            f_props, f_head, f_rear = find_female(
                    image = self.video_settings[ix].arena.background_image,
                    female = self.video_settings[ix].female,
                    lp_threshold = self.video_settings[ix].tight_threshold
                )

            # update female based on props we just found --
            # this ensures that the ellipse used to mask the female is 
            # not biased by variation in user-defined ellipses.
            tighten_female_ellipse(
                    female = self.video_settings[ix].female,
                    female_props = f_props
                )

            for frame_ix in xrange(self.video_settings[ix].video.get_n_frames()):

                frame, ts = self.video_settings[ix].video.get_frame(frame_ix)

                try:
                    male_props = find_male(
                            image = frame, 
                            female = self.video_settings[ix].female,
                            arena = self.video_settings[ix].arena,
                            lp_threshold = self.video_settings[ix].tight_threshold)
                except NoPropsDetected as NPD:
                    self.progress_log.append('\t' + NPD.message + ' Body @ frame {}'.format(frame_ix))
                    male.timestamps[frame_ix] = ts
                    female.timestamps[frame_ix] = ts
                    continue

                wing_props = find_wings(
                        image = frame,
                        female = self.video_settings[ix].female,
                        arena = self.video_settings[ix].arena,
                        male_props = male_props,
                        loose_threshold = self.video_settings[ix].loose_threshold,
                        logger = self.progress_log,
                        frame_ix = frame_ix
                    )
                
                if wing_props is None:
                    male.timestamps[frame_ix] = ts
                    female.timestamps[frame_ix] = ts
                    continue

                set_male_props(male, wing_props, frame_ix)
                male.body.centroid.y[frame_ix], male.body.centroid.x[frame_ix] = male_props.centroid
                male.body.orientation[frame_ix] = male_props.orientation
                male.timestamps[frame_ix] = ts

                set_female_props(female, f_props, f_head, f_rear, frame_ix)
                female.timestamps[frame_ix] = ts

                # wing_annotation_img = draw_tracked_wings(
                # 		image = frame,
                # 		left_centroid = np.array([
                # 			male.left_wing.centroid.y[frame_ix],
                # 			male.left_wing.centroid.x[frame_ix]]),
                # 		right_centroid = np.array([
                # 			male.right_wing.centroid.y[frame_ix],
                # 			male.right_wing.centroid.x[frame_ix]]),
                # 		head_centroid = np.array([
                # 			male.body.head.y[frame_ix],
                # 			male.body.head.x[frame_ix]]),
                # 		tail_centroid = np.array([
                # 			male.body.rear.y[frame_ix],
                # 			male.body.rear.x[frame_ix]]),
                # 		female_head = f_head,
                # 		female_rear = f_rear
                # 	)

                # cv2.imshow('frame', wing_annotation_img)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # 	break

                percent_complete = (frame_ix + 1.) / self.video_settings[ix].video.get_n_frames() * 100
                self.tracking_progress.emit(
                    percent_complete,
                    'Tracking video {}/{}.'.format(ix + 1, len(self.video_settings)))

            # update the tracking settings dictionary with male and female items.
            tracking_settings.update({'male': male, 'female': female})
            tracking_summary.set_attributes(**tracking_settings)

            save_file  = self.video_settings[ix].save_file
            save_type = save_file.split('.')[-1]
            if save_type == 'xlsx':
                tracking_summary.to_xlsx(save_file)
            elif save_type == 'fcts':
                with open(save_file, 'wb') as SAVE:
                    pickle.dump(tracking_summary, SAVE)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_hrs = int(elapsed_time / 3600)
            time_mins = int((elapsed_time - time_hrs * 3600) / 60)
            time_secs = int(elapsed_time - time_hrs * 3600 - time_mins * 60)

            self.progress_log.append('End Time: {}\nTotal Time Elapse: {}'.format(
                    time.strftime('%H:%M:%S', time.localtime(end_time)),
                    '{:02d}:{:02d}:{:02d}'.format(time_hrs, time_mins, time_secs)
                ))
        self.progress_log.append('TRACKING COMPLETE')

    @pyqtSlot(int, int)
    def update_group(self, row, col):
        """Updates the group attribute in video_settings if a user changes the 
        'Group' cell in self.table_widget.

        This is a slot for self.table_widget.cellChanged.

        Parameters
        ----------
        row : int 
            Row of cell within self.table_widget that user clicked.

        col : int
            Column of cell within self.table_widget that user clicked.
        """
        if col == 5:
            self.video_settings[row].group = str(self.table_widget.item(row, col).text())

    def update_view(self):
        """Updates the table widget based on changing user settings coming from
        other widgets in the StackedLayout.
        """
        self.table_widget.setRowCount(len(self.video_settings))
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(self.headers)
        self.table_widget.horizontalHeader().setResizeMode(QHeaderView.Stretch)

        for row in xrange(len(self.video_settings)):
            items = [
                QTableWidgetItem(QString(self.video_settings[row].video_file), 0),
                QTableWidgetItem(QString(self.video_settings[row].save_file), 0),
                QTableWidgetItem(QString(self.video_settings[row].save_file.split('.')[-1]), 0),
                QTableWidgetItem(QString(str(self.video_settings[row].tight_threshold)), 0),
                QTableWidgetItem(QString(str(self.video_settings[row].loose_threshold)), 0),
                QTableWidgetItem(QString(self.video_settings[row].group), 0)
            ]
            
            for col, item in enumerate(items):
                if col == 0 or col == 1:
                    item.setTextAlignment(Qt.AlignRight)
                else:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table_widget.setItem(row, col, item)


