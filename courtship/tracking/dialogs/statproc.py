"""
statproc.py

Dialog window for doing statistical processing on waveforms/signals stored in a .fcts file.
"""
import os, pickle, json
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import numpy as np
import pandas as pd

from pyqtgraph import PlotItem, PlotWidget, mkPen

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from widgets.batch import *
from widgets.statistics import *
from widgets.video import *

from canal.objects.experiment import FixedCourtshipTrackingSummary as FCTS
import canal.statistics.centroid as centroid_stats
import canal.statistics.social as social_stats
import canal.statistics.wing as wing_stats


class StatProcessing(QMainWindow):
    """Dialog for processing signals.

    Parameters
    ----------
    root_folder : string (default = '~/')
        Path to original directory opened by the user.

    Attributes
    ----------
    selected_data : list of lists
        All data that has been selected by the user.

    mean_pen : QPen
        Pen object that defines the aesthetics of the mean statistic line.

    signal_window : StatProcessingWindow
        QWidget that holds the signal.

    spreadsheet : QTableWidget
        QWidget that displays (numerically) the selected_data.

    plot_widget : QPlotWidget
        QWidget that displays (graphically) the selected data.
    """
    def __init__(self, root_folder = '~/', parent=None):
        super(StatProcessing, self).__init__(parent)
        self.selected_data = []
        self.start_ixs = []
        self.stop_ixs = []
        self.signal_window_connected = False
        self.mean_pen = mkPen('r', width=2)
        self.root_folder = root_folder

        self.menu_bar = QMenuBar(self)
        self.file_menu = self.menu_bar.addMenu('&File')
        self.video_menu = self.menu_bar.addMenu('Video')

        self.set_status_bar()
        self.set_file_menu()
        self.set_video_menu()

        self.signal_window = StatProcessingWindowWidget(window_size=760)
        self.signal_window.setEnabled(False)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.signal_window.update_stat_label)
        self.signal_window.scrolled_on_window.connect(self.update_slider)

        self.stat_spreadsheet = QTableWidget()
        self.index_spreadsheet = QTableWidget()
        self.plot_widget = PlotWidget()

        self.video_widget = MainVideoPlayer()
        self.video_dock_widget = QDockWidget()
        self.video_dock_widget.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        self.video_dock_widget.setAllowedAreas(Qt.NoDockWidgetArea)
        self.video_dock_widget.setWidget(self.video_widget)
        self.video_dock_widget.hide()

        self.central_widget = QWidget()
        central_layout = QGridLayout()
        central_layout.setMenuBar(self.menu_bar)
        central_layout.addWidget(self.signal_window, 1, 0, 1, 4)
        central_layout.addWidget(self.slider, 0, 0, 1, 4)
        central_layout.addWidget(self.stat_spreadsheet, 2, 0, 2, 2)
        central_layout.addWidget(self.index_spreadsheet, 4, 0, 2, 2)
        central_layout.addWidget(self.plot_widget, 2, 2, 4, 2)
        self.central_widget.setLayout(central_layout)

        self.setCentralWidget(self.central_widget)
        self.setWindowTitle('Statistical Processing')

    def set_file_menu(self):
        open_action = QAction('Open Tracking Summary', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip('Open a Tracking Summary for Analysis.')
        open_action.triggered.connect(self.open_file)
        self.file_menu.addAction(open_action)

        export_action = QAction('Export Data', self)
        export_action.setStatusTip('Export data in table to file.')
        export_action.triggered.connect(self.export_data)
        self.file_menu.addAction(export_action)

    def set_video_menu(self):
        open_action = QAction('Open Video', self)
        open_action.setStatusTip('Open Video File Associated with .fcts file.')
        open_action.triggered.connect(self.open_video)
        self.video_menu.addAction(open_action)

    def set_status_bar(self):
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(True)

        self.status_label = QLabel()
        self.status_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.status_label.setText('File: ')

        status_bar.addPermanentWidget(self.status_label)
        
    def open_file(self, file_name=None):
        """Opens a .fcts file and places it in the StatProcessingWindowWidget."""
        if not isinstance(file_name, str):
            file_dialog = QFileDialog(self)
            file_name = str(file_dialog.getOpenFileName(
                caption = 'Open Tracking Summary File',
                filter = 'Tracking Summary File (*.fcts *.xlsx)',
                directory = self.root_folder
                ))

        if file_name.split('.')[-1] == 'fcts':
            with open(file_name, 'rb') as f: 
                tracking_summary = pickle.load(f)
        else:
            tracking_summary = FCTS().from_xlsx(file_name)

        self.selected_data = []
        self.start_ixs = []
        self.stop_ixs = []

        self.tracking_summary = tracking_summary

        #calculate statistics here that will get placed into the statics window.
        lwa, rwa = wing_stats.individual_wing_angles(tracking_summary.male)
        fwa = wing_stats.full_wing_angle(tracking_summary.male)
        wd = wing_stats.wing_distances(tracking_summary.male)
        nn = social_stats.nearest_neighbor_centroid(tracking_summary.male, tracking_summary.female)
        n2e, t2e = social_stats.nose_and_tail_to_ellipse(tracking_summary.male, tracking_summary.female)
        vel = centroid_stats.centroid_velocity(tracking_summary.male)
        
        self.signal_window.statistics = {
            'LeftWingAngle': lwa,
            'RightWingAngle': rwa,
            'FullWingAngle': fwa,
            'WingDistance': wd,
            'Centroid-to-Centroid': nn,
            'Nose-to-Ellipse': n2e,
            'Rear-to-Ellipse': t2e,
            'CentroidVelocity': vel
        }

        self.signal_window.raw_statistics = self.signal_window.statistics.copy()

        self.signal_window.n_frames = tracking_summary.male.n_frames - 1
        self.signal_window.update_combobox()
        self.signal_window.update_stat_label(0)
        if self.signal_window_connected == False:
            self.signal_window.region_selected.connect(self.update_table)
            self.signal_window_connected = True
        self.signal_window.setEnabled(True)

        self.clear_views()

        self.status_label.setText('File: {}'.format(file_name))
        self.slider.setEnabled(True)
        self.slider.setMaximum(tracking_summary.male.n_frames - 1)

    def open_video(self):
        """Opens video file associated with fcts file."""
        file_dialog = QFileDialog(self)
        open_name = str(file_dialog.getOpenFileName(
            caption = 'Open Video File',
            filter = 'Video File (*.fmf)',
            directory = self.root_folder,
            ))

        self.video_widget.set_video(open_name)
        self.video_widget.tracking_summary = self.tracking_summary
        self.video_widget.image_annotation = True 
        self.slider.valueChanged.connect(self.video_widget.slider_update_label)
        
        self.video_dock_widget.setWindowTitle(open_name)
        self.video_dock_widget.show()

    def export_data(self):
        """Exports data to file."""
        file_dialog = QFileDialog(self)
        save_name = str(file_dialog.getSaveFileName(
            caption = 'Save File',
            filter = '(*.txt .*json)',
            directory = self.root_folder
            ))

        if os.path.basename(save_name).split('.')[-1] == 'txt':
            with open(save_name, 'wb') as f:
                for row in self.selected_data:
                    for i, col in enumerate(row):
                        if i != len(row) - 1:
                            f.write('{}, '.format(col))
                        else:
                            f.write('{}'.format(col))
                    f.write('\n')

        json_data = {}
        
        for k, stat in self.signal_window.raw_statistics.iteritems():
            json_data[k] = []
            for i in xrange(len(self.start_ixs)):
                json_data[k].append(np.asarray(stat[self.start_ixs[i]:self.stop_ixs[i]]).tolist())

        json_data['start_ixs'] = self.start_ixs
        json_data['stop_ixs'] = self.stop_ixs

        if os.path.basename(save_name).split('.')[-1] == 'json':
            with open(save_name, 'wb') as f:
                json.dump(json_data, f)

    def clear_views(self):
        self.plot_widget.clear()
        self.stat_spreadsheet.clear()	
        self.index_spreadsheet.clear()
        self.video_dock_widget.hide()
        
    @pyqtSlot(int)
    def update_slider(self, ix):
        self.slider.setValue(ix)

    @pyqtSlot(str, int, int, list)
    def update_table(self, stat_name, start_ix, stop_ix, stat_values):
        """Adds values to spreadsheet and plots.

        Parameters
        ----------
        stat_name : string
            Name of statistic being passed to table/plot.

        stat_values : list
            List of numbers containing raw statistic values.
        """
        self.selected_data.append(stat_values)
        self.start_ixs.append(start_ix)
        self.stop_ixs.append(stop_ix)

        nrows = len(self.selected_data)
        ncols = np.max([len(d) for d in self.selected_data])

        self.stat_spreadsheet.setRowCount(nrows)
        self.stat_spreadsheet.setColumnCount(ncols)

        for r, row in enumerate(self.selected_data):
            for c, val in enumerate(row):
                self.stat_spreadsheet.setItem(r, c, QTableWidgetItem('{:02f}'.format(val)))

        self.index_spreadsheet.setRowCount(len(self.start_ixs))
        self.index_spreadsheet.setColumnCount(2)
        self.index_spreadsheet.setHorizontalHeaderLabels(['Start Ix','Stop Ix'])

        for i in xrange(len(self.start_ixs)):
            self.index_spreadsheet.setItem(i, 0, QTableWidgetItem('{}'.format(self.start_ixs[i])))
            self.index_spreadsheet.setItem(i, 1, QTableWidgetItem('{}'.format(self.stop_ixs[i])))

        min_val = np.min([np.min(d) for d in self.selected_data])
        max_val = np.max([np.max(d) for d in self.selected_data])

        self.mean = [0] * ncols
        for col in xrange(ncols):
            s = 0
            for item in xrange(nrows):
                if col < len(self.selected_data[item]):
                    s += self.selected_data[item][col]
                self.mean[col] = s * 1. / nrows

        self.plot_widget.clear()
        self.plot_widget.setXRange(0, ncols)
        self.plot_widget.setYRange(min_val, max_val)
        
        for selection in self.selected_data:
            self.plot_widget.plot(x = np.arange(0, len(selection)), y = selection)
        
        self.plot_widget.plot(x = np.arange(0, ncols), y = self.mean, pen=self.mean_pen)