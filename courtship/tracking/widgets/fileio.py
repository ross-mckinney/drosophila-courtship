"""
file.py

Widgets for navigating folders and adding files.
"""
import os 

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from objects.settings import VideoSettings

class FileExplorer(QWidget):
    """Widget for navigating through folders/files.

    Attributes
    ----------
    tree_view : QTreeView
        View of file tree.

    model : QFileSystemModel
        Model containing directory structure for display in
        tree_view.

    Signals
    -------
    open_video : string
        Signal holding the file name of a to-be-opened
        FlyMovie, selected from a CustomContextMenu.
    """

    open_video = pyqtSignal(str)

    def __init__(self, parent = None):
        super(FileExplorer, self).__init__(parent)

        self.tree_view = QTreeView()
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.mouse_click)
        self.model = QFileSystemModel()

        layout = QGridLayout()
        layout.addWidget(self.tree_view, 0, 0)
        self.setLayout(layout)

    def set_path(self, path):
        """Updates the current FileSystemModel given a new root path."""
        self.model.setRootPath(path)
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(path))

    def mouse_click(self, position):
        """Opens CustomContextMenu following right-click of file."""
        selected_file = str(self.model.filePath(self.tree_view.currentIndex()))
        file_suffix = os.path.basename(selected_file).split('.')[-1]

        if file_suffix == 'fmf':
            menu = QMenu()
            open_video_action = menu.addAction('Open Video')
            action = menu.exec_(self.tree_view.mapToGlobal(position))

            if action == open_video_action:
                self.open_video.emit(selected_file)

