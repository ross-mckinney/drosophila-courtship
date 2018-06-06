"""
text.py

Label and text widgets.
"""
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class BatchStepList(QWidget):
    """List displayed to the left of the main Batch widgets to show the
    user which tracking setup step they are on."""
    active_style = "QLabel { background-color : #eff0f1; color : #31363b; }"
    inactive_style = "QLabel { background-color : #31363b; color : #eff0f1; }"

    def __init__(self, highlight=0, parent=None):
        super(BatchStepList, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.highlight = highlight
        self.size_policy = QSizePolicy()
        self.size_policy.setHorizontalPolicy(QSizePolicy.Minimum)
        self.size_policy.setVerticalPolicy(QSizePolicy.Expanding)

        self.labels = [QLabel(l) for l in \
            ['1. Select Files',
            '2. Define Arena',
            '3. Define Female',
            '4. Set Tight Threshold',
            '5. Set Loose Threshold',
            '6. Track']]

        self.update_layout()
        
    def update_layout(self):
        for i, label in enumerate(self.labels):
            if i == self.highlight:
                label.setStyleSheet(self.active_style)
            else:
                label.setStyleSheet(self.inactive_style)
            label.setSizePolicy(self.size_policy)
            label.setFixedWidth(160)
            self.layout.addWidget(label)
        self.setLayout(self.layout)

