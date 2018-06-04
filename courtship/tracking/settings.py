
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class TrackingSettings(QObject):
    """Container to keep track of settings for a particular video.

    Attributes
    ----------
    video_file : string 
        Path to video file to track.

    video : motmot.FlyMovieFormat.FlyMovie
        Opened video to track.

    arena : Arena object
        Arena of video to track.

    female : Female object
        Contains information about the female mask.

    loose_threshold : int 
        Threshold value which lets entire fly be observed.

    tight_threshold : int
        Threshold value which only lets fly abdomen be observed.

    group : string (default = 'None')
        String to identify to which group (if any) this pair of flies belong.
    """
    def __init__(self):
        self.video_file = None
        self.save_file = None
        self.video = None
        self.arena = None
        self.female = None
        self.loose_threshold = 5
        self.tight_threshold = 35
        self.group = 'None'
