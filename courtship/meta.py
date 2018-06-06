# -*- coding: utf-8 -*-

"""
.. module:: courtship
   :synopsis: Classes holding meta data about different objects used
              during tracking.

.. moduleauthor:: Ross McKinney
"""
import os
from datetime import datetime
import numpy as np
import motmot.FlyMovieFormat.FlyMovieFormat as FMF


class Meta(object):
    """Base class for Meta objects."""
    def __str__(self):
        class_str = ''
        for key, val in sorted(self.__dict__.iteritems()):
            class_str += '  {}: {}\n'.format(key, val)
        return class_str


class VideoMeta(Meta):
    """Provides meta data about a particular (tracked) video.

    Parameters
    ----------
    filename : string
        Name of recorded video.

    fps : int
        Frame rate (frames-per-second) of recorded video.

    start_time : string
        Recording start time of video ("YYYY-MM-DD HH:MM:SS").

    end_time : string
        Recording end time of video ("YYYY-MM-DD HH:MM:SS").

    duration_seconds : float
        Total duration of video in seconds.

    duration_frames : int
        Total duration of video in frames.

    pixels_per_mm : int
        Number of pixels in 1mm of absolute distance.

    timestamps : np.ndarray of float
        Timestamps (in seconds from epoch) associated with each frame.
    """
    def __init__(self, filename=None, fps=None, start_time=None, end_time=None,
                 duration_seconds=None, duration_frames=None, pixels_per_mm=None,
                 timestamps=None):
        self.filename = filename
        self.fps = fps
        self.start_time = start_time
        self.end_time = end_time
        self.duration_seconds = duration_seconds
        self.duration_frames = duration_frames
        self.pixels_per_mm = pixels_per_mm
        self.timestamps = timestamps

    def __str__(self):
        class_str = 'Video Meta Data:\n'
        return class_str + super(VideoMeta, self).__str__()

    def update_meta_from_video(self):
        """Gets summary statistics about a this class' associated video file.

        If the video file associated with this VideoMeta object exists on disk,
        the following attributes will be updated: (1) fps, (2) start_time, (3)
        end_time, (4) timestamps.
        """
        if not os.path.exists(self.filename):
            raise AttributeError(
                'self.filename -- {} '.format(self.filename) +
                '-- does not exist on disk. Please update filename with a ' +
                'valid file path.'
                )

        video = FMF.FlyMovie(self.filename)

        self.timestamps = video.get_all_timestamps()
        self.fps = 1. / np.mean(np.diff(self.timestamps))
        self.start_time = datetime.fromtimestamp(self.timestamps[0]).strftime(
            '%Y-%m-%d %H:%M:%S')
        self.end_time = datetime.fromtimestamp(self.timestamps[-1]).strftime(
            '%Y-%m-%d %H:%M:%S')


class ArenaMeta(Meta):
    """Provides meta data about a particular arena used during tracking.

    Parameters
    ----------
    shape : string
        Descriptor of arena shape. Should be either 'circular' or 'polygonal'.

    center_pixel_rr : int
        Row index of the center pixel of the arena.

    center_pixel_cc : int
        Column index of the center pixel of the arena.

    diameter_mm : int or None
        Diameter of the arena (if circular) in mm.

    radius_mm : int or None
        Radius of the arena (if circular) in mm.

    vertices : list or None
        Vertices [(rr, cc), ..] of arena (if polygonal).
    """
    def __init__(self, shape=None, center_pixel_rr=None, center_pixel_cc=None,
                 diameter_mm=None, radius_mm=None, vertices=None):
        self.shape = shape
        self.center_pixel_rr = center_pixel_rr
        self.center_pixel_cc = center_pixel_cc
        self.diameter_mm = diameter_mm
        self.radius_mm = radius_mm
        self.vertices = vertices

    def __str__(self):
        class_str = 'Arena Meta Data:\n'
        return class_str + super(ArenaMeta, self).__str__()


class SoftwareTrackingMeta(Meta):
    """Provides meta data about the software version and parameters used
    during tracking.

    Parameters
    ----------
    version : string
        Version of software used for tracking.

    tight_threshold : int
        Tight threshold used during tracking.

    loose_threshold : int
        Loose threshold used during tracking.

    date_tracked : string
        Date ("YYYY-MM-DD HH:MM:SS") tracked.
    """
    def __init__(self, version=None, tight_threshold=None, loose_threshold=None,
                 date_tracked=None):
        self.version = version
        self.tight_threshold = tight_threshold
        self.loose_threshold = loose_threshold
        self.date_tracked = date_tracked

    def __str__(self):
        class_str = 'Software/Tracking Meta Data:\n'
        return class_str + super(SoftwareTrackingMeta, self).__str__()
