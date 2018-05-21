# -*- coding: utf-8 -*-

"""
.. module:: courtship
   :synopsis: Classes holding meta data about different objects used
              during tracking.

.. moduleauthor:: Ross McKinney
"""

class Meta(object):
    """Base class for Meta objects."""
    def __str__(self):
        for key, val in self.__dict__.iteritems():
            print '  ' + key + ': ' + val


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

    duration_seconds : int 
        Total duration of video in seconds.

    duration_frames : int
        Total duration of video in frames.

    pixels_per_mm : int
        Number of pixels in 1mm of absolute distance.
    """
    def __init__(self, filename=None, fps=None, start_time=None, end_time=None,
        duration_seconds=None, duration_frames=None, pixels_per_mm=None):
        self.filename = filename
        self.fps = fps
        self.start_time = start_time
        self.end_time = end_time
        self.duration_seconds = duration_seconds
        self.duration_frames = duration_frames
        self.pixels_per_mm = pixels_per_mm

    def __str__(self):
        print "Video Meta Data:"
        super(VideoMeta, self).__str__()


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
        print "Arena Meta Data:"
        super(ArenaMeta, self).__str__()


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
        print "Software/Tracking Meta Data:"
        super(SoftwareTrackingMeta, self).__str__()