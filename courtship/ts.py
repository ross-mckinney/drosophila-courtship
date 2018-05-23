# -*- coding: utf-8 -*-

"""
.. module:: courtship
   :synopsis: Class to hold tracking summaries (ts). 

.. moduleauthor:: Ross McKinney
"""
from .meta import (
    VideoMeta,
    ArenaMeta,
    SoftwareTrackingMeta
)


class TrackingSummary(object):
    """Holds information about a particular tracked video.

    Attributes
    ----------
    video : VideoMeta

    arena : ArenaMeta

    software : SoftwareMeta

    group : string or None (default=None)
        String denoting to which group tracked objects belong.
    """
    def __init__(self):
        self.video = VideoMeta()
        self.arena = ArenaMeta()
        self.software = SoftwareTrackingMeta()
        self.group = None

    def __str__(self):
        class_str = (
            'Tracking Summary\n' +
            '----------------\n' +
            'Group: {}\n'.format(self.group) +
            self.video.__str__() + '\n' +
            self.arena.__str__() + '\n' +
            self.software.__str__()
            )
        return class_str

    def meta_data(self):
        """Returns a dictionary of all meta data associated with this object.

        Returns
        -------
        dict
            Meta data (VideoMeta, ArenaMeta, SoftwareMeta) associated with
            this TrackingSummary. Each key is a string, formatted with its
            type and attribute named joined with a '.'; each value is the
            attribute value.

        Examples
        --------
        >>> ts = TrackingSummary()
        >>> ts.meta_data()
        {'arena.center_pixel_cc': None,
         'arena.center_pixel_rr': None,
         'arena.diameter_mm': None,
         'arena.radius_mm': None,
         'arena.shape': None,
         'arena.vertices': None,
         'group': None,
         'software.date_tracked': None,
         'software.loose_threshold': None,
         'software.tight_threshold': None,
         'software.version': None,
         'video.duration_frames': None,
         'video.duration_seconds': None,
         'video.end_time': None,
         'video.filename': None,
         'video.fps': None,
         'video.pixels_per_mm': None,
         'video.start_time': None,
         'video.timestamps': None}
        """
        meta = {
            'video': self.video,
            'arena': self.arena,
            'software': self.software
        }
        meta_dict = {}
        for attr_meta_type, datum in meta.iteritems():
            named_data = {}
            for key, val in datum.__dict__.iteritems():
                named_data['.'.join([attr_meta_type, key])] = val
            meta_dict.update(named_data)
        meta_dict.update({'group': self.group})
        return meta_dict


class FixedCourtshipTrackingSummary(TrackingSummary):
    def __init__(self):
        super(FixedCourtshipTrackingSummary, self).__init__()
        self.male = None
        self.female = None

    def __str__(self):
        class_str = ''
        return class_str + super(FixedCourtshipTrackingSummary, self).__str__()
