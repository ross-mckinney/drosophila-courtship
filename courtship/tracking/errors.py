# -*- coding: utf-8 -*-

class ZeroRegionProps(object):
    def __init__(self):
        self.major_axis_length = 0
        self.minor_axis_length = 0
        self.orientation = 0
        self.area = 0
        self.centroid = (0, 0)

class TrackingError(Exception):
    """Base class for all errors that could occur during tracking."""
    pass

class NoPropsDetected(TrackingError):
    """Raised if no region properties were detected for wings in a binary frame.

    Parameters
    ----------
    message : string
        What went wrong.

    Attributes
    ----------
    props : ZeroRegionProps object (from this module)
        Mock skimage.measure.RegionProps object with attributes set to 0.
    """

    def __init__(self, message):
        super(NoPropsDetected, self).__init__()
        self.message = message
        self.props = ZeroRegionProps()

