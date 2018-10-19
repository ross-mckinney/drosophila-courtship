# -*- coding: utf-8 -*-

"""
.. module:: courtship
   :synopsis: Class to hold tracking summaries (ts). 

.. moduleauthor:: Ross McKinney
"""
import h5py
import numpy as np
import pandas as pd

from courtship.fly import Fly

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

    def to_xlsx(self, savename):
        """Saves this TrackingSummary as an .xlsx file.

        The saved .xlsx file will have the following sheet names:
            1. meta
            2. male
            3. female

        Parameters
        ----------
        savename : string
            Valid file path for saving this TrackingSummary.
        """
        meta_data = self.meta_data()
        timestamps = meta_data.pop('video.timestamps')

        meta_df = pd.DataFrame(meta_data, index=[0])
        male_df = self.male.to_df()
        female_df = self.female.to_df()

        writer = pd.ExcelWriter(savename, engine='xlsxwriter')
        meta_df.to_excel(writer, sheet_name='meta', index=False)
        male_df.to_excel(writer, sheet_name='male', index=False)
        female_df.to_excel(writer, sheet_name='female', index=False)

        writer.save()

    def _from_dfs(self, meta_df, male_df, female_df):
        """Loads all attributes into this object from DataFrames.

        Parameters
        ----------
        meta_df : pd.DataFrame
            Contains meta data to add to `fcts`.

        male_df : pd.DataFrame
            Contains male data to add to `fcts`.

        female_df : pd.DataFrame
            Contains female data to add to `fcts`.
        """
        self.male = Fly.from_df(male_df)
        self.female = Fly.from_df(female_df)
        self.video.timestamps = self.male.timestamps

        # set all meta data
        for colname in meta_df.columns:
            parsed_name = colname.split('.')
            if parsed_name[0] == 'arena':
                if parsed_name[-1] == 'vertices':
                    setattr(
                        self.arena, 
                        parsed_name[-1], 
                        meta_df[colname].values.tolist()
                        )
                else:
                    setattr(self.arena, parsed_name[-1], meta_df[colname][0])
            elif parsed_name[0] == 'video':
                setattr(self.video, parsed_name[-1], meta_df[colname][0])
            elif parsed_name[0] == 'software':
                setattr(self.software, parsed_name[-1], meta_df[colname][0])
            elif parsed_name[0] == 'group':
                self.group = meta_df[colname][0]

    @classmethod
    def from_xlsx(cls, filename):
        """Opens a .xlsx file and loads all attributes from it.

        Parameters
        ----------
        filename : string
            Valid path to .xslx file containing data.

        Returns
        -------
        fcst : FixedCourtshipTrackingSummary
        """
        meta_df = pd.read_excel(filename, sheet_name='meta')
        male_df = pd.read_excel(filename, sheet_name='male')
        female_df = pd.read_excel(filename, sheet_name='female')

        fcts = cls()
        fcts._from_dfs(meta_df, male_df, female_df)
        return fcts

    @classmethod
    def from_hdf5(cls, filename):
        """Opens a .hdf5 file and generates a FixedCourtshipTrackingSummary.

        Parameters
        ----------
        filename : string
            Path to valid .hdf5 file.

        Returns
        -------
        fcts : FixedCourtshipTrackingSummary
        """
        meta_df = pd.DataFrame()
        with h5py.File(filename, 'r') as f:
            # load meta values (these are stored as attributes in `f`)
            for key, val in f['meta'].attrs.iteritems():
                meta_df[key] = pd.Series([val])

            meta_df.reindex_axis(sorted(meta_df.columns), axis=1)
            male_df = pd.DataFrame(
                f['male/df'][()],
                columns=f['male/df'].attrs['columns']
                )
            female_df = pd.DataFrame(
                f['female/df'][()],
                columns=f['female/df'].attrs['columns']
            )

        fcts = cls()
        fcts._from_dfs(meta_df, male_df, female_df)
        return fcts