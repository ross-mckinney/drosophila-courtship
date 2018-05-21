# -*- coding: utf-8 -*-

"""
.. module:: courtship
   :synopsis: Classes for Fly objects.

.. moduleauthor:: Ross McKinney
"""
import numpy as np
import pandas as pd


class Point(object):
    """Stores 2D coordinates as attributes.

    Attributes
    ----------
    row : np.ndarray or None (default = None)
        Array to store row-coordinates of point

    col : np.ndarray or None (default = None)
        Array to store col-coordinates of point
    """
    def __init__(self):
        self.row = None
        self.col = None

    def coords(self):
        """Stacks row and col coordinates into an [N, 2] np.ndarray.
        
        Returns
        -------
        coords : np.ndarray of shape [N, 2].
            First column contains row-coordinates. Second column contains
            col-coordinates.
        """
        return np.vstack((self.row, self.col)).T


class Ellipse(object):
    """Base class that will be used to describe body and wing positions.

    Attributes
    ----------
    centroid : Point object
        Contains centroid coordinates for Ellipse.

    minor_axis_length : np.ndarray or None (default = None)
        Contains minor axis length of Ellipse.

    major_axis_length : np.ndarray or None (default = None)
        Contains major axis length of Ellipse.

    orientation : np.ndarray or None (default = None)
        Contains angle (from -np.pi/2 to np.pi/2) major_axis_length
        of Ellipse makes with Cartesian x-axis.
    """
    def __init__(self):
        self.centroid = Point()
        self.major_axis_length = None
        self.minor_axis_length = None
        self.orientation = None

    def init_params(self, n):
        """Initilizes space to hold ellipse data.

        All parameters will be initilized as np.zeros(n).

        .. warning:: Any values held within an attribute will
           be overriden.

        Parameters
        ----------
        n : int
            Length of array to initilize for all parameters.
        """
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], Point):
                self.__dict__[key].row = np.zeros(n)
                self.__dict__[key].col = np.zeros(n)
            else:
                self.__dict__[key] = np.zeros(n)

    def _dig(self, d, all_params):
        """Recursively digs into objects/dictionaries to return all parameters.

        Parameters
        ----------
        d : dictionary
            Highest level parameter dictionary. Initialize as self.__dict__
            to generate a full - deep - parameter list.
        all_params : dictionary
            Initialize as empty. This will be the final dictionary containing
            all parameters.
        """
        for k, v in d.iteritems():
            if isinstance(v, Point):
                all_params[k] = self._dig(v.__dict__, dict())
            else:
                all_params[k] = v

        return all_params

    def _combine_keys(self, k, d):
        """Adds string, k, to all keys within dictionary, d.

        Parameters
        ----------
        k : string
            String to add to all keys within d.
        d : dictionary
            Dicitionary
        """
        nd = dict()
        for key, val in d.iteritems():
            nd[k + '_' + key] = val
        return nd

    def get_params(self, return_dict=True):
        """Gets all parameters as a dictionary.

        Parameters
        ----------
        return_dict : bool (default = True)
            Whether to return a dictionary or a pandas DataFrame
            containing Ellipse data.
        """
        params = self._dig(self.__dict__, dict())
        if return_dict:
            return params

        df = pd.DataFrame()
        for k1, v1 in params.iteritems():
            if isinstance(v1, np.ndarray):
                df[k1] = v1
            if isinstance(v1, dict):
                for k2, v2 in self._combine_keys(k1, v1).iteritems():
                    df[k2] = v2

        # return DataFrame where columns have been sorted alphabetically.
        return df.reindex(sorted(df.columns), axis=1)


class Body(Ellipse):
    """The body is a distinct type of ellipse with directionality.

    Attributes
    ----------
    rotation_angle : np.ndarray or None (default = None)
        Angle (from 0 to 2*np.pi) needed to rotate Ellipse such that
        the ellipse is oriented with the rear-to-head axis pointing
        to the right along the Cartesian x-axis.

    head : Point object
        Coordinates of head of ellipse.

    rear : Point object
        Coordinates of rear of ellipse.
    """
    def __init__(self):
        Ellipse.__init__(self)
        self.rotation_angle = None
        self.head = Point()
        self.rear = Point()


class Wing(Ellipse):
    """Instance of Ellipse."""
    def __init__(self):
        Ellipse.__init__(self)


class Fly(object):
    """Class used to keep track of features during tracking.

    Attributes
    ----------
    body : Body object
        Ellipse fitted to body of fly (excludes wings).

    left_wing : Wing object
        Ellipse fitted to left wing of fly.

    right_wing : Wing object
        Ellipse fitted to right wing of fly.
    """
    def __init__(self):
        """A fly is composed of three ellipses fitted to (1) the body,
        (2) the right wing, and (3) the left wing.
        """
        self.body = Body()
        self.right_wing = Wing()
        self.left_wing = Wing()

    def init_params(self, n):
        """Initilizes space for all parameters.

        .. warning:: Any values held within an attribute will
           be overriden. Therefore, only call this function during the
           initilization of a Fly object.

        Parameters
        ----------
        n : int
            Number of frames to initilize space for each of the
            following parameters:
                body, left_wing and right_wing.
        """
        self.body.init_params(n)
        self.left_wing.init_params(n)
        self.right_wing.init_params(n)

    def from_csv(self, csv_file):
        """Allows the creation of a Fly from a csv file.

        .. note:: see Fly.to_df for a list of required column names.

        Parameters
        ----------
        csv_file : string
            Path to file containing fly data.
        """
        df = pd.read_csv(csv_file)
        fly = self.from_df(df)
        return fly

    @classmethod
    def from_df(cls, df):
        """Generates a fly object from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame object
            DataFrameta_dict.update()me from which to generate Fly object.

        Returns
        -------
        fly : Fly object
            Fly contained within DataFrame.
        """
        fly = cls()

        for colname in df.columns.values.tolist():
            col_id = colname.split('_')

            if col_id[0] == 'body':
                if 'row' not in col_id and 'col' not in col_id:
                    setattr(
                        fly.body,
                        '_'.join(col_id[1:]),
                        df[colname].values
                    )
                else:
                    if 'centroid' in col_id:
                        setattr(
                            fly.body.centroid,
                            col_id[-1],
                            df[colname].values
                        )
                    elif col_id[1] == 'head':
                        setattr(
                            fly.body.head,
                            col_id[-1],
                            df[colname].values
                        )
                    else:
                        setattr(fly.body.rear, col_id[-1], df[colname].values)
            elif col_id[0] == 'left':
                if 'row' not in col_id[-1] and 'col' not in col_id[-1]:
                    setattr(
                        fly.left_wing,
                        '_'.join(col_id[1:]),
                        df[colname].values
                    )
                else:
                    setattr(
                        fly.left_wing.centroid,
                        col_id[-1],
                        df[colname].values
                    )
            elif col_id[0] == 'right':
                if 'row' not in col_id and 'col' not in col_id:
                    setattr(
                        fly.right_wing,
                        '_'.join(col_id[1:]),
                        df[colname].values
                    )
                else:
                    setattr(
                        fly.right_wing.centroid,
                        col_id[-1],
                        df[colname].values
                    )

        return fly

    def to_csv(self, csv_file):
        """Save Fly in .csv format.

        Parameters
        ----------
        csv_file : string
            File path to save Fly.
        """

        df = self.to_df()
        df.to_csv(csv_file, index=False)

    def to_df(self):
        """Returns a pandas.DataFrame object containing all information
        about this Fly.

        .. note:: Columns will have the following names:
           1. body_centroid_col
           2. body_centroid_row
           3. body_head_col
           4. body_head_row
           5. body_major_axis_length
           6. body_minor_axis_length
           7. body_orientation
           8. body_rear_col
           9. body_rear_row
           10. body_rotation_angle
           11. left_centroid_col
           12. left_centroid_row
           13. left_major_axis_length
           14. left_minor_axis_length
           15. left_orientation
           16. right_centroid_col
           17. right_centroid_row
           18. right_major_axis_length
           19. right_minor_axis_length
           20. right_orientation

           These names correspond to the ellipse fitted to the body,
           left wing, and right wing of the fly.

        Returns
        -------
        df : pandas.DataFrame object
        """
        # get parameters for all instances of Ellipse as DataFrames, and
        # append a string descriptor to the head of each column in each
        # DataFrame.
        body_params = self.body.get_params(return_dict=False)
        body_params.columns = [
            'body_' + c_name for c_name in body_params.columns.values]

        left_wing_params = self.left_wing.get_params(return_dict=False)
        left_wing_params.columns = [
            'left_' + c_name for c_name in left_wing_params.columns.values
        ]

        right_wing_params = self.right_wing.get_params(return_dict=False)
        right_wing_params.columns = [
            'right_' + c_name for c_name in right_wing_params.columns.values
        ]

        df = pd.concat([
            body_params, left_wing_params, right_wing_params], axis=1)

        return df
