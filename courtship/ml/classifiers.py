# -*- coding: utf-8 -*-

"""
.. module:: ml
   :synopsis: Contains class for generating classifiers.

.. moduleauthor:: Ross McKinney
"""
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc


def from_csv(csv_file):
    """Reads in training/testing/validation data from a .csv file.

    .. note:: Classified frames from all files will be concatenated
    into a single feature matrix and classification array.

    Parameters
    ----------
    csv_file : string
        Path to a .csv file to read.
        This file should have the following column names:
        1. feature_matrix_file
        2. start
        3. stop
        4. classification
        5. behavior_name

    Returns
    -------
    X : np.ndarray | shape = [n_frames, n_features]
        Feature matrix for all classified frames in csv_file.

    y : np.ndarray | shape = [n_frames]
        Classification array (0 or 1) for all classified frames
        in csv_file.
    """

    # load csv into pandas DataFrame
    df = pd.read_csv(csv_file)

    # loop through all unique feature_matrix_files
    for i, unique_file in enumerate(df['feature_matrix_file'].unique()):
        subset = df[df['feature_matrix_file'] == unique_file]

        start = subset['start'].values
        stop = subset['stop'].values
        classifications = subset['classification'].values

        for j in range(len(start)):
            if j == 0:
                ix = np.arange(start[j], stop[j] + 1)
                y = np.repeat(classifications[j], stop[j] - start[j] + 1)
            else:
                ix = np.hstack((ix, np.arange(start[j], stop[j] + 1)))
                y = np.hstack(
                    (y, np.repeat(classifications[j], stop[j] - start[j] + 1))
                )

        with open(unique_file, 'rb') as f:
            fmat = pickle.load(f)

        X = fmat.get_X()
        if i == 0:
            all_X = X[ix, :]
            all_y = y
        else:
            all_X = np.vstack((X[ix, :], all_X))
            all_y = np.hstack((y, all_y))

    return all_X, all_y


def list_from_csv(csv_file):
    """Reads a csv file and returns all {X_train, y_train} as a list of dicts.

    Parameters
    ----------
    csv_file : string
        Path to a .csv file to read.
        This file should have the following column names:
        1. feature_matrix_file
        2. start
        3. stop
        4. classification
        5. behavior_name

    Returns
    -------
    [{'X': X, 'y': y}, ...] : list of dictionaries
        Contains all X and y for each unique feature_matrix_file
        in csv_file.
    """

    # load csv into pandas DataFrame
    # and drop any rows that have np.nan in them
    df = pd.read_csv(csv_file).dropna(how='all')

    all_data = []

    # loop through all unique feature_matrix_files
    for i, unique_file in enumerate(df['feature_matrix_file'].unique()):
        subset = df[df['feature_matrix_file'] == unique_file]

        start = subset['start'].values
        stop = subset['stop'].values
        classifications = subset['classification'].values

        for j in range(len(start)):
            if j == 0:
                ix = np.arange(start[j], stop[j] + 1)
                y = np.repeat(classifications[j], stop[j] - start[j] + 1)
            else:
                ix = np.hstack((ix, np.arange(start[j], stop[j] + 1)))
                y = np.hstack(
                    (y, np.repeat(classifications[j], stop[j] - start[j] + 1))
                    )

        ix = ix.astype(np.int)
        print 'loading: ', type(unique_file), unique_file

        with open(unique_file, 'rb') as f:
            fmat = pickle.load(f)

        X = fmat.get_X()
        all_data.append({'X': X[ix, :], 'y': y})

    return all_data


class Classifier(AdaBoostClassifier):
    """Adaboost classifier with convenience functions.

    Classifier inherits from sklearn.ensemble.AdaBoostClassifier,
    and calls super with the following parameters:

    super(
        sklearn.tree.DecisionTreeClassifier(
            max_depth = 2,
            min_samples_leaf = 1
            ),
        algorithm = 'SAMME',
        n_estimators = 100
    )

    Parameters
    ----------
    behavior_name : string
        Name of behavior that classifier is being used for.

    Attributes
    ----------
    training_data : list of dictionaries
        Each dictionary should be organized as follows, and
        represents ground truth data from one video/fly pair:
        {
        'X': np.ndarray | shape = [n_frames, n_features],
        'y': np.ndarray | shape = [n_frames]
        }

    validations : list of dictionaries
        Contains information about any cross validaition
        made with data in self.training_data. Keys are as
        follows: 'accuracy', 'fpr', 'tpr', 'auc', 'predicted_classifications',
        'true_classifications'.
    """
    def __init__(self, behavior_name):
        dt_stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
        super(Classifier, self).__init__(
            dt_stump,
            algorithm="SAMME",
            n_estimators=100
        )
        self.behavior_name = behavior_name
        self.training_data = None
        self.validations = []

    def load_training_data(self, training_file):
        """Load training data from file.

        This will set the attribute self.training_data
        to be a list containing dictionaries as entries.
        [{'X': X, 'y': y}, ...].

        Parameters
        ----------
        training_file : string
            Path to .csv file containing training data.
        """
        self.training_data = list_from_csv(training_file)

    def get_training_data(self):
        """Combines all training data into a single feature matrix, X,
        and classification array, y.

        Returns
        -------
        X : np.ndarray | shape = [n_frames, n_features]
            Feature matrix generated from all X in self.training_data.

        y : np.ndarray | shape = [n_frames]
            Classification array (0 or 1) generated from all y in
            self.training_data.
        """
        if self.training_data is None:
            print "Need to set/load training data first."
            return
        else:
            for i in xrange(len(self.training_data)):
                if i == 0:
                    X = self.training_data[i]['X']
                    y = self.training_data[i]['y']
                else:
                    X = np.vstack((self.training_data[i]['X'], X))
                    y = np.hstack((self.training_data[i]['y'], y))

        return X, y

    def leave_one_out(self, chunk_to_leave_out):
        """Splits data into two training/testing sets for leave-one-out
        cross validation.

        Parameters
        ----------
        chunk_to_leave_out : int
            Which item within self.training_data to leave out. This item will
            be used as testing data. Note that chunk_to_leave_out must be less
            than len(self.training_data).

        Returns
        -------
        X_train : np.ndarray | shape = [n_frames, n_features]
            Training feature matrix containing all data within
            self.training_data other than that specified by chunk_to_leave_out.

        y_train : np.ndarray | shape = [n_frames]
            Training classification array containing all data within
            self.training_data other than that specified by chunk_to_leave_out.

        X_test : np.ndarray | shape = [m_frames, m_features]
            Testing feature matrix. This should be the feature matrix contained
            within self.training_data at index chunk_to_leave_out.

        y_test : np.ndarray | shape = [m_frames]
            Testing classification array. This should be the classification
            array contained within self.training_data at index
            chunk_to_leave_out.
        """

        if self.training_data is None:
            print "Need to set/load training data first."
            return

        if chunk_to_leave_out >= len(self.training_data):
            print (
                "Error in Classifier.Classifier.leave_one_out: \n" +
                "chunk_to_leave_out > len(self.training_data)"
            )
            return

        X_train = np.zeros(1)
        y_train = np.zeros(1)

        for i in xrange(len(self.training_data)):
            if i == chunk_to_leave_out:
                X_test = self.training_data[i]['X']
                y_test = self.training_data[i]['y']
            else:
                if X_train.size == 1:
                    X_train = self.training_data[i]['X']
                    y_train = self.training_data[i]['y']
                else:
                    X_train = np.vstack((self.training_data[i]['X'], X_train))
                    y_train = np.hstack((self.training_data[i]['y'], y_train))

        return X_train, y_train, X_test, y_test

    def cross_validate(self):
        """Performs leave-one-out cross validation using the training
        data contained within self.training_data.

        Updates self.validations with new validation data after each
        iteration.
        """

        if self.training_data is None:
            print "Training data has not been set. Aborting."
            return

        dt_stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
        classifier = AdaBoostClassifier(
            dt_stump,
            algorithm="SAMME",
            n_estimators=100
        )

        for i in xrange(len(self.training_data)):
            X_train, y_train, X_test, y_test = self.leave_one_out(i)
            classifier.fit(X_train, y_train)

            proba = classifier.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])

            self.validations.append({
                'accuracy': classifier.score(X_test, y_test),
                'fpr': fpr,
                'tpr': tpr,
                'auc': np.round(auc(fpr, tpr), decimals=2),
                'predicted_classifications': classifier.predict(X_test),
                'true_classifications': y_test
                })

            print '\tAccuracy: {}'.format(
                np.round(self.validations[i]['accuracy'], 2)
            )
            print '\tAuc: {}'.format(
                self.validations[i]['auc']
            )
            sys.stdout.flush()

    def plot_cross_validation(self, ax, colors=['k', 'm']):
        """Generates plots to see how well the classifier is working.

        Parameters
        ----------
        ax : matplotlib.pyplot axis object
            Plots will be generated within this axis handle.

        colors : list of string, or list of tuple
            Colors to use for ground truth and predicted bars
            in eventplot.
        """

        true_where = []
        predicted_where = []

        for val in self.validations:
            true_where.append(
                np.where(val['true_classifications'])[0]
            )
            predicted_where.append(
                np.where(val['predicted_classifications'])[0]
            )

        true_plot_positions = np.arange(0, len(true_where)) * 3
        predicted_plot_positions = np.arange(0, len(true_where)) * 3 + 1

        ax.eventplot(
            true_where,
            lineoffsets=true_plot_positions,
            colors=colors[0]
        )
        ax.eventplot(
            predicted_where,
            lineoffsets=predicted_plot_positions,
            colors=colors[1]
        )
