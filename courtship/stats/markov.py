# -*- coding: utf-8 -*-

import string
import numpy as np


def sort_and_merge(arr1, arr2, arr1_labels, arr2_label=1):
    """Sorts and merges two arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        Array to insert values into.

    arr2 : np.ndarray
        Array containing values to insert into arr1.

    arr1_labels: np.ndarray
        Array containing labels for arr1.

    arr2_label : int
        New label, given to arr2, to insert into arr1_labels.

    Returns
    -------
    merged_arr : np.ndarray
        Sorted merger of arr1 and arr2.

    labeled_arr : np.ndarray
        Labeled array that has been sorted the same as merged_arr.
    """
    ixs = np.searchsorted(arr1, arr2)
    arr2_labels = np.ones(arr2.size) * arr2_label

    merged_arr = np.insert(arr1.copy(), ixs, arr2)
    labeled_arr = np.insert(arr1_labels, ixs, arr2_labels)

    return merged_arr, labeled_arr


def get_transition_probas(summary, *states):
    """Finds the probabilities of transitioning between behavioral states.

    Parameters
    ----------
    summary : FixedCourtshipTrackingSummary object.

    *states : list of string
        Each member of *states should be a valid behavior name in `summary`.

    Returns
    -------
    probas : np.ndarray of shape [len(*states), len(*states)]
        Probabilities of transitioning between behavioral states for
        an animal. This is the Markov transition matrix.

        .. note:: np.nan is returned if there were no observed behaviors.
    """

    for i, state in enumerate(*states):
        state_arr_boundary = summary.male.get_behavior(state).ixs()
        state_boundary_midpoint = np.mean(state_arr_boundary, axis=1)

        if i == 0:
            final_state_sorted = state_boundary_midpoint
            state_labels = np.zeros(state_boundary_midpoint.size)
            continue

        final_state_sorted, state_labels = sort_and_merge(
            final_state_sorted,
            state_boundary_midpoint,
            state_labels,
            i
        )

    n_states = len(*states)
    probas = np.zeros(shape=(n_states, n_states))
    for r in xrange(n_states):
        n_transitions = []
        for c in xrange(n_states):
            init_state = np.where(state_labels == r)[0]
            final_state = c
            n = 0
            for init in init_state:
                if init == state_labels.size - 1:
                    break
                if state_labels[init + 1] == final_state:
                    n += 1

            n_transitions.append(n)
        probas[r, :] = np.asarray(n_transitions) / (1. * np.sum(n_transitions))

    return probas


def get_transition_matrix(exp, group_name, bnames):
    """Gets a markov transition matrix for a given FixedCourtshipExperiment
    and list of behavioral states.

    Parameters
    ----------
    exp : FixedCourtshipExperiment

    group_name : string
        Name of group to get transition matrix for.

    bnames : list of string of shape [N_b]
        Each behavior name must be a valid behavior key in each TrackingSummary
        present in the passed `exp`.

    Returns
    -------
    transition_matrix : np.ndarray of shape [N_b, N_b, len(group)]
        The matrix represents the transitional probability from behaviors
        along the rows to behaviors along the columns. Each matrix within
        the returned transition matrix (that is, all matrices along dimension
        2) represent the transition matrix for a single TrackingSummary within
        specified experimental group. Therefore, to get the mean transitional
        probabilities for a group, just take the mean along dimension 2.
    """
    group = getattr(exp, group_name)

    N_b = len(bnames)
    N_ts = len(group)

    tm = np.zeros(shape=(N_b, N_b, N_ts))

    for i, ts in enumerate(group):
        probas = get_transition_probas(ts, bnames)
        tm[:, :, i] = probas

    return tm


def print_transition_matrix(p, labels=['Tap', 'Ori', 'Sci'], round_digits=3):
    """Prints a more nicely formatted version of a markov transition table.

    Parameters
    ----------
    p : np.ndarray of shape [N, N]
        This should be a square matrix containing the probability of
        transitioning between behaviors. This matrix shows transitions from
        behaviors along the rows to behaviors along the columns. This is a
        markov transition matrix.

    labels : list of string of length [N]
        The behavior corresponding to each row/column index in `p`.

    round_digits : int, optional (default=3)
        How many digits should the transitional probabilities be rounded to.
    """
    spacing = 8
    print ' ' * (spacing + 1),
    print string.center('Transitioning To', spacing * len(labels))

    print ' ' * (spacing + 1),
    for l in labels:
        print string.center(l, spacing),

    print
    print ' ' * (spacing + 1),
    for i in xrange(len(labels)):
        print string.center('-----', spacing),

    print
    for i, row in enumerate(p):
        for j, col in enumerate(row):
            if j == 0:
                print string.center(labels[i] + ' | ', spacing),
            print string.center('{}'.format(np.round(col, round_digits)), spacing),
        print


if __name__ == '__main__':
    pass
