# test_experiment.py

from context import (
    experiment
)

if __name__ == '__main__':
    exp = experiment.FixedCourtshipTrackingExperiment.load_from_fcts(
        ['D:/lc-tnt-control/fcts2'],
        groups={'control': ['tnt-minus'], 'experimental': ['tnt-plus']},
        order=['control', 'experimental']
    )
    exp.add_behavior_from_csv(
        behavior_name='courtship_gt',
        csv_filename='D:/lc-tnt-control/data/hand-scored-courtship.csv')

    # exp.get_ang_location_summary('scissoring')
    # exp.get_ang_rvals('scissoring')
    # exp.get_ang_rayleighvals('scissoring')
    # exp.get_ang_watsonwilliams('scissoring')

    # exp.get_behavioral_distances('scissoring')
    # exp.get_behavioral_distances_peak_ratio('scissoring')
    # exp.get_behavioral_indices('scissoring')
    # exp.get_behavioral_index_as_fraction_of_courtship(
    #     behavior_name='scissoring',
    #     courtship_behavior_name='courtship_gt')
    # exp.get_behavioral_matrices('courtship_gt')
    # exp.get_behavioral_latencies('scissoring')

    # exp.get_binned_forward_velocities('courtship_gt')
    # exp.get_binned_sideways_velocities('courtship_gt')
    # exp.get_binned_abs_sideways_velocities('courtship_gt')

    # exp.save_behavioral_data(
    #     'test_summary.csv',
    #     {'tap': 'tapping', 'sci': 'scissoring', 'ori': 'orienting'},
    #     'courtship_gt')

    exp.save_behavioral_matrices(
        savename='test_matrices.csv', behavior_name='courtship_gt')