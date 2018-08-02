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

    exp.get_ang_location_summary('scissoring')
    exp.get_ang_rvals('scissoring')
    exp.get_ang_rayleighvals('scissoring')
    exp.get_ang_watsonwilliams('scissoring')
    exp.get_behavioral_distances('scissoring')