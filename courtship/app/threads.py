# threads.py
#

import time
import pickle
from datetime import datetime

import numpy as np

from PyQt5.QtCore import QThread

from .. import __version__
from ..fly import Fly
from ..ts import FixedCourtshipTrackingSummary
from .tracking import *


class TrackingThread(QThread):
    """Worker thread to run tracking algorithm.
    
    Parameters
    ----------
    video_settings : list of TrackingSettings
        Each TrackingSettings object should be completely filled before 
        creating an instance of this object, and running this thread.

    logger : QTextEdit
        To store information about video being currently tracked.
        This may need to be changed to prevent "QObject::connect: Cannot
        queue arguments of type 'QTextBlock' from being displayed.
    """
    def __init__(self, video_settings, logger, progress, parent=None):
        super(TrackingThread, self).__init__(parent)
        self.video_settings = video_settings
        self.logger = logger
        self.tracking_progress = progress
    
    def run(self):
        for ix in xrange(len(self.video_settings)):
            start_time = time.time()

            settings = self.video_settings[ix]
            video = settings.video
            n_frames = video.get_n_frames()
            timestamps = video.get_all_timestamps()
            fps = (1. / np.mean(np.diff(timestamps)))

            male = Fly()
            female = Fly()
            tracking_summary = FixedCourtshipTrackingSummary()

            male.init_params(n_frames)
            female.init_params(n_frames)
            male.timestamps = timestamps
            female.timestamps = timestamps

            # load video attributes into FixedCourtshipTrackingSummary
            tracking_summary.video.filename = settings.video_file
            tracking_summary.video.timestamps = timestamps
            tracking_summary.video.fps = fps
            tracking_summary.video.duration_frames = n_frames
            tracking_summary.video.duration_seconds = (n_frames * 1.) / fps
            tracking_summary.video.start_time = datetime.fromtimestamp(
                timestamps[0]).strftime('%Y-%m-%d %H:%M:%S')
            tracking_summary.video.end_time = datetime.fromtimestamp(
                timestamps[-1]).strftime('%Y-%m-%d %H:%M:%S')
            tracking_summary.video.pixels_per_mm = settings.arena.pixels_to_mm

            # load arena attributes into FixedCourtshipTrackingSummary
            tracking_summary.arena.shape = 'circular'
            tracking_summary.arena.center_pixel_rr = settings.arena.center[0]
            tracking_summary.arena.center_pixel_cc = settings.arena.center[1]
            tracking_summary.arena.radius_mm = settings.arena.radius
            tracking_summary.arena.diameter_mm = 2 * settings.arena.radius

            # load software attributes into FixedCourtshipTrackingSummary
            tracking_summary.software.tight_threshold = settings.tight_threshold
            tracking_summary.software.loose_threshold = settings.loose_threshold
            tracking_summary.software.date_tracked = datetime.today(
                ).strftime('%Y-%m-%d %H:%M:%S')
            tracking_summary.software.version = __version__

            # set the `group` attribute for the FixedCourtshipTrackingSummary
            tracking_summary.group = settings.group

            self.logger.append(
                'Tracking started for video: {} \nStart Time: {}'.format(
                    settings.video_file,
                    time.strftime('%H:%M:%S', time.localtime(start_time))
                ))

            # get the location and region properties that define
            # the fixed female.
            f_props, f_head, f_rear = find_female(
                    image=settings.arena.background_image,
                    female=settings.female,
                    lp_threshold=settings.tight_threshold
                )

            # update female based on props we just found --
            # this ensures that the ellipse used to mask the female is
            # not biased by variation in user-defined ellipses.
            tighten_female_ellipse(
                    female=settings.female,
                    female_props=f_props
                )

            # loop through each frame in the video, and find the male.
            for frame_ix in xrange(n_frames):
                frame_ix = long(frame_ix)
                frame, ts = video.get_frame(frame_ix)

                try:
                    male_props = find_male(
                            image=frame,
                            female=settings.female,
                            arena=settings.arena,
                            lp_threshold=settings.tight_threshold
                            )
                except NoPropsDetected as NPD:
                    self.logger.append(
                        '\t' + NPD.message +
                        ' Body @ frame {}'.format(frame_ix)
                        )
                    continue

                wing_props = find_wings(
                        image=frame,
                        female=settings.female,
                        arena=settings.arena,
                        male_props=male_props,
                        loose_threshold=settings.loose_threshold,
                        logger=self.logger,
                        frame_ix=frame_ix
                    )
                # if wing_props is None:
                #     # male.timestamps[frame_ix] = ts
                #     # female.timestamps[frame_ix] = ts
                #     continue

                male.body.centroid.row[frame_ix] = male_props.centroid[0]
                male.body.centroid.col[frame_ix] = male_props.centroid[1]
                male.body.orientation[frame_ix] = male_props.orientation
                set_male_props(male, wing_props, frame_ix)

                set_female_props(female, f_props, f_head, f_rear, frame_ix)

                percent_complete = (frame_ix + 1.) / n_frames * 100
                self.tracking_progress.emit(
                    percent_complete,
                    'Tracking video {}/{}.'.format(
                        ix + 1, len(self.video_settings)))

            # update the tracking settings dictionary with male and female items.
            # tracking_settings.update({'male': male, 'female': female})
            # tracking_summary.set_attributes(**tracking_settings)
            tracking_summary.male = male
            tracking_summary.female = female

            save_file = settings.save_file
            save_type = save_file.split('.')[-1]
            if save_type == 'xlsx':
                tracking_summary.to_xlsx(save_file)
            elif save_type == 'fcts':
                with open(save_file, 'wb') as SAVE:
                    pickle.dump(tracking_summary, SAVE)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_hrs = int(elapsed_time / 3600)
            time_mins = int((elapsed_time - time_hrs * 3600) / 60)
            time_secs = int(elapsed_time - time_hrs * 3600 - time_mins * 60)

            self.logger.append(
                'End Time: {}\nTotal Time Elapse: {}'.format(
                    time.strftime('%H:%M:%S', time.localtime(end_time)),
                    '{:02d}:{:02d}:{:02d}'.format(
                        time_hrs, time_mins, time_secs))
                )
        self.logger.append('TRACKING COMPLETE')