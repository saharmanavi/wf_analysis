import pandas as pd
import numpy as np
import six
from scipy.interpolate import interp1d

from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator import foraging
from imaging_behavior.core.utilities import get_nearest_time


def choose_date(func):
    '''decorator to allow functions in the class object that require a specific date to automatically pull the date specified when 
    creating the class object or to specify a different date when running the individual function. 
    '''
    def date_wrapper(self, date=None, *args, **kwargs):
        if isinstance(date, six.string_types):
            return func(self, date, *args, **kwargs)
        else:
            date = self.date
            return func(self, date, *args, **kwargs)
    return date_wrapper

def date_and_objects(func):
    '''similar to choose_date decorator, except this allows for flexible creation of path and session manifest objects where needed'''
    def date_object_wrapper(self, date=None, path=None, session_manifest=None, *args, **kwargs):
        if isinstance(date, six.string_types):
            path, session_manifest = self.check_session_files(date)
            return func(self, date, path, session_manifest, *args, **kwargs)
        else:
            date = self.date
            path = self.path
            session_manifest = self.session_manifest
            return func(self, date, path, session_manifest, *args, **kwargs)
    return date_object_wrapper


def beh_pkl_path_to_df(path):
    data = pd.read_pickle(path)
    time_arr = np.hstack((0, np.cumsum(data['vsyncintervals']) / 1000.))
    core_data = foraging.data_to_change_detection_core(data, time_arr)       
    df = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks = core_data['licks'],
        time=core_data['time'])       
    return df

def get_jcam_index(times,visual_frame):
    """
    Given a dictionary of times from the session object and an input visual frame, this will return the nearest jcam frame
    modified from imaging_behavior.core.utilities
    """
    #use the "frame display times" to account for display lag
    time_values = times['frame_display_times']
    visual_frame_numbers = np.arange(0,len(times['frame_display_times'])) 
    f = interp1d(visual_frame_numbers,time_values,bounds_error=False)   
    time_in_timeline = f(visual_frame)   
    return get_nearest_time(times['fluorescence_read_times'],time_in_timeline)


def add_jcam_to_df(df_in,session):
    """
    adds column to dataframe that contains correspoding jcam frames for trial start and stimulus display
    """
    df_in['start_frame_jcam'] = None
    df_in['stim_frame_jcam'] = None
    for idx in df_in.index:
        # find the nearest jcam index for every trial start
        df_in.loc[idx,'start_frame_jcam'] = get_jcam_index(session.timeline.times,df_in.iloc[idx].startframe)

         #If a change occured, find the nearest jcam index
        if pd.isnull(df_in.loc[idx,'change_frame'])==False:
            change_frame = df_in.iloc[idx]['change_frame']
            df_in.loc[idx,'stim_frame_jcam'] = get_jcam_index(session.timeline.times,change_frame)
    return df_in


