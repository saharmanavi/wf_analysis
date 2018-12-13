
import os
import json
import h5py
import pandas as pd

from IPython.display import Image
from imaging_behavior import load_session_from_manifest
from imaging_behavior.core.slicer import BinarySlicer
from WF_utilities import choose_date, date_and_objects, beh_pkl_path_to_df, add_jcam_to_df, get_jcam_index
from visual_behavior.translator.foraging import load_running_speed

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


class FancyDataPackage(object):
	def __init__(self, mouse_id, xfer_dir, start_dir='default', date=None):
		if start_dir=='default':
			self.start_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData"
		else:
			self.start_dir = start_dir
		self.mouse_id = mouse_id
		self.xfer_dir = xfer_dir
		self.generate_manifest()
		if date == None:
			self.dates = self.date_list
		else:
			self.dates = list(date)

	def generate_manifest(self):
		folder_list = []
		for d in os.listdir(self.main_dir):
			if (self.mouse_id.lower() in d.lower()) and ('.json' not in d.lower()):
				folder_list.append(d)
		manifest_dict = {}
		for f in folder_list:
			if '-' in f:
				key = f.split('-')[0]
			elif '_' in f:
				key = f.split('_')[0]
			manifest_dict[key] = f
		date_list = [k for k in manifest_dict.keys()]
		self.manifest_dict = manifest_dict
		self.date_list = date_list		
		
		return self.manifest_dict, self.date_list

	def get_session_path(self, date):
		path = os.path.join(self.start_dir, self.manifest_dict[date], 'DoC')
		if os.path.exists(path)==False:
			path = os.path.join(self.start_dir, self.manifest_dict[date])
		self.path = path
		return self.path

	def generate_session_dfs(self, direct_path=None):
		'''currently written to only handle single dates'''
		path = self.get_session_path(self.dates[0])
		for f in os.listdir(path):
			if ('.pkl' in f) & ('task' in f):
				pkl_path = os.path.join(path, f)

		df = beh_pkl_path_to_df(pkl_path)
		pkl = pd.read_pickle(pkl_path)
		df_jcam = add_jcam_to_df(df, session)
		running_speed = load_running_speed(pkl)
		stim_log = pd.DataFrame.from_dict(pkl['stimuluslog'])

		if len(stim_log.image_name.unique()) > 1:
			ori_col = 'image_name'
		else:
			ori_col = 'ori'

		df_jcam['lick_frames_jcam'] = None
		df_jcam['run_speeds'] = None
		df_jcam['test_ori'] = None
		df_jcam['reference_ori'] = None
		df_jcam['responses'] = None
		df_jcam['ori_by_frame'] = None
		df_jcam['stim_by_frame'] = None
		for idx in df_jcam.index:
			start_f = df_jcam.at[idx, 'startframe']
			try:
				end_f = df_jcam.at[idx+1, 'startframe']
			except:
				end_f = df_jcam.startframe.iloc[-1]
			speed = list(running_speed[running_speed['frame'].between(start_f-1, end_f, inclusive=False)].speed)
			df_jcam.at[idx, 'run_speeds'] = speed

			oris = list(stim_log[stim_log['frame'].between(start_f-1, end_f, inclusive=False)][ori_col].astype(int))
			df_jcam.at[idx, 'ori_by_frame'] = oris





			if len(df_jcam.iloc[idx]['lick_frames']) > 0:
				lick_frames = df_jcam.iloc[idx]['lick_frames']
				licks = []
				for l in lick_frames:
					lick = get_jcam_index(self.session.timeline.times,l)
					licks.append(lick)     
				df_jcam.at[idx, 'lick_frames_jcam'] = licks
		if save==True:
			df_name = '{}_{}_beh_jcam_df.pkl'.format(self.date, self.mouse_id)
			pd.to_pickle(df_jcam, os.path.join(self.path, df_name))

		self.df = df_jcam


