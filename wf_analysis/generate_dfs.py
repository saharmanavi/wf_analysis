
import os
import json
import h5py
import pandas as pd
import numpy as np
from collections import OrderedDict

from imaging_behavior import load_session_from_manifest
from WF_utilities import beh_pkl_path_to_df, get_jcam_index, rad_to_dist, generate_mouse_manifest
from visual_behavior.utilities import flatten_list

class AnalysisDataFrames(object):
	def __init__(self, mouse_id, dates, main_dir='default', save=False, run_all=False):
		if main_dir=='default':
			self.main_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData"
		else:
			self.main_dir = main_dir
		self.mouse_id = mouse_id

		self.manifest_dict, self.date_list = generate_mouse_manifest(self.mouse_id, self.main_dir)
		self.dates = dates
		if type(self.dates) is not list:
			self.dates= [dates]
		if run_all==True:
			self.dates = self.date_list
		
		self.ori_dict = {90: [90, -270, 450, 810],
		                360: [0, 360, -360, 720],
		                -90: [-90, 270, -450, 630],
		                -360: [180, -180, 540, 900]}

		self.res_type_dict = {'HIT': 1, 
			                 'MISS': 2,
			                 'FA': 3,
			                 'CR': 4,
			                 'AUTOREWARDED': 99,
			                 'EARLY_RESPONSE': 0}
		
		for d in self.dates:
			print d
			self.get_session_path(d)
			self.generate_session_object()
			self.generate_matrix()
			# self.generate_trial_df()

			if save==True:
				self.save_files(d)

	def get_session_path(self, date):
		path = os.path.join(self.main_dir, self.manifest_dict[date], 'DoC')
		if os.path.exists(path)==False:
			path = os.path.join(self.main_dir, self.manifest_dict[date])
		self.path = path

		return self.path

	def generate_session_object(self):
		for f in os.listdir(self.path):
			if 'autogenerated.json' in f:
				manifest_path = os.path.join(self.path, f)

		session = load_session_from_manifest(manifest_path)
		self.session = session

	def generate_matrix(self):
		pkl = pd.read_pickle(self.session.behavior_log_file_path)

		fixed_dict = {}
		for k in self.ori_dict.keys():
		    for o in self.ori_dict[k]:
		        fixed_dict[o] = k
		stimlog = pd.DataFrame.from_dict(pkl['stimuluslog'])
		stimlog['cam_frame'] = get_jcam_index(self.session.timeline.times, stimlog['frame'])
		if len(stimlog.image_name.unique()) > 1:
		    ori_col = 'image_name'
		else:
		    ori_col = 'ori'
		stimlog['image_id'] = pd.to_numeric(stimlog[ori_col])
		stimlog['image_id'].replace(fixed_dict, inplace=True)

		run_df = pd.DataFrame()
		run_df['speed'] = rad_to_dist(self.session.timeline.values['running_speed_radians_per_sec'])
		run_df['cam_time'] = np.round(self.session.timeline.times['running_speed_cm_per_sec'], 2)
		run_df = run_df.groupby('cam_time').mean().reset_index()

		trial_df = beh_pkl_path_to_df(self.session.behavior_log_file_path)
		trial_df['jcam_start_frame'] = None
		trial_df['jcam_change_frame'] = None
		trial_df['jcam_rew_frame'] = None
		trial_df['jcam_lick_frame'] = None
		trial_df['trial_number'] = trial_df.index
		for idx in trial_df.index:
			trial_df.at[idx,'jcam_start_frame'] = get_jcam_index(self.session.timeline.times,trial_df.at[idx, 'startframe'])

			if pd.isnull(trial_df.at[idx,'change_frame'])==False:
				trial_df.at[idx,'jcam_change_frame'] = get_jcam_index(self.session.timeline.times,trial_df.at[idx, 'change_frame'])

			if len(trial_df.at[idx, 'lick_frames']) > 0:
				lick_frames = trial_df.at[idx, 'lick_frames']
				licks = []
				for l in lick_frames:
					lick = get_jcam_index(self.session.timeline.times,l)
					licks.append(lick)     
				trial_df.at[idx, 'jcam_lick_frame'] = licks

			if len(trial_df.at[idx, 'reward_frames']) > 0:
				rew_frames = trial_df.at[idx, 'reward_frames']
				rewards = []
				for r in rew_frames:
					reward = get_jcam_index(self.session.timeline.times,r)
					rewards.append(reward)     
				trial_df.at[idx, 'jcam_rew_frame'] = rewards

			if trial_df.at[idx, 'trial_type']=='autorewarded':
				trial_df.at[idx, 'response_type'] = 'AUTOREWARDED'	

		reward_df = pd.DataFrame()
		reward_df['rew_frames'] = flatten_list(list(trial_df['jcam_rew_frame']))

		lick_df = pd.DataFrame()
		lick_df['lick_frames'] = flatten_list(list(trial_df['jcam_lick_frame']))

		cam_frame_times = pd.DataFrame()
		cam_frame_times['cam_time'] = np.round(self.session.timeline.values['fluorescence_read_times'], 2)
		cam_frame_times['cam_frame'] = cam_frame_times.index

		matrix = cam_frame_times.merge(trial_df['jcam_start_frame'].to_frame(), left_on='cam_frame', right_on='jcam_start_frame', how='left')
		matrix = matrix.merge(trial_df['jcam_change_frame'].to_frame(), left_on='cam_frame', right_on='jcam_change_frame', how='left')
		matrix = matrix.merge(reward_df, left_on='cam_frame', right_on='rew_frames', how='left')
		matrix = matrix.merge(lick_df, left_on='cam_frame', right_on='lick_frames', how='left')
		matrix = matrix.groupby(['cam_frame', 'cam_time']).count()[['jcam_start_frame', 'jcam_change_frame', 'rew_frames', 'lick_frames']].reset_index()
		matrix = matrix.merge(trial_df[['response_type', 'jcam_start_frame', 'trial_number']], left_on='cam_frame', right_on='jcam_start_frame', how='left').drop('jcam_start_frame_y', 1)
		matrix = matrix.merge(stimlog[['state', 'frame', 'cam_frame', 'image_id']], on='cam_frame', how='left')
		matrix = matrix.merge(run_df, on='cam_time', how='left')

		matrix[['state', 'frame', 'response_type', 'trial_number', 'image_id']] = matrix[['state', 'frame', 'response_type', 'trial_number', 'image_id']].fillna(method='ffill')

		# self.matrix = matrix
		# return self.matrix

		beh_start = matrix[matrix.frame==0].index[0]
		matrix.loc[beh_start:, ['state', 'image_id']] = matrix.loc[beh_start:, ['state', 'image_id']].astype(int)

		matrix['response_type'].replace(self.res_type_dict, inplace=True)

		rename_dict = {'state': 'stim_state',
		              'frame': 'beh_frame',
		              'image_id': 'ori',
		              'speed': 'cm_s',
		              'jcam_start_frame_x': 'trial_start',
		              'jcam_change_frame': 'change',
		              'rew_frames': 'reward',
		              'lick_frames': 'lick',
		              }
		matrix = matrix.rename(columns=rename_dict)
		self.matrix = matrix[['cam_frame', 'cam_time', 'beh_frame', 'trial_number', 'stim_state', 'ori', 'response_type',
								'trial_start', 'change', 'reward', 'lick', 'cm_s']]
		return self.matrix

	def generate_trial_df(self):
		trial_df = pd.DataFrame()
		for c in self.matrix.columns:
		    if c!='trial_number':
		        s = self.matrix.groupby('trial_number')[c].apply(list).reset_index()
		        trial_df[c] = s[c]

		trial_df['trial_number'] = trial_df.index
		for idx in trial_df.index:
		    start = trial_df.trial_start[idx].index(1L)
		    trial_df.at[idx, 'trial_start'] = int(trial_df.cam_frame[idx][start])
		    
		    licks = [i for i, x in enumerate(trial_df.lick[idx]) if x == 1L]
		    trial_df.at[idx, 'lick'] = list(np.array(trial_df.cam_frame[idx])[licks])
		    
		    trial_df.at[idx, 'response_type'] = np.median(trial_df.at[idx, 'response_type'])
		    try:
		        change = trial_df.change[idx].index(1L)
		        rew = trial_df.reward[idx].index(1L)
		        trial_df.at[idx, 'change'] = int(trial_df.cam_frame[idx][change])
		        trial_df.at[idx, 'reward'] = int(trial_df.cam_frame[idx][rew])
		    except:
		        trial_df.at[idx, 'change'] = np.nan
		        trial_df.at[idx, 'reward'] = np.nan
		        
		    oris = list(OrderedDict.fromkeys(trial_df.ori[idx]))
		    trial_df.at[idx, 'reference_ori'] = oris[0]
		    try:
		        trial_df.at[idx, 'test_ori'] = oris[1]
		    except IndexError:
		        trial_df.at[idx, 'test_ori'] = oris[0]

		col_order = ['trial_number', 'response_type', 'trial_start', 'reference_ori', 'test_ori', 'change', 'reward', 'lick', 
					'cam_frame','cam_time', 'beh_frame', 'stim_state', 'ori', 'cm_s']

		self.trial_df = trial_df[col_order]
		return self.trial_df

	def save_files(self, date):
		pd.to_pickle(self.session, os.path.join(self.path, 
											"{}_{}_session_object.pkl".format(date, self.mouse_id)))
		pd.to_pickle(self.matrix, os.path.join(self.path, 
											"{}_{}_DoC_matrix_df.pkl".format(date, self.mouse_id)))
		pd.to_pickle(self.trial_df, os.path.join(self.path, 
											"{}_{}_DoC_trial_df.pkl".format(date, self.mouse_id)))
		print 'Files for {} on {} saved at {}.'.format(self.mouse_id, date, self.path)