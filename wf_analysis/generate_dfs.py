
import os
import json
import h5py
import glob2
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from imaging_behavior import load_session_from_manifest
from WF_utilities import beh_pkl_path_to_df, get_jcam_index, rad_to_dist, generate_mouse_manifest
from visual_behavior.utilities import flatten_list
import visual_behavior.translator.foraging as fg
import imaging_behavior.io as ibio
import imaging_behavior.core.processing as ibp
import imaging_behavior.core.utilities as ut

def extract_read_times(read_time_list, sampling_rate, test_fcn=lambda x: np.where(x>=1.5)):
	'''modified from imaging behavior'''
	readdf = np.diff(read_time_list) #differentiate the signal
	readdf = np.insert(readdf, 0,readdf[0]) #replace lost first element
	readtime_inds = test_fcn(readdf) #Find all points where derivative exceeds 1.5
	#convert to timestamps, throw out first element which corresponds to 
	#start of first exposure, not a read
	readtimes = readtime_inds[0][:]/sampling_rate 
	return readtimes


class AnalysisDataFrames(object):
	def __init__(self, mouse_id, dates, main_dir='default', run_all=False):
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

		self.fixed_dict = {}
		for k in self.ori_dict.keys():
			for o in self.ori_dict[k]:
				self.fixed_dict[o] = k

		self.res_type_dict = {'HIT': 1, 
		                     'MISS': 2,
		                     'FA': 3,
		                     'CR': 4,
		                     'AUTOREWARDED': 5,
		                     'EARLY_RESPONSE': 6}
		self.channels = ['photodiode2',
					      'read_backscatter',
					      'read',
					      'trigger',
					      'vid1',
					      'vid0',
					      'visualFrame',
					      'runningRef',
					      'runningSig',
					      'reward',
					      'licking',
					      'vid2']
		
		for d in self.dates:
			self.date = d
			self.get_session_path()
			self.get_movie_type()
			self.generate_session_object()
			self.generate_matrix()
			if 'DoC' in self.path:
				self.save_path = os.path.split(self.path)[0]
				self.extract_hemo_timestamps()
				self.generate_blank_matrix()
				self.save_files(self.save_path)

	def get_session_path(self):
		path = os.path.join(self.main_dir, self.manifest_dict[self.date], 'DoC')
		if os.path.exists(path)==False:
			path = os.path.join(self.main_dir, self.manifest_dict[self.date])
		self.path = path
		print self.path
		return self.path

	def get_movie_type(self):
		mov = [fn for fn in glob2.glob(os.path.join(self.path, "*2_2_1*")) if ".tiff" not in fn]
		movie_type = mov[0].split(".")[-1]
		self.movie_type = ".{}".format(movie_type)
		return self.movie_type

	def generate_session_object(self):
		try:
			session = pd.read_pickle(glob2.glob(os.path.join(self.path, "*session_object.pkl"))[0])
		
		except:
			manifest_path = ut.autoGenerateManifest(self.path, animal_id=self.mouse_id, filetype=self.movie_type)
			session = load_session_from_manifest(manifest_path)
			pd.to_pickle(session, os.path.join(self.path, 
									"{}_{}_session_object.pkl".format(self.date, self.mouse_id)))
		self.session = session
		self.pkl_path = self.session.behavior_log_file_path
		self.pkl = pd.read_pickle(self.pkl_path)
		return self.session, self.pkl_path, self.pkl

	def make_trials_df(self):
		trials_df = beh_pkl_path_to_df(self.pkl_path)
		trials_df['jcam_startframe'] = get_jcam_index(self.session.timeline.times, trials_df['startframe'])
		for idx in trials_df.index:
		    if trials_df.at[idx, 'trial_type']=='autorewarded':
		        trials_df.at[idx, 'response_type'] = 'AUTOREWARDED'
		self.trials_df = trials_df
		return self.trials_df

	def make_change_df(self):
		change_df = pd.DataFrame()
		change_df['frame'] = self.trials_df['change_frame'].dropna()
		change_df['jcam_changeframe'] = get_jcam_index(self.session.timeline.times, change_df['frame'])
		self.change_df = change_df
		return self.change_df

	def make_stim_df(self):
		stim_df = fg.load_visual_stimuli(self.pkl)
		stim_df['jcam_onframe'] = get_jcam_index(self.session.timeline.times, stim_df['frame'])
		stim_df['jcam_offframe'] = get_jcam_index(self.session.timeline.times, stim_df['end_frame'])
		stim_df['image_id'] = None
		if stim_df['image_name'][0] is not None:
			stim_df['image_id'] = stim_df['image_name']
		else:
			stim_df['image_id'] = stim_df['orientation']
		self.stim_df = stim_df
		return self.stim_df

	def make_licks_df(self):
		licks_df = fg.load_licks(self.pkl)
		licks_df['jcam_lick_frame'] = get_jcam_index(self.session.timeline.times, licks_df['frame'])
		self.licks_df = licks_df
		return self.licks_df

	def make_rewards_df(self):
		rewards_df = fg.load_rewards(self.pkl)
		rewards_df['jcam_rew_frame'] = get_jcam_index(self.session.timeline.times, rewards_df['frame'])
		self.rewards_df = rewards_df
		return self.rewards_df

	def make_run_df(self):
		run_df = pd.DataFrame()
		run_df['speed'] = rad_to_dist(self.session.timeline.values['running_speed_radians_per_sec'])
		run_df['cam_time'] = np.round(self.session.timeline.times['running_speed_cm_per_sec'], 2)
		run_df = run_df.groupby('cam_time').mean().reset_index()
		self.run_df = run_df
		return self.run_df

	def generate_matrix(self):
		self.make_trials_df()
		self.make_change_df()
		self.make_stim_df()
		self.make_licks_df()
		self.make_rewards_df()
		self.make_run_df()

		cam_frame_times = pd.DataFrame()
		cam_frame_times['cam_time'] = np.round(self.session.timeline.values['fluorescence_read_times'], 2)
		cam_frame_times['cam_frame'] = cam_frame_times.index

		matrix = cam_frame_times.merge(self.trials_df['jcam_startframe'].to_frame(), left_on='cam_frame', right_on='jcam_startframe', how='left')
		matrix = matrix.merge(self.change_df['jcam_changeframe'].to_frame(), left_on='cam_frame', right_on='jcam_changeframe', how='left')
		matrix = matrix.merge(self.rewards_df['jcam_rew_frame'].to_frame(), left_on='cam_frame', right_on='jcam_rew_frame', how='left')
		matrix = matrix.merge(self.licks_df['jcam_lick_frame'].to_frame(), left_on='cam_frame', right_on='jcam_lick_frame', how='left')
		matrix = matrix.groupby(['cam_frame', 'cam_time']).count()[['jcam_startframe', 'jcam_changeframe', 'jcam_rew_frame', 'jcam_lick_frame']].reset_index()
		matrix = matrix.merge(self.trials_df[['response_type', 'jcam_startframe', 'index']], left_on='cam_frame', right_on='jcam_startframe', how='left').drop('jcam_startframe_y', 1)



		matrix = matrix.merge(self.stim_df[['jcam_onframe', 'image_id']], left_on='cam_frame', right_on='jcam_onframe', how='left')
		matrix = matrix.merge(self.stim_df[['jcam_offframe', 'image_id']], left_on='cam_frame', right_on='jcam_offframe', how='left')
		matrix['image_id'] = matrix['image_id_x'].combine(matrix['image_id_y'], lambda s1, s2: s2 if pd.isnull(s2)==False else s1)
		matrix = matrix.merge(self.run_df, on='cam_time', how='left')
		for idx in matrix.index:
			if (pd.isnull(matrix.at[idx, 'jcam_offframe'])==False) & (matrix.at[idx, 'jcam_offframe']!=0):
				matrix.at[idx+1, 'jcam_offframe'] = 0
		matrix['stim_state'] = matrix['jcam_onframe'].combine(matrix['jcam_offframe'], lambda s1, s2: s1 if pd.isnull(s1)==False else s2)

		b_start = self.trials_df.jcam_startframe.iloc[0]
		b_end = self.stim_df.jcam_offframe.iloc[-1]
		matrix.loc[b_start:b_end, 'stim_state'] = matrix.loc[b_start:b_end, 'stim_state'].fillna(method='ffill')
		matrix.loc[b_start:b_end, 'stim_state'] = matrix.loc[b_start:b_end, 'stim_state'].where(cond=matrix['stim_state']==0, other=1)
		matrix.loc[b_start:b_end, 'image_id'] = matrix.loc[b_start:b_end, 'image_id'].fillna(method='ffill')
		matrix.loc[b_start:b_end, 'image_id'] = matrix.loc[b_start:b_end, 'image_id'].fillna(method='bfill')
		matrix.loc[b_start:b_end, 'image_id'] = matrix.loc[b_start:b_end, 'image_id'].astype(int)
		matrix.loc[b_start:b_end, 'index'] = matrix.loc[b_start:b_end, 'index'].fillna(method='ffill')
		matrix.loc[b_start:b_end, 'response_type'] = matrix.loc[b_start:b_end, 'response_type'].fillna(method='ffill')

		del matrix['jcam_onframe']
		del matrix['jcam_offframe']
		del matrix['image_id_x']
		del matrix['image_id_y']

		matrix['response_type'].replace(self.res_type_dict, inplace=True)
		rename_dict = {'frame': 'beh_frame',
		              'image_id': 'ori',
		              'speed': 'cm_s',
		              'jcam_startframe_x': 'trial_start',
		              'jcam_changeframe': 'change',
		              'jcam_rew_frame': 'reward',
		              'jcam_lick_frame': 'lick',
		               'index': 'trial_number'
		              }
		matrix = matrix.rename(columns=rename_dict)
		matrix.fillna(-99, inplace=True)
		matrix.ori.replace(to_replace=self.fixed_dict, inplace=True)
		self.matrix = matrix
		self.matrix.to_csv(path_or_buf=os.path.join(self.path, 
								"{}_{}_doc_matrix_df.csv".format(self.date, self.mouse_id)))
		return self.matrix



	def extract_hemo_timestamps(self):
		jphys = glob2.glob(os.path.join(self.path, "*JPhys*"))[0]
		jphys_dict = ibio.importRawJPhys(jphys,channels=self.channels)
		readtimes = extract_read_times(jphys_dict['read_backscatter'], 
		                              		jphys_dict['sampling_rate'],
		                              		test_fcn=lambda x: np.where(x>=1.5))
		rt = ut.remove_vals_in_window(readtimes, .005)
		self.doc_hemo_timestamps = rt
		return self.doc_hemo_timestamps


	def generate_blank_matrix(self):
		folder = os.path.join(self.main_dir, self.manifest_dict[self.date], 'blank')
		jphys = glob2.glob(os.path.join(folder, "*JPhys*"))[0]
		pkl = pd.read_pickle(glob2.glob(os.path.join(folder, "*task=*.pkl"))[0])
		stimlog = pd.DataFrame(pkl['stimuluslog'])

		jphys_dict = ibio.importRawJPhys(jphys,channels=self.channels)
		master_times = np.arange(jphys_dict['number_of_samples'])/jphys_dict['sampling_rate']
		
		#get camera frame times
		readtimes = extract_read_times(jphys_dict['read'], jphys_dict['sampling_rate'],)
		rt = ut.remove_vals_in_window(readtimes, .005)

		#get visual frame times
		filtered_visual_frame_signal = medfilt(jphys_dict['visualFrame'], kernel_size=5)
		vistimes = ut.remove_vals_in_window(extract_read_times(filtered_visual_frame_signal, 
																jphys_dict['sampling_rate'], 
																test_fcn=lambda x: np.where(x<=-1.5)),
											window = 0.005)
		if vistimes[1]-vistimes[0] > 0.25:
			vistimes = vistimes[1:]
		photodiode_times, photodiode_problem = ibp.extract_photodiode_times(jphys_dict)
		photodiode_times = ut.remove_vals_in_window(photodiode_times, window = 0.9)
		photodiode_times = photodiode_times[photodiode_times > vistimes[0]]
		mean_display_lag = ibp.calculateMeanDisplayLag(photodiode_times,vistimes)
		frame_display_times = vistimes + mean_display_lag
		time_values = frame_display_times
		visual_frame_numbers = np.arange(0,len(time_values)) 
		f = interp1d(visual_frame_numbers,time_values,bounds_error=False)   
		time_in_timeline = f(stimlog['frame']) 
		beh_cam_frames = ut.get_nearest_time(rt,time_in_timeline)

		#get licks
		licks_df = fg.load_licks(pkl)
		if len(licks_df) > 0 :
			time_in_timeline = f(licks_df['frame']) 
			licks_df['jcam_lick_frame'] = ut.get_nearest_time(rt,time_in_timeline)
		elif len(licks_df) == 0:
			licks_df['jcam_lick_frame'] = None

		#get run times/speed
		running_speed_radians_per_sec = ut.extract_running_speed_jphys(jphys_dict['runningSig'],
											                           jphys_dict['runningRef'], 
											                           master_times, 
											                           running_threshold=100, 
											                           sampling_rate=jphys_dict['sampling_rate'])
		run_df = pd.DataFrame()
		run_df['speed'] = rad_to_dist(running_speed_radians_per_sec)
		run_df['cam_time'] = np.round(master_times, 2)
		run_df = run_df.groupby('cam_time').mean().reset_index()

		#generate matrix
		cam_frame_times = pd.DataFrame()
		cam_frame_times['cam_time'] = np.round(rt, 2)
		cam_frame_times['cam_frame'] = cam_frame_times.index
		matrix = cam_frame_times.merge(licks_df['jcam_lick_frame'].to_frame(), left_on='cam_frame', right_on='jcam_lick_frame', how='left')
		matrix = matrix.groupby(['cam_frame', 'cam_time']).count()[['jcam_lick_frame']].reset_index()
		matrix = matrix.merge(run_df, on='cam_time', how='left')
		matrix['stim_state'] = None
		matrix.loc[beh_cam_frames[0]:beh_cam_frames[-1]+1, 'stim_state'] = 0
		matrix.rename(columns={'jcam_lick_frame':'lick', 'speed':'cm_s'}, inplace=True)
		matrix.fillna(value=-99, inplace=True)\
		
		hemo_readtimes = extract_read_times(jphys_dict['read_backscatter'], 
		                              			jphys_dict['sampling_rate'],
		                              			test_fcn=lambda x: np.where(x<=-1.5))
		hemo_rt = ut.remove_vals_in_window(hemo_readtimes, .005)
		self.blank_hemo_timestamps = hemo_rt
		self.blank_matrix = matrix

		return self.blank_matrix, self.blank_hemo_timestamps                           

	def save_files(self, save_path):		
		self.blank_matrix.to_csv(path_or_buf=os.path.join(save_path, 
											"{}_{}_blank_matrix_df.csv".format(self.date, self.mouse_id)))
		np.save(os.path.join(save_path, "{}_{}_blank_hemo_movie_timestamps.npy".format(self.date, self.mouse_id)),
											self.blank_hemo_timestamps)
		np.save(os.path.join(save_path, "{}_{}_doc_hemo_movie_timestamps.npy".format(self.date, self.mouse_id)),
									self.doc_hemo_timestamps)
											



	# def generate_trial_df(self):
	# 	trial_df = pd.DataFrame()
	# 	for c in self.matrix.columns:
	# 	    if c!='trial_number':
	# 	        s = self.matrix.groupby('trial_number')[c].apply(list).reset_index()
	# 	        trial_df[c] = s[c]

	# 	trial_df['trial_number'] = trial_df.index
	# 	for idx in trial_df.index:
	# 	    start = trial_df.trial_start[idx].index(1L)
	# 	    trial_df.at[idx, 'trial_start'] = int(trial_df.cam_frame[idx][start])
		    
	# 	    licks = [i for i, x in enumerate(trial_df.lick[idx]) if x == 1L]
	# 	    trial_df.at[idx, 'lick'] = list(np.array(trial_df.cam_frame[idx])[licks])
		    
	# 	    trial_df.at[idx, 'response_type'] = np.median(trial_df.at[idx, 'response_type'])
	# 	    try:
	# 	        change = trial_df.change[idx].index(1L)
	# 	        rew = trial_df.reward[idx].index(1L)
	# 	        trial_df.at[idx, 'change'] = int(trial_df.cam_frame[idx][change])
	# 	        trial_df.at[idx, 'reward'] = int(trial_df.cam_frame[idx][rew])
	# 	    except:
	# 	        trial_df.at[idx, 'change'] = np.nan
	# 	        trial_df.at[idx, 'reward'] = np.nan
		        
	# 	    oris = list(OrderedDict.fromkeys(trial_df.ori[idx]))
	# 	    trial_df.at[idx, 'reference_ori'] = oris[0]
	# 	    try:
	# 	        trial_df.at[idx, 'test_ori'] = oris[1]
	# 	    except IndexError:
	# 	        trial_df.at[idx, 'test_ori'] = oris[0]

	# 	col_order = ['trial_number', 'response_type', 'trial_start', 'reference_ori', 'test_ori', 'change', 'reward', 'lick', 
	# 				'cam_frame','cam_time', 'beh_frame', 'stim_state', 'ori', 'cm_s']

	# 	self.trial_df = trial_df[col_order]
	# 	return self.trial_df
