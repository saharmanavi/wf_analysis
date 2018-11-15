
import os
import json
import h5py
import pandas as pd

from IPython.display import Image
from imaging_behavior import load_session_from_manifest
from imaging_behavior.core.slicer import BinarySlicer
from WF_utilities import choose_date, date_and_objects, beh_pkl_path_to_df, add_jcam_to_df, get_jcam_index
from visual_behavior.translator.foraging import load_running_speed


class GetWFDataset(object):
	def __init__(self, mouse_id, main_directory, date=None, manifest='save', manifest_save_loc=None):
		'''
		mouse_id is a string with format M123456
		main_directory is the directory where all the session folders are, eg //allen/.../IntrinsicImageData
		manifest refers to a dictionary of folder names of every session for the mouse_id specified. 
			can be 'save' or 'temp'
			'save' will save a JSON file to the location specified in manifest_save_loc (or if none is specified, then in the main_directory)
			'temp' will simply keep the dictionary as a class object but not save a copy anywhere
		Date, optional, refers to session-specific items. Those items can be called later with different dates as necessary.
		
		Session-specific functions available once the class object is loaded. If no date is specified or a different date is desired, 
		class's date will be used. If class date is None, specify date = 'YYMMDD' for each function. 
		# get_session_path()
		# check_session_files()
		# get_direct_file_paths()
		# get_IB_session_object()
		# get_session_dataframe()
		# get_dff_movie()
		# get_raw_movie()
		# display_summary_fig()


		'''

		self.mouse_id = mouse_id
		self.main_dir = main_directory
		self.date = date
		self.manifest = manifest
		self.manifest_save_loc = manifest_save_loc
		self.manifest_name = '{}_WFdata_manifest.json'.format(self.mouse_id)

		self.generate_manifest()
		if self.manifest=='save':
			self.save_manifest()
		self.numpy_or_h5()

		if self.date is not None:
			self.check_session_files(self.date)
		
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
		self.manifest_dict = manifest_dict		
		return self.manifest_dict

	def save_manifest(self):
		if self.manifest_save_loc is not None:
			save_loc = os.path.join(self.manifest_save_loc, self.manifest_name)
		else:
			save_loc = os.path.join(self.main_dir, self.manifest_name)

		with open(save_loc, "w") as write_file:
			json.dump(self.manifest_dict, write_file)
		print 'Manifest for all sessions saved at:'
		print save_loc

	def numpy_or_h5(self):
		date = self.manifest_dict.keys()[-1]
		check_dir = os.path.join(self.main_dir, self.manifest_dict[date])

		if os.path.exists(os.path.join(check_dir, 'DoC')):
			ext = 'h5'
		else:
			for f in os.listdir(check_dir):
				if '16_16_1.npy' in f:
					ext = 'npy'
		self.ext = ext
		return self.ext

	@choose_date
	def get_session_path(self, date):
		path = os.path.join(self.main_dir, self.manifest_dict[date], 'DoC')
		if os.path.exists(path)==False:
			path = os.path.join(self.main_dir, self.manifest_dict[date])
		self.path = path
		return self.path
	
	@choose_date
	def check_session_files(self, date):
		self.path = self.get_session_path(date)

		session_manifest = {'hemo_2x': None,
							'gcamp_2x': None,
							'hemo_16x': None,
							'gcamp_16x': None,
							'session_jphys': None,
							'behavior_pkl': None,
							'dff_movie': None,
							'behavior_summary_fig': None,
							'WF_summary_fig': None,
							'behavior_cam': None,
							'eye_cam': None,
							'lick_cam': None,
							'ib_manifest': None,
							}
		create_dict = {'behavior_summary_fig' : [f for f in os.listdir(self.path) if ('task' in f) & ('.png' in f)],
						'hemo_2x' : [f for f in os.listdir(self.path) if ('2_2_1' in f) and ('cam1' in f)],
						'gcamp_2x' : [f for f in os.listdir(self.path) if ('cam1' not in f) and ('2_2_1' in f) and ('tiff' not in f)], 
						'hemo_16x' : [f for f in os.listdir(self.path) if ('16_16_1' in f) and ('cam1' in f)],
						'gcamp_16x' : [f for f in os.listdir(self.path) if ('cam1' not in f) and ('16_16_1' in f)],
						'behavior_pkl' : [f for f in os.listdir(self.path) if '.pkl' in f],
						'dff_movie' : [f for f in os.listdir(self.path) if '_dff_' in f],
						'WF_summary_fig' : [f for f in os.listdir(self.path) if 'summary_figure.png' in f],
						'behavior_cam' : [f for f in os.listdir(self.path) if '0.avi' in f],
						'eye_cam' : [f for f in os.listdir(self.path) if '1.avi' in f],
						'lick_cam' : [f for f in os.listdir(self.path) if '2.avi' in f],
						'ib_manifest' : [f for f in os.listdir(self.path) if 'autogenerated.json' in f],
						}	
		try: 

			if self.ext == 'npy':
				jphys_num = create_dict['gcamp_2x'][0].split('_2_2_1.npy')[0][-3:]
			if self.ext == 'h5':
				jphys_num = '100'
			create_dict['session_jphys'] = [f for f in os.listdir(self.path) if ('jphys' in f.lower()) and (jphys_num in f)]
		except:
			pass
		
		for key in create_dict.keys():
			try:
				session_manifest[key] = create_dict[key][0]
			except:
				pass
						
		self.session_manifest = session_manifest
		save_loc = os.path.join(self.path, 'session_manifest.json')
		with open(save_loc, "w") as write_file:
			json.dump(self.session_manifest, write_file)
		# print 'Manifest for {} session saved at:'.format(date)
		# print save_loc
		return self.path, self.session_manifest

	@date_and_objects
	def get_direct_file_paths(self, date, path, session_manifest, manifest_item, rtrn=False):
		if type(manifest_item) is not list:
			manifest_item = [manifest_item]

		for i in manifest_item:
			try:
				full_path = os.path.join(path,session_manifest[i])
				print '{}:'.format(i)
				print full_path
				if rtrn==True:
					return full_path
			except KeyError:
				print '{} does not exist as a key. Check session_manifest for list of keys.'.format(i)
			except TypeError:
				print '{} file does not (yet) exist.'.format(i)
			
	@date_and_objects
	def get_IB_session_object(self, date, path, session_manifest):
		manifest_path = os.path.join(path, session_manifest['ib_manifest'])
		session = load_session_from_manifest(manifest_path)
		self.session = session
		return self.session

	@date_and_objects
	def get_session_dataframe(self, date, path, session_manifest, save=False):
		if (date == self.date):
			try:
				session = self.session
			except AttributeError:
				session = self.get_IB_session_object(date, path, session_manifest)
		else:
			session = self.get_IB_session_object(date, path, session_manifest)
		
		pkl_path = os.path.join(self.path, self.session_manifest['behavior_pkl']) 
		df = beh_pkl_path_to_df(pkl_path)
		df_jcam = add_jcam_to_df(df, session)
		running_speed = load_running_speed(pd.read_pickle(pkl_path))

		df_jcam['lick_frames_jcam'] = None
		df_jcam['run_speeds'] = None
		for idx in df_jcam.index:
			start_f = df_jcam.at[idx, 'startframe']
			try:
				end_f = df_jcam.at[idx+1, 'startframe']
			except:
				end_f = df_jcam.startframe.iloc[-1]
			speed = list(running_speed[running_speed['frame'].between(start_f-1, end_f, inclusive=False)].speed)
			df_jcam.at[idx, 'run_speeds'] = speed

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
		return self.df

	@date_and_objects
	def get_dff_movie(self, date, path, session_manifest):
		movie_path = os.path.join(path, session_manifest['dff_movie'])
		if os.path.exists(movie_path):
			if self.ext == 'h5':
				f = h5py.File(movie_path, 'r')
				dff_movie = f['data']
			if self.ext == 'npy':
				dff_movie = BinarySlicer(movie_path)	
			self.dff_movie = dff_movie
			return self.dff_movie
		else:
			print 'oops, this DFF movie has not been saved/generated'
		
	@date_and_objects
	def get_raw_movie(self, date, path, session_manifest, size='16x', cam='gcamp'):
		'''size can be 16x or 2x
		cam can be hemo or gcamp'''
		if size == '2x':
			print "You are attempting to load the 2x raw movie file. This is a VERY BIG file."
		if (cam == 'hemo') and (self.ext == 'npy'):
			print 'This mouse predates the hemodynamics recordings.'

		key_name = '{}_{}'.format(cam, size)
		movie_path = os.path.join(path, session_manifest[key_name])
		if self.ext == 'h5':
			f = h5py.File(movie_path, 'r')
			raw_movie = f['data']
		if self.ext == 'npy':
			raw_movie = BinarySlicer(movie_path)
		self.raw_movie = raw_movie
		return self.raw_movie

	@choose_date
	def display_summary_fig(self, date, fig_type):
		'''fig_type is either beh for behavior summary, or WF for widefield summary'''
		if fig_type=='beh':
			manifest_item = 'behavior_summary_fig'
		if fig_type=='WF':
			manifest_item = 'WF_summary_fig'
		fig_path  = self.get_direct_file_paths(date=date, manifest_item=manifest_item, rtrn=True)
		try:
			return Image(filename=fig_path)
		except Exception as e:
			print 'Error: {}'.format(e)

