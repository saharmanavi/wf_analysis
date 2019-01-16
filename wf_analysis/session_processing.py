import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator import foraging
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
from visual_behavior.visualization.extended_trials.mouse import make_summary_figure
import visual_behavior_research.utilities as dro

from imaging_behavior.core.slicer import BinarySlicer
from imaging_behavior import load_session_from_manifest
from imaging_behavior.core.utilities import autoGenerateManifest, normalize_movie, get_nearest_time
from imaging_behavior.io.manifest import load_manifest

from scipy.interpolate import interp1d
from IPython.display import Image


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


class ProcessWFData(object):
	def __init__(self, mouse_id, date, main_directory, load_movies=True, summary_figure=True):


		self.mouse_id = mouse_id
		self.date = date
		self.main_dir = main_directory
		self.get_session_path()
		self.numpy_or_h5()
		self.get_manifest()
		# self.manifest_dict
		# self.manifest_path
		self.session = load_session_from_manifest(self.manifest_path)
		self.get_session_dataframe()
		if load_movies==True:
			self.get_movie_file()
			self.make_get_dff_movie()
		if summary_figure==True:
			self.create_behavior_summary_fig()
		# self.display_behavior_summary_fig()
		# self.display_multisession_summary_fig()

	def get_session_path(self):
		try:
			path = os.path.join(self.main_dir, '{}-{}'.format(self.date, self.mouse_id), 'DoC')
		except IOError:
			path = os.path.join(self.main_dir, '{}_{}'.format(self.date, self.mouse_id))
		self.path = path
		return self.path

	def numpy_or_h5(self):
		if 'DoC' in self.path:
			ext = 'h5'
		else:
			for f in os.listdir(self.path):
				if '16_16_1.npy' in f:
					ext = 'npy'
		self.ext = ext
		return self.ext

	def get_manifest(self):	    
	    manifest_path = autoGenerateManifest(path = self.path, 
	                                            animal_id = self.mouse_id, 
	                                            filetype = ".{}".format(self.ext))
	    manifest_dict = load_manifest(manifest_path)
	    self.manifest_path = manifest_path
	    self.manifest_dict = manifest_dict
	    return self.manifest_path, self.manifest_dict

	def get_session_dataframe(self):
		df = beh_pkl_path_to_df(self.manifest_dict['behavioral_log_file_path'])
		df_jcam = add_jcam_to_df(df, self.session)
		self.dataframe = df_jcam
		return self.dataframe

	def get_movie_file(self):
		if self.ext=='h5':
			moviefile = h5py.File(self.manifest_dict['decimated_jcam_movie_file_path'], 'r')
			movie = np.transpose(moviefile['data'], (0, 2, 1))
		elif self.ext=='npy':
			movie = BinarySlicer(str(self.manifest_dict['decimated_jcam_movie_file_path']))
		self.movie = movie
		return self.movie

	def make_get_dff_movie(self):
		moviename = os.path.split(str(self.manifest_dict['decimated_jcam_movie_file_path']))[-1]
		dff_moviename = moviename.replace('16_16_1.{}'.format(self.ext),'16-16-1_dff_rolling_gaussian.{}'.format(self.ext))

		if os.exists(os.path.join(self.path, dff_moviename)):
			if self.ext == 'h5':
				f = h5py.File(os.path.join(self.path, dff_moviename), 'r')
				dff_movie = f['data']
			if self.ext == 'npy':
				dff_movie = BinarySlicer(str(os.path.join(self.path, dff_moviename)))
		else:
			print 'Making DFF movie'
			dff_movie = normalize_movie(self.movie,
			                           mask_data=False,
			                           show_progress=True)
			filename = os.path.join(self.path,dff_moviename)
			if self.ext == 'h5':
				hf = h5py.File(filename, 'w')
				hf.create_dataset('data', data=dff_movie)
				hf.close()
			if self.ext == 'npy':
				np.save(filename, dff_movie)
		self.dff_movie = dff_movie
		return self.dff_movie

	def create_behavior_summary_fig(self):		
		figname = self.manifest_dict['behavioral_log_file_path'].replace('.pkl', '.png')
		if os.path.exists(os.path.join(self.path, figname))==False:
			make_daily_figure(self.dataframe)
			plt.savefig(os.path.join(self.path, figname))
			plt.savefig(os.path.join(r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data",
									self.mouse_id,
									"figures",
									figname))
		self.figname = figname
		return self.figname

	def display_behavior_summary_fig(self):
		fig_path = os.path.join(self.path, self.figname)
		try:
			return Image(filename=fig_path)
		except IOError:
			print 'Image file not found, try checking the behavior data folder at'
			print "\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data\{}\figures".format(self.mouse_id)

	def display_multisession_summary_fig(self, update_summary=False):
		behavior_data = r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data"
		if update_summary==True:
			df_all = dro.load_from_folder(os.path.join(behavior_data,self.mouse_id,"output"),
								          load_existing_dataframe=True,
								          save_dataframe=True,
								          filename_contains='full_field_gratings')
			make_summary_figure(df_all, self.mouse_id)
			plt.savefig(os.path.join(behavior_data,
								self.mouse_id,
								"figures",
								'multisession_behavior_summary.png'))
		fig_path = os.path.join(behavior_data, self.mouse_id, "figures", 'multisession_behavior_summary.png')	




