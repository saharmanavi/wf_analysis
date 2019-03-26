from __future__ import print_function
import os, sys
import h5py
import numpy as np
import pandas as pd
import glob2
import shutil
# import wf_analysis as wf


class FindIssues(object):
	def __init__(self, processed_folder, data_type):		
		self.profo = processed_folder
		self.log = open(os.path.join(self.profo, "check_log.txt"), "w")
		self.data_type = data_type

		self.mouse = os.path.split(self.profo)[1].split('_')[1]
		self.date = os.path.split(self.profo)[1].split('_')[0]
		
		print("checking files for session {}-{}".format(self.date, self.mouse), file=self.log)
		print('', file=self.log)

		self.beh_dir = os.path.join(r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data", self.mouse)

		root_imaging_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData"
		self.imaging_dir = os.path.join(root_imaging_dir, "{}-{}".format(self.date, self.mouse))
		if os.path.exists(self.imaging_dir):
			self.imaging_dir = self.imaging_dir
		elif os.path.exists(os.path.join(root_imaging_dir, '{}_{}'.format(self.date, self.mouse))):
			self.imaging_dir = 	os.path.join(root_imaging_dir, '{}_{}'.format(self.date, self.mouse))
		else:
			print("No such directory. Check your date", file = self.log)
			sys.exit()

		self.check_processed_files()
		print('', file=self.log)
		self.find_summary_figs()
		print('', file=self.log)
		self.check_movies()

		print('', file=self.log)
		self.check_processed_files()
		print('', file=self.log)
		for k in self.files_dict.keys():
			print('{} : {}'.format(k, self.files_dict[k]), file=self.log)
		
		print('', file=self.log)
		print('atypical files:', file=self.log)
		for f in os.listdir(self.profo):
		    if (f not in self.dict_list) and ("log" not in f):
		        print(f, file=self.log)

		if pd.isnull(self.files_dict.values()).any():
			sys.stdout = sys.__stdout__
			print('Check on session {} for {}'.format(self.date, self.mouse))

	def check_processed_files(self, print_logs=True):
		self.file_names=['*dff_rolling_gaussian*',
						'*summary_figure.png',
						'*task=*.png',
						'*gcamp_doc_cam2_256x256_tc1.h5',
						'*gcamp_doc_cam2_256x256_tc5.h5',
						'*gcamp_doc_still_frame.npy',
						'*doc_matrix_df.csv',
						'*JPhysdoc',
						'*16_16_1.npy']
		if self.data_type=='h5':
			self.file_names.pop(self.file_names.index('*16_16_1.npy'))
			self.file_names.extend(['*JCamF_cam2_200.dcimg_16_16_1.h5',
									'*JPhysblank',
									'*blank_hemo_movie_timestamps.npy',
									'*blank_matrix_df.csv',
									'*doc_hemo_movie_timestamps.npy',
									'*gcamp_blank_cam2_256x256_tc1.h5',
									'*gcamp_blank_cam2_256x256_tc5.h5',
									'*hemo_blank_cam1_256x256_tc1.h5',
									'*hemo_doc_cam1_256x256_tc1.h5',
									'*hemo_doc_still_frame.npy'])

		files_dict = {}
		for name in self.file_names:
			try:
				files_dict[name] = glob2.glob(os.path.join(self.profo, name))[0]
			except IndexError:
				files_dict[name] = np.nan
				if print_logs==True:
					print('{} NOT found'.format(name), file=self.log)

		self.dict_list = [files_dict[k].split("\\")[-1] for k in files_dict.keys() if pd.isnull(files_dict[k])==False]
		self.files_dict = files_dict
		return self.files_dict, self.dict_list

	def find_summary_figs(self):
		for k in self.files_dict.keys():
			if (pd.isnull(self.files_dict[k])) and ('.png' in k):
				fig_dir = os.path.join(self.beh_dir, 'figures')
				files = glob2.glob(os.path.join(fig_dir, "{}*{}*".format(self.date, self.mouse)))
				for f in files:
					shutil.copy2(f, self.profo)
					print("transferring {} to processed folder".format(os.path.split(f)[1]), file=self.log)
				self.check_processed_files(print_logs=False)

	def check_movies(self):
		movie_shapes = {'*gcamp_blank_cam2_256x256_tc1.h5': '(62000, 256, 256)',
						'*gcamp_blank_cam2_256x256_tc5.h5': '(12400, 256, 256)',
						'*hemo_blank_cam1_256x256_tc1.h5': '(31000, 256, 256)',
						'*hemo_doc_cam1_256x256_tc1.h5': '(181000, 256, 256)',
						'*gcamp_doc_cam2_256x256_tc1.h5': '(362000, 256, 256)',
						'*gcamp_doc_cam2_256x256_tc5.h5': '(72400, 256, 256)',}

		for k in self.files_dict.keys():
			if ('.h5' in k) and ('dcimg_16' not in k):
				try:
					print('{} should have shape {}'.format(k, movie_shape[k]), file=self.log)
					a = h5py.File(self.files_dict[k], 'r+')
					movie = a['data']
					print("actual movie shape is {}".format(movie.shape), file=self.log)
					if 'hemo_doc_cam1_256x256_tc1' in k:
						if pd.isnull(self.files_dict['*hemo_doc_still_frame.npy']):
							print('creating hemo still_frame', file=self.log)
							still_frame = np.rot90(movie[30,:,:], 1)
							np.save(os.path.join(self.profo, '{}_{}_hemo_doc_still_frame.npy'.format(self.date, self.mouse)), still_frame)
					if 'gcamp_doc_cam2_256x256_tc1' in k:
						self.movie_len = movie.shape[0]
						if pd.isnull(self.files_dict['*gcamp_doc_still_frame.npy']):
							print('creating gcamp still_frame', file=self.log)
							still_frame = np.transpose(self.movie[frame,:,:], (1,0))
							np.save(os.path.join(self.profo, '{}_{}_gcamp_doc_still_frame.npy'.format(self.date, self.mouse)), still_frame)
					a.close()
					del movie
				except:
					pass
		    	self.check_processed_files(print_logs=False)

# if __name__ == "__main__":
# 	FindIssues(r"C:\Users\saharm\Desktop\movie_folder\170126_M279793", data_type='npy')

# 	# for f in os.listdir(r"M:\M395926"):
# 	# 	FindIssues(os.path.join(r"M:\M395926", f))

		