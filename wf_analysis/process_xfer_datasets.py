import os
import shutil
import numpy as np
import glob2
import time
from generate_dfs import AnalysisDataFrames
from downsample_movies import DownsampleMovies
from WF_utilities import generate_mouse_manifest

class FancyDataPackage(object):
	def __init__(self, mouse_id, dates, xfer_dir, start_dir='default', run_all=False):		
		self.mouse_id = mouse_id
		self.xfer_dir = xfer_dir
		if start_dir=='default':
			self.start_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData"
		else:
			self.start_dir = start_dir

		self.manifest_dict, self.date_list = generate_mouse_manifest(self.mouse_id, self.start_dir)
		self.dates = dates
		if run_all==True:
			self.dates = self.date_list		
		if type(self.dates) is not list:
			self.dates = [dates]

		for d in self.dates:
			self.t0=time.time()
			self.date = d
			print 'starting {} {}'.format(self.mouse_id, self.date)
			self.create_folders()
			self.sess = AnalysisDataFrames(mouse_id = self.mouse_id,
											dates = self.date,
											main_dir = self.start_dir)
			print 'dataframes took {} seconds to make'.format(time.time()-self.t0)
			self.session_path = self.sess.path
			if 'DoC' in self.session_path:
				self.session_path = os.path.split(self.session_path)[0]
			self.find_movies()
			self.downsample_movies()
			self.xfer_del_files(self.session_path)
			print "processing {} took {} seconds".format(self.date, time.time()-self.t0)
			print

	def create_folders(self):
		folder_name = "{}_{}".format(self.date, self.mouse_id)
		folder = os.path.join(self.xfer_dir, folder_name)
		if not os.path.exists(folder):
		    os.makedirs(folder)

		chunk_dir = os.path.join(r"C:\Users\saharm\Desktop\movie_folder", '{}_{}'.format(self.date, self.mouse_id))
		self.folder = folder
		self.chunk_dir = chunk_dir
		return self.folder, self.chunk_dir

	def find_movies(self):
		if 'DoC' in self.sess.path:
			movie_dict = {
						'{}_{}_gcamp_doc'.format(self.date, self.mouse_id): glob2.glob(os.path.join(self.session_path, 'DoC', "*cam2*2_2_1.h5"))[0],
						'{}_{}_hemo_doc'.format(self.date, self.mouse_id): glob2.glob(os.path.join(self.session_path, 'DoC', "*cam1*2_2_1.h5"))[0],
						'{}_{}_gcamp_blank'.format(self.date, self.mouse_id): glob2.glob(os.path.join(self.session_path, 'blank', "*cam2*2_2_1.h5"))[0],
						'{}_{}_hemo_blank'.format(self.date, self.mouse_id): glob2.glob(os.path.join(self.session_path, 'blank', "*cam1*2_2_1.h5"))[0],
						}
		else:
			movie_dict = {'{}_{}_gcamp_doc'.format(self.date, self.mouse_id): glob2.glob(os.path.join(self.session_path, "*2_2_1.npy"))[0]}
			self.f_num = glob2.glob(os.path.join(self.session_path, "*2_2_1.npy"))[0].split('JCamF')[1].split('_')[0]

		self.movie_dict = movie_dict
		return self.movie_dict

	def downsample_movies(self):
		for k in self.movie_dict.keys():
			try:
				print 'downsampling {} 100hz'.format(k)
				t_movie = time.time()
				DownsampleMovies(name_str = k,
								spatial_compression = 2, 
			                    temporal_compression = 1, 
			                    raw_movie_path = self.movie_dict[k], 
			                    chunk_dir = self.chunk_dir,
			                    final_dir = self.folder,
			                    create=True, concat=True)
				print "{} took {} seconds to make".format(k, time.time()-t_movie)


				if 'gcamp' in k:
					print 'downsampling {} 20hz'.format(k)
					t_movie = time.time()
					DownsampleMovies(name_str = k,
									spatial_compression = 2, 
					                temporal_compression = 5, 
					                raw_movie_path = self.movie_dict[k], 
					                chunk_dir = self.chunk_dir,
					                final_dir = self.folder,
					                create=True, concat=True)
					print "{} took {} seconds to make".format(k, time.time()-t_movie)
			except:
				print 'something went wrong with {}'.format(k)
				pass

		
	def xfer_del_files(self, location):
		print 'xferring files'
		shutil.copy2(glob2.glob(os.path.join(location, "**", "*doc_matrix_df.csv"))[0], self.folder)
		shutil.copy2(glob2.glob(os.path.join(location, "**", "*dff_rolling_gaussian*"))[0], self.folder)
		
		if 'DoC' in self.sess.path:
			shutil.copy2(glob2.glob(os.path.join(location, "*blank_matrix_df.csv"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "*blank_hemo_movie_time*"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "*doc_hemo_movie_time*"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "DoC", "*cam2*16_16_1.h5"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "DoC", "*JPhys*"))[0], os.path.join(self.folder, "{}JPhysdoc".format(self.date)))
			shutil.copy2(glob2.glob(os.path.join(location, "blank", "*JPhys*"))[0], os.path.join(self.folder, "{}JPhysblank".format(self.date)))
		else:
			shutil.copy2(glob2.glob(os.path.join(location, "*16_16_1.npy"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "*JPhys{}".format(self.f_num)))[0], os.path.join(self.folder, "{}JPhysdoc".format(self.date)))

		try:
			shutil.copy2(glob2.glob(os.path.join(location, "**", "*summary_figure.png"))[0], self.folder)
			shutil.copy2(glob2.glob(os.path.join(location, "**", "*task=*.png"))[0], self.folder)
		except:
			pass
		if os.path.exists(self.chunk_dir):
			shutil.rmtree(self.chunk_dir)
		print 'all files in {}:'.format(self.folder)
		for f in os.listdir(self.folder):
			print f
	

# if __name__ == "__main__":
# 	FancyDataPackage(mouse_id = 'M395926', 
# 					dates = , 
# 					xfer_dir = r"E:\wf_dataset", 
# 					start_dir='default', 
# 					run_all=False)

	












