import os
import shutil
import h5py
import numpy as np
from generate_dfs import AnalysisDataFrames
from cam2_downsample_movies import decimate_JCamF, concat_all_chunks
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
		if type(self.dates) is not list:
			self.dates = [dates]
		if run_all==True:
			self.dates = self.date_list

		for d in self.dates:
			self.date = d
			self.create_folders()
			#for DoC
			self.sess_type = "DoC"
			self.sess = AnalysisDataFrames(mouse_id = self.mouse_id,
											dates = self.date,
											main_dir = self.start_dir,
											save = True)
			self.session_path = self.sess.session_path
			
			self.find_2x_movie()
			self.chunks = np.arange(0, self.movie_2x.shape[0], 10000)

			self.downsample_movie()
			self.concatinate_movie_chunks()



		def create_folders(self):
			folder_name = "{}-{}".format(self.date, self.mouse_id)
			folder = os.path.join(self.xfer_dir, folder_name)
		    if not os.path.exists(folder):
		        os.makedirs(folder)

		    self.folder = folder
		    return self.folder

		def find_2x_movie(self):
			for f in os.listdir(self.session_path):
				# if "2_2_1.npy" in f:
				# 	movie_loc = os.path.join(self.session_path, f)
				#	movie_2x = np.load(movie_loc)
				#check if that's correct
				if ("2_2_1.h5" in f) and ("cam2" in f):
					movie_loc = os.path.join(self.session_path, f)
					m = h5py.File(movie_loc, 'r')
					movie_2x = m['data']
				
				self.movie_2x = movie_2x
				return self.movie_2x

		def downsample_movie(self, hzs=[20,100]):
			chunk_folders = []
			for hz in hzs:
				tc = int(100./hz)

				temp_path = os.path.join(r"C:\\some_local_path", self.sess_type, "hz={}".format(hz))
				chunk_folders.append(temp_path)

				for n, c in enumerate(self.chunks):
			        try:
			            v1 = h[self.chunks[n]:self.chunks[n+1], :, :]
			            output_name = "{}_{}_{}_{}_{}_256x256_{}hz.h5".format(self.chunks[n]/tc, self.chunks[n+1]/tc, self.date, self.mouse_id, self.sess_type, hz)
			        except IndexError:
			            v1 = h[self.chunks[n]:, :, :]
			            output_name = "{}_end_{}_{}_{}_256x256_{}hz.h5".format(self.chunks[n]/tc, self.date, self.mouse_id, self.sess_type, hz)

			        print 'decimating {}'.format(n) 
			        decimate_JCamF(input_array = v1, 
			                        output_file = os.path.join(temp_path, output_name), 
			                        spatial_compression = 2, 
			                        temporal_compression = tc)
			    	del v1

			    if hz == 20:
					for f in os.listdir(temp_path):
						s = int(f.split("_")[0])/5
						try:
							e = int(f.split("_")[1])/5
						except ValueError:
							e = "end"
						os.rename(os.path.join(temp_path,f), os.path.join(temp_path,"{}_{}_{}_{}_{}_256x256_20hz.h5".format(s, e, self.date, self.mouse_id, self.sess_type)))

			        
			self.chunk_folders = chunk_folders
			return self.chunk_folders

		def concatinate_movie_chunks(self):
			for fol in self.chunk_folders:
				hz = int(os.path.split(fol)[0].split("=")[1])
				file_name = "{}_{}_{}_256x256_{}hz.h5".format(self.date, self.mouse_id, self.sess_type, hz)
				output_file = os.path.join(self.folder, file_name)
				chunks_loc = fol
				length = int(self.movie_2x.shape[0] / (100./hz))
				concat_all_chunks(output_file, chunks_loc, length)

		def xfer_del_files(self):
			for f in os.listdir(self.session_path):
				if "matrix" in f:
					matrix_df = os.path.join(self.session_path, f)
				if "trial_df" in f:
					trial_df = os.path.join(self.session_path, f)

			shutil.copy2(matrix_df, self.folder)
			shutil.copy2(trial_df, self.folder)

			file_20 = os.path.join(self.folder, "{}_{}_{}_256x256_20hz.h5".format(self.date, self.mouse_id, self.sess_type))
			file_100 = os.path.join(self.folder, "{}_{}_{}_256x256_100hz.h5".format(self.date, self.mouse_id, self.sess_type))

			if os.exists(file_20):
				fol_20 = [cf for cf in self.chunk_folders if "20hz" in cf][0]
				shutil.rmtree(fol_20)
			if os.exists(file_100):
				fol_100 = [cf for cf in self.chunk_folders if "100hz" in cf][0]
				shutil.rmtree(fol_100)			














