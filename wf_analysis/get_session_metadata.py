import os
from WF_utilities import choose_date


class SessionMetadata(object):
	def __init__(self, mouse_id, date, main_dir='default'):
		if main_dir=='default':
			self.main_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\CorticalMapping\IntrinsicImageData"
		else:
			self.main_dir = main_dir		
		
		self.mouse_id = mouse_id
				
		if type(self.dates) is not list:
			self.dates = [dates]
		else:
			self.dates = dates
		self.generate_manifest()
		self.numpy_or_h5()
		



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
	




