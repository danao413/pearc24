import os
import torch
from torch_geometric.data import InMemoryDataset

class MoleculeDataset(InMemoryDataset):
	def __init__(
		self,
		data_path,
		processed_path = 'processed',
		transform = None,
		pre_transform = None
		):
		self.data_path = data_path
		self.processed_path = processed_path
		super(MoleculeDataset, self).__init__(data_path, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_path[0])

	@property 
	def raw_file_names(self):
		return []

	@property 
	def processed_file_names(self):
		file_names = ['data.pt']
		return file_names

	@property 
	def processed_dir(self):
		return os.path.join(self.data_path, self.processed_path)


class GetY(object):
	def __init__(self, index=0):
		self.index = index 
	def __call__(self, data):
		if self.index != -1:
			data.y = data.y[0].item()
		return data 

def figure_out_dataset(data_path, processed_path, processing_args, transforms):

	full_processed_path = os.path.join(data_path, processed_path)

	if os.path.exists(os.path.join(full_processed_path, 'data.pt')):
		dataset = MoleculeDataset(
			data_path,
			processed_path,
			transforms
			)
	else:
		process_data(data_path, processed_path, processing_args)
		if os.path.exists(os.path.join(full_processed_path, 'data.pt')):
			dataset = MoleculeDataset(
				data_path, 
				processed_path,
				transforms
				)

	return dataset
