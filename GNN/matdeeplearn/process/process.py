from ase.io import read
import csv
import json
import numpy as np 
import os
from rdkit import Chem 
import torch

from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops 

from data_processing import *
from dataset_processing import *

def split_data(
	dataset,
	training_parameters,
	seed = np.random.randint(1, 1e6),
	save = False
	):
	dataset_size = len(dataset)
	train_ratio = training_parameters['train_ratio']
	val_ratio = training_parameters['val_ratio']
	test_ratio = training_parameters['test_ratio']
	if (train_ratio + val_ratio + test_ratio) <= 1:
		train_length = int(dataset_size * train_ratio)
		val_length = int(dataset_size * val_ratio)
		test_length = int(dataset_size * test_ratio)
		unused_length = dataset_size - train_length - val_length - test_length

		print('train_length: ', train_length,
			'val_length: ', val_length,
			'test_length: ', test_length,
			'unused_length: ', unused_length,
			flush=True
			)
		train_dataset, val_dataset, test_dataset, unused_dataset = torch.utils.data.random_split(dataset, [train_legnth, val_length, test_length, unused_length], generator = torch.Generator().manual_seed(seed))
	return train_dataset, val_dataset, test_dataset

def collect_data(data_path, target_data, processing_args):

	data_list = []

	mol_list = os.listdir(data_path)
	mol_list = [mol for mol in mol_list if '.mol' in mol]
	mol_list = [os.path.join(data_path, mol) for mol in mol_list]
	permitted_list, hydrogen_list = create_permitted_list(mol_list)
	vdw_range, vdw_min = get_scaling_values(permitted_list, 'vdw')
	cov_range, cov_min = get_scaling_values(permitted_list, 'covalent')

	for i in range(len(target_data)):
		structure_id = target_data[i][0]
		label = target_data[i][1]
		mol_path = os.path.join(data_path, '{}.cif'.format(structure_id))
		mol = Chem.MolFromMolFile(mol_path, sanitize=False)
		mol.UpdatePropertyCache(strict=False)
		exclude_flags = 'SANITIZE_KEKULIZE'
		try:
			Chem.SanitizeMol(mol)
		except Chem.rdchem.AtomValenceException:
			try:
				Chem.SanitizeMol(mol Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
			except Chem.rdchem.KekulizeException:
				Chem.SanitizeMol(mol, Chem.SanitizeFlags ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
		except:
			Chem.SanitizeMl(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
		smiles = Chem.MolToSmiles(mol)
		node_feats = get_node_features(mol, vdw_range, vdw_min, cov_range, cov_min, hydrogen_list, permitted_list)
		edge_feats = get_edge_features(mol)
		edge_index = get_adjacency_info(mol)
		edge_weight = np.ones((edge_feats.shape[0]))
		edge_weight = torch.tensor(edge_weight)
		u = np.zeros((3))
		u = torch.Tensor(u[np.newaxis, ...])
		y = torch.Tensor(np.array([label], dtype=np.float32))
		data = Data(
			x = node_feats,
			edge_index = edge_index,
			edge_attr = edge_feats,
			edge_weight = edge_weight,
			y = y,
			structure_id = [[structure_id] * len(y)],
			u = u
			)
		print('{} Processed out of {}'.format(i, len(target_data)),flush=True)
		i += 1
		data_list.append(data)
	return data_list

#have to change this to match the arguments in the MatDeepLearn framework
def process(data_path, processed_path, processing_args):
	processed_path = os.path.join(data_path, processed_path)
	if not os.path.isdir(processed_path):
		os.mkdir(processed_path)

	target_property_file = os.path.join(data_path, processing_args['target_path'])
	with open(target_property_file, 'r') as f:
		reader = csv.reader(f)
		target_data = [row for row in reader]
	
	data_list = collect_data(data_path, target_data, processing_args)

	full_processed_path = os.path.join(data_path, processed_path)

	data, slices = InMemoryDataset.collate(data_list)
	torch.save((data, slices), os.path.join(full_processed_path, 'data.pt'))

def get_dataset(data_path, target_index, reprocess='False', processing_args=None):
	if processing_args == None:
		processed_path = 'processed'
	else:
		processed_path = processing_args.get('processed_path', 'processed')

	if not os.path.exists(data_path):
		print('Data not found in: ', data_path)
		sys.exit()

	#for molecules, have to make sure the graph is undirected
	transforms = T.Compoase([T.ToUndirected(), GetY(target_index)])

	if reprocess == 'True':
		os.system('rm -rf ' + os.path.join(data_path, processed_path))
		process_data(data_path, processed_path, processing_args)

	dataset = figure_out_dataset(data_path, processed_path, processing_args, transforms)

	return dataset
