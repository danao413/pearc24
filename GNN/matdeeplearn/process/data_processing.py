from ase.io import read
import numpy as np 
import os
from rdkit import Chem
import torch
from torch_geometric.utils import add_self_loops

def one_hot_encoding(x, permitted_list):
	if x not in permitted_list:
		x = permitted_list[-1]

	binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

	return binary_encoding

def create_permitted_list(mol_list):
	permitted_dict = {}
	hydrogen_max = 0
	hydrogen_min = 10000
	for mol in mol_list:
		hydrogen_count = 0
		try:
			s = read(mol)
		except:
			continue
		symbols = s.get_chemical_symbols()
		numbers = s.get_atomic_numbers()
		for idx, symbol in enumerate(symbols):
			if symbol == 'H':
				hydrogen_count += 1
			if symbol not in permitted_dict:
				permitted_dict[symbol] = numbers[idx]
		if hydrogen_count > hydrogen_max:
			hydrogen_max = hydrogen_count
		if hydrogen_count < hydrogen_min:
			hydrogen_min = hydrogen_count
	permitted_dict = {k: v for k, v in sorted(permitted_dict.items(), key=lambda item: item[1])}
	permitted_list = list(permitted_dict.keys())
	hydrogen_list = list(range(hydrogen_min, hydrogen_max))
	return permitted_list, hydrogen_list

def get_scaling_values(permitted_list, cat):
	#get some of the scaling values
	#for later data preprocessing
	max_atom = permitted_list[-1]
	min_atom = permitted_list[0]
	if cat == 'vdw':
		max_value = Chem.GetPeriodicTable().GetRvdw(max_atom)
		min_value = Chem.GetPeriodicTable().GetRvdw(min_atom)
	elif cat == 'covalent':
		max_value = Chem.GetPeriodicTable().GetRcovalent(max_atom)
		min_value = Chem.GetPeriodicTable().GetRcovalent(min_atom)
	value_range = max_value - min_value 
	return value_range, min_value 

def create_atom_features(atom, vdw_range, vdw_min, covalent_range, covalent_min, hydrogen_list, permitted_list):
	atom_fea = one_hot_encoding(str(atom.GetSymbol()), permitted_list)
	n_heavy_neighbors = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, 'MoreThanFour'])
	formal_charge = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, 'Extreme'])
	hybrid = one_hot_encoding(str(atom.GetHybridization()), ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'])
	radical = one_hot_encoding(int(atom.GetNumRadicalElectrons()), [1, 2, 3, 4, 'MoreThanFour'])
	ring = [int(atom.IsInRing())]
	aromatic = [int(atom.GetIsAromatic())]
	vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - vdw_min) / vdw_range)]
	covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - covalent_min) / covalent_range)]
	chirality = one_hot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'])
	n_hydrogens = one_hot_encoding(int(atom.GetTotalNumHs()), hydrogen_list) 
	atom_fea.extend(n_heavy_neighbors)
	atom_fea.extend(formal_charge)
	atom_fea.extend(hybrid)
	atom_fea.extend(radical)
	atom_fea.extend(ring)
	atom_fea.extend(aromatic)
	atom_fea.extend(vdw_radius_scaled)
	atom_fea.extend(covalent_radius_scaled)
	atom_fea.extend(chirality)
	atom_fea.extend(n_hydrogens)
	return atom_fea

def create_edge_features(bond):
	bond_fea = []
	allowed_bonds = [1, 1.5, 2, 3]
	bond_type = one_hot_encoding(int(bond.GetBondTypeAsDouble()), allowed_bonds)
	bond_conj = [int(bond.GetIsConjugated())]
	bond_ring = [int(bond.IsInRing())]
	bond_stereo = one_hot_encoding(str(bond.GetStereo()), ['STEREOZ', 'STEREOE', 'STEREOANY', 'STEREOONE'])
	bond_fea.extend(torch.tensor(bond_type))
	bond_fea.extend(bond_conj)
	bond_fea.extend(bond_ring)
	bond_fea.extend(bond_stereo)
	return bond_fea

def get_node_features(mol, vdw_range, vdw_min, cov_range, cov_min, hydrogen_list, permitted_list):
	total_node_feat = []
	for atom in mol.GetAtoms():
		atom_fea = create_atom_features(
			atom,
			vdw_range,
			vdw_min,
			cov_range,
			cov_min,
			hydrogen_list,
			permitted_list
			)
		total_node_feat.append(atom_fea)
	total_node_feat = np.asarray(atom_fea)
	total_node_feat = torch.from_numpy(total_node_feat)
	return total_node_feat

def get_edge_features(mol):
	total_edge_feat = []
	for bond in mol.GetBonds():
		edge_feat = create_edge_features(bond)
		total_edge_feat += [edge_feat, edge_feat]
	total_edge_feat = np.asarray(total_edge_feat)
	total_edge_feat = torch.from_numpy(total_edge_feat)
	return total_edge_feat

def get_adjacency_info(mol):
	edge_indices = []
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		edge_indices += [[i, j], [j, i]]
	edge_indices = torch.tensor(edge_indices).view(2, -1)
	edge_indices, edge_weight = add_self_loops(
		edge_indices, num_nodes = len(mol.GetAtoms())
		)
	edge_indices = edge_indices.to(torch.long)
	return edge_indices
