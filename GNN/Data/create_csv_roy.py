from ase import Atoms
from ase.io import write
import csv
import os 
import pandas as pd

geometry = {}
lattice_vectors = {}

def extract_geometry(out_file):
	with open(out_file, 'r') as f:
		for line in f:
			if 'Lengths:' in line:
				lattice_vectors['x'] = float(line.split()[1])
				lattice_vectors['y'] = float(line.split()[2])
				lattice_vectors['z'] = float(line.split()[3])
			if '( 0.0000,  0.0000,  0.0000)' in line:
				geo_key = line.split()[0] + line.split()[1]
				x = float(line.split()[2])
				y = float(line.split()[3])
				z = float(line.split()[4])
				geometry[geo_key] = [x, y, z]
			if 'Gap:' in line:
				bandgap = float(line.split()[1])
		f.close()

	return geometry, lattice_vectors, bandgap 

def build_atoms(geometry, lattice_vectors):
	positions = list(geometry.values())
	lattice_vectors = list(lattice_vectors.values())
	elements = list(geometry.keys())

	new_elements = []

	for element in elements:
		split_idx = 0
		int_split = element.split()[split_idx]
		try:
			ele_int = int(int_split)
			new_ele = str(element[split_idx])
			new_elements.append(new_ele)
		except ValueError:
			split_idx -= 1
			new_ele = str(element[split_idx:])
			new_elements.append(new_ele)

	ase_object = Atoms(positions=positions, symbols=new_elements, cell=lattice_vectors, pbc=True)
	return ase_object



dft_dir = os.path.abspath('ROY_Calculations')
extract_dir = os.path.abspath('Final_ROY')
if not os.path.isdir(extract_dir):
	os.mkdir(extract_dir)
dft_list = os.listdir(dft_dir)

gap_dict = {}

os.chdir(dft_dir)

for dft in dft_list:
	os.chdir(dft)
	file_list = os.listdir(os.getcwd())
	file_list = [file_name for file_name in file_list if 'j' in file_name]
	file_list = [file_name.split('.')[0] for file_name in file_list]
	file_list = [int(file_name.split('_')[-1]) for file_name in file_list]
	file_list.sort()
	out_file = file_list[-1]
	out_file = 'j_{}.out'.format(out_file)
	geometry, lattice_vectors, bandgap = extract_geometry(out_file)
	atoms = build_atoms(geometry, lattice_vectors)
	gap_dict[dft] = bandgap
	write_path = os.path.join(extract_dir, dft)
	write('{}.cif'.format(write_path), atoms, format='cif')
	os.chdir(dft_dir)

os.chdir(extract_dir)

with open('targets.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	for key, value in gap_dict.items():
		writer.writerow([key, value])
csvfile.close()

