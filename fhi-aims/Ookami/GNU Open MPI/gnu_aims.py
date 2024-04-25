from ase.calculators.aims import Aims 
from ase.io import read
import os

struct_file = 'geometry.in'

s = read(struct_file)

os.environ['AIMS_SPECIES_DIR'] = '/lustre/scratch/danaoconnor/FHI-aims/fhi-aims.231212/species_defaults/defaults_2020/light'
os.environ['ASE_AIMS_COMMAND'] = 'mpiexec -n 48 /lustre/scratch/danaoconnor/FHI-aims/bin/aims.231212.scalapack.mpi.x > aims.out' 

calc = Aims(xc = 'PBE',
			spin = 'none',
			charge = 0,
			relativistic = ('atomic_zora', 'scalar'),
			occupation_type = ('gaussian', 0.01),
			mixer = 'pulay',
			n_max_pulay = 8,
			charge_mix_param = 0.02,
			sc_accuracy_rho = 0.0001,
			sc_accuracy_eev = 0.01,
			sc_accuracy_etot = 1e-06,
			KS_method = 'parallel',
			empty_states = 6,
			basis_threshold = 1e-05,
			k_grid = (3, 3, 3),
			vdw_correction_hirshfeld = ''
	)

s.calc = calc 

s.get_potential_energy()
