The code is separated by the system: Bridges-2 (AMD processor) and Ookami (A64FX processor).
Included under each system is the compiler/scalapack build used, which contains a text file that describes the modules loaded to compile the code as well as the cmake file used to compile. 
We also include a python script that interfaces with the atomic simulation environment (ASE) to run the compiled FHI-aims code.
We also provide a text file on how to install an anaconda environment with ASE. This should be the only additional python package needed to run the code.
The only changes that should have to be made to the code are:
-the structure file being read in (IMPORTANT: the file format must be acceptable to ase) 
-the "ASE_SPECIES_DIR" which should point to wherever the species defaults live in your compilation of FHI-aims
-the "ASE_RUN_COMMAND" which should be "mpiexec -n <number of cores> <PATH_TO_AIMS_EXECUTABLE> > <output file>.out"
