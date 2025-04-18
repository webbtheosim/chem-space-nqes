The input simulation files in this repository can be used to run OpenMM 7.7.0 with additional changes in the source files.

These changes include implementation of Waldman-Hagler mixing rules, a modified .lammpstrj reporter, and trajectory and thermodynamic reporters for PIMD simulations to obtain information on all beads.

Before running any simulations, copy the contents of 'pdbfile.py', 'pdbreporter.py', 'statedatareporter.py', and '__init__.py' in this directory to their respective files in the $PATH_TO_OPENMM/openmm/app directory within the environment it was installed.
