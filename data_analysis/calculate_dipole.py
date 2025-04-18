import os
import numpy as np
import time
from joblib import Parallel,delayed
import itertools
import sys

# Function to read box length from trajectory file header
def read_header(text_file):
    for i in range(9):
        line = text_file.readline()
        if i==5:  
            Lsplit = line.split()
            L = float(Lsplit[1]) - float(Lsplit[0])
    return L

# Get command line arguments for molecule name, simulation number and type
MOL = sys.argv[1]
SIM_NO = int(sys.argv[2])
simulation_type = str(sys.argv[3])

# Set number of beads based on simulation type
# P=32 for path integral simulations, P=1 for classical
if simulation_type in ['PI_H','PI_D']:
    P = 32
elif simulation_type in ['Classical']:
    P = 1

def dipole_calculator(MOL,SIM_NO):
    # Read atomic charges from force field file
    charges = []
    atom_classes = []
    type_to_class = dict()
    ff_file = open('{}/ff_{}.xml'.format(MOL,MOL),'r')
    ff_lines = ff_file.readlines()
    for ff_line in ff_lines:
        if '<Atom name=' in ff_line:
            atom_classes.append(ff_line.split('"')[-2])
        if '<Atom type' in ff_line:
            type_to_class[ff_line.split('"')[1]] = float(ff_line.split('"')[3])
    for i in atom_classes:
        charges.append(type_to_class[i])
    atomspermolecule = len(charges)
    atomcount = int(np.ceil(5000/atomspermolecule)) * atomspermolecule

    # Determine trajectory files to read based on simulation type
    if P == 1: # Classical simulation
        files = ['{}/{}/trajectory.lammpstrj'.format(MOL,SIM_NO)] 
        linecount = float(os.popen('wc -l "{}/{}/trajectory.lammpstrj"'.format(MOL,SIM_NO)).read().split()[0])
        frames = [1000]

    elif P > 1: # Path integral simulation
        if os.path.isfile('{}/{}/Ptrajectory.lammpstrj'.format(MOL,SIM_NO)):
            files = ['{}/{}/Ptrajectory.lammpstrj'.format(MOL,SIM_NO)] 
            linecount = float(os.popen('wc -l "{}/{}/Ptrajectory.lammpstrj"'.format(MOL,SIM_NO)).read().split()[0])
            frames = [1000]

        else:
            # Handle split trajectory files
            for file_count in [2,3,4]:
                if os.path.isfile('{}/{}/Ptrajectory{}_{}.lammpstrj'.format(MOL,SIM_NO,file_count,file_count)) == True:
                    break
            files = ['{}/{}/Ptrajectory{}_{}.lammpstrj'.format(MOL,SIM_NO,file_count,file_number) for file_number in range(1,file_count+1)]
            linecounts = [float(os.popen('wc -l "{}/{}/Ptrajectory{}_{}.lammpstrj"'.format(MOL,SIM_NO,file_count,file_number)).read().split()[0]) for file_number in range(1,file_count+1)]
            frames = [float(os.popen('wc -l "{}/{}/Ptrajectory{}_{}.lammpstrj"'.format(MOL,SIM_NO,file_count,file_number)).read().split()[0])/((atomcount*P)+9) for file_number in range(1,file_count+1)]
            frames = np.array(frames).astype(int)

    # Prepare arrays for dipole calculation
    charges = np.array([charges,]*3).transpose()
    moleculecount = int(atomcount/atomspermolecule)
    dipoles = np.zeros((np.sum(frames),3))
    M2list = []
    frame_no = 0
    # Loop through trajectory files
    for index,obj in enumerate(files):
        with open(obj,'r') as text_file:
            for i in range(int(frames[index])):
                # Read box size and atom coordinates
                L = read_header(text_file)
                halfL = L/2
                N = atomcount*P
                frame_coords = np.zeros((N,3))
                for atom_iter in range(N):
                    line = text_file.readline()
                    frame_coords[atom_iter] = np.array([float(x) for x in line.split()[-3:]])
                dipole_sums = np.zeros(3)
                # Calculate dipole moment for each bead number
                for bead_no in range(P):
                    totaldipole = np.zeros(3)
                    for j in range(moleculecount):
                        # Get coordinates for one molecule
                        molecule_coords = np.zeros((atomspermolecule,3))
                        for k in range(atomspermolecule):
                            molecule_coords[k] = frame_coords[j*P*atomspermolecule+(P*k+bead_no)]
                        # Apply minimum image convention
                        first_atom = molecule_coords[0].copy()
                        for k in range(3):
                            molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])>(halfL),molecule_coords[:,k]-L,molecule_coords[:,k])
                            molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])<(-halfL),molecule_coords[:,k]+L,molecule_coords[:,k])
                        
                        # Calculate molecular dipole
                        molecule_coords = molecule_coords * charges
                        totaldipole += np.sum(molecule_coords,axis=0)
                    dipole_sums += totaldipole

                # Calculate molecular dipole
                frame_dipole = dipole_sums/P
                dipoles[frame_no] = frame_dipole
                M2list.append(np.sum(frame_dipole**2))
                frame_no += 1
    return dipoles, np.array(M2list)

# Calculate dipoles and M^2
dipoles,M2 = dipole_calculator(MOL,SIM_NO)

# Save results
dipole_output = open('{}_dipole_outputs/{}_{}.di'.format(simulation_type,MOL,SIM_NO),'wb')
np.save(dipole_output,dipoles)
np.save(dipole_output,M2)
dipole_output.close()