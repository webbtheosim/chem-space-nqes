import time
import numpy as np
import os
import mdtraj as md
import pickle
from itertools import chain
import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import sys

def compute_nonbonded_energy(acceptor_member,donor_member,frame_positions,L):

    interaction_atom_type = {'acceptor':types[acceptor_member],'donor':types[donor_member]}

    interaction_atom_properties = {}

    for atom_type in ['acceptor','donor']:
        for atom_property in ['charge','sigma','epsilon']:
            interaction_atom_properties[atom_type,atom_property] = type_properties[interaction_atom_type[atom_type],atom_property]

    epsilon = 2 * np.sqrt(interaction_atom_properties['acceptor','epsilon']*interaction_atom_properties['donor','epsilon']) \
    *((interaction_atom_properties['acceptor','sigma']**3)*(interaction_atom_properties['donor','sigma']**3))/((interaction_atom_properties['acceptor','sigma']**6)+(interaction_atom_properties['donor','sigma']**6))

    sigma = (((interaction_atom_properties['acceptor','sigma']**6)+(interaction_atom_properties['donor','sigma']**6))/2)**(1/6)
    
    pos1 = frame_positions[acceptor_member]
    pos2 = frame_positions[donor_member]
    delta = abs(pos1-pos2)
    delta = np.where(delta>L/2,L-delta,delta)
    distance = np.linalg.norm(delta)

    coulomb = (1.60217663e-19**2)*(interaction_atom_properties['acceptor','charge'])*(interaction_atom_properties['donor','charge'])/(distance*1e-10*8.8541878128e-12*4*np.pi)/1000*(6.02214076e23)
    LJ = 4*epsilon*((sigma/distance)**12 - (sigma/distance)**6)

    return coulomb+LJ

molinput = sys.argv[1]
simno = sys.argv[2]

for molecule in [molinput]:
    os.chdir('/scratch/gpfs/bu9134/Classical_TAFFI_Simulations/{}/{}'.format(molecule,simno))

    u = mda.Universe('modified_init.pdb','modified_trajectory.lammpstrj', format='LAMMPSDUMP', dt=10)
    residue_count = len(u.residues)

    t = md.load('modified_trajectory.lammpstrj',top='modified_init.pdb')

    f = open('HB_analysis.out','rb')
    acceptors = np.load(f,allow_pickle=True)
    hydrogendonors_unique = np.load(f,allow_pickle=True)


    start = 0
    for acceptor in acceptors:
        for hydrogendonor in hydrogendonors_unique:
            HB_count = 0
            if start == 0:
                HB_arrayog = np.load(f,allow_pickle=True)
                start = 1
            else:
                HB_arrayog = np.concatenate((np.load(f,allow_pickle=True),HB_arrayog))

    com_trajectory = md.load('com.lammpstrj',top='com_init.pdb')

    atomtype1= np.arange(0,residue_count)
    residue_pair_list = [sorted(x) for x in itertools.combinations(atomtype1,2)]

    ######
    ff_file = open('ff_{}.xml'.format(molecule),'r').readlines()
    type_properties = {}

    MOLECULE_N_ATOMS = 0
    for line in ff_file:
        if '<Atom name' in line:
            MOLECULE_N_ATOMS += 1
        if '<Atom type=' in line:
            split = line.split('"')
            atom_type = split[1][1:]
            type_properties[atom_type,'charge'] = float(split[3])
            type_properties[atom_type,'sigma'] = float(split[5]) * 10 #converted to Angstroms
            type_properties[atom_type,'epsilon'] = float(split[7])
        if '<Type name' in line:
            split = line.split('"')
            atom_type = split[1][1:]
            type_properties[atom_type,'element'] = split[5]
        if  '<Bond atom' in line:
            split = line.split('"')
            bond_atoms = [split[1][0],split[3][0]]
            bond_atom_indices = [split[1],split[3]]
            if 'C' in  bond_atoms and 'O' in bond_atoms:
                C_in_bond = bond_atoms.index('C')
                O_in_bond = bond_atoms.index('O')
                C_index = int(bond_atom_indices[C_in_bond][1:])-1
                O_index = int(bond_atom_indices[O_in_bond][1:])-1

    pdb_file = open('modified_init.pdb','r').readlines()
    types = [line.split()[2] for line in pdb_file if 'ATOM ' in line]
    ######

    analysis_frames = np.arange(0,1000,100)

    CO_pairs = [[C_index+i*MOLECULE_N_ATOMS,O_index+i*MOLECULE_N_ATOMS] for i in range(residue_count)]
    directors = md.compute_displacements(t[analysis_frames],CO_pairs)

    pairwise_distances = md.compute_distances(com_trajectory,residue_pair_list) #given in nm, converted to Angstroms later
    
    interaction_energies = []
    interaction_distances = []
    interaction_angles = []
    interaction_HB = []
    for frame_order,frame in enumerate(analysis_frames): #sample every 1 ns
        L = u.trajectory[frame].dimensions[0] #in Angstroms
        frame_positions = u.trajectory[frame].positions #in Angstroms

        HB_array = HB_arrayog[HB_arrayog[:,0]==frame]
        residues = np.zeros((len(HB_array),2))
        AccDon_array = HB_array[:,[1,3]]
        for index1,row in enumerate(AccDon_array):
            for index2,column in enumerate(row):
                residues[index1,index2] = u.atoms[int(AccDon_array[index1,index2])].residue.ix #int(str(t.topology.atom(int(AccDon_array[index1,index2])).residue)[3:]) - 1
        residues = [sorted(i) for i in residues]
        residues_set = {tuple(i) for i in residues}

        for index,pair in enumerate(residue_pair_list):

            interaction_distance = pairwise_distances[frame][index]*10

            if interaction_distance > 9:
                continue

            interaction_distances.append(interaction_distance) #converted to Angstrom here


            if tuple(pair) in residues_set:
                interaction_HB.append(1)
            else:
                interaction_HB.append(0)

            frame_directors = directors[frame_order]
            direction1 = frame_directors[pair[0]]
            direction2 = frame_directors[pair[1]]
            interaction_angles.append(dot(direction1,direction2)/norm(direction1)/norm(direction2))

            pairwise_energy = 0
            acceptor_group = np.array([x.index for x in u.residues[pair[0]].atoms])
            donor_group = np.array([x.index for x in u.residues[pair[1]].atoms])
            for acceptor_member in acceptor_group:
                for donor_member in donor_group:
                    pairwise_energy += compute_nonbonded_energy(acceptor_member,donor_member,frame_positions,L)
            interaction_energies.append(pairwise_energy)

    
    output = open('pairwise_interaction_analysis.out','wb')
    np.save(output,np.array(interaction_energies))
    np.save(output,np.array(interaction_distances))
    np.save(output,np.array(interaction_angles))
    np.save(output,interaction_HB)
    output.close()