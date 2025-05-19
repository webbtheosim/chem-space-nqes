import os
import numpy as np
import time
from joblib import Parallel,delayed
import itertools
import sys
from itertools import combinations
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import re

def HB_analysis(dirname,simno):

    path = '/scratch/gpfs/bu9134/Classical_TAFFI_Simulations/{}/{}'.format(dirname,simno)
    os.chdir(path)

    #Check Presence of H-bonds, identify donors, hydrogens, and acceptors
    f = open('ff_{}.xml'.format(dirname)).readlines()

    Hbondcheck = False

    hydrogendonors = []
    acceptors = []

    #convert atom types to elements to identify H-bonds later
    type_to_element = dict()
    for line in f:
        if '<Type name' in line:
            split = line.split('"')
            type_to_element[split[1][1:]] = split[5]
            if split[5] in ['O','N','F']:
                acceptors.append(split[1][1:])

    #identify presence of OH, NH or HF groups
    for line in f:
        if '<Bond class' in line:
            split = line.split('"')
            type1 = split[1][1:]
            type2 = split[3][1:]
            element1 = type_to_element[type1]
            element2 = type_to_element[type2]
            if sorted([element1,element2]) in [['H','N'],['H','O'],['F','H']]:
                if element1 == 'H':
                    hydrogendonors.append([type1,type2])
                elif element2 == 'H':
                    hydrogendonors.append([type2,type1])
                #confirm presence of H-bonds
                Hbondcheck = True
    
    #Remove duplicate hydrogen-donor pairs in case the system has two identical H-bond functional groups (don't want them treated differently)
    hydrogendonors_unique = []
    [hydrogendonors_unique.append(x) for x in hydrogendonors if x not in hydrogendonors_unique] 
    hydrogendonors_unique = np.array(hydrogendonors_unique,dtype=object)
    

    if Hbondcheck == True:
        f = open('HB_analysis.out', 'wb')
        np.save(f,acceptors,allow_pickle=True)
        np.save(f,hydrogendonors_unique,allow_pickle=True)
        f.close()
        for acceptor in acceptors:
            for hydrogendonor in hydrogendonors_unique:
                f = open('HB_analysis.out', 'ab')
                u = mda.Universe('modified_init.pdb','modified_trajectory.lammpstrj', format='LAMMPSDUMP', dt=10)
                
                hydrogen = hydrogendonor[0]
                donor = hydrogendonor[1]

                HB = HydrogenBondAnalysis(
                universe=u,
                donors_sel= "name {}".format(donor),
                hydrogens_sel="name {}".format(hydrogen),
                acceptors_sel="name {}".format(acceptor),
                d_a_cutoff=3.6,
                d_h_a_angle_cutoff=150,
                update_selections=False)
                
                HB.run(
                start=None,
                stop=None,
                step=None,
                verbose=True)
            
                HB_results = HB.results.hbonds

                np.save(f, HB_results,allow_pickle=True)
                f.close()
        print(dirname,simno)
