from __future__ import print_function
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
from openmmtools import integrators
import time

start = time.time()

step_count = 100000
timestep = 0.0005#picoseconds
temperature = 298.15  #kelvin
pressure = 1.01325 #bar
thermo_freq = 2000 
coords_freq = 20000
print_velocities = False #'velocity.lammpstrj'   #set True to print velocities
vel_seed = 4862
P = 32

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

pdb = PDBFile('init.pdb') 
list_of_files = os.listdir(os.getcwd()) #list of files in the current directory
for each_file in list_of_files:
    if '.xml' in each_file:
        ff_file = each_file
    if 'rpmd' in each_file and '.save' in each_file:
        save_file = each_file
forcefield = ForceField(ff_file)

#####
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system.addForce(RPMDMonteCarloBarostat(pressure*bar,25))
system = WH.WaldmanHagler_LJ(system)

integrator = RPMDIntegrator(P,temperature*kelvin, 1/picosecond, timestep*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

Restart.load_simulation(save_file,simulation,P)

integrator.setTemperature((temperature+5)*kelvin)

simulation.step(steps=1000000)

simulation.reporters.append(StateDataReporter(True,"thermoplus.avg", thermo_freq, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))

simulation.step(steps=3000000)

Restart.save_simulation('plus.save',simulation,P)

#####

system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system.addForce(RPMDMonteCarloBarostat(pressure*bar,25))
system = WH.WaldmanHagler_LJ(system)

integrator = RPMDIntegrator(P,temperature*kelvin, 1/picosecond, timestep*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

Restart.load_simulation(save_file,simulation,P)

integrator.setTemperature((temperature-5)*kelvin)

simulation.step(steps=1000000)

simulation.reporters.append(StateDataReporter(True,"thermominus.avg", thermo_freq, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))

simulation.step(steps=3000000)

Restart.save_simulation('minus.save',simulation,P)
