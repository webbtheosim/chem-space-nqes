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

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

pdb = PDBFile('init.pdb') 
list_of_files = os.listdir(os.getcwd()) #list of files in the current directory
for each_file in list_of_files:
    if '.xml' in each_file:
        ff_file = each_file
        break
forcefield = ForceField(ff_file)

##### NVE Relaxation
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system = WH.WaldmanHagler_LJ(system)

integrator = VerletIntegrator(timestep*picoseconds)
simulation = Simulation(topology=pdb.topology,system=system,integrator=integrator,platform=platform,platformProperties=properties)
simulation.context.setPositions(positions=pdb.positions)
simulation.minimizeEnergy()
simulation.step(steps=200000)

simulation.context.setVelocitiesToTemperature(400*kelvin,vel_seed)
relaxed_state = simulation.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)
relaxed_positions = relaxed_state.getPositions()
relaxed_velocities = relaxed_state.getVelocities()

post_minimize = time.time()
print('Minimization: {:.3f} s'.format(post_minimize-start))
#####

##### NVT Annealing
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system = WH.WaldmanHagler_LJ(system)
integrator = LangevinIntegrator(400*kelvin,1/picosecond,timestep*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
simulation.context.setPositions(relaxed_positions)
simulation.context.setVelocities(relaxed_velocities)

simulation.step(steps=400000) 
for i in range(100):
    simulation.step(steps=int(1600000/100))
    anneal_temp = 400 - (i+1)*(400-temperature)/100
    integrator.setTemperature(anneal_temp*kelvin)

annealed_state = simulation.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)
annealed_positions = annealed_state.getPositions()
annealed_velocities = annealed_state.getVelocities()

post_anneal = time.time()
print('Anneal: {:.3f} s'.format(post_anneal-post_minimize))
#####

##### NPT Simulation
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system.addForce(MonteCarloBarostat(pressure*bar,temperature*kelvin,25))
system = WH.WaldmanHagler_LJ(system)

integrator = LangevinIntegrator(temperature*kelvin,1/picosecond,timestep*picoseconds)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

simulation.context.setPositions(annealed_positions)
simulation.context.setVelocities(annealed_velocities)

simulation.step(steps=6000000)

post_equil = time.time()
print('Equilibration: {:.3f} s'.format(post_equil-post_anneal))

simulation.reporters.append(PDBReporter('trajectory.lammpstrj',print_velocities,coords_freq))
simulation.reporters.append(StateDataReporter(False,"thermo.avg", thermo_freq, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))
simulation.step(steps=20000000)  

post_data = time.time()
print('Data Collection: {:.3f} s'.format(post_data-post_equil))
#####

Restart.save_simulation('classical.save',simulation,'classical')
