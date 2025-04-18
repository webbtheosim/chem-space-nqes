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

TIMESTEP = 0.0005 # in picoseconds
TEMPERATURE = 298.15 # Kelvin
PRESSURE = 1.01325 # bar (=1 atm)
THERMO_FREQ = 1/TIMESTEP # every 1 ps
COORDS_FREQ = 10/TIMESTEP # every 10 ps
PRINT_VELOCITIES = False # set True to print velocities
VELOCITY_SEED = np.random.randint(0,10000)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

PDB = PDBFile('init.pdb') 
MOLECULE_NAME = str(os.getcwd().split('/')[-2])
FF_FILE = 'ff_{}.xml'.format(MOLECULE_NAME)
FF = ForceField(FF_FILE)

##### Minimization & NVE
system = FF.createSystem(topology=PDB.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system = WH.WaldmanHagler_LJ(system)

integrator = VerletIntegrator(TIMESTEP*picoseconds)
simulation = Simulation(topology=PDB.topology,system=system,integrator=integrator,platform=platform,platformProperties=properties)
simulation.context.setPositions(positions=PDB.positions)

simulation.minimizeEnergy() # OpenMM Minimization Procedure
simulation.step(steps=200000) # 0.1 ns NVE simulation

simulation.context.setVelocitiesToTemperature(400*kelvin,VELOCITY_SEED)
relaxed_state = simulation.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)
relaxed_positions = relaxed_state.getPositions()
relaxed_velocities = relaxed_state.getVelocities()

post_minimize = time.time()
print('Minimization: {:.3f} s'.format(post_minimize-start))
#####

##### NVT Annealing from 400 to 298.15 K
system = FF.createSystem(topology=PDB.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system = WH.WaldmanHagler_LJ(system)

integrator = LangevinIntegrator(400*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=PDB.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
simulation.context.setPositions(relaxed_positions)
simulation.context.setVelocities(relaxed_velocities)

simulation.step(steps=400000) # NVT simulation at 400 K for 0.2 ns
for i in range(100): # Anneal from 400 to 298.15 K for 0.8 ns
    simulation.step(steps=int(1600000/100))
    anneal_temp = 400 - (i+1)*(400-TEMPERATURE)/100
    integrator.setTemperature(anneal_temp*kelvin)

annealed_state = simulation.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)
annealed_positions = annealed_state.getPositions()
annealed_velocities = annealed_state.getVelocities()

post_anneal = time.time()
print('Anneal: {:.3f} s'.format(post_anneal-post_minimize))
#####

##### NPT Equilibration & Data Collection
system = FF.createSystem(topology=PDB.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system.addForce(MonteCarloBarostat(PRESSURE*bar,TEMPERATURE*kelvin,25))
system = WH.WaldmanHagler_LJ(system)

integrator = LangevinIntegrator(TEMPERATURE*kelvin,1/picosecond,TIMESTEP*picoseconds)
simulation = Simulation(topology=PDB.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)

simulation.context.setPositions(annealed_positions)
simulation.context.setVelocities(annealed_velocities)

simulation.step(steps=6000000) # 3 ns NPT equilibration

post_equil = time.time()
print('Equilibration: {:.3f} s'.format(post_equil-post_anneal))

simulation.reporters.append(PDBReporter('trajectory.lammpstrj',PRINT_VELOCITIES,COORDS_FREQ))
simulation.reporters.append(StateDataReporter(False,"thermo.avg", THERMO_FREQ, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))
simulation.step(steps=20000000) # 10 ns NPT data collection

post_data = time.time()
print('Data Collection: {:.3f} s'.format(post_data-post_equil))
#####

Restart.save_simulation('classical.save',simulation,'classical')