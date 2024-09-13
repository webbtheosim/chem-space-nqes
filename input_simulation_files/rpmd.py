from __future__ import print_function
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
import time

start = time.time()

step_count = 100000
timestep = 0.0005#picoseconds
temperature = 298.15  #kelvin
pressure = 1.01325 #bar
thermo_freq = 2000
coords_freq = 20000
print_velocities = False #'velocity.lammpstrj'   #set True to print velocities
P = 32
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

##### NPT Equilibration
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system = WH.WaldmanHagler_LJ(system)

integrator = RPMDIntegrator(P,temperature*kelvin, 1/picosecond, timestep*picoseconds)
barostat = RPMDMonteCarloBarostat(pressure*bar,25)
system.addForce(barostat)
simulation = Simulation(topology=pdb.topology, system=system, integrator=integrator, platform=platform, platformProperties=properties)
simulation.context.setPositions(annealed_positions)
simulation.context.setVelocities(annealed_velocities)

simulation.step(steps=6000000) 

post_equil = time.time()
print('Equilibration: {:.3f} s'.format(post_equil-post_anneal))
#####

##### NPT Simulation
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=PME,nonbondedCutoff=14*angstrom,removeCMMotion=True)
system.addForce(RPMDMonteCarloBarostat(pressure*bar,25))
system = WH.WaldmanHagler_LJ(system)


state = dict()
for i in range(P):
    state[i] = simulation.integrator.getState(i,getPositions=True,getVelocities=True,enforcePeriodicBox=True)
vectors = state[0].getPeriodicBoxVectors()

integrator2 = RPMDIntegrator(P,temperature*kelvin, 1/picosecond, timestep*picoseconds)
simulation2 = Simulation(topology=pdb.topology, system=system, integrator=integrator2, platform=platform, platformProperties=properties)
simulation2.context.setPeriodicBoxVectors(vectors[0],vectors[1],vectors[2])

simulation2.reporters.append(pdbreporter.RPMDReporter('trajectory.lammpstrj',print_velocities,coords_freq))
simulation2.reporters.append(StateDataReporter(True,"thermo.avg", thermo_freq, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))

for i in range(P):
    simulation2.integrator.setPositions(i,state[i].getPositions())
    simulation2.integrator.setVelocities(i,state[i].getVelocities())

output_file = open('QKE.data','w')
totalPE = []
for i in range(int(20000000/2000)):
    simulation2.step(steps=2000)
    totalPE.append(simulation2.integrator.getTotalEnergy().value_in_unit(kilojoules_per_mole))
output_file.write('QKE = {}'.format(str(totalPE)))

post_data = time.time()
print('Data Collection: {:.3f} s'.format(post_data-post_equil))
#####

Restart.save_simulation('rpmd.save',simulation2,P)
