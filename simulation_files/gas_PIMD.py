from __future__ import print_function
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmm.app.internal.unitcell import computeLengthsAndAngles
import re
import numpy as np
from scipy.stats import sem
import time

TE_output = open('TEoutput','w')

def mom_angmom_rescale_rpmd(simulation):
    #Resample all Velocities
    state = simulation.integrator.getState(0,getPositions=True)
    velocity = np.array([np.random.normal(0,1,3)*prefactor[i] for i in range(atom_count)])

    #Remove Linear Momentum
    linear_momentum = np.sum(velocity*masses,axis=0)
    velocity = velocity - linear_momentum/np.sum(masses)

    #Remove Angular Momentum
    positions = np.array([np.array([x.value_in_unit(nanometer) for x in i]) for i in state.getPositions()])
    com = np.sum(positions*masses,axis=0)/sum(masses)
    relative_pos = positions-com
    angular_momentum = np.sum(masses*np.cross(relative_pos,velocity),axis=0)

    Ixx = np.sum(masses*np.sum(relative_pos[:,1:]**2,axis=1).reshape(-1,1))
    Iyy = np.sum(masses*np.sum(relative_pos[:,[0,2]]**2,axis=1).reshape(-1,1))
    Izz = np.sum(masses*np.sum(relative_pos[:,:2]**2,axis=1).reshape(-1,1))
    Ixy = -np.sum(masses*(relative_pos[:,0]*relative_pos[:,1]).reshape(-1,1))
    Iyz = -np.sum(masses*(relative_pos[:,1]*relative_pos[:,2]).reshape(-1,1))
    Ixz = -np.sum(masses*(relative_pos[:,0]*relative_pos[:,2]).reshape(-1,1))
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    Iinverse = np.linalg.inv(I)
    omega = np.dot(Iinverse,angular_momentum)
    velocity = velocity - np.cross(omega,relative_pos)

    #Rescale to Desired Temperature
    actualKE = np.sum(0.5*masses*1000*velocity**2) #in kJ/mol (convert to kJ by dividing by 1000, convert to m/s by multiplying by 1000^2)
    desiredKE = 3*atom_count*kb*avogadro/1000*P*temperature/2 #(3/2)*NkT converted to kJ/mol
    multiplier = np.sqrt(desiredKE/actualKE)
    velocity = multiplier*velocity
    velocity = Quantity(value=velocity, unit=nanometer/picosecond)
    return velocity


timestep = 0.0005#picoseconds
temperature = 298.15  #kelvin
pressure = 1.01325 #bar
thermo_freq = 2000
coords_freq = 20000
print_velocities = False #'velocity.lammpstrj'   #set True to print velocities
P = 32
kb = 1.380649e-23
avogadro = 6.02214076e23

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

##########
system = forcefield.createSystem(topology=pdb.topology, nonbondedMethod=CutoffPeriodic,nonbondedCutoff=15*angstrom,removeCMMotion=False)
system = WH.WaldmanHagler_LJ(system)
system = Coulomb.No_Long(system)
for i in range(system.getNumForces()):
    if 'CustomNonbonded' in str(system.getForce(i)):
        system.getForce(i).setUseLongRangeCorrection(False)

masses = []
for i in range(system.getNumParticles()):
    masses.append(float(str(system.getParticleMass(i))[:-3]))
masses = np.array(masses)/(1000*avogadro) #masses in kg/particle
prefactor = np.sqrt(kb*temperature*P/masses) / 1000 #convert to nm/ps from m/s
masses = masses*avogadro #convert masses to kg/mol for Angular Momentum and Rescaling
masses = masses.reshape(-1,1)
atom_count = len(masses)

integrator = RPMDIntegrator(P,298.15*kelvin, 10/picosecond, timestep*picoseconds)
simulation = Simulation(topology=pdb.topology,system=system,integrator=integrator,platform=platform,platformProperties=properties)
simulation.context.setPositions(positions=pdb.positions)
simulation.context.setVelocitiesToTemperature(298.15*kelvin,np.random.choice(range(0,10000)))
simulation.step(1)
simulation.integrator.setApplyThermostat(False)
simulation.minimizeEnergy()

for i in range(P):
    modified_velocity = mom_angmom_rescale_rpmd(simulation)
    simulation.integrator.setVelocities(i,modified_velocity)

simulation.step(2000000)

simulation.reporters.append(StateDataReporter(True,"gas_thermo.avg", thermo_freq, step=True, time=True, density=True, totalEnergy=True, kineticEnergy=True, volume=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(pdbreporter.RPMDReporter('gas_trajectory.lammpstrj',print_velocities,coords_freq))

stepct = 8000000 #4 ns
st = time.time()

collision_frequency = 10 #in /ps
collision_probability = 1-np.exp(-timestep*collision_frequency)
N = P*atom_count
batch_size = 400
batch_count = int(stepct/batch_size)
step_count = 0
TE = [] 
for batch in range(batch_count):
    indices = np.where(np.random.uniform(low=0, high=1, size=(N*batch_size,))<collision_probability)
    indices = np.array([(i//(N),(i%N)//atom_count,(i%N)%atom_count) for i in indices[0]])
    for simstep in range(batch_size):
        if simstep in indices[:,0]:
            step_array = indices[np.where(indices[:,0] == simstep)]
            for row in step_array:
                velocity = simulation.integrator.getState(row[1],getVelocities=True).getVelocities()
                velocity[row[2]] = Quantity(value=np.random.normal(0,1,3)*prefactor[row[2]], unit=nanometer/picosecond)
                simulation.integrator.setVelocities(row[1],velocity)
        simulation.step(1)
        step_count += 1
        if step_count % 2000 == 0:
            TE.append(simulation.integrator.getTotalEnergy().value_in_unit(kilojoules_per_mole))
            for bead in range(P):
                reset_velocity = mom_angmom_rescale_rpmd(simulation)
                simulation.integrator.setVelocities(bead,reset_velocity)
TE_output.write('TE = {}'.format(str(TE)))
print('4 ns Data Collection:', time.time()-st)
Restart.save_simulation('gas_rpmd.save',simulation,P)
