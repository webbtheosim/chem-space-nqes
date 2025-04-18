Before running simulations, make sure to follow the instructions in chem-space-nqes/openmm_modifications/README.md

This directory includes the classical and path-integral molecular dynamics simulation input files for liquid-phase systems, gas-phase systems, and thermal expansion coefficient simulations.

Note that all simulation input files are configured to run on the CUDA implementation of OpenMM and require a GPU.

For liquid-phase simulations, change directory for the specified molecule and simulation number

$cd chem-space-nqes/input_simulation_files/Liquid_Phase/$molecule/$simulation_number

and run

$python chem-space-nqes/input_simulation_files/classical.py
or
$python chem-space-nqes/input_simulation_files/PIMD.py

For gas-phase simulations, change directory for the specified molecule

$cd chem-space-nqes/input_simulation_files/Gas_Phase/$molecule/1

and run

$python chem-space-nqes/input_simulation_files/gas_classical.py
or
$python chem-space-nqes/input_simulation_files/gas_PIMD.py


Once liquid-phase simulations are completed, thermal expansion coefficient simulations should be run. Change directory for the specified molecule and simulation number

$cd chem-space-nqes/input_simulation_files/Liquid_Phase/$molecule/$simulation_number

and run

$python chem-space-nqes/input_simulation_files/alpha_classical.py
or
$python chem-space-nqes/input_simulation_files/alpha_PIMD.py