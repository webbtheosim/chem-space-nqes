This directory contains the analysis files used to process the raw simulation trajectory and thermodynamic outputs.

calculate_dipole.py: This file, used to calculate the dipole moments of each system, should be run first, as other files may depend on its analysis. The analysis takes in three input parameters, molecule name, simulation number, and the simulation type ('Classical', 'PI_H', 'PI_D'), to output the dielectric information for each simulation. It outputs a file ($molecule_$simulation_number.di) for each molecule and simulation number containing the numpy array for the dipole moments and the M2 values. It can be run via

$python calculate_dipole.py $molecule_name $simulation_number $simulation_type

process_thermo.py: This file processes the thermodynamic data (thermo.avg) for each molecule and its four simulations. As the code loops through every molecule and simulation number, it only requires one input parameter, the simulation type. It outputs $simulation_type_processed_thermo.npy containing a dictionary with the processed thermodynamic data. It can be run via

$python process_thermo.py $simulation_type

calculate_alpha.py: Once thermal expansion coefficient simulations are completed, this file should be run to calculate and compile all thermal expansion coefficients for each molecule. It requires one input parameter, the simulation type. It outputs $simulation_type_alphas.npy containing a dictionary with the processed thermodynamic data. It can be run via

$python calculate_alpha.py $simulation_type

gas_analysis.py: This file processes the thermodynamic data from gas simulations for all molecules and simulation types, and outputs gas_output.npy that contains a dictionary with the processed data. It can be run via

$python gas_analysis.py

pairwise_interaction_analysis.py: This file analyzes two specific systems, 2-methylpropan-2-ol and butan-1-ol, to obtain interaction distances, angles, and strengths for all pairwise interactions. It requires molecule name and simulation number as the input, and outputs pairwise_interaction_analysis.out. It can be run via

$python pairwise_interaction_analysis.py $molecule_name $simulation_number

All analyzed data from these files have been compiled in chem-space-nqes/analyzed_data, and are used to generate the the figures in this study along with Supplementary_Data.csv.