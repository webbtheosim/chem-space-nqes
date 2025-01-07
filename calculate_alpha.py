import os
import numpy as np
from scipy.stats import sem
import sys

molecules = sorted(['butane-1,4-diol', '2-iodopropane', 'propan-2-one', 'propane-1,2,3-triol', '2,6-dimethylheptan-4-one', '1,2-dibromoethane', 'methanal', '2-(2-hydroxyethylamino)ethanol', 'chloroethane', '(2-hydroxyethoxy)ethan-2-ol', '1-butoxybutane', '2-methylpropan-2-ol', '(E)-hex-2-ene', 'methanoic_acid', 'N,N-dimethylformamide', 'bromoethane', 'N-methylacetamide', 'prop-2-enenitrile', 'pentan-3-ol', '2-aminoethanol', 'diethyl_propanedioate', 'pentanenitrile', 'pentane-1,5-diol', 'butan-1-ol', '1,1,2-trichloroethane', 'pentane-2,4-dione', 'N-butylbutan-1-amine', '1,2-dibromopropane', 'propan-1-amine', 'pentan-1-ol', 'N-ethylethanamine', '1-bromopropane', 'butane-1-thiol', '2-methylpropane', '1,2-ethanedithiol', '2-chloroethanol', 'butan-1-amine', '2-methylpropan-2-amine', 'ethenyl_acetate', 'methyl_2-methylprop-2-enoate', 'dibromomethane', 'methoxymethane', '2,4-dimethylpentan-3-one', 'ethyl_propanoate', 'chloroform', '2-propan-2-yloxypropane', '1,1,2,2-tetrachloroethane', '1,1-dichloroethene', 'ethanamide', 'methyl_formate', 'ethanol', '1-chlorobutane', 'ethoxyethene', 'ethylsulfanylethane', 'dimethoxymethane', 'dichloro(fluoro)methane', '1-methoxy-2-(2-methoxyethoxy)ethane', 'octan-1-ol', 'N,N-diethylethanamine', 'propan-2-amine', 'diethyl_carbonate', 'hexan-2-one', 'N-propan-2-ylpropan-2-amine', 'dichloromethane', 'ethane-1,2-diamine', 'methanamide', '1,4-dichlorobutane', 'ethyl_acetate', 'methylsulfinylmethane', 'heptan-2-one', '1,1,1,2,2-pentachloroethane', 'methylsulfanylmethane', 'methyl_acetate', '2-methylbutan-2-ol', 'methyldisulfanylmethane', '1-bromobutane', 'propanenitrile', 'N,N-dimethylacetamide', 'N-methylformamide', '1,3-dichloropropane', '1,1-dichloroethane', 'pentan-3-amine', 'acetyl_acetate', 'methanol', 'acetonitrile', 'bromomethane', '1,2-dichloroethane'])

simulation_type = str(sys.argv[1])
if simulation_type in ['PI_H','PI_D']:
    P = 32
    thermo_name = 'thermo_combined'
    SIM_TYPE = simulation_type
elif simulation_type in ['Classical']:
    P = 1
    thermo_name = 'thermo'
    SIM_TYPE = 'cl_data'

alpha_dict = {}
for molecule in molecules:
    alphas = []
    sems = []
    for simno in [1,2,3,4]:
        delta_temp = 5 #Kelvin
    
        plus = np.mean([float(i.split(',')[-2]) for i in open('{}/{}/thermoplus.avg'.format(molecule,simno),'r').readlines()[1:]])
        minus = np.mean([float(i.split(',')[-2]) for i in open('{}/{}/thermominus.avg'.format(molecule,simno),'r').readlines()[1:]])
       
        thermo = np.mean([float(i.split(',')[-2]) for i in open('{}/{}/{}.avg'.format(molecule,simno,thermo_name),'r').readlines()[1::P]])
    
        slope, b = np.polyfit([298.15-delta_temp,298.15,298.15+delta_temp],[minus,thermo,plus], 1)
        alpha = slope/thermo
        alphas.append(alpha)
    alpha_dict[molecule,'alpha'] = np.mean(alphas)
    alpha_dict[molecule,'alpha_SEM'] = sem(alphas)

f_out = open('{}_alphas.npy'.format(simulation_type),'wb')
np.save(f_out,alpha_dict)
f_out.close()
