import os
import numpy as np
import sys
import ast

AVOGADRO = 6.0221409e23
ATM_TO_PA = 101325 # atm to pascals
PRESSURE = 1 #atm
PRESSURE = PRESSURE * 1E-27 * ATM_TO_PA * AVOGADRO / 1000  # nm3 to m3, atm to Pa, J to kJ, per system to per mol

molecules = sorted(['butane-1,4-diol', '2-iodopropane', 'propan-2-one', 'propane-1,2,3-triol', '2,6-dimethylheptan-4-one', '1,2-dibromoethane', 'methanal', '2-(2-hydroxyethylamino)ethanol', 'chloroethane', '(2-hydroxyethoxy)ethan-2-ol', '1-butoxybutane', '2-methylpropan-2-ol', '(E)-hex-2-ene', 'methanoic_acid', 'N,N-dimethylformamide', 'bromoethane', 'N-methylacetamide', 'prop-2-enenitrile', 'pentan-3-ol', '2-aminoethanol', 'diethyl_propanedioate', 'pentanenitrile', 'pentane-1,5-diol', 'butan-1-ol', '1,1,2-trichloroethane', 'pentane-2,4-dione', 'N-butylbutan-1-amine', '1,2-dibromopropane', 'propan-1-amine', 'pentan-1-ol', 'N-ethylethanamine', '1-bromopropane', 'butane-1-thiol', '2-methylpropane', '1,2-ethanedithiol', '2-chloroethanol', 'butan-1-amine', '2-methylpropan-2-amine', 'ethenyl_acetate', 'methyl_2-methylprop-2-enoate', 'dibromomethane', 'methoxymethane', '2,4-dimethylpentan-3-one', 'ethyl_propanoate', 'chloroform', '2-propan-2-yloxypropane', '1,1,2,2-tetrachloroethane', '1,1-dichloroethene', 'ethanamide', 'methyl_formate', 'ethanol', '1-chlorobutane', 'ethoxyethene', 'ethylsulfanylethane', 'dimethoxymethane', 'dichloro(fluoro)methane', '1-methoxy-2-(2-methoxyethoxy)ethane', 'octan-1-ol', 'N,N-diethylethanamine', 'propan-2-amine', 'diethyl_carbonate', 'hexan-2-one', 'N-propan-2-ylpropan-2-amine', 'dichloromethane', 'ethane-1,2-diamine', 'methanamide', '1,4-dichlorobutane', 'ethyl_acetate', 'methylsulfinylmethane', 'heptan-2-one', '1,1,1,2,2-pentachloroethane', 'methylsulfanylmethane', 'methyl_acetate', '2-methylbutan-2-ol', 'methyldisulfanylmethane', '1-bromobutane', 'propanenitrile', 'N,N-dimethylacetamide', 'N-methylformamide', '1,3-dichloropropane', '1,1-dichloroethane', 'pentan-3-amine', 'acetyl_acetate', 'methanol', 'acetonitrile', 'bromomethane', '1,2-dichloroethane'])
simulation_numbers = [1,2,3,4]

simulation_type = str(sys.argv[1])
if simulation_type in ['PI_H','PI_D']:
    P = 32
    thermo_name = 'thermo_combined'
elif simulation_type in ['Classical']:
    P = 1
    thermo_name = 'thermo'

PROPERTIES = ['Potential Energy', 'Kinetic Energy', 'Total Energy', 'Temperature', 'Box Volume', 'Density'] + ['dHdV','dV2','M','M2']
molecules=['ethanol']
def thermo_analysis(MOL):
    all_simulations = {}
    
    for PROPERTY in PROPERTIES:
        all_simulations[PROPERTY] = []

    for SIM_NO in simulation_numbers:
        simulation_thermo_data = {}

        thermo_file = '{}/{}/{}.avg'.format(MOL,SIM_NO,thermo_name)
        first_line = open(thermo_file,'r').readline()
        column_names = [i.split(' (')[0] for i in first_line.split('"')[1::2]]
        
        thermo_array = np.loadtxt(thermo_file,delimiter=',')
        for column_index,column_name in enumerate(column_names):
            thermo_array_P = np.array([np.sum(thermo_array[:,column_index][i:i+P])/P for i in np.arange(0,len(thermo_array),P)])
            simulation_thermo_data[column_name] = thermo_array_P[9::10]
        
        if P > 1:
            QKE = open('{}/{}/QKE.data'.format(MOL,SIM_NO),'r').readline()
            QKE = np.array(ast.literal_eval(QKE[6:]))
            if len(QKE) == 10000:
                QKE = QKE[::10]
            if simulation_type == 'PI_H':
                simulation_thermo_data['Potential Energy'] = QKE/P
                simulation_thermo_data['Total Energy'] = simulation_thermo_data['Potential Energy'] + simulation_thermo_data['Kinetic Energy']
            elif simulation_type == 'PI_D':
                simulation_thermo_data['Potential Energy'] = (QKE/P-simulation_thermo_data['Kinetic Energy'])
                simulation_thermo_data['Total Energy'] = QKE/P

        H = simulation_thermo_data['Potential Energy'] + PRESSURE * simulation_thermo_data['Box Volume']
        dH = H - np.mean(H)
        dV = simulation_thermo_data['Box Volume'] - np.mean(simulation_thermo_data['Box Volume'])
        simulation_thermo_data['dHdV'] = dH * dV
        simulation_thermo_data['dV2'] = (dV)**2

        dipole_file = open('{}_dielectric_outputs/{}_{}.di'.format(simulation_type,MOL,SIM_NO),'rb')
        simulation_thermo_data['M'] = np.load(dipole_file)
        simulation_thermo_data['M2'] = np.load(dipole_file)

        for PROPERTY in PROPERTIES:
            if PROPERTY == 'M':
                all_simulations[PROPERTY].append(list(np.mean(simulation_thermo_data[PROPERTY],axis=0)))
            else:
                all_simulations[PROPERTY].append(np.mean(simulation_thermo_data[PROPERTY]))
    return all_simulations

processed_thermo = {}

for molecule in molecules:
    print(molecule)
    analyzed_thermo = thermo_analysis(molecule)
    for key in analyzed_thermo.keys():
        processed_thermo[molecule,key] = np.array(analyzed_thermo[key])
print((processed_thermo['ethanol','Box Volume']))


f = open('ethanol/1/thermo.avg','r').readlines()[1:]
vs=[]
for i in f:
    vs.append(float(i.split(',')[-]))
print(np.mean(vs))

print(fsdbsdvjbhsdfjbhfsd)
output_file = open('{}_processed_thermo.npy'.format(simulation_type),'wb')
np.save(output_file,processed_thermo)
output_file.close()
