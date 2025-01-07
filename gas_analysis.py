import numpy as np
from scipy.stats import sem

molecules = sorted(['butane-1,4-diol', '2-iodopropane', 'propan-2-one', 'propane-1,2,3-triol', '2,6-dimethylheptan-4-one', '1,2-dibromoethane', 'methanal', '2-(2-hydroxyethylamino)ethanol', 'chloroethane', '(2-hydroxyethoxy)ethan-2-ol', '1-butoxybutane', '2-methylpropan-2-ol', '(E)-hex-2-ene', 'methanoic_acid', 'N,N-dimethylformamide', 'bromoethane', 'N-methylacetamide', 'prop-2-enenitrile', 'pentan-3-ol', '2-aminoethanol', 'diethyl_propanedioate', 'pentanenitrile', 'pentane-1,5-diol', 'butan-1-ol', '1,1,2-trichloroethane', 'pentane-2,4-dione', 'N-butylbutan-1-amine', '1,2-dibromopropane', 'propan-1-amine', 'pentan-1-ol', 'N-ethylethanamine', '1-bromopropane', 'butane-1-thiol', '2-methylpropane', '1,2-ethanedithiol', '2-chloroethanol', 'butan-1-amine', '2-methylpropan-2-amine', 'ethenyl_acetate', 'methyl_2-methylprop-2-enoate', 'dibromomethane', 'methoxymethane', '2,4-dimethylpentan-3-one', 'ethyl_propanoate', 'chloroform', '2-propan-2-yloxypropane', '1,1,2,2-tetrachloroethane', '1,1-dichloroethene', 'ethanamide', 'methyl_formate', 'ethanol', '1-chlorobutane', 'ethoxyethene', 'ethylsulfanylethane', 'dimethoxymethane', 'dichloro(fluoro)methane', '1-methoxy-2-(2-methoxyethoxy)ethane', 'octan-1-ol', 'N,N-diethylethanamine', 'propan-2-amine', 'diethyl_carbonate', 'hexan-2-one', 'N-propan-2-ylpropan-2-amine', 'dichloromethane', 'ethane-1,2-diamine', 'methanamide', '1,4-dichlorobutane', 'ethyl_acetate', 'methylsulfinylmethane', 'heptan-2-one', '1,1,1,2,2-pentachloroethane', 'methylsulfanylmethane', 'methyl_acetate', '2-methylbutan-2-ol', 'methyldisulfanylmethane', '1-bromobutane', 'propanenitrile', 'N,N-dimethylacetamide', 'N-methylformamide', '1,3-dichloropropane', '1,1-dichloroethane', 'pentan-3-amine', 'acetyl_acetate', 'methanol', 'acetonitrile', 'bromomethane', '1,2-dichloroethane'])

gas_thermo_dict = {}

for molecule in ['ethanol']:#molecules:
    print(molecule)
    # gas_thermo_file = open('/scratch/gpfs/bu9134/Gas_Classical_TAFFI_Simulations/{}/1/gas_thermo.avg'.format(molecule),'r')
    # thermo_array = np.loadtxt(gas_thermo_file,delimiter=',')
    # gas_thermo_dict[molecule,'classical','Potential Energy'] = [np.mean(thermo_array[:,2][i:i+1000]) for i in list(range(0,4000,1000))]

    # gas_thermo_file = open('/projects/WEBB/eser/Gas_RPMD_TAFFI_Simulations/{}/1/gas_thermo.avg'.format(molecule),'r')
    # thermo_array = np.loadtxt(gas_thermo_file,delimiter=',')
    # gas_thermo_dict[molecule,'PI_H','Kinetic Energy Sum'] = np.array([sum(thermo_array[:,3][index:index+32]) for index in range(0,len(thermo_array),32)])
    # exec(open('/projects/WEBB/eser/Gas_RPMD_TAFFI_Simulations/{}/1/TEoutput'.format(molecule),'r').readlines()[0])
    # PIMD_Total_Energy = np.array(TE)
    # gas_thermo_dict[molecule,'PI_H','Potential Energy'] = (PIMD_Total_Energy - gas_thermo_dict[molecule,'PI_H','Kinetic Energy Sum'])/32
    # gas_thermo_dict[molecule,'PI_H','Potential Energy'] = [np.mean(gas_thermo_dict[molecule,'PI_H','Potential Energy'][i:i+1000]) for i in list(range(0,4000,1000))]

    gas_thermo_file = open('/projects/WEBB/eser/Gas_Deuterated_TAFFI_Simulations/{}/gas_thermo.avg'.format(molecule),'r')
    thermo_array = np.loadtxt(gas_thermo_file,delimiter=',')
    gas_thermo_dict[molecule,'PI_D','Kinetic Energy Sum'] = np.array([sum(thermo_array[:,3][index:index+32]) for index in range(0,len(thermo_array),32)])
    exec(open('/projects/WEBB/eser/Gas_Deuterated_TAFFI_Simulations/{}/TEoutput'.format(molecule),'r').readlines()[0])
    PIMD_Total_Energy = np.array(TE)
    gas_thermo_dict[molecule,'PI_D','Potential Energy'] = (PIMD_Total_Energy - gas_thermo_dict[molecule,'PI_D','Kinetic Energy Sum'])/32
    gas_thermo_dict[molecule,'PI_D','Potential Energy'] = [np.mean(gas_thermo_dict[molecule,'PI_D','Potential Energy'][i:i+1000]) for i in list(range(0,4000,1000))]
    print(gas_thermo_dict['ethanol','PI_D','Potential Energy'] )

f_out = open('gas_output.npy','wb')
np.save(f_out,gas_thermo_dict)
f_out.close()