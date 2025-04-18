import os
import numpy as np
from rdkit import Chem
from scipy.stats import sem
import copy
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors
from rdkit import RDLogger
import pandas as pd
import dill
from rdkit.Chem import AllChem
from umap.umap_ import UMAP
from sklearn import svm
from sklearn.cluster import KMeans
import warnings
import shap
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
from matplotlib import colors as mcolors
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D
from cairosvg import svg2png
from matplotlib.ticker import MaxNLocator
import matplotlib
import matplotlib.cm as cm
import colorsys
import seaborn as sns
import json

# Set random seed for reproducibility
np.warnings = warnings
np.random.seed(18)

# List of molecules to analyze - sorted for consistency
molecules = sorted(['butane-1,4-diol', '2-iodopropane', 'propan-2-one', 'propane-1,2,3-triol', '2,6-dimethylheptan-4-one', '1,2-dibromoethane', 'methanal', '2-(2-hydroxyethylamino)ethanol', 'chloroethane', '(2-hydroxyethoxy)ethan-2-ol', '1-butoxybutane', '2-methylpropan-2-ol', '(E)-hex-2-ene', 'methanoic_acid', 'N,N-dimethylformamide', 'bromoethane', 'N-methylacetamide', 'prop-2-enenitrile', 'pentan-3-ol', '2-aminoethanol', 'diethyl_propanedioate', 'pentanenitrile', 'pentane-1,5-diol', 'butan-1-ol', '1,1,2-trichloroethane', 'pentane-2,4-dione', 'N-butylbutan-1-amine', '1,2-dibromopropane', 'propan-1-amine', 'pentan-1-ol', 'N-ethylethanamine', '1-bromopropane', 'butane-1-thiol', '2-methylpropane', '1,2-ethanedithiol', '2-chloroethanol', 'butan-1-amine', '2-methylpropan-2-amine', 'ethenyl_acetate', 'methyl_2-methylprop-2-enoate', 'dibromomethane', 'methoxymethane', '2,4-dimethylpentan-3-one', 'ethyl_propanoate', 'chloroform', '2-propan-2-yloxypropane', '1,1,2,2-tetrachloroethane', '1,1-dichloroethene', 'ethanamide', 'methyl_formate', 'ethanol', '1-chlorobutane', 'ethoxyethene', 'ethylsulfanylethane', 'dimethoxymethane', 'dichloro(fluoro)methane', '1-methoxy-2-(2-methoxyethoxy)ethane', 'octan-1-ol', 'N,N-diethylethanamine', 'propan-2-amine', 'diethyl_carbonate', 'hexan-2-one', 'N-propan-2-ylpropan-2-amine', 'dichloromethane', 'ethane-1,2-diamine', 'methanamide', '1,4-dichlorobutane', 'ethyl_acetate', 'methylsulfinylmethane', 'heptan-2-one', '1,1,1,2,2-pentachloroethane', 'methylsulfanylmethane', 'methyl_acetate', '2-methylbutan-2-ol', 'methyldisulfanylmethane', '1-bromobutane', 'propanenitrile', 'N,N-dimethylacetamide', 'N-methylformamide', '1,3-dichloropropane', '1,1-dichloroethane', 'pentan-3-amine', 'acetyl_acetate', 'methanol', 'acetonitrile', 'bromomethane', '1,2-dichloroethane']) \
+ ['9C','10C','11C','14C','15C']  # Add alkane chains with specified carbon counts

# Calculate molecule counts based on force field files - determines how many molecules are in the simulation
molecule_counts = dict()
for molecule in molecules:
    # Extract count by reading the force field XML file and counting lines containing 'Atom name'
    molecule_counts[molecule] = np.ceil(5000/len([i for i in open('ff_files/ff_{}.xml'.format(molecule)).readlines() if 'Atom name' in i]))

# Load simulation data for different methods            
PI_H_data = np.load(open('../analyzed_data/PI_H_processed_thermo.npy','rb'),allow_pickle='TRUE').item()  # Path-Integral H data
PI_D_data = np.load(open('../analyzed_data/PI_D_processed_thermo.npy','rb'),allow_pickle='TRUE').item()  # Path-Integral D (deuterium) data
cl_data = np.load(open('../analyzed_data/Classical_processed_thermo.npy','rb'),allow_pickle='TRUE').item()  # Classical simulation data

# Define SMILES strings for molecules
molecules_SMILES = ['OCCOCCO', 'CCC/C=C/C', 'ClC(Cl)C(Cl)(Cl)Cl', 'ClC(Cl)C(Cl)Cl', 'ClCC(Cl)Cl', 'CC(Cl)Cl', 'ClC(Cl)=C', 'BrCCBr', 'CC(Br)CBr', 'ClCCCl', 'SCCS', 'ClCCCCl', 'ClCCCCCl', 'CCCCBr', 'CCCBr', 'CCCCOCCCC', 'CCCCCl', 'COCCOCCOC', 'CC(C)C(=O)C(C)C', 'CC(C)CC(=O)CC(C)C', 'OCCNCCO', 'NCCO', 'OCCCl', 'CC(C)I', 'CCC(C)(C)O', 'CC(C)(C)N', 'CC(C)(C)O', 'CC(C)C', 'CC(C)OC(C)C', 'CCN(CC)CC', 'CN(C)C(C)=O', 'CN(C)C=O', 'CCCCNCCCC', 'CCNCC', 'CNC(C)=O', 'CNC=O', 'CC(C)NC(C)C', 'CC#N', 'CC(=O)OC(C)=O', 'CCBr', 'CBr', 'CCCCN', 'CCCCO', 'OCCCCO', 'CCCCS', 'CCCl', 'ClC(Cl)Cl', 'BrCBr', 'FC(Cl)Cl', 'ClCCl', 'CCOC(=O)OCC', 'CCOC(=O)CC(=O)OCC', 'COCOC', 'CC(N)=O', 'NCCN', 'CCO', 'CC(=O)OC=C', 'CCOC=C', 'CCOC(C)=O', 'CCOC(=O)CC', 'CCSCC', 'CCCCCC(C)=O', 'CCCCC(C)=O', 'C=O', 'NC=O', 'OC=O', 'CO', 'COC', 'COC(=O)C(C)=C', 'COC(C)=O', 'COC=O', 'CSSC', 'CSC', 'C[S](C)=O', 'CCCCCCCCO', 'CCCCCO', 'CCC(N)CC', 'CCC(O)CC', 'OCCCCCO', 'CC(=O)CC(C)=O', 'CCCCC#N', 'C=CC#N', 'CCCN', 'CC(C)N', 'CC(C)=O', 'OCC(O)CO', 'CCC#N']
molecules_SMILES += ['CCCCCCCCC','CCCCCCCCCC','CCCCCCCCCCC','CCCCCCCCCCCCCC','CCCCCCCCCCCCCCC']  # Add SMILES for alkane chains

# Define atomic masses for molecular weight calculations
element_masses = {'H': 1.00794, 'D': 2.01355321270, 'C': 12.011, 'N': 14.00674,'O': 15.9994, 'F': 18.998403, 'S': 32.066, 'Cl': 35.4527, 'Br': 79.904, 'I': 126.90447}

# Calculate molecular weights for both H and D systems
molecules_MW = []  # Regular molecular weights
molecules_MW_D = []  # Deuterated molecular weights
for molecule in molecules:
    # Read PDB file and calculate molecular weight
    one_molecule_pdb = 'one_molecule_pdbs/one_{}.pdb'.format(molecule)
    one_molecule_pdb_file = open(one_molecule_pdb,'r').readlines()
    MW = 0
    MW_D = 0
    for line in one_molecule_pdb_file:
        if 'ATOM' in line or 'HETATM' in line:  # Process only atom lines from PDB
            element = line.split()[-1]
            MW += element_masses[element]  # Add atomic mass to regular MW
            MW_D += element_masses[element]  # Add atomic mass to deuterated MW
            if element == 'H':
                MW_D += element_masses['D'] - element_masses['H']  # Add mass difference for deuterium
    molecules_MW.append(MW)
    molecules_MW_D.append(MW_D)
molecules_MW = np.array(molecules_MW)
molecules_MW_D = np.array(molecules_MW_D)

# Define physical constants used in calculations
k_B = 8.31446261815324e-3  # Boltzmann constant in units of kJ/mol K
Jtohartree = 2.2937104486906e17  # Joule to hartree conversion
T = 298.15  # Temperature in Kelvin
AVOGADRO = 6.0221409e23  # Avogadro's number
k_B_JK = 1.380649e-23  # Boltzmann constant in J/K

# Thermal Expansion Coefficient - load from files
cl_data.update(np.load('../analyzed_data/Classical_alphas.npy',allow_pickle='TRUE').item())
PI_H_data.update(np.load('../analyzed_data/PI_H_alphas.npy',allow_pickle='TRUE').item())
PI_D_data.update(np.load('../analyzed_data/PI_D_alphas.npy',allow_pickle='TRUE').item())

# Calculate Isothermal Compressibility for each simulation type
conversion = AVOGADRO*(10**-27)/(1000*1e-9)  # Convert from per mole to total basis, nm^3 to m^3, kJ to J, Pa to GPa
for SIM_TYPE in [cl_data,PI_H_data,PI_D_data]:
    for molecule in molecules:
        kappa_array = conversion*SIM_TYPE[molecule,'dV2']/(k_B*T*SIM_TYPE[molecule,'Box Volume'])
        SIM_TYPE[molecule,'kappa'] = np.mean(kappa_array)  # Calculate mean
        SIM_TYPE[molecule,'kappa_SEM'] = sem(kappa_array)  # Calculate standard error of the mean

# Calculate Dielectric Constant for each simulation type
for SIM_TYPE in [cl_data,PI_H_data,PI_D_data]:
    for molecule in molecules:
        dielectric = []
        for SIM_NO in range(4):  # Process 4 simulations
            # Calculate dipole variance
            dipole_variance = SIM_TYPE[molecule,'M2'][SIM_NO] - np.sum(SIM_TYPE[molecule,'M'][SIM_NO]**2)
            dielectric.append(1+(4/3*np.pi*dipole_variance)/(SIM_TYPE[molecule,'Box Volume'][SIM_NO]*1000*k_B_JK*1.88973*T*Jtohartree))
        SIM_TYPE[molecule,'epsilon'] = np.mean(dielectric)  # Calculate mean
        SIM_TYPE[molecule,'epsilon_SEM'] = sem(dielectric)  # Calculate standard error

# Calculate Heat of Vaporization
gas_data = np.load('../analyzed_data/gas_output.npy',allow_pickle='TRUE').item()  # Load gas phase energy data
sim_type_labels = ['classical','PI_H','PI_D']
for SIM_TYPE_ix, SIM_TYPE in enumerate([cl_data,PI_H_data,PI_D_data]):
    for molecule in molecules:
        dHvap = np.array(gas_data[molecule,sim_type_labels[SIM_TYPE_ix],'Total Energy']) +\
        k_B*T - SIM_TYPE[molecule,'Total Energy']/molecule_counts[molecule]
        SIM_TYPE[molecule,'dHvap'] = np.mean(dHvap)  # Calculate mean
        SIM_TYPE[molecule,'dHvap_SEM'] = sem(dHvap)  # Calculate standard error
        
# Calculate Molar Volume
for SIM_TYPE_index, SIM_TYPE in enumerate([cl_data,PI_H_data,PI_D_data]):
    if SIM_TYPE_index == 2:  # For PI_D data, use deuterated molecular weights
        MW_list = copy.deepcopy(molecules_MW_D)
    else:  # For other methods, use regular molecular weights
        MW_list = copy.deepcopy(molecules_MW)
    for molecule_index,molecule in enumerate(molecules):
        molar_volume_array = MW_list[molecule_index]/SIM_TYPE[molecule,'Density']
        SIM_TYPE[molecule,'molar_volume'] = np.mean(molar_volume_array)  # Calculate mean
        SIM_TYPE[molecule,'molar_volume_SEM'] = sem(molar_volume_array)  # Calculate standard error

# Store average density values for each molecule
for SIM_TYPE in [cl_data,PI_H_data,PI_D_data]:
    for molecule in molecules:
        density_array = SIM_TYPE[molecule,'Density']
        SIM_TYPE[molecule,'density'] = np.mean(density_array)  # Calculate mean
        SIM_TYPE[molecule,'density_SEM'] = sem(density_array)  # Calculate standard error


# Define properties to analyze for NQEs and EIEs
PROPERTIES = ['density','alpha','kappa','epsilon','molar_volume','dHvap']

# Calculate Nuclear Quantum Effects (NQEs) and Isotope Effects (EIEs)
NQEs = {}  # Dictionary to store NQEs
EIEs = {}  # Dictionary to store EIEs
for PROPERTY in PROPERTIES:
    for molecule in molecules:
        # Extract property values for each simulation type
        PI_H_value = PI_H_data[molecule,PROPERTY]
        PI_D_value = PI_D_data[molecule,PROPERTY]
        cl_value = cl_data[molecule,PROPERTY]
        
        # Extract standard errors
        PI_H_SEM = PI_H_data[molecule,PROPERTY+'_SEM']
        PI_D_SEM = PI_D_data[molecule,PROPERTY+'_SEM']
        cl_SEM = cl_data[molecule,PROPERTY+'_SEM']
        
        # Calculate NQE as percentage
        NQE = 100*(PI_H_value-cl_value)/(PI_H_value)

        # Propagate errors for NQE calculation
        numerator_SEM = np.sqrt(cl_SEM**2+PI_H_SEM**2)
        denominator_SEM = PI_H_SEM
        NQE_SEM = np.abs(NQE * np.sqrt((numerator_SEM/(PI_H_value-cl_value))**2 + (denominator_SEM/(PI_H_value))**2))
        
        # Store NQE values
        NQEs[molecule,PROPERTY] = NQE
        NQEs[molecule,PROPERTY+'_SEM'] = NQE_SEM
        
        # Calculate EIE
        EIE = 100*(PI_H_value-PI_D_value)/(PI_D_value)
        
        # Propagate errors for EIE calculation
        numerator_SEM = np.sqrt(PI_D_SEM**2+PI_H_SEM**2)
        denominator_SEM = PI_D_SEM
        EIE_SEM = np.abs(EIE * np.sqrt((numerator_SEM/(PI_H_value-PI_D_value))**2 + (denominator_SEM/(PI_D_value))**2))
        
        # Store EIE values
        EIEs[molecule,PROPERTY] = EIE
        EIEs[molecule,PROPERTY+'_SEM'] = EIE_SEM

# Set up molecular descriptor calculation using Mordred
descriptor_list = {'AMW','nH'}  # AMW = average molecular weight, nH = number of hydrogen atoms
calculator = Calculator(descriptors, ignore_3D=True)  # Initialize calculator
calculator.descriptors = [d for d in calculator.descriptors if str(d) in descriptor_list]  # Filter only desired descriptors

# Process molecules with RDKit to create molecule objects from SMILES strings
RDKIT_mols = [Chem.MolFromSmiles(molecules_SMILES[index]) for index,molecule in enumerate(molecules)]

# Create pandas DataFrame with molecular descriptors
df_processed = calculator.pandas(RDKIT_mols)  # Calculate descriptors and store in DataFrame
df_processed['Name'] = pd.DataFrame(molecules)  # Add molecule names
df_processed['SMILES'] = pd.DataFrame(molecules_SMILES)  # Add SMILES strings

# Add all calculated properties to DataFrame
for PROPERTY in PROPERTIES:
    df_processed[f'classical_{PROPERTY}'] = pd.DataFrame([cl_data[molecule,f'{PROPERTY}'] for molecule in molecules])
    df_processed[f'classical_{PROPERTY}_SEM'] = pd.DataFrame([cl_data[molecule,f'{PROPERTY}_SEM'] for molecule in molecules])

    df_processed[f'PI_H_{PROPERTY}'] = pd.DataFrame([PI_H_data[molecule,f'{PROPERTY}'] for molecule in molecules])
    df_processed[f'PI_H_{PROPERTY}_SEM'] = pd.DataFrame([PI_H_data[molecule,f'{PROPERTY}_SEM'] for molecule in molecules])
    
    df_processed[f'PI_D_{PROPERTY}'] = pd.DataFrame([PI_D_data[molecule,f'{PROPERTY}'] for molecule in molecules])
    df_processed[f'PI_D_{PROPERTY}_SEM'] = pd.DataFrame([PI_D_data[molecule,f'{PROPERTY}_SEM'] for molecule in molecules])
    
    df_processed[f'NQE_{PROPERTY}'] = pd.DataFrame([NQEs[molecule,f'{PROPERTY}'] for molecule in molecules])
    df_processed[f'NQE_{PROPERTY}_SEM'] = pd.DataFrame([NQEs[molecule,f'{PROPERTY}_SEM'] for molecule in molecules])

    df_processed[f'EIE_{PROPERTY}'] = pd.DataFrame([EIEs[molecule,f'{PROPERTY}'] for molecule in molecules])
    df_processed[f'EIE_{PROPERTY}_SEM'] = pd.DataFrame([EIEs[molecule,f'{PROPERTY}_SEM'] for molecule in molecules])
    
# Calculate hydrogen atom density
df_processed['nHperV'] = df_processed['nH']/df_processed['classical_molar_volume']
# Calculate standard error for the hydrogen density using error propagation
df_processed['nHperV_SEM'] = np.array(df_processed['nHperV'])*(np.array(df_processed['classical_molar_volume_SEM'])/np.array(df_processed['classical_molar_volume']))

# Open and load ChEMBL SMILES data for UMAP visualization
read_ChEMBL = open('ChEMBL_SMILES.json','r')
UMAP_smiles = json.load(read_ChEMBL)
# Create RDKit molecule objects from the ChEMBL SMILES strings
UMAP_mols = [Chem.MolFromSmiles(i) for i in UMAP_smiles]

# Add our molecules to the UMAP dataset, excluding the alkane chains (last 5 entries)
for SMILE in molecules_SMILES[:-5]:
    UMAP_mols.append(Chem.MolFromSmiles(SMILE))
    UMAP_smiles.append(SMILE)
    
# Enable RDKit logging for debugging
RDLogger.EnableLog('rdApp.*')

# Define function to process MMFF (Merck Molecular Force Field) atom types for molecules
def process_MMFF(molecules_SMILES_input):
    MMFF_atom_types = []  # List to store unique MMFF atom types
    MMFF_dict = dict()    # Dictionary to store atom type counts for each molecule
    for index,SMILE in enumerate(molecules_SMILES_input):
        mol = Chem.MolFromSmiles(SMILE)  # Create molecule from SMILES
        mol_with_H = Chem.AddHs(mol)     # Add explicit hydrogens
        properties = AllChem.MMFFGetMoleculeProperties(mol_with_H)  # Get MMFF properties
        if properties == None:
            continue  # Skip if MMFF properties can't be calculated
        ATOM_COUNT = mol_with_H.GetNumAtoms()  # Get total atom count
        
        # Count occurrences of each MMFF atom type in the molecule
        for i in range(ATOM_COUNT):
            MMFFtype = properties.GetMMFFAtomType(i)  # Get MMFF type for atom i
            if MMFFtype in MMFF_atom_types:
                MMFF_dict[MMFFtype][index] += 1  # Increment count if type exists
            else:
                MMFF_atom_types.append(MMFFtype)  # Add new type to list
                MMFF_dict[MMFFtype] = np.zeros(len(molecules_SMILES_input))  # Initialize count array
                MMFF_dict[MMFFtype][index] += 1  # Set count for current molecule
    MMFF_df = pd.DataFrame(MMFF_dict)  # Convert dictionary to DataFrame
    return MMFF_df

# Process atom types for all molecules in UMAP dataset
MMFF_df = process_MMFF(UMAP_smiles)
features = list(MMFF_df.columns)  # Extract feature names (atom types)

# Perform UMAP visualization specifically for molar volume
for PROPERTY in ['molar_volume']:

    df = MMFF_df  # Use processed MMFF dataframe
    # Add property values: 0 for ChEMBL molecules, NQE values for our molecules
    df[PROPERTY] = pd.DataFrame([0 for i in UMAP_smiles]+[NQEs[molecule,PROPERTY] for molecule in molecules])
    
    x = df.loc[:, features].values  # Extract feature matrix
    y = np.array(df[PROPERTY])      # Extract property values
    
    # Configure and run UMAP dimensionality reduction
    reducer = UMAP(n_neighbors=100, n_epochs=1000, min_dist=1.0, random_state=42)
    X_trans = reducer.fit_transform(x)  # Transform high-dimensional data to 2D
    
    # Combine transformed coordinates with property values
    arr_concat=np.concatenate((X_trans, y.reshape(y.shape[0],1)), axis=1)
    
    # Create DataFrame with UMAP coordinates and property values
    df=pd.DataFrame(arr_concat, columns=['x', 'y', PROPERTY])
    umap_x = list(df['x'])  # Extract x coordinates
    umap_y = list(df['y'])  # Extract y coordinates
    data_smiles = UMAP_smiles  # Store SMILES strings

# Rearrange columns to put 'nH' and 'AMW' at the end of the DataFrame
for column in ['nH','AMW']:
    df_processed = df_processed[[col for col in df_processed.columns if col != column] + [column]]

# Save the processed dataset to CSV file
df_processed.to_csv('NQE_dataset.csv', index=False)

# Define helper function for customizing violin plot appearance
def patch_violinplot(input_axis,color):
    from matplotlib.collections import PolyCollection
    for art in input_axis.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor(color)  # Set edge color for violin plots

# Define colors for different properties in violin plots
violin_colors = {'molar_volume':'lightblue','alpha':'silver','kappa':'lightgreen','epsilon':'goldenrod','dHvap':'salmon'}

# List of properties to include in violin plots
VIOLIN_PROPERTIES = ['molar_volume','alpha','kappa','epsilon','dHvap']

# Create violin plots showing distribution of effects (NQEs and EIEs)
for EFFECTix, EFFECT in enumerate([NQEs,EIEs]):  # Loop over both effect types
    # Create figure with 2 subplots of different widths
    fig, ax = plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios': [1, 4]},dpi=100)

    # Prepare data for violin plots
    df_violin = {}
    for PROPERTY in VIOLIN_PROPERTIES:
        df_violin[PROPERTY] = [EFFECT[molecule,PROPERTY] for molecule in molecules]  # Extract property values
        df_violin[PROPERTY+'_SEM'] = [EFFECT[molecule,PROPERTY+'_SEM'] for molecule in molecules]  # Extract standard errors

    df_violin = pd.DataFrame.from_dict(df_violin)

    # Create violin plots for all properties except molar_volume
    others_violin = sns.violinplot(data=df_violin[VIOLIN_PROPERTIES[1:]], palette=violin_colors,width=0.9,ax=ax[1],linecolor='black')
    # Create separate violin plot for molar_volume
    molar_volume_violin = sns.violinplot(data=df_violin[['molar_volume']], palette=violin_colors,width=0.9,ax=ax[0],linecolor='black')
    
    # Add horizontal line at y=0 for reference
    for plot_no in [0,1]:
        ax[plot_no].axhline(0,linestyle='dotted',lw=1.5,zorder=-100,color='black')
    
    # Add error bands to show average SEM for each property
    for PROPERTY_INDEX,PROPERTY in enumerate(VIOLIN_PROPERTIES):
        SEM = df_violin[f'{PROPERTY}_SEM']
        if PROPERTY == 'molar_volume':  # Special handling for molar_volume plot
            ax[0].fill_between([-1,1], y2=-np.mean(np.abs(SEM)),\
                       y1=np.mean(np.abs(SEM)), color='gray',alpha=0.2,zorder=-1,lw=0)
        else:  # For other properties
            ax[1].fill_between([PROPERTY_INDEX-1.5,PROPERTY_INDEX-0.5], y2=-np.mean(SEM),\
                               y1=np.mean(SEM), color='gray',alpha=0.2,zorder=-1,lw=0)

    # Add vertical lines to separate properties in second subplot
    for line in [0.5,1.5,2.5]:
        ax[1].axvline(line,color='black',linestyle='--',lw=2.5)
        
    # Set axis limits and ticks for first subplot (molar_volume)
    ax[0].set_xlim(-1,1)
    ax[0].set_ylim(-5,10)
    ax[0].set_yticks(np.arange(-5,15,5))

    # Set axis limits and ticks for second subplot (other properties)
    ax[1].set_xlim(-0.5,3.5)
    ax[1].set_yticks(np.arange(-50,60,25))
    ax[1].set_ylim(-50,50)
        
    # Format both subplots: remove tick labels, set tick positions, add borders
    for plot_no in [0,1]:
        ax[plot_no].set_xticks(ax[plot_no].get_xticks(),[])  # Remove x tick labels
        ax[plot_no].set_yticks(ax[plot_no].get_yticks(),[])  # Remove y tick labels
        ax[plot_no].yaxis.set_ticks_position('both')  # Show ticks on both sides
        ax[plot_no].patch.set_edgecolor('black')  # Set border color
        ax[plot_no].patch.set_linewidth('2.5')  # Set border width
        ax[plot_no].tick_params(width=2,direction='in',labelsize=18,length=6)  # Format tick marks
        
    # Customize violin plot elements (boxes, medians, etc.)
    for PROPERTY_INDEX,PROPERTY in enumerate(VIOLIN_PROPERTIES[:-1]):
        box_color = 'black'
        median_color = 'white'
        border_color = 'black'
        for plot_no in [1,0]:
            # Adjust box and median colors and line widths
            ax[plot_no].get_children()[1+PROPERTY_INDEX*4].set_color(box_color)  # Box color
            ax[plot_no].get_children()[2+PROPERTY_INDEX*4].set_color(box_color)  # Box color
            ax[plot_no].get_children()[3+PROPERTY_INDEX*4].set_color(median_color)  # Median color
            ax[plot_no].get_children()[1+PROPERTY_INDEX*4].set_lw(3)  # Box line width
            ax[plot_no].get_children()[2+PROPERTY_INDEX*4].set_lw(7)  # Box line width
            ax[plot_no].get_children()[0+PROPERTY_INDEX*4].set_lw(2)  # Violin border width
            patch_violinplot(ax[plot_no],border_color)  # Set violin border color
            if PROPERTY != 'molar_volume':
                break 
            
    # Adjust layout and save figure
    fig.tight_layout(pad=4)
    plt.savefig('Figure1{}.png'.format(['A','B'][EFFECTix]))  # Save as Figure1A (NQEs) or Figure1B (EIEs)
    plt.show()

# Set up color normalization for UMAP plot
norm = mcolors.Normalize(vmin=0,vmax=3, clip=True)  # Normalize values between 0 and 3

# Create color mapper using blue color scale
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
# Generate node colors based on molar volume NQE values
node_color = [(r, g, b, 1) for r, g, b, a in mapper.to_rgba(df_processed['NQE_molar_volume'])]

# Create UMAP visualization plot
fig, ax = plt.subplots(figsize=(14, 8),dpi=100)

# Convert UMAP coordinates to numpy arrays
umap_x,umap_y = np.array(umap_x),np.array(umap_y)

# Find indices of hydrocarbons (alkane chains)
hydrocarbon_indices = [index for index,obj in enumerate(UMAP_smiles) if obj in molecules_SMILES[-5:]]
# Plot our molecules with colors based on NQE values
our_molecules = ax.scatter(list(umap_x[-87:])+list(umap_x[hydrocarbon_indices]),list(umap_y[-87:])+list(umap_y[hydrocarbon_indices]),
                           color=node_color,s=250,edgecolors=['black' for i in molecules_SMILES],zorder=1000)

# Plot ChEMBL molecules in background
grid = ax.scatter(umap_x[:-87],umap_y[:-87],color='moccasin',s=80)

# Format plot appearance
ax.patch.set_edgecolor('black')  # Set border color 
ax.patch.set_linewidth('7')  # Set border width
ax.tick_params(width=7,length=13,direction='in')  # Format tick marks

# Configure axis ticks
ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Use integer ticks for y-axis
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides of y-axis
ax.xaxis.set_ticks_position('both')  # Show ticks on both sides of x-axis

# Get all tick labels
labels = ax.get_xticklabels() + ax.get_yticklabels()

# Set tick label size
ax.tick_params(labelsize=27)

# Add color bar for NQE values
color_bar = fig.colorbar(mapper,pad=0.01)
color_bar.ax.tick_params(labelsize=25)  # Set color bar tick label size
color_bar.ax.set_yscale('linear')  # Use linear scale for color bar
color_bar.ax.tick_params(width=5,length=7)  # Format color bar tick marks
color_bar.outline.set_linewidth('5')  # Set color bar border width

# Set axis limits and ticks
ax.set_ylim(30,55)
ax.set_xlim(15,45)
ax.set_yticks(np.arange(30,60,5))
ax.set_xticks(np.arange(15,55,10))

# Remove tick labels
ax.set_xticks(ax.get_xticks(),['' for i in ax.get_xticks()])
ax.set_yticks(ax.get_yticks(),['' for i in ax.get_yticks()])
color_bar.ax.set_yticks(color_bar.ax.get_yticks(),[' ' for i in color_bar.ax.get_yticks()])

# Save figure and display
plt.savefig('Figure2A.png')
plt.show()

MOLECULES_indices = range(92)  # Define indices of molecules to use

# Define features for training the model
TRAIN_FEATURES = ['nHperV','AMW']  # Hydrogen density and average molecular weight

NO_CYCLES = 5 # Number of cycles for Random Forest regression
outliers = []  # List to store outliers

# This loop processes molar_volume using Random Forest regression with SHAP analysis
for PROPERTY in ['molar_volume']:
    # Initialize lists to store R² values and SHAP values
    R2_list = []  # Stores R² values for each cycle
    # Create a matrix to store SHAP values for each feature (rows) and molecule (columns)
    shap_list = [[0 for x in MOLECULES_indices] for i in TRAIN_FEATURES + ['classical_alpha']]
    # Store actual feature values for each molecule to be used later in SHAP visualization
    shap_value_list = [[df_processed[FEATURE][index] for index in MOLECULES_indices] for FEATURE in ['classical_alpha']+TRAIN_FEATURES]
    # Store predictions for each molecule across multiple cycles
    pred_list_of_lists = [[] for i in MOLECULES_indices]
    
    # Run multiple cycles of the Random Forest model to get statistical robustness
    for R2cycle in range(NO_CYCLES):
        np.random.seed(R2cycle)  # Set random seed for reproducibility
        pred_list = []  # Store predictions for current cycle
        actual_list = []  # Store actual values for current cycle
        
        # Leave-one-out approach - train on all molecules except one
        for ix, test_molecule_index in enumerate(MOLECULES_indices):
            # Initialize Random Forest model with 20 trees
            regr = RandomForestRegressor(n_estimators=20, max_features=len(TRAIN_FEATURES), random_state = np.random.randint(10000))
            
            # Define test and training indices
            iTest = np.array([test_molecule_index])  # Test on one molecule
            iTrain = np.array([x for x in MOLECULES_indices if x!=test_molecule_index and x not in outliers])  # Train on all others except outliers
            
            # Train the model - note we're predicting NQE_property/classical_alpha
            regression = regr.fit(df_processed[TRAIN_FEATURES].iloc[iTrain], (df_processed[f'NQE_{PROPERTY}']/df_processed['classical_alpha']).iloc[iTrain])
            
            # Make predictions and multiply by classical_alpha to get actual property value
            train_pred = regr.predict(df_processed[TRAIN_FEATURES].iloc[iTest])
            pred_list.append(train_pred[0] * df_processed['classical_alpha'][ix])
            pred_list_of_lists[ix].append(train_pred[0] * df_processed['classical_alpha'][ix])
            
            # Store actual values for comparison
            actual_list.append(list((df_processed[f'NQE_{PROPERTY}']).iloc[iTest])[0])
            
            # Define a function for SHAP explainer that includes multiplication by classical_alpha
            def func(inputs):
                return regr.predict(inputs[TRAIN_FEATURES]) * inputs['classical_alpha']
            
            # Create SHAP explainer and compute SHAP values for the test molecule
            explainer = shap.Explainer(func,
                                       df_processed[['classical_alpha']+TRAIN_FEATURES].iloc[iTrain])
            shap_values = explainer(df_processed[['classical_alpha']+TRAIN_FEATURES].iloc[iTest])

            # Accumulate SHAP values across cycles
            for index, obj in enumerate(shap_values.values[0][:]):
                shap_list[index][ix] += obj

        # Calculate R² score excluding outliers
        actual_for_r2 = [actual_list[index] for index, obj in enumerate(molecules) if index not in outliers]
        pred_for_r2 = [pred_list[index] for index, obj in enumerate(molecules) if index not in outliers]
        R2_list.append(r2_score(actual_for_r2, pred_for_r2))
    
    # Average SHAP values across all cycles
    shap_list = [[x/NO_CYCLES for x in i] for i in shap_list]
    # Print average R² value with standard error
    print('R^2 value = {} +- {}'.format(np.mean(R2_list), sem(R2_list)))

# Calculate means and standard errors of predictions for each molecule
RF_means = np.array([np.mean(i) for i in pred_list_of_lists])
RF_sems = np.array([sem(i) for i in pred_list_of_lists])

# Create scatter plot of Random Forest predictions vs actual values
fig, ax = plt.subplots(figsize=(6,6), dpi=100)

# Plot data points
scatter = ax.scatter(np.abs(df_processed['NQE_molar_volume']),
                     np.abs(RF_means),
                     color=node_color, edgecolor='black', linewidth=1, s=150)

# Add error bars for both x and y values
ax.errorbar(x=np.abs(df_processed['NQE_molar_volume']), y=RF_means,
            xerr=[NQEs[molecule, PROPERTY+'_SEM'] for molecule in molecules],
            yerr=RF_sems, fmt='None', color='black', zorder=-1, capsize=2)

# Add identity line (x=y)
ax.plot(np.linspace(0,6), np.linspace(0,6), color='black', linewidth=5, label='x=y', linestyle='--', zorder=0)

# Set plot limits and formatting
ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_yticks([0,1,2,3,4,5,6])
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth('5') 
ax.tick_params(width=5, length=9, direction='in')
ax.tick_params(labelsize=20)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
# Hide tick labels
ax.set_yticks(ax.get_yticks(), ['' for i in ax.get_yticks()])
ax.set_xticks(ax.get_xticks(), ['' for i in ax.get_xticks()])
# Save figure and display
plt.savefig('Figure2B.png')
plt.show()


# Define a function to scale data using Yeo-Johnson power transformation
def scale(data):
    data = np.array(data)
    data = data.reshape((len(data),1))  # Reshape for the transformer
    power = PowerTransformer(method='yeo-johnson', standardize=True)  # Initialize transformer
    data_trans = power.fit_transform(data)  # Transform data
    scaled = np.concatenate(data_trans).ravel().tolist()  # Flatten and convert to list
    return scaled

# Create visualization for SHAP analysis
fig, ax = plt.subplots(figsize=(5.555, 6.666), dpi=100)
    
# Calculate y-positions for features, spaced evenly
SHAP_y = np.array([(len(TRAIN_FEATURES)-i)*2 for i in range(len(TRAIN_FEATURES+['classical_alpha']))])

# Scale all feature values for color mapping
shap_value_list = [scale(i) for i in shap_value_list]

# Use coolwarm colormap for visualization
cmap = plt.cm.coolwarm

# Plot SHAP values for each feature and molecule
for FEATURE_INDEX in range(len(TRAIN_FEATURES+['classical_alpha'])):
    for index, obj in enumerate(shap_list[FEATURE_INDEX]):
        # Add jitter to y-position for better visualization
        SHAP_plot = ax.scatter(shap_list[FEATURE_INDEX][index], 
                              SHAP_y[FEATURE_INDEX]+np.random.uniform(-0.65, 0.65), 
                              c=shap_value_list[FEATURE_INDEX][index],
                              vmin=min(shap_value_list[FEATURE_INDEX]),
                              vmax=max(shap_value_list[FEATURE_INDEX]),
                              cmap=cmap, s=120, clip_on=False)

# Add colorbar
colorbar = fig.colorbar(SHAP_plot, pad=0.025)
colorbar.ax.patch.set_edgecolor('black') 
colorbar.outline.set_linewidth('3')
colorbar.ax.get_yaxis().set_ticks([])  # Hide colorbar ticks

# Format plot
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth('5') 

# Add zero line to indicate importance direction
ax.axvline(0, linewidth=5, linestyle='--', color='black', zorder=-10)
ax.tick_params(width=5, length=9, direction='in', labelsize=20)
ax.xaxis.set_ticks_position('both')
ax.set_xticks([-2.2,-1.1,0,1.1,2.2])
ax.set_yticks([0,2,4])
ax.set_xlim(-2.2,2.2)

# Hide tick labels
ax.set_yticks(ax.get_yticks(), ['' for i in ax.get_yticks()])
ax.set_xticks(ax.get_xticks(), ['' for i in ax.get_xticks()])
labels = ax.get_xticklabels() + ax.get_yticklabels()

# Save figure and display
plt.savefig('Figure2C.png')
plt.show()

# K-means clustering and SVM analysis for visualization
# Define molecules to analyze
ANALYSIS_molecules = molecules
ANALYSIS_indices = [molecules.index(i) for i in ANALYSIS_molecules]

# Create figure
fig, ax = plt.subplots(figsize=(15, 9), dpi=100)
cmap = plt.cm.Reds  # Use red colormap

# Extract x and y values with their error bars
x = [df_processed['classical_alpha'][index]*1000 for index in ANALYSIS_indices]  # Scale alpha by 1000
x_err = [df_processed['classical_alpha_SEM'][index]*1000 for index in ANALYSIS_indices]
y = [df_processed['NQE_molar_volume'][index] for index in ANALYSIS_indices]
y_err = [df_processed['NQE_molar_volume_SEM'][index] for index in ANALYSIS_indices]

# Get values for coloring points based on hydrogen density (nHperV)
values_for_coloring = [df_processed['nHperV'][index] for index in ANALYSIS_indices]

# Perform K-means clustering with 4 clusters
data = list(zip(values_for_coloring, [1 for i in x]))  # Use nHperV as clustering feature
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_

# Group molecules by cluster and calculate average NQE_molar_volume for each cluster
molecules_in_label = {}
label_averages = {}
for i in [0,1,2,3]:
    molecules_in_label[i] = [molecule for index, molecule in enumerate(ANALYSIS_molecules) if kmeans.labels_[index] == i]
    label_averages[i] = np.mean([obj for index, obj in enumerate(y) if kmeans.labels_[index] == i])

# Get average values for each molecule's cluster for coloring
averages = [label_averages[x] for x in kmeans.labels_]
# Normalize values for colormap
normalized_averages = [(v - min(averages)) / (max(averages) - min(averages)) for v in averages]

# Create scatter plot, colored by cluster average
xy_scatter = ax.scatter(x, y, c=[label_averages[i] for i in kmeans.labels_], 
                        cmap=cmap, edgecolor='black', s=500, linewidth=2)

# Convert colors to hex for fill_between
rgba_colors = [cmap(i) for i in normalized_averages]
hex_colors = [
    f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
    for r, g, b, _ in rgba_colors]
hex_colors = list(np.unique(hex_colors))[::-1]  # Unique colors in reverse order

# Add error bars
ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='None',
            zorder=-100, ecolor='black', linewidth=3, capsize=4, markeredgewidth=2)

# Create SVM decision boundaries between clusters
for pair_index, pair in enumerate([[0,2], [0,1], [1,3]]):
    # Extract data for the current pair of clusters
    svm_x = []
    svm_y = []
    for labelno in pair:
        for molecule in molecules_in_label[labelno]:
            molecule_index = molecules.index(molecule)
            svm_x.append([df_processed['classical_alpha'][molecule_index]*1000, 
                         df_processed['NQE_molar_volume'][molecule_index]])
            svm_y.append(labelno)

    # Train SVM to find decision boundary
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(svm_x, svm_y)
    w = clf.coef_[0]
    boundary_slope = -w[0] / w[1]  # Calculate boundary line slope

    # Define x-range for boundary line
    if pair_index == 0:
        boundary_x = np.linspace(0.31, 2.1, 1000)
    else:
        boundary_x = np.linspace(0.31, 2.691, 1000)
    
    # Calculate y-values for boundary line
    boundary_y = boundary_slope * boundary_x - clf.intercept_[0] / w[1]

    # Set alpha for all fill regions
    all_alphas = 0.5
    upper_bound = 3.465  # Upper y-limit for fill
    lower_bound = 0.033  # Lower y-limit for fill
    
    # Fill regions between boundaries for different pairs of clusters
    if pair_index == 1:
        ax.fill_between(boundary_x, y2=boundary_y, y1=[lower_bound for i in boundary_y], 
                       color=hex_colors[1], alpha=all_alphas, zorder=-1000, lw=0, interpolate=True)
        ax.fill_between(boundary_x, y2=boundary_y, y1=[upper_bound for i in boundary_y], 
                       color=hex_colors[2], alpha=all_alphas, zorder=-1000, lw=0, interpolate=True)
    if pair_index == 0:
        ax.fill_between(boundary_x, y2=boundary_y, y1=[upper_bound for i in boundary_y], 
                       color=hex_colors[-1], alpha=all_alphas, zorder=-900, lw=0, interpolate=True)
        ax.fill_between(boundary_x, y2=boundary_y, y1=[upper_bound for i in boundary_y], 
                       color='white', alpha=1, zorder=-999, lw=0)
    if pair_index == 2:
        ax.fill_between(boundary_x, y2=boundary_y, y1=[lower_bound for i in boundary_y], 
                       color=hex_colors[0], alpha=all_alphas, zorder=-900, lw=0)
        ax.fill_between(boundary_x, y2=boundary_y, y1=[lower_bound for i in boundary_y], 
                       color='white', alpha=1, zorder=-999, lw=0)

# Format plot
ax.tick_params(labelsize=20, direction='in')
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('9') 
ax.tick_params(width=9, length=12, zorder=10000)
ax.set_xticks([0.3, 0.9, 1.5, 2.1, 2.7])
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
ax.set_ylim(0, 3.5)
ax.set_xlim(0.3, 2.7)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
# Hide tick labels
ax.set_xticks(ax.get_xticks(), ['' for i in ax.get_xticks()])
ax.set_yticks(ax.get_yticks(), ['' for i in ax.get_yticks()])
ax.set_axisbelow(False)  # Ensure gridlines don't overlay data

bounds = np.array([0.008,0.045,0.078,0.106,0.139])*100

# Set up colors for the colorbar
colors = hex_colors
cmap = mcolors.ListedColormap(colors)

# Create a normalization to map values to colors
norm = mcolors.BoundaryNorm(bounds, len(colors))

# Add a secondary axis for the colorbar
ax2 = fig.add_axes([0.75, 0.30, 0.06, 0.4])  # [left, bottom, width, height]
color_bar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, 
                                           boundaries=bounds, spacing='proportional')

# Format the colorbar
ax2.tick_params(width=1, length=3, labelsize=20, axis='y', which='both', right=False)
plt.setp(ax2.spines.values(), linewidth=4)  # Set border width
ax2.set_yticks(ax2.get_yticks(), ['' for i in ax2.get_yticks()])  # Hide tick labels

# Save the figure and display
plt.savefig('Figure3.png')
plt.show()

# Create a new figure for comparing hydrogen-bonding systems
fig, ax = plt.subplots(figsize=(15,9), dpi=100)

# Select molecules for hydrogen-bonding analysis
molecules_to_plot = ['butan-1-ol', 'propane-1,2,3-triol', 'butane-1,4-diol', 
                    'pentane-1,5-diol', '1-bromobutane', '1-chlorobutane', 'butane-1-thiol']
MOLECULE_indices = [molecules.index(i) for i in molecules_to_plot]


x = [df_processed['nHperV'][index] for index in MOLECULE_indices]  # Hydrogen density
y = [df_processed['classical_alpha'][index] for index in MOLECULE_indices]  # Classical thermal expansion
x_err = [df_processed['nHperV_SEM'][index] for index in MOLECULE_indices]
y_err = [df_processed['classical_alpha_SEM'][index] for index in MOLECULE_indices]

# Set up color mapping based on NQE molar volume
cmap = cm.Blues
norm = mcolors.Normalize(vmin=1.4, vmax=2.4, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
color_list = [(r, g, b, 1) for r, g, b, a in 
          mapper.to_rgba([df_processed['NQE_molar_volume'][index] for index in MOLECULE_indices])]

# Create scatter plot with error bars
ax.scatter(x, y, alpha=1, s=500, edgecolor='black', color=color_list)
ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='None', zorder=-100, 
            ecolor='black', linewidth=3, capsize=4, markeredgewidth=2)

# Set plot limits and ticks
ax.set_xlim(0.08, 0.13)
ax.set_ylim(0.0002, 0.0014)

ax.set_xticks([0.08, 0.09, 0.10, 0.11, 0.12, 0.13])
ax.set_yticks([i/10000 for i in [2, 6, 10, 14]])  # Convert to decimal values

# Format the plot
ax.tick_params(labelsize=25, direction='in')
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth('9')
ax.tick_params(width=9, length=16, direction='in')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

# Adjust the alignment of the first tick label
ticklabels = ax.get_xticklabels()
ticklabels[0].set_ha("left")

# Add colorbar to the figure
bounds = np.array([0, 0.04, 0.074, 0.11, 0.13])*100
ax2 = fig.add_axes([0.16, 0.19, 0.25, 0.1])  # [left, bottom, width, height]

color_bar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', orientation='horizontal')

# Format the colorbar
ax2.set_xticks([1.4, 2.4])
ax2.tick_params(width=1, length=3, labelsize=25, axis='x', which='both', 
               bottom=False, top=False, labelbottom=False)
plt.setp(ax2.spines.values(), linewidth=4)

# Hide tick labels
ax2.set_xticks(ax2.get_xticks(), ['' for i in ax2.get_xticks()])
ax.set_yticks(ax.get_yticks(), ['' for i in ax.get_yticks()])
ax.set_xticks(ax.get_xticks(), ['' for i in ax.get_xticks()])

# Save and display the figure
plt.savefig('Figure4A.png')
plt.show()

# Create new figure for comparing linear vs. branched chemically similar systems
fig, ax = plt.subplots(figsize=(15,9), dpi=100)

# Select molecules for linear vs branched comparison
molecules_to_plot = ['butan-1-ol', 'butan-1-amine', '2-methylpropan-2-amine', '2-methylpropan-2-ol', 
                    'propan-1-amine', 'propan-2-amine', '2-methylbutan-2-ol', 'pentan-1-ol']
MOLECULE_indices = [molecules.index(i) for i in molecules_to_plot]

x = [df_processed['nHperV'][index] for index in MOLECULE_indices]
y = [df_processed['classical_alpha'][index] for index in MOLECULE_indices]
x_err = [df_processed['nHperV_SEM'][index] for index in MOLECULE_indices]
y_err = [df_processed['classical_alpha_SEM'][index] for index in MOLECULE_indices]

# Add error bars
ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='None', zorder=-100, 
            ecolor='black', linewidth=3, capsize=4, markeredgewidth=2)

# Set up color mapping based on NQE molar volume
norm = mcolors.Normalize(vmin=2.3, vmax=3.3, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
color_list = [(r, g, b, 1) for r, g, b, a in 
          mapper.to_rgba([df_processed['NQE_molar_volume'][index] for index in MOLECULE_indices])]

# Create scatter plot
ax.scatter(x, y, color=color_list, s=500, edgecolor='black')

# Set plot limits and ticks
ax.set_yticks([i/10000 for i in [8, 10, 12, 14, 16]])
ax.set_xticks([0.112, 0.114, 0.116, 0.118, 0.120])
ax.set_xlim(0.112, 0.12)
ax.set_ylim(0.0008, 0.0016)

# Format the plot
ax.tick_params(labelsize=25, width=9, length=16, direction='in')
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth('9')
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

# Add colorbar
bounds = np.array([0, 0.04, 0.074, 0.25, 0.1])*100
ax2 = fig.add_axes([0.16, 0.19, 0.25, 0.1])  # [left, bottom, width, height]

color_bar = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', orientation='horizontal')

# Format colorbar
ax2.tick_params(width=1, length=3, labelsize=25, axis='x', which='both',
                bottom=False, top=False, labelbottom=False)
plt.setp(ax2.spines.values(), linewidth=4)

# Hide tick labels
ax2.set_yticks(ax2.get_yticks(), ['' for i in ax2.get_yticks()])
ax.set_yticks(ax.get_yticks(), ['' for i in ax.get_yticks()])
ax.set_xticks(ax.get_xticks(), ['' for i in ax.get_xticks()])

# Save and display figure
plt.savefig('Figure4B.png')
plt.show()

# Detailed analysis of effect of branching - comparing two specific molecules
plot_molecules = ['butan-1-ol', '2-methylpropan-2-ol']  # Linear vs branched alcohols

# Initialize dictionary to store pairwise interaction data
interaction_dictionary = {}

# Set up distance bins for analysis
N_bins = 50
distance_bins = np.linspace(3.3, 9, N_bins)  # From 3.3 to 9 Angstroms
distance_increments = distance_bins[1] - distance_bins[0]  # Bin width

# Initialize dictionaries to store mean values for each molecule and each property
energy_means, angle_means, HB_means, counts_within_bin = {}, {}, {}, {}
for molecule in plot_molecules:
    energy_means[molecule] = np.zeros((4, N_bins))  # 4 simulations, N_bins distance bins
    angle_means[molecule] = np.zeros((4, N_bins))
    counts_within_bin[molecule] = np.zeros((4, N_bins))
    HB_means[molecule] = np.zeros((4, N_bins))
    
# Load and process data for each molecule and simulation
for molecule in plot_molecules:
    for SIM_NO in range(4):
        # Load previously analyzed data from file
        read_f = open('../analyzed_data/branching_analysis/{}/{}/pairwise_interaction_analysis.out'.format(molecule, SIM_NO+1), 'rb')
        interaction_dictionary['energy', molecule, SIM_NO] = np.load(read_f)  # Interaction energies
        interaction_dictionary['distance', molecule, SIM_NO] = np.load(read_f)  # Distances
        interaction_dictionary['angle', molecule, SIM_NO] = np.load(read_f)  # Angles
        interaction_dictionary['HB', molecule, SIM_NO] = np.load(read_f)  # Hydrogen bonds
        
        # Calculate mean values within each distance bin
        for bin_index, distance_bin in enumerate(distance_bins):
            # Find all pairs within current bin
            indices_within_bin = np.where((interaction_dictionary['distance', molecule, SIM_NO] < distance_bin+distance_increments) & 
                                         (interaction_dictionary['distance', molecule, SIM_NO] > distance_bin))
            
            # Calculate mean energy for pairs in this bin
            energy_means[molecule][SIM_NO][bin_index] = np.mean(interaction_dictionary['energy', molecule, SIM_NO][indices_within_bin])
            
            # Calculate mean angle for pairs in this bin
            angle_means[molecule][SIM_NO][bin_index] = np.mean(interaction_dictionary['angle', molecule, SIM_NO][indices_within_bin])
            
            HB_means[molecule][SIM_NO][bin_index] = sum(interaction_dictionary['HB', molecule, SIM_NO][indices_within_bin])/(cl_data[molecule, 'Box Volume'][SIM_NO]*10)
            
            # Count number of pairs in this bin
            counts_within_bin[molecule][SIM_NO][bin_index] = len(indices_within_bin[0])

# Filter bins with too few counts and calculate statistics
filtered_distance_bins = {}
for molecule in plot_molecules:
    # Average counts across simulations
    counts_within_bin[molecule] = np.mean(counts_within_bin[molecule], axis=0)
    # Only keep bins with more than 10 counts (for statistical significance)
    count_filter = np.where(counts_within_bin[molecule] > 10)[0]
    
    # Calculate standard error of the mean (SEM) for each property and filter bins
    energy_means[molecule, 'SEM'] = np.array([sem(energy_means[molecule][:, i]) for i in range(N_bins)])[count_filter]
    energy_means[molecule] = np.mean(energy_means[molecule], axis=0)[count_filter]
    
    angle_means[molecule, 'SEM'] = np.array([sem(angle_means[molecule][:, i]) for i in range(N_bins)])[count_filter]
    angle_means[molecule] = np.mean(angle_means[molecule], axis=0)[count_filter]
    
    HB_means[molecule, 'SEM'] = np.array([sem(HB_means[molecule][:, i]) for i in range(N_bins)])[count_filter]
    HB_means[molecule] = np.mean(HB_means[molecule], axis=0)[count_filter]
    
    # Keep track of which distance bins are used for each molecule
    filtered_distance_bins[molecule] = distance_bins[count_filter]

# Create figure with 3 subplots sharing x-axis
fig, axes = plt.subplots(figsize=(15, 15), nrows=3, sharex=True, dpi=100)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.12)

# Assign axes to variables for easier reference
ax1 = axes[0]  # Hydrogen bonds
ax2 = axes[1]  # Energy
ax3 = axes[2]  # Angle

# Define visual parameters
tick_width = 7
plot_width = 7
tick_length = 13

# Define colors for each molecule
color_dict = {'2-methylpropan-2-ol': 'crimson', 'butan-1-ol': 'darkslategray'}

# Plot data for each molecule
for molecule in plot_molecules:
    # Plot mean values for each property with dashed lines
    ax1.plot(filtered_distance_bins[molecule], HB_means[molecule], 
             color=color_dict[molecule], lw=plot_width, linestyle=(0, (5, 1)))
    ax2.plot(filtered_distance_bins[molecule], energy_means[molecule], 
             color=color_dict[molecule], lw=plot_width, linestyle=(0, (5, 1)))
    ax3.plot(filtered_distance_bins[molecule], angle_means[molecule], 
             color=color_dict[molecule], lw=plot_width, linestyle=(0, (5, 1)), label='                ')

    # Add shaded regions for standard error
    ax1.fill_between(filtered_distance_bins[molecule], 
                     HB_means[molecule] - HB_means[molecule, 'SEM'], 
                     HB_means[molecule] + HB_means[molecule, 'SEM'],
                     color=color_dict[molecule], alpha=0.3)

    ax2.fill_between(filtered_distance_bins[molecule], 
                     energy_means[molecule] - energy_means[molecule, 'SEM'], 
                     energy_means[molecule] + energy_means[molecule, 'SEM'],
                     color=color_dict[molecule], alpha=0.3)

    ax3.fill_between(filtered_distance_bins[molecule], 
                     angle_means[molecule] - angle_means[molecule, 'SEM'], 
                     angle_means[molecule] + angle_means[molecule, 'SEM'],
                     color=color_dict[molecule], alpha=0.3)

# Add horizontal line at y=0 for reference in angle plot
ax3.axhline(0, linestyle='dotted', lw=2, zorder=-1000, color='gray')

# Format all subplots
for plot_no in [0, 1, 2]:
    axes[plot_no].yaxis.set_ticks_position('both')
    axes[plot_no].xaxis.set_ticks_position('both')
    axes[plot_no].tick_params(width=tick_width, length=tick_length, direction='in')
    plt.setp(axes[plot_no].spines.values(), lw=tick_width)
    
# Set limits and ticks for each subplot
ax1.set_xlim(3, 8)  # Set x-axis limits for distances
ax1.set_ylim(0, 1.2)  # Hydrogen bond density
ax1.set_yticks([0, 0.6, 1.2])
ax2.set_yticks([0, -15, -30])  # Energy values
ax3.set_yticks([-0.6, -0.4, -0.2, 0, 0.2])  # Angle values
ax2.set_ylim(-30, 0)  # Energy range
ax3.set_ylim(-0.6, 0.2)  # Angle range

# Hide tick labels for cleaner presentation
for plot_no in [0, 1, 2]:
    axes[plot_no].set_yticks(axes[plot_no].get_yticks(), ['' for i in axes[plot_no].get_yticks()])
ax3.set_xticks(ax3.get_xticks(), ['' for i in ax3.get_xticks()])

# Save and display the figure
plt.savefig('Figure5.png')
plt.show()