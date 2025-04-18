This repository contains all analysis and data files from our investigation of nuclear quantum effects (NQEs) and equilibrium isotope effects (EIEs) across 92 chemically diverse molecular liquids.

To repeat any of the reported simulations, modify the OpenMM source files in accordance with the files stored in openmm_modifications/ 

All simulation input files are stored in simulation_files/ 

Data that already has been analyzed are stored in analyzed_data/ 

All thermodynamic data analysis files are located in data_analysis/ 

Finally, to generate the figures displayed in our manuscript, relocate to generate_figures/ and run generate_figures.py.

The module versions used in this project are listed below:
python==3.9.7\
dill==0.3.6\
seaborn==0.12.2\
rdkit==2022.09.3\
cairosvg==2.7.1\
matplotlib==3.5.2\
sklearn==1.0.2\
shap==0.44.1\
numpy==1.24.4\
mordred==1.2.0\
pandas==1.4.4\
umap==0.5.3\
MDAnalysis==2.1.0\
openmm==7.7.0\