import os
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def smiles_to_3d_mol(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if len(''.join([c for c in smiles if c.isupper()])) < 8:
        AllChem.UFFOptimizeMolecule(mol)
    filename = f"{name}.mol"
#    if name not in ['mol_{}'.format(x) for x in [87,88,89,90,91]]:
    #Chem.MolToMolFile(mol, filename)
    return filename

molecules_SMILES = ['OCCOCCO', 'CCC/C=C/C', 'ClC(Cl)C(Cl)(Cl)Cl', 'ClC(Cl)C(Cl)Cl', 'ClCC(Cl)Cl', 'CC(Cl)Cl', 'ClC(Cl)=C', 'BrCCBr', 'CC(Br)CBr', 'ClCCCl', 'SCCS', 'ClCCCCl', 'ClCCCCCl', 'CCCCBr', 'CCCBr', 'CCCCOCCCC', 'CCCCCl', 'COCCOCCOC', 'CC(C)C(=O)C(C)C', 'CC(C)CC(=O)CC(C)C', 'OCCNCCO', 'NCCO', 'OCCCl', 'CC(C)I', 'CCC(C)(C)O', 'CC(C)(C)N', 'CC(C)(C)O', 'CC(C)C', 'CC(C)OC(C)C', 'CCN(CC)CC', 'CN(C)C(C)=O', 'CN(C)C=O', 'CCCCNCCCC', 'CCNCC', 'CNC(C)=O', 'CNC=O', 'CC(C)NC(C)C', 'CC#N', 'CC(=O)OC(C)=O', 'CCBr', 'CBr', 'CCCCN', 'CCCCO', 'OCCCCO', 'CCCCS', 'CCCl', 'ClC(Cl)Cl', 'BrCBr', 'FC(Cl)Cl', 'ClCCl', 'CCOC(=O)OCC', 'CCOC(=O)CC(=O)OCC', 'COCOC', 'CC(N)=O', 'NCCN', 'CCO', 'CC(=O)OC=C', 'CCOC=C', 'CCOC(C)=O', 'CCOC(=O)CC', 'CCSCC', 'CCCCCC(C)=O', 'CCCCC(C)=O', 'C=O', 'NC=O', 'OC=O', 'CO', 'COC', 'COC(=O)C(C)=C', 'COC(C)=O', 'COC=O', 'CSSC', 'CSC', 'C[S](C)=O', 'CCCCCCCCO', 'CCCCCO', 'CCC(N)CC', 'CCC(O)CC', 'OCCCCCO', 'CC(=O)CC(C)=O', 'CCCCC#N', 'C=CC#N', 'CCCN', 'CC(C)N', 'CC(C)=O', 'OCC(O)CO', 'CCC#N', 'CCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCCC', 'CCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCC'][:]

color_string = '''color gray, elem C
color white, elem H
color red, elem O
color blue, elem N
color white, elem H
color yellow, elem S
color green, elem Cl
color cyan, elem F
color brown, elem Br
color purple, elem I
'''

mol_files = [smiles_to_3d_mol(s, f"mol_{i}") for i, s in enumerate(molecules_SMILES)]

def generate_pymol_script(mol_files, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    with open("render_pymol.pml", "w") as f:
        f.write("bg_color white\n")
        f.write("set depth_cue, 0  ;# ← DISABLE fog effect\n")
        for mol in mol_files[-5:]:######################
            obj_name = os.path.splitext(os.path.basename(mol))[0]
            output_img = os.path.join(output_dir, obj_name + ".png")
            f.write(f"load {mol}, {obj_name}\n")
            f.write(f"util.cbac\n")
            f.write(f"set orthoscopic, on\n")
            f.write(color_string)
            f.write(f"hide everything, {obj_name}\n")
            f.write(f"show sticks, {obj_name}\n")
            f.write(f"show spheres, {obj_name}\n")
            f.write(f"set depth_cue, 0, {obj_name}\n")
            f.write(f"set transparency, 0, {obj_name}\n")
            f.write(f"set sphere_transparency, 0, {obj_name}\n")
            f.write(f"set stick_transparency, 0, {obj_name}\n")
            f.write(f"set sphere_scale, 0.25, {obj_name}\n")
            f.write(f"orient {obj_name}\n")
            f.write(f"zoom {obj_name}, buffer=1.5\n")
            f.write(f"ray 10000, 10000\n")
            f.write(f"png {output_img}\n")
            f.write(f"delete {obj_name}\n")
        f.write("quit\n")

generate_pymol_script(mol_files)


