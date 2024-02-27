import rdkit
from rdkit import Chem
from rdkit.Chem import SMARTS
from edfp import generate_edfp

def maccs_to_smiles(maccs):
  maccs = maccs(generate_edfp)
  mol = Chem.MolFromMACCS(maccs)
  return Chem.MolToSmiles(mol)

def get_molecular_weight(smiles):

  mol = Chem.MolFromSmiles(smiles)
  return Chem.GetMolWt(mol)

def get_structure_image(smiles):

  mol = Chem.MolFromSmiles(smiles)
  return Chem.DrawMolToImage(mol)

if __name__ == "__main__":
  maccs = maccs

  smiles = maccs_to_smiles(maccs)

  molecular_weight = get_molecular_weight(smiles)

  structure_image = get_structure_image(smiles)

  print("SMILES:", smiles)
  print("MW:", molecular_weight)
  print("PNG:", structure_image)
  filtered_molecules = [mol for mol in molecules if mol.GetNumAtomsMatchingSmarts(smarts_query) > 0]