from rdkit import Chem
from utils.logger import setup_logging

logger = setup_logging()

# map for common atoms
COMMON_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Si", "Se"]
ATOM_MAP = {a: i for i, a in enumerate(COMMON_ATOMS)}

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4
}

def one_hot(x, choices):
    out = [0] * len(choices)
    if x in choices:
        out[choices.index(x)] = 1
    return out
