import numpy as np
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

def atom_to_feature_vector(atom):
    symbol = atom.GetSymbol()
    atom_onehot = one_hot(symbol, COMMON_ATOMS)
    degree = atom.GetDegree()  # int
    formal_charge = atom.GetFormalCharge()
    num_hs = atom.GetTotalNumHs()
    aromatic = 1 if atom.GetIsAromatic() else 0
    hybrid = HYBRIDIZATION_MAP.get(atom.GetHybridization(), -1)
    hybrid_oh = one_hot(hybrid, list(range(len(HYBRIDIZATION_MAP)+1)))  # include -1 as index 0?
    vec = atom_onehot + [degree, formal_charge, num_hs, aromatic] + hybrid_oh
    return np.array(vec, dtype=np.float32)