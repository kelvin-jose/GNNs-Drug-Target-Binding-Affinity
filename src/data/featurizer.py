import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
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

def featurize_rdkit_mol(mol, use_explicit_hs = True):
    if mol is None:
        logger.error("featurize_rdkit_mol received None")
        return None

    if use_explicit_hs:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass

    num_atoms = mol.GetNumAtoms()
    node_feats = np.vstack([atom_to_feature_vector(a) \
                            for a in mol.GetAtoms()]) if num_atoms > 0 \
                            else np.zeros((0, len(COMMON_ATOMS)+5+len(HYBRIDIZATION_MAP)+1), 
                            dtype=np.float32)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        is_aromatic = 1 if bond.GetIsAromatic() else 0
        
        bt_oh = [
            1 if bt == Chem.rdchem.BondType.SINGLE else 0,
            1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
        ]
        edge_attr = bt_oh + [is_aromatic]
        edges.append((i, j))
        edges.append((j, i))
        edge_attrs.append(edge_attr)
        edge_attrs.append(edge_attr)
    
    if len(edges) > 0:
        edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
        edge_attr = np.array(edge_attrs, dtype=np.float32)
    else:
        edge_index = np.zeros((2,0), dtype=np.int64)
        edge_attr = np.zeros((0,5), dtype=np.float32)

    pos = np.zeros((num_atoms, 3), dtype=np.float32)
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer(0)
        for i in range(num_atoms):
            p = conf.GetAtomPosition(i)
            pos[i] = [p.x, p.y, p.z]
    else:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
            conf = mol.GetConformer()
            for i in range(num_atoms):
                p = conf.GetAtomPosition(i)
                pos[i] = [p.x, p.y, p.z]
        except Exception:
            pass
    
    # numpy to tensors
    node_feats = torch.tensor(node_feats, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    return {
        "x": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos
    }