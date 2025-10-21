import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, is_aa
from torch_geometric.data import Data
from utils.logger import setup_logging

logger = setup_logging()

# amino acid representation in PDB format
AA3 = [
    "ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]
AA3_TO_IDX = {aa:i for i, aa in enumerate(AA3)}

def parse_pocket_residues(pdb_path):
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        logger.error(f"pdb file not found: {pdb_path}")
        return []

    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as e:
        logger.exception(f"Failed to parse PDB {pdb_path}: {e}")
        return []

    residues = []
    for model in struct:
        for chain in model:
            chain_id = chain.get_id()
            for res in chain:
                # skip hetatms that are not amino acids (but include if flagged as amino acid)
                if not is_aa(res, standard=True):
                    continue
                resseq = res.get_id()[1]
                res_uid = f"{chain_id}_{res.get_resname().strip()}_{resseq}"
                residues.append((res_uid, res))
    logger.info(f"Parsed {len(residues)} residues from {pdb_path.name}")
    return residues

def residue_to_feature(res):
    # amino acid one-hot
    resname = res.get_resname().upper()
    onehot = np.zeros(len(AA3), dtype=np.float32)
    if resname in AA3_TO_IDX:
        onehot[AA3_TO_IDX[resname]] = 1.0
    # residue index (residue id within chain)
    try:
        resseq = res.get_id()[1]  # integer sequence number
    except Exception:
        resseq = 0
    # normalize later relative to chain length if desired; here we scale by 1e-2 to keep small
    seq_feat = np.array([resseq], dtype=np.float32)
    # average bfactor across atoms in residue
    b_factors = [atom.get_bfactor() for atom in res.get_atoms()]
    avg_b = np.mean(b_factors).astype(np.float32) if len(b_factors) > 0 else np.array([0.0], dtype=np.float32)
    feat = np.concatenate([onehot, seq_feat/100.0, np.array([avg_b/100.0], dtype=np.float32)])
    return feat

def get_ca_coord(res):
    for atom in res:
        name = atom.get_name()
        if name == "CA":
            coord = atom.get_coord()
            return np.array(coord, dtype=np.float32)
    # fallback: centroid of heavy atoms
    coords = [a.get_coord() for a in res if a.element != "H"]
    if len(coords) == 0:
        # last fallback: any atom
        coords = [a.get_coord() for a in res]
    centroid = np.mean(coords, axis=0) if len(coords) > 0 else np.zeros(3, dtype=np.float32)
    return np.array(centroid, dtype=np.float32)

def build_residue_graph_from_pdb(pdb_path, residue_cutoff = 8.0):
    residues = parse_pocket_residues(pdb_path)
    if len(residues) == 0:
        raise RuntimeError(f"No residues parsed from {pdb_path}")

    node_feats = []
    positions = []
    res_ids = []
    for res_uid, res in residues:
        node_feats.append(residue_to_feature(res))
        positions.append(get_ca_coord(res))
        res_ids.append(res_uid)
    x = torch.tensor(np.vstack(node_feats), dtype=torch.float32)
    pos = torch.tensor(np.vstack(positions), dtype=torch.float32)

    # compute pairwise distances and edges
    coords = pos.numpy()
    N = coords.shape[0]
    edge_index = []
    edge_attr = []
    cutoff = float(residue_cutoff)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= cutoff:
                # add both directions
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([d])
                edge_attr.append([d])
    if len(edge_index) == 0:
        # no edges found (e.g., single residue) -> create self-loop to keep PyG happy
        edge_index = [[0,0]]
        edge_attr = [[0.0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    # add metadata
    data.metadata = {
        "complex_id": Path(pdb_path).parent.name,
        "pdb_path": str(pdb_path),
        "residue_ids": res_ids,
        "node_type": "residue",
        "residue_cutoff": residue_cutoff
    }
    return data