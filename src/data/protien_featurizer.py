import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, is_aa
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