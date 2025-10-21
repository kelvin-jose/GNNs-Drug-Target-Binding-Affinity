from pathlib import Path
from Bio.PDB import PDBParser, is_aa
from utils.logger import setup_logging

logger = setup_logging()

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