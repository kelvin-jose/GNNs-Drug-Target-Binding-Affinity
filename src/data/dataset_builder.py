import pandas as pd
from pathlib import Path
from utils.logger import setup_logging

logger = setup_logging()

def parse_index_file(index_path: Path):
    records = []
    with open(index_path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) < 5:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pdb_id = parts[0].lower()
                resolution = float(parts[1])
                year = int(parts[2])
                affinity = float(parts[3])
                records.append({
                    "complex_id": pdb_id,
                    "resolution": resolution,
                    "year": year,
                    "affinity": affinity,
                    "measure": "pKd",
                })
            except Exception as e:
                logger.warning(f"Skipping malformed line: {line.strip()} ({e})")

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} entries from {index_path.name}")
    return df

def build_metadata_table(refined_dir, index_df, output_csv):
    data, missing = [], 0
    for _, row in index_df.iterrows():
        cid = row["complex_id"]
        subdir = refined_dir / "refined-set"
        ligand_file = subdir / cid/ f"{cid}_ligand.sdf"
        protein_file = subdir / cid / f"{cid}_protein.pdb"
        pocket_file = subdir / cid / f"{cid}_pocket.pdb"

        if not ligand_file.exists() or not protein_file.exists():
            missing += 1
            continue

        data.append({
            "complex_id": cid,
            "ligand_file": str(ligand_file),
            "protein_file": str(protein_file),
            "pocket_file": str(pocket_file) if pocket_file.exists() else None,
            "affinity": row["affinity"],
            "measure": row["measure"],
            "resolution": row["resolution"],
            "year": row["year"],
        })

    df = pd.DataFrame(data)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} entries -> {output_csv}")
    logger.info(f"Skipped {missing} complexes (missing files)")
    return df

def build_dataset(version="v2020", index_type="refined"):
    logger.info(f"Building PDBBind {version} ({index_type}) dataset...")
    refined_dir = Path(f"data/raw/PDBbind_{version}_{index_type}")
    index_file = Path(f"data/raw/PDBbind_{version}_{index_type}_plain_text_index/index/INDEX_{index_type}_data.{version.split('v')[-1]}")

    if not refined_dir.exists():
        logger.error(f"Missing directory: {refined_dir}")
        return None
    if not index_file.exists():
        logger.error(f"Missing index file: {index_file}")
        return None
    
    index_df = parse_index_file(index_file)
    output_csv = Path(f"data/processed/{index_type}_dataset_metadata.csv")
    df = build_metadata_table(refined_dir, index_df, output_csv)
    logger.info(f"Final dataset size: {len(df)} entries.")
    return df