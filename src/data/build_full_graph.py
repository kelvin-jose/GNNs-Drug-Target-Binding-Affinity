import torch
import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logging

logger = setup_logging()

def build_all_complex_graphs(metadata_csv="data/processed/refined_dataset_metadata.csv",
                             ligand_dir="data/processed/ligand_graphs",
                             protein_dir="data/processed/protein_graphs",
                             output_dir="data/processed/complex_graphs",
                             contact_cutoff=5.0):
    df = pd.read_csv(metadata_csv)
    ligand_dir = Path(ligand_dir)
    protein_dir = Path(protein_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    logger.info(f"Building complex graphs for {len(df)} entries...")

    for _, row in df.iterrows():
        cid = row["complex_id"]
        affinity = float(row["affinity"])
        ligand_file = ligand_dir / f"{cid}.pt"
        protein_file = protein_dir / f"{cid}.pt"
        out_file = output_dir / f"{cid}.pt"

        if out_file.exists():
            continue

        if not ligand_file.exists() or not protein_file.exists():
            logger.warning(f"Missing graphs for {cid}, skipping.")
            skipped += 1
            continue

        try:
            ligand_data = torch.load(ligand_file)
            protein_data = torch.load(protein_file)
            complex_data = combine_graphs(
                ligand_data, protein_data, affinity, cutoff=contact_cutoff
            )
            torch.save(complex_data, out_file)
        except Exception as e:
            logger.error(f"Failed to build complex {cid}: {e}")
            skipped += 1

    logger.info(f"Complex graph building complete. Skipped: {skipped}. Output → {output_dir}")
