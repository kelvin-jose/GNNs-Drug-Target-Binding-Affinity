import time
from pathlib import Path
from src.train.train_coregnn import train
from src.utils.logger import setup_logging
from src.data.dataset_builder import build_dataset
from src.data.pdbbind_downloader import download_pdbbind
from src.data.protein_featurizer import featurize_all_proteins
from src.data.ligand_featurizer import featurize_and_save_all_ligands
from src.data.build_full_graph import build_all_complex_graphs

def main():
    logger = setup_logging()
    start_time = time.time()
    logger.info("Starting full DTBA pipeline...")

    # --- Directories ---
    raw_dir = Path("data/raw/PDBbind_v2020_refined")
    processed_dir = Path("data/processed")
    ligand_dir = processed_dir / "ligand_graphs"
    protein_dir = processed_dir / "protein_graphs"
    complex_dir = processed_dir / "complex_graphs"
    runs_dir = Path("experiments/runs/coregnn")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Download Dataset (skip if exists) ---
    if not raw_dir.exists():
        logger.info("Downloading PDBbind refined set...")
        download_pdbbind(version="v2020", subset="refined")
    else:
        logger.info("Raw dataset already exists. Skipping download.")

    # --- 2. Build Metadata ---
    metadata_csv = processed_dir / "refined_dataset_metadata.csv"
    if not metadata_csv.exists():
        logger.info("Building dataset metadata...")
        df = build_dataset(version="v2020", index_type="refined")
        if df is None or len(df) == 0:
            logger.error("Dataset build failed — check index files.")
            return
    else:
        logger.info(f"Metadata found at {metadata_csv}")

    # --- 3. Ligand Featurization ---
    if not ligand_dir.exists() or len(list(ligand_dir.glob("*.pt"))) == 0:
        logger.info("Processing ligand graphs...")
        featurize_and_save_all_ligands()
    else:
        logger.info("Ligand graphs already exist. Skipping.")

    # --- 4. Protein Featurization ---
    if not protein_dir.exists() or len(list(protein_dir.glob("*.pt"))) == 0:
        logger.info("Processing protein graphs...")
        featurize_all_proteins()
    else:
        logger.info("Protein graphs already exist. Skipping.")

    # --- 5. Build Complex Graphs ---
    if not complex_dir.exists() or len(list(complex_dir.glob("*.pt"))) == 0:
        build_all_complex_graphs()
        logger.info(f"Complex graphs saved to {complex_dir}")
    else:
        logger.info("Complex graphs already exist. Skipping.")

    # --- 6. Train & Evaluate Model ---
    logger.info("Training and evaluating ComplexGNN...")
    train(epochs=1, batch_size=4, lr=1e-4, hidden_dim=128, encoder_layers=3, patience=10)

    total_time = (time.time() - start_time) / 60
    logger.info(f"Pipeline completed successfully in {total_time:.2f} min.")
    logger.info(f"Results saved under: {runs_dir}")


if __name__ == "__main__":
    main()