import torch
import pandas as pd
from rdkit import Chem
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

from src.utils.logger import setup_logging

logger = setup_logging()

class PDBDataset(InMemoryDataset):
    def __init__(self, metadata_csv = "data/processed/refined_dataset_metadata.csv",
        complex_dir = "data/processed/complex_graphs",
        root = "data/processed/full_dataset",
        transform=None,
        pre_transform=None,
        force_rebuild=False):
        self.metadata_csv = Path(metadata_csv)
        self.complex_dir = Path(complex_dir)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.force_rebuild = force_rebuild
        self.cache_file = self.root / f"full_dataset.pt"
        super().__init__(self.root, transform, pre_transform)

        if self.cache_file.exists() and not self.force_rebuild:
            logger.info(f"Loading cached dataset from {self.cache_file}")
            with torch.serialization.safe_globals([Data]):
                self.data, self.slices = torch.load(self.cache_file, weights_only=False)
        else:
            logger.info("Processing dataset (this may take a while)...")
            self.process()
            self.data, self.slices = torch.load(self.cache_file)

    @property
    def raw_file_names(self):
        return [self.metadata_csv.name]

    @property
    def processed_file_names(self):
        return [self.cache_file.name]
    
    def process(self):
        df = pd.read_csv(self.metadata_csv)
        data_list = []
        skipped = 0
        for _, row in df.iterrows():
            cid = row["complex_id"]
            graph_file = self.complex_dir / f"{cid}.pt"
            affinity = float(row["affinity"])

            if not graph_file.exists():
                logger.warning(f"Complex graph missing for {cid}")
                skipped += 1
                continue

            try:
                with torch.serialization.safe_globals([Data]):
                    graph_data = torch.load(graph_file, weights_only=False)
                graph_data.y = torch.tensor([affinity], dtype=torch.float32)
                graph_data.complex_id = cid
                data_list.append(graph_data)
            except Exception as e:
                logger.error(f"Failed to load complex {cid}: {e}")
                skipped += 1
                continue

            if len(data_list) % 200 == 0:
                logger.info(f"Loaded {len(data_list)} complex graphs...")

        logger.info(f"Finished. Loaded {len(data_list)} complexes. Skipped: {skipped}")

        if len(data_list) == 0:
            raise RuntimeError("No valid complex graphs loaded. Check paths and files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.cache_file)
        logger.info(f"Saved processed dataset to {self.cache_file}")

    def get(self, idx):
        return super().get(idx)

    def len(self):
        return super().len()
