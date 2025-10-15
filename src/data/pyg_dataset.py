import torch
from pandas as pd
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import SDMolSupplier
from utils.logger import setup_logging
from torch_geometric.data import InMemoryDataset, Data

from data.featurizer import featurize_rdkit_mol

logger = setup_logging()

class PDBBindLigandDataset(InMemoryDataset):
    def __init__(self, metadata_csv="data/processed/refined_dataset_metadata.csv", 
                 root="data/processed/graphs", transform=None, 
                 pre_transform=None, force_rebuild=False):
        self.metadata_csv = Path(metadata_csv)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.root / "dataset.pt"
        self.force_rebuild = force_rebuild
        super().__init__(self.root, transform, pre_transform)

        if self.cache_file.exists() and not self.force_rebuild:
            logger.info(f"Loading cached dataset from {self.cache_file}")
            self.data, self.slices = torch.load(self.cache_file)
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
            ligand_path = Path(row["ligand_file"])
            affinity = float(row["affinity"])
            if not ligand_path.exists():
                logger.warning(f"Ligand file missing for {cid}: {ligand_path}")
                skipped += 1
                continue

            # read as RDKit Mol (sdf or mol2 may be present—try sdf first)
            mol = None
            try:
                if ligand_path.suffix.lower() == ".sdf":
                    supplier = SDMolSupplier(str(ligand_path), removeHs=False)
                    mol = supplier[0]
                else:
                    mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
            except Exception as e:
                logger.warning(f"RDKit read failed for {cid}: {e}")
                mol = None

            if mol is None:
                logger.warning(f"Failed to parse ligand for {cid}, skipping.")
                skipped += 1
                continue

            feats = featurize_rdkit_mol(mol)
            if feats is None:
                skipped += 1
                continue

            data = Data(x = feats["x"], edge_index = feats["edge_index"],
                edge_attr = feats["edge_attr"], pos = feats["pos"] if feats["pos"].shape[0] > 0 else None,
                y = torch.tensor([affinity], dtype=torch.float32), complex_id = cid)
            data_list.append(data)

            if len(data_list) % 500 == 0:
                logger.info(f"Processed {len(data_list)} molecules...")

        logger.info(f"Finished processing. Total processed: {len(data_list)}. Skipped: {skipped}")
        if len(data_list) == 0:
            raise RuntimeError("No data processed. Check metadata and ligand files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.cache_file)
        logger.info(f"Saved processed dataset to {self.cache_file}")