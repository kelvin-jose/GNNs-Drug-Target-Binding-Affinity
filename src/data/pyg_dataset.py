from pathlib import Path
from utils.logger import setup_logging
from torch_geometric.data import InMemoryDataset

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