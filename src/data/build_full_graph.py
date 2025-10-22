import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from src.utils.logger import setup_logging

logger = setup_logging()

def build_cross_edges(ligand_pos, protein_pos, cutoff=5.0):
    cross_edges = []
    n_lig = ligand_pos.shape[0]
    n_pro = protein_pos.shape[0]

    for i in range(n_lig):
        for j in range(n_pro):
            dist = np.linalg.norm(ligand_pos[i] - protein_pos[j])
            if dist <= cutoff:
                cross_edges.append((i, n_lig + j))  # ligand -> protein
                cross_edges.append((n_lig + j, i))  # protein -> ligand
    if len(cross_edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(cross_edges, dtype=np.int64).T
    return edge_index

def combine_graphs(ligand_data, protein_data, affinity, cutoff=5.0):
    # extract and shift indices
    num_lig_nodes = ligand_data.x.size(0)
    num_pro_nodes = protein_data.x.size(0)

    # ligand–protein edges
    cross_edge_index = build_cross_edges(
        ligand_data.pos.cpu().numpy(),
        protein_data.pos.cpu().numpy(),
        cutoff=cutoff
    )

    # merge intra-graph edges
    ligand_edge_index = ligand_data.edge_index
    protein_edge_index = protein_data.edge_index + num_lig_nodes  # shift protein indices

    # combine all edge indices
    all_edges = torch.cat([
        ligand_edge_index,
        protein_edge_index,
        torch.tensor(cross_edge_index, dtype=torch.long)
    ], dim=1) if cross_edge_index.shape[1] > 0 else torch.cat([
        ligand_edge_index,
        protein_edge_index
    ], dim=1)

    # combine features and positions
    x = torch.cat([ligand_data.x, protein_data.x], dim=0)
    pos = torch.cat([ligand_data.pos, protein_data.pos], dim=0)

    data = Data(
        x=x,
        edge_index=all_edges,
        pos=pos,
        y=torch.tensor([affinity], dtype=torch.float32),
        node_type=torch.cat([torch.zeros(num_lig_nodes, dtype=torch.long), torch.ones(num_pro_nodes, dtype=torch.long)])
    )
    return data

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
            with torch.serialization.safe_globals([Data]):
                ligand_data = torch.load(ligand_file, weights_only=False)
                protein_data = torch.load(protein_file, weights_only=False)
            
            complex_data = combine_graphs(
                ligand_data, protein_data, affinity, cutoff=contact_cutoff
            )
            torch.save(complex_data, out_file)
        except Exception as e:
            logger.error(f"Failed to build complex {cid}: {e}")
            skipped += 1

    logger.info(f"Complex graph building complete. Skipped: {skipped}. Output → {output_dir}")

if __name__ == "__main__":
    build_all_complex_graphs()
