# GNNs-Drug-Target-Binding-Affinity

### Project Goal
Proteins are the molecular machines of life, and drugs are small molecules designed to interact with them. The strength of this interaction—how tightly a drug binds to a protein target—is known as its binding affinity. Accurately predicting binding affinity is a cornerstone of drug discovery, but experimentally measuring it is slow, expensive, and limited to a small number of compounds.

The goal is to build a robust geometric GNN that predicts binding affinity from <b>protein-ligand</b> 3D complexes, and demonstrate imporved accuracy and interpretability vs classical docking and standard 2D GNN baselines. 

### Key Milestones
#### Phase 0
- Create a repo skeleton
- Setup environment management
- Version control with git
- Establish experiment tracking

#### Phase 1
- Download dataset
- Preprocess downloaded files
- Save output files

#### Phase 2
- Train a ligand-only GNN model
- Evaluate the trained model
- Save best model files 

#### Phase 3
- Featurize proteins and ligands
- Generate complex graphs
- More sophisticated GNN based model
- Full train and eval pipeline

### Architectire Overview
```bash
 ┌──────────────────────────┐
 │      Dataset Builder     │
 │  - Parse INDEX_refined   │
 │  - Map ligand/protein    │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │     Featurizers          │
 │  Ligand -> atom graph     │
 │  Protein -> residue graph │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Complex Graph Builder    │
 │  - Merge ligand+protein  │
 │  - Add inter-edges (≤5Å) │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │     CoreGNN Model     │
 │  - Dual Graph Encoders   │
 │  - Cross-Attention Layer │
 │  - Affinity Regression   │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Trainer + Evaluator      │
 │  - RMSE, Pearson         │
 │  - TensorBoard logging   │
 └──────────────────────────┘

```

### Directory Structure
```bash
GNNs-Drug-Target-Binding-Affinity/
│
├── data/                  # all source data live here
│   ├── raw/               # raw data downloaded data
│   └── processed/         # data after preprocessing
├── src/                   # all source code lives here
│   ├── data/              # source code to process data
│   │   ├── pdbbind_downloader.py   # helper functions to download data
│   │   ├── ligand_featurizer.py    # helper functions to convert rdkit molecules to feature vectors
│   │   ├── protein_featurizer.py   # helper functions to convert proteins to features
│   │   ├── pyg_dataset.py          # helper functions to create PDBBindDataset data
│   │   ├── build_full_graph.py     # helper functions to create protein - ligand cross edges
│   │   └── dataset_builder.py      # helper functions to build train 
│   ├── train/                      # source code for training
│   │   ├── train_coregnn.py        # train and evaluate a core GNN model
│   │   └── train_baseline.py       # helper functions to train a ligand-only GNN baseline model
│   ├── models/                     # source code for model files
│   │   ├── core_gnn.py             # Main GNN model
│   │   └── gnn.py                  # SAGEConv based simple baseline 
dataset
│   ├── utils/  
│   │   └── logger.py               # helper functions (logging)
│   └── main.py                     # main entry point
│
├── experiments/           # experiment outputs, checkpoints TensorBoard logs
├── notebooks/             # notebook artifacts 
├── requirements.txt
├── .gitignore
├── README.md
└── setup_logging.yaml     # logging configuration file
```
### To run locally
##### E2E pipeline
```bash
python -m src.main
```
##### Logging in tensorboard
```bash
tensorboard --logdir experiments/runs
```
### Evaluation Metrics
| Metric |  Meaning  |
|:-----|:--------:|
| `RMSE`   | Root Mean Square Error |
| `R²`   |  Coefficient of Determination  |
| `Pearson`   | Linear correlation between predicted and actual |

### Citation
```bash
@software{coregnn_2025,
  author = {Kelvin Jose},
  title = {Residue-Level Graph Neural Network for DTBA Prediction},
  year = {2025},
  url = {https://github.com/kelvin-jose/GNNs-Drug-Target-Binding-Affinity}
}
```