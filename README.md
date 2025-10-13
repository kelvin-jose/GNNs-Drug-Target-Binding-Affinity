# GNNs-Drug-Target-Binding-Affinity

### Project Goal
Build a robust geometric GNN that predicts binding affinity from <b>protein-ligand</b> 3D complexes, and demonstrate imporved accuracy and interpretability vs classical docking and standard 2D GNN baselines. 

### Phase 0
- Create a repo skeleton
- Setup environment management
- Version control with git
- Establish experiment tracking

### Phase 1
- Download dataset
- Preprocess downloaded files
- Save output files

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
│   │   └── dataset_builder.py      # helper functions to build train dataset
│   ├── utils/  
│   │   └── logger.py               # helper functions (logging)
│   └── main.py                     # main entry point
│
├── experiments/           # experiment outputs, checkpoints TensorBoard logs
├── requirements.txt
├── .gitignore
├── README.md
└── setup_logging.yaml     # logging configuration file
```
#### To run locally
```bash
python src/main.py

tensorboard --logdir experiments/runs
```