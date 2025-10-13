# GNNs-Drug-Target-Binding-Affinity

### Project Goal
Build a robust geometric GNN that predicts binding affinity from <b>protein-ligand</b> 3D complexes, and demonstrate imporved accuracy and interpretability vs classical docking and standard 2D GNN baselines. 

### Phase 0
- Create a repo skeleton
- Setup environment management
- Version control with git
- Establish experiment tracking

### Directory Structure
```bash
GNNs-Drug-Target-Binding-Affinity/
│
├── src/                   # all source code lives here
│   ├── utils/             # helper functions (logging)
│   └── main.py            # entry point
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