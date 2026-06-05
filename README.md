# AHP-STG

Core implementation for **Adaptive Hierarchical Patching for Large-Scale Traffic Forecasting**.

This repository contains the public core code for AHP-STG, an adaptive hierarchical patching framework for large-scale traffic forecasting. The method organizes traffic sensors into spatial patches using a KDTree-based hierarchy and introduces lightweight adaptive refinements to better capture dynamic spatial correlations while keeping large-network forecasting efficient.

## Paper

**Title:** Adaptive Hierarchical Patching for Large-Scale Traffic Forecasting

**Authors:** Songtao Liu, Jingcheng Wang, Maolin Wang, Xingyuan Dai, Xiangyang Xiao, Hongxia Zhao, Yu Zhang, Xiaoyan Gong, Xiaoliang Xing, and Yisheng Lv

**Keywords:** Traffic forecasting; Spatio-temporal modeling; Adaptive spatial patching; Hierarchical spatial interaction; Large-scale traffic networks

## Highlights

- Adaptive hierarchical patching for scalable spatial dependency modeling.
- KDTree-based spatial hierarchy as an interpretable structural prior.
- Learnable adaptive refinement for dynamic traffic-state-aware spatial interactions.
- Future-time-aware decoding with target time metadata.
- Optional adjacency-derived spatial attention bias.
- Soft adaptive merge refinement for stable patch-level interaction.

## Repository Contents

```text
.
├── config/                  # Dataset and ablation configurations
├── imgs/                    # Framework and patching illustrations
├── lib/                     # Data loading, preprocessing, and metrics
├── models/                  # Core AHP-STG model implementation
├── results/                 # Lightweight experiment records
├── scripts/                 # Utility scripts for local baseline evaluation
├── main.py                  # Training, validation, and testing entry point
├── TRAIN_COMMANDS.txt       # Example training commands
├── Test_COMMANDS.txt        # Example testing command
├── requirements.txt         # Python dependencies
└── LICENSE
```

Large datasets, checkpoints, raw prediction arrays, logs, and manuscript files are intentionally excluded from this public repository.

## Requirements

```bash
pip install -r requirements.txt
```

The experiments were developed with PyTorch. The original local environment used the conda environment name `patchstg`.

## Data

Place downloaded LargeST-style datasets under `./data` before running training or evaluation, for example:

```text
data/
├── SD/
├── GBA/
├── GLA/
└── CA/
```

Model checkpoints should be placed under `./cpt` when evaluating pretrained weights.

## Quick Start

Train or evaluate a dataset-specific configuration with:

```bash
python main.py --config ./config/SD_future.conf
```

Other dataset configurations are provided under `config/`, including SD, GBA, GLA, and CA variants.

## Experiment Records

The lightweight JSON files and `results/optimization_log.md` record the local optimization stages and final selected results. Large raw prediction files are not included.

## License

This project is released under the MIT License. The implementation was developed from the PatchSTG codebase and substantially adapted for the AHP-STG study.
