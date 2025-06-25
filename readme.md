# Synthesis Planning in Reaction Space (SPRS)

This repository contains the code and resources for the paper "Synthesis Planning in Reaction Space: A Study on Success, Robustness and Diversity" (Under review).

## Overview

This project investigates synthesis planning approaches, with a focus on measuring success rates, robustness of the synthesis planning algorithms, and diversity of discovered routes. Our implementation builds upon several existing frameworks, primarily AiZynthFinder, with custom search algorithms for computer-aided synthesis planning.

## Installation

The installation process involves setting up multiple systems to replicate our results. Follow these steps carefully:

### Creating a Conda Environment

We recommend setting up a dedicated conda environment based on the AiZynthFinder development environment, where Models Matter will be installed.

### 1. AiZynthFinder Setup

- Either use the provided code or copy the search algorithms from `aizynthfinder/search/*` to your own AiZynthFinder implementation.
- You can switch between algorithm versions by renaming folders:
  - Rename folders to either `mcts` or `retrostar` depending on which algorithm you want to use. For example, rename `mcts_distance` or `mcts_expansion_clustering` to `mcts` to use our distance-based/clustering-based MCTS implementation.
- Install AiZynthFinder in development mode as described in the [official repository](https://github.com/MolecularAI/aizynthfinder).

### 2. Models Matter Framework

- Clone & install the Models Matter framework in the same conda environment as AiZynthFinder.
- This is necessary to use the Dual Value Networks / Retro* template-based model.
- Install the Retro* benchmark support: `pip install git+https://github.com/AustinT/syntheseus-retro-star-benchmark`
- Copy all files from `code/models_matter/*` to your Models Matter installation at `modelsmatter_modelzoo/ssbenchmark/ssmodels/*`.
  - Note: These custom model implementations will be merged into the main Models Matter repository after publication.

### 3. Dual Value Networks (Optional)

- For evaluation with Dual Value Networks, refer to their repository: [https://github.com/DiXue98/PDVN](https://github.com/DiXue98/PDVN).
- We provide a newly trained self-play model in our data package.

## Data

All necessary data is available on Figshare [https://figshare.com/s/44f7c0d150d0dbfd15fe]:

- Evaluation datasets:
  - ChEMBL1000 and ChEMBL100 (subsample)
  - USPTO190
  - GDB17-1000
- Building blocks:
  - eMolecules
  - eMolecules without best-first routes
- Pre-trained models:
  - Template-based model from Retro*
  - Self-play model from Dual Value Networks

Find a detailed visualization under figshare.md.

## Troubleshooting

Common issues and solutions:

### Installation Issues
- **Missing dependencies**: Make sure your conda environment has all required packages. Check the AiZynthFinder repository for detailed requirements.
- **Import errors**: Ensure all packages are installed in the same conda environment and paths are correctly set.

### Runtime Issues
- **Model/Data loading fails**: Verify that all paths to mode/data files are absolute and correct.

If you encounter persistent issues, please open an issue in this repository with details about your environment and the specific error messages.

## Running Experiments

### HPC

We provide example templates for running experiments on HPC infrastructure:

1. Copy the template directory: `data/run/example_run_template`
2. Adjust the pathways in the configuration files:
   - `data/run/example_run_template/original_file/dual_value_networks_azf_config_retrostar.yml`
   - `data/run/example_run_template_/temp/runs/run.slurm`
3. Submit the job: `sbatch run.slurm`

For other datasets, split the provided evaluation data and join the resulting HDF files as needed.

### Dual Value Networks

We also provide scripts to reproduce the Dual Value Networks results under `data/run/dpvn/`

## Citation

If you use this code or data in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## Contributing

We welcome contributions to this project. Please feel free to open issues or submit pull requests.


## Funding

This study was partially funded by the European Union's Horizon 2020 research and innovation program under the Marie Skłodowska-Curie Innovative Training Network European Industrial Doctorate grant agreement No. 956832 "Advanced machine learning for Innovative Drug Discovery".
