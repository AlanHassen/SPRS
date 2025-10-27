# Synthesis Planning in Reaction Space (SPRS)

This repository contains the code and resources for the paper "Synthesis Planning in Reaction Space: A Study on Success, Robustness and Diversity" (Under review).

## Overview

This project investigates synthesis planning approaches, with a focus on measuring success rates, robustness of the synthesis planning algorithms, and diversity of discovered routes. Our implementation builds upon several existing frameworks, primarily AiZynthFinder, with custom search algorithms for computer-aided synthesis planning.

## Installation

The installation process involves setting up multiple systems to replicate our results. Follow these steps carefully:

### Creating a Conda Environment

We recommend setting up a dedicated conda environment based on the AiZynthFinder development environment, where Models Matter will be installed.

# SPRS Conda Environment Installation

## 0. Install SPRS Conda Environment

We have simplified the installation process. Please run the following command in your terminal:
'''./install.sh'''

**Note:** The dependencies are optimized for an M1 Mac. It might be necessary to adjust some packages depending on your environment. However, for simply reproducing the plots with Jupyter notebooks, this setup should be sufficient. For more details, we refer interested readers to the respective repositories.
### What `install.sh` Does:
The installation script automates the following steps:
1. **Install and activate the `sprs` conda environment.**
2. **Clone the required "Models Matter" repository** into the `external/` directory.
3. **Install the `modelsmatter_modelzoo`.**
   * **Important:** If this step fails, you may need to remove the `deepspeed` and `fairscale` dependencies from `external/modelsmatter/external/modelsmatter_modelzoo/pyproject.toml`, as they are not required for this purpose. After removing them, please rerun the script.

4. **Install the provided AiZynthFinder** that includes the new algorithms.
5. **Install Syntheseus Retro* benchmark** for the single-step model.
6. **Install some Mac-specific packages.**

## 1. AiZynthFinder & Models Matter

**Important:** These steps are only necessary if you want to run AiZynthFinder. We strongly suggest you do this on an HPC with Slurm and not locally.

- Either use the provided code or copy the search algorithms from `aizynthfinder/search/*` to your own AiZynthFinder implementation.
- You can switch between algorithm versions by renaming folders:
  - Rename folders to either `mcts` or `retrostar` depending on which algorithm you want to use in `aizynthfinder/aizynthfinder/search/`. For example, rename `mcts_distance` or `mcts_expansion_clustering` to `mcts` to use our distance-based/clustering-based MCTS implementation.

### Steps:

1.  Download the `data` folder from Figshare and place the content directly under your `/data/` directory. A content description is available under 'figshare.md'
    - **Caution:** Make sure the structure is `/data/*`, not `/data/data/*`.
2.  Download the retro_star model used in dual value networks into the following path: `/data/models/dual_value_networks/template_based/model/origin.txt`.
3.  Adjust the AiZynthFinder configuration files (e.g., `example_experiment/dual_value_networks_azf_config_retrostar.yml`) to include the correct full paths to your data folder.
4.  You have to activate the conda environment (`conda activate sprs`) and export the external models plugin path for AiZynthFinder before running an experiment e.g., `export PYTHONPATH=/Users/alankaihassen/development/diversity_search_rebuttal/updated_code/SPRS/aizynthfinder/plugins/`
4.  You can now run inference with the command (or use example_experiments):
    ```bash
    aizynthcli --config dual_value_networks_azf_config_retrostar.yml --stocks emolecules --policy retrostar_mlp --smiles smiles.txt
    ```

## 2. Dual Value Networks (Optional)
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
