# Running a Default Experiment

This guide outlines the steps to perform a default retrosynthesis search using AiZynthfinder via `aizynthcli`.

### 1. Installation

First, ensure you have the `sprs` conda environment installed and activated.

```bash
conda activate sprs
```

### 2. Configuration

Before running an experiment, you must adapt the paths in the configuration YAML file (e.g., `dual_value_networks_azf_config_retrostar.yml`). Update the file to point to the correct locations for your models, data, and output directories.

### 3. Running the Search

Export the plugin folder for AiZynthfinder (change path):

```bash
export PYTHONPATH=/Users/alankaihassen/development/diversity_search_rebuttal/updated_code/SPRS/aizynthfinder/plugins/
```

Execute the search from the command line. The following command runs a default search using the Retro* algorithm on the targets defined in `chembl_1.txt`, with eMolecules as the building block stock.

```bash
aizynthcli --config dual_value_networks_azf_config_retrostar.yml --stocks emolecules --policy retrostar_mlp --smiles chembl_1.txt
```

**Command Breakdown:**
*   `--config`: Specifies the configuration file for the experiment.
*   `--stocks`: Defines the building block library to use (e.g., `emolecules`).
*   `--policy`: Selects the policy network (e.g., `retrostar_mlp`).
*   `--smiles`: Points to the input file containing the target SMILES strings.