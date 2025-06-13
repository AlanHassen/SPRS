## Figshare Repository Structure

```
data
├── evaluation_data [Evaluation data from dual value networks and chembl1000 subsample]
├── models
│   └── dual_value_networks [self play model retrained by us]
│   └── template-based model [without self play]
├── results
│   ├── azf [AiZynthFinder results]
│   └── dpvn [Dual Value Networks Results]
├── run [Configuration files and scripts]
└── stocks [Building blocks datasets]
```

## Data Details

### Evaluation Data

The evaluation data is sourced from dual value networks ([https://github.com/DiXue98/PDVN](https://github.com/DiXue98/PDVN)) and includes:
- chembl1000
- uspto190
- gdb17-1000
- chembl1000 subsample (chembl100) that we created by subsampling solved molecules from chembl1000 

### Models

Contains self-play model retrained by us:
- Self-play models
- Template-based models (can be downloaded from a separate figshare repository)

### Results

The results section is divided into two main categories:

#### AiZynthFinder (azf) Results

- **chembl100**: Results on the Chembl100 subsample with various algorithms
- **full_building_blocks**: Results using all available building blocks for multiple datasets
- **removed_best_first_building_blocks**: Results with the reduced building block set

#### Dual Value Networks (dpvn) Results

- **full_building_blocks**: Using all building blocks for chembl1000, gdb17, and uspto_190 datasets
- **no_best_first_building_blocks**: Using the building blocks without best first results

### Run Configurations

Contains example configurations and scripts to run evaluations:
- AZF MCTS and RetroStar configuration files
- Scripts for running evaluations with dual value networks
- Example template for evaluating a dataset using AiZynthFinder

### Stocks (Building Blocks)

Contains two versions of building block datasets:
- Full building blocks
- Reduced building blocks dataset (removed building blocks)


### Full data provided
```
├── evaluation_data [Evaluation data from dual value networks [https://github.com/DiXue98/PDVN] (chembl1000, uspto190, gdb17-1000) and also the chembl1000 subsample (chembl100)]
│   ├── chembl_1000_subsample_emolecules_stock_100_targets.csv
│   └── dual_value_networks
│       ├── dual_value_networks_chembl1000_evaluation_smiles.csv
│       ├── dual_value_networks_gdb17_evaluation_smiles.csv
│       └── dual_value_networks_uspto_190_hard_molecules.csv
├── models
│   └── dual_value_networks [self play model retrained by us]
│       ├── self_play
│       │   └── final_results
│       │       └── model
│       │           └── 742788_mols
│       │               └── one_step
│       │                   ├── prior_fn.ckpt
│       │                   ├── rollout_model.ckpt
│       │                   └── value_fn.ckpt
│       └── template_based [Template based model used as a basis can be downloaded from a different figshare repository]
│           └── model
│               └── origin.txt
├── results
│   ├── azf
│   │   ├── chembl100 [Results on the Chembl100 subsample]
│   │   │   ├── 0_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf
│   │   │   ├── 0_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster_route_statistics.csv
│   │   │   ├── 1_dpvn_ssm_chembl_100_subsample_all_routes_default_mcts_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf
│   │   │   ├── 1_dpvn_ssm_chembl_100_subsample_all_routes_default_mcts_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster_route_statistics.csv
│   │   │   ├── 2_dpvn_ssm_chembl_100_subsample_all_routes_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf
│   │   │   ├── 2_dpvn_ssm_chembl_100_subsample_all_routes_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster_route_statistics.csv
│   │   │   ├── 3_dpvn_ssm_chembl_100_subsample_all_routes_expansion_clustering_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf
│   │   │   ├── 3_dpvn_ssm_chembl_100_subsample_all_routes_expansion_clustering_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster_route_statistics.csv
│   │   │   ├── 5_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf
│   │   │   └── 5_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster_route_statistics.csv
│   │   ├── full_building_blocks [Results using all building blocks]
│   │   │   ├── 0_dpvn_ssm_chembl_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 0_dpvn_ssm_gdb17_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 0_dpvn_ssm_uspto190_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 1_dpvn_ssm_chembl_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules.hdf
│   │   │   ├── 1_dpvn_ssm_gdb17_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 1_dpvn_ssm_uspto190_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 2_dpvn_ssm_chembl_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 2_dpvn_ssm_gdb17_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 2_dpvn_ssm_uspto190_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 5_dpvn_ssm_chembl_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 5_dpvn_ssm_gdb17_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 5_dpvn_ssm_uspto190_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_25000_iterations.hdf
│   │   │   ├── 6_dpvn_ssm_chembl_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   ├── 6_dpvn_ssm_gdb17_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   │   └── 6_dpvn_ssm_uspto190_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30.hdf
│   │   └── removed_best_first_building_blocks [Results with the reduced building block set]
│   │       ├── 0_dpvn_ssm_chembl_retrostar_0_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30_25000_iterations.hdf
│   │       ├── 0_dpvn_ssm_gdb17_retrostar_0_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30_25000_iterations.hdf
│   │       ├── 0_dpvn_ssm_uspto190_retrostar_0_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30_25000_iterations.hdf
│   │       ├── 1_dpvn_ssm_chembl_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30.hdf
│   │       ├── 1_dpvn_ssm_gdb17_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30.hdf
│   │       ├── 1_dpvn_ssm_uspto190_DEFAULT_mcts_50_wide_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30.hdf
│   │       ├── 2_dpvn_ssm_chembl_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30.hdf
│   │       ├── 2_dpvn_ssm_gdb17_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30.hdf
│   │       ├── 2_dpvn_ssm_uspto190_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30.hdf
│   │       ├── 5_dpvn_ssm_chembl_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30_25000_iterations.hdf
│   │       ├── 5_dpvn_ssm_gdb17_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30_25000_iterations.hdf
│   │       ├── 5_dpvn_ssm_uspto190_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30_25000_iterations.hdf
│   │       ├── 6_dpvn_ssm_chembl_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30.hdf
│   │       ├── 6_dpvn_ssm_gdb17_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_0_depth_30.hdf
│   │       └── 6_dpvn_ssm_uspto190_expansion_cluster_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_without_retrostar_depth_30.hdf
│   └── dpvn [Dual Value Networks Results]
│       ├── full_building_blocks [Using all building blocks]
│       │   ├── chembl1000_with_realistic_filter
│       │   │   ├── evaluation_metrics.json
│       │   │   ├── result_iter.json
│       │   │   └── routes_dict
│       │   │       ├── Routes generated by dual value networks
│       │   ├── gdb17_with_realistic_filter
│       │   │   ├── evaluation_metrics.json
│       │   │   ├── result_iter.json
│       │   │   └── routes_dict
│       │   │       ├── Routes generated by dual value networks
│       │   ├── note.txt
│       │   └── uspto_190
│       │       ├── evaluation_metrics.json
│       │       └── result_iter.json
│       └── no_best_first_building_blocks [Using the building blocks without best first results]
│           ├── chembl_realistic_filter
│           │   ├── evaluation_metrics.json
│           │   ├── result_iter.json
│           │   └── routes_dict
│           ├── gdb17_realistic_filter
│           │   ├── evaluation_metrics.json
│           │   ├── result_iter.json
│           │   └── routes_dict
│           │       ├── Routes generated by dual value networks
│           └── uspto190
│               ├── evaluation_metrics.json
│               └── result_iter.json
├── run
│   ├── configs [example run configs how to run evaluations using AZF]
│   │   ├── chembl100_azf_mcts_config.yml
│   │   ├── chembl100_azf_retrostar_config.yml
│   │   ├── dual_value_networks_azf_config_mcts.yml
│   │   └── dual_value_networks_azf_config_retrostar.yml
│   ├── dpvn [configs how to run an evaluate with dual value networks]
│   │   ├── 0_train_auto_dpvn.sh
│   │   ├── 1_uspto_without_best_first_routes_evaluation.sh
│   │   ├── 4_chembl_evaluation_with_reactlistic_filter.sh
│   │   ├── 5_chembl_evaluation_without_best_first_routes_with_realistic_filter.sh
│   │   ├── 7_gdb17_evaluation_with_realistic_filter.sh
│   │   └── 8_gdb17_evaluation_without_best_first_routes_with_realistic_filter.sh
│   └── example_run_template_uspto190 [an example template how to evaluate dataset using Aizynthfinder]
│       ├── original_file
│       │   ├── dual_value_networks_azf_config_retrostar.yml
│       │   └── dual_value_networks_uspto_190_hard_molecules.csv
│       ├── results
│       └── temp
│           ├── result_hdfs
│           ├── result_route_statistics
│           ├── runs
│           │   └── run.slurm
│           └── splitted_data
│               ├── dual_value_networks_uspto_190_hard_molecules_part_0.csv
│               ├── dual_value_networks_uspto_190_hard_molecules_part_1.csv
│               └── dual_value_networks_uspto_190_hard_molecules_part_2.csv
└── stocks [building blocks datasets]
    ├── full_building_blocks
    │   └── dual_value_networks_emolecules_stock.hdf5
    └── removed_building_blocks
        ├── emolecules_without_retrostar_building_blocks_azf_format.hdf5
        └── emolecules_without_retrostar_building_blocks_dpvn_format.csv
```