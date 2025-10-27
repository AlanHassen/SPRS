from aizynthfinder.reactiontree import ReactionTree
import pandas


class SynthesisRoute:
    """A wrapper around the aizynthfinder reaction tree class for building block extraction"""

    def __init__(self, reaction_tree: ReactionTree):
        self.reaction_tree = reaction_tree

    def get_root_molecules_smiles(self):
        """Returns the smiles of the root molecule"""
        root_smiles = self.reaction_tree.root.smiles
        assert root_smiles is not None
        return root_smiles

    def is_solved(self):
        """Check if the route is solved"""
        return self.reaction_tree.is_solved

    def get_in_stock_building_blocks(self):
        """
        Extract building blocks (leaf molecules) and classify them as in-stock or missing.
        
        Returns:
            tuple: (in_stock, missing_stock) - two lists of SMILES strings
        """
        route_leafs = [mol for mol in self.reaction_tree.leafs()]

        in_stock = []
        missing_stock = []
        for leaf in route_leafs:
            if self.reaction_tree.in_stock(leaf):
                in_stock.append(leaf.smiles)
            else:
                missing_stock.append(leaf.smiles)

        return in_stock, missing_stock


class MoleculeResult:
    """A class for extracting building blocks from synthesis routes of a molecule"""

    def __init__(self, synthesis_routes_df: pandas.DataFrame) -> None:
        """
        Args:
            synthesis_routes_df: DataFrame with a column named 'synthesis_route' containing SynthesisRoute objects
        """
        self.synthesis_routes_df = synthesis_routes_df

    def get_synthesis_routes(self):
        """Get list of SynthesisRoute objects"""
        return self.synthesis_routes_df["synthesis_route"].tolist()

    def get_all_building_blocks(self):
        """
        Extract all building blocks (in stock and missing) from all synthesis routes.
        
        Returns:
            pandas.DataFrame: DataFrame with columns:
                - route_index: index of the route
                - in_stock_building_blocks: list of SMILES strings for available building blocks
                - missing_building_blocks: list of SMILES strings for missing building blocks
                - is_solved: whether the route is fully solved
        """
        results = []
        
        for idx, synthesis_route in enumerate(self.get_synthesis_routes()):
            in_stock, missing = synthesis_route.get_in_stock_building_blocks()
            results.append({
                'route_index': idx,
                'in_stock_building_blocks': in_stock,
                'missing_building_blocks': missing,
                'is_solved': synthesis_route.is_solved()
            })
        
        return pandas.DataFrame(results)

    def get_solved_routes_building_blocks(self):
        """
        Extract building blocks only from solved synthesis routes.
        
        Returns:
            pandas.DataFrame: DataFrame with building blocks from solved routes only
        """
        results = []
        
        for idx, synthesis_route in enumerate(self.get_synthesis_routes()):
            if synthesis_route.is_solved():
                in_stock, missing = synthesis_route.get_in_stock_building_blocks()
                results.append({
                    'route_index': idx,
                    'in_stock_building_blocks': in_stock,
                    'missing_building_blocks': missing
                })
        
        return pandas.DataFrame(results)
    
class AiZynthfinderResults:
    """ a class for calculating the results of AiZynthfinder """

    def __init__(self, aizynthfinder_results, additional_route_data_column = None):
        self.aizynthfinder_results = aizynthfinder_results
        self.additional_route_data_column = additional_route_data_column
        self.extract_all_reaction_trees()

    def extract_all_reaction_trees(self):
        # loop through the results and convert them to ReactionTree objects
        molecule_results = []
        for index, row in self.aizynthfinder_results.iterrows():
            molecule_results.append(self._extract_molecule_results(row))
        
        self.aizynthfinder_results["molecule_results"] = molecule_results

    def _extract_molecule_results(self, molecule_result_row) -> MoleculeResult:
        """Calculate the routes for the trees of a singular molecule

        Args:
            molecule_result_row: a row of data from AiZynthfinder containing 'trees'

        Returns:
            MoleculeResult: object containing synthesis routes with building block extraction methods
        """
        synthesis_routes = []

        for tree in molecule_result_row["trees"]:
            reaction_tree = ReactionTree.from_dict(tree)
            synthesis_route = SynthesisRoute(reaction_tree)
            synthesis_routes.append(synthesis_route)
        
        # Create a DataFrame with the synthesis routes
        synthesis_routes_df = pandas.DataFrame({'synthesis_route': synthesis_routes})
        
        # Create and return MoleculeResult object
        molecule_result = MoleculeResult(synthesis_routes_df)
        
        return molecule_result


from pathlib import Path

# Get the current directory as a Path object
current_dir = Path.cwd()

# Get the parent directory
repository_directory = current_dir.parent
result_path = str(repository_directory) +   "/data/results/azf/chembl100/"
save_path = str(repository_directory) +  "/results/visualizations"

print(f"Repository Directory:  {repository_directory}")
print(f"Result Path:  {result_path}")
print(f"Save Path:  {save_path}")

data_info = {
    # default retrostar
    "0_chembl100_default_retrostar_0": result_path + "0_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf",

    # default mcts
    "1_chembl100_default_mcts": result_path + "1_dpvn_ssm_chembl_100_subsample_all_routes_default_mcts_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf",

    # diversity forcing mcts
    "2_chembl100_diversity_forcing_mcts": result_path + "2_dpvn_ssm_chembl_100_subsample_all_routes_mcts_diversity_forcing_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf",    
    
    # expansion clustering mcts
    "3_chembl100_expansion_clustering_mcts": result_path + "3_dpvn_ssm_chembl_100_subsample_all_routes_expansion_clustering_mcts_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf",
    
    # expansion clustering retro*-0
    "4_chembl100_expansion_clustering_retrostar_0": result_path + "/5_dpvn_ssm_chembl_100_subsample_all_routes_retrostar_0_clustered_first_occurence_policy_uspto_template_based_stock_emolecules_depth_30_cluster_top_100_routes_max_5_cluster.hdf"    
    }

# load data from hdf

import pandas as pd

def read_hdf(path):
    data = pd.read_hdf(path, key="table")
    return data

def get_molecule_building_blocks(molecule_result: MoleculeResult, solved_only: bool = True) -> list:
    """
    Get all building blocks from a single molecule's synthesis routes.
    
    Args:
        molecule_result: MoleculeResult object containing synthesis routes
        solved_only: If True, only extract building blocks from solved routes (default: True)
        unique: If True, return only unique building blocks (default: True)
    
    Returns:
        list: List of building block SMILES strings
    """
    # Get building blocks from either solved routes or all routes
    if solved_only:
        building_blocks_df = molecule_result.get_solved_routes_building_blocks()
        # assert that the missing_building_blocks column is empty
        assert building_blocks_df['missing_building_blocks'].apply(len).sum() == 0
    else:
        building_blocks_df = molecule_result.get_all_building_blocks()
    
    # Flatten the lists of building blocks from all routes
    all_building_blocks = []
    for blocks_list in building_blocks_df['in_stock_building_blocks']:
        all_building_blocks.append(blocks_list)
    
    return all_building_blocks

for key, hdf_path in data_info.items():
    print(f"Processing {key}...")
    
    # Read the HDF file
    df = read_hdf(hdf_path)
    
    # Extract building blocks using AiZynthfinderResults
    azf_results = AiZynthfinderResults(df)

    results_df = azf_results.aizynthfinder_results
    del df
    # Add building blocks column
    results_df['solved_routes_building_blocks_nested'] = results_df['molecule_results'].apply(
        lambda molecule: get_molecule_building_blocks(molecule, solved_only=True)
    )
    
    # Add count of building blocks
    results_df['num_routes'] = results_df['solved_routes_building_blocks_nested'].apply(len)
    
    # drop the trees and molecule_results to save memory
    results_df = results_df.drop(columns=['trees','molecule_results'])
    
    print(f"  - Processed {len(results_df)} molecules")
    print()

    # save the results with building blocks to visualizations folder / building blocks
    results_df.to_parquet(f"{save_path}/building_blocks_nested/{key}_building_blocks.parquet", index=False)

    # --- This is what you're asking about ---
    del azf_results
    del results_df