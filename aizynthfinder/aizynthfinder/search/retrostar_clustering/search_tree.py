""" Module containing a class that holds the tree search
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AffinityPropagation

from aizynthfinder.chem.serialization import MoleculeDeserializer, MoleculeSerializer
from aizynthfinder.search.andor_trees import AndOrSearchTreeBase, SplitAndOrTree
from aizynthfinder.search.retrostar.cost import MoleculeCost
from aizynthfinder.search.retrostar.nodes import MoleculeNode
from aizynthfinder.utils.exceptions import RejectionException
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.reactiontree import ReactionTree
    from aizynthfinder.utils.type_utils import List, Optional, Sequence


class SearchTree(AndOrSearchTreeBase):
    """
    Encapsulation of the Retro* search tree (an AND/OR tree).

    :ivar config: settings of the tree search algorithm
    :ivar root: the root node

    :param config: settings of the tree search algorithm
    :param root_smiles: the root will be set to a node representing this molecule, defaults to None
    """

    def __init__(
        self, config: Configuration, root_smiles: Optional[str] = None
    ) -> None:
        super().__init__(config, root_smiles)
        self._mol_nodes: List[MoleculeNode] = []
        self._logger = logger()
        self.molecule_cost = MoleculeCost(config)

        if root_smiles:
            self.root: Optional[MoleculeNode] = MoleculeNode.create_root(
                root_smiles, config, self.molecule_cost
            )
            self._mol_nodes.append(self.root)
        else:
            self.root = None

        self._routes: List[ReactionTree] = []

        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
            "clustered_tree_width": [],
            "tree_width": [],
        }

        print("RETROSTAR EXPANSION CLUSTERING")

    @classmethod
    def from_json(cls, filename: str, config: Configuration) -> SearchTree:
        """
        Create a new search tree by deserialization from a JSON file

        :param filename: the path to the JSON node
        :param config: the configuration of the search tree
        :return: a deserialized tree
        """

        def _find_mol_nodes(node):
            for child_ in node.children:
                tree._mol_nodes.append(child_)  # pylint: disable=protected-access
                for grandchild in child_.children:
                    _find_mol_nodes(grandchild)

        tree = cls(config)
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        mol_deser = MoleculeDeserializer(dict_["molecules"])
        tree.root = MoleculeNode.from_dict(
            dict_["tree"], config, mol_deser, tree.molecule_cost
        )
        tree._mol_nodes.append(tree.root)  # pylint: disable=protected-access
        for child in tree.root.children:
            _find_mol_nodes(child)
        return tree

    @property
    def mol_nodes(self) -> Sequence[MoleculeNode]:  # type: ignore
        """Return the molecule nodes of the tree"""
        return self._mol_nodes

    def one_iteration(self) -> bool:
        """
        Perform one iteration of
            1. Selection
            2. Expansion
            3. Update

        :raises StopIteration: if the search should be pre-maturely terminated
        :return: if a solution was found
        :rtype: bool
        """
        if self.root is None:
            raise ValueError("Root is undefined. Cannot make an iteration")

        self._routes = []

        next_node = self._select()

        if not next_node:
            self._logger.debug("No expandable nodes in Retro* iteration")
            raise StopIteration

        self._expand(next_node)

        if not next_node.children:
            next_node.expandable = False

        self._update(next_node)

        return self.root.solved

    def routes(self) -> List[ReactionTree]:
        """
        Extracts and returns routes from the AND/OR tree

        :return: the routes
        """
        if self.root is None:
            return []
        if not self._routes:
            self._routes = SplitAndOrTree(self.root, self.config.stock).routes
        return self._routes

    def serialize(self, filename: str) -> None:
        """
        Seralize the search tree to a JSON file

        :param filename: the path to the JSON file
        :type filename: str
        """
        if self.root is None:
            raise ValueError("Cannot serialize tree as root is not defined")

        mol_ser = MoleculeSerializer()
        dict_ = {"tree": self.root.serialize(mol_ser), "molecules": mol_ser.store}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=2)

    def _expand(self, node: MoleculeNode) -> None:
        reactions, priors = self.config.expansion_policy([node.mol])
        self.profiling["expansion_calls"] += 1
        
        if not reactions:
            return

        costs = -np.log(np.clip(priors, 1e-3, 1.0))
        reactions_to_expand = []
        reaction_costs = []
        for reaction, cost in zip(reactions, costs):
            # try to get the reactants
            try:
                self.profiling["reactants_generations"] += 1
                _ = reaction.reactants
            except:  # pylint: disable=bare-except
                continue
            # if there are no reactants, skip the reaction
            if not reaction.reactants:
                continue

            for idx, _ in enumerate(reaction.reactants):
                rxn_copy = reaction.copy(idx)
                #print(f"Reaction: {rxn_copy.reaction_smiles()}")
                if self._filter_reaction(rxn_copy):
                    continue
                reactions_to_expand.append(rxn_copy)
                reaction_costs.append(cost)

        # clustering of the alternatives
        reaction_fingerprints = []
        negative_fingeprint = [-1] * 256
        # loop through the reactions and get the ones that we should expand
        for reaction in reactions_to_expand:
            try:
                reaction_fingerprint = reaction.fingerprint(2,256)
            except Exception as e:
                print(f"An error occurred: {e}")
                reaction_fingerprint = negative_fingeprint
            reaction_fingerprints.append(reaction_fingerprint)

        reaction_fingerprints = np.array(reaction_fingerprints)

        disconnection_importance, cluster_centers_indices, labels, n_clusters_ = self._calculate_reaction_cluster(reaction_fingerprints)

        # they must be of equal length
        assert len(disconnection_importance) == len(reactions_to_expand)
        assert len(disconnection_importance) == len(reaction_costs)

        # filter out the reactions that are not important
        reactions_to_expand_clustered = [rxn for rxn, imp in zip(reactions_to_expand, disconnection_importance) if imp == 1]
        reaction_costs_clustered = [cost for cost, imp in zip(reaction_costs, disconnection_importance) if imp == 1]

        if len(reactions_to_expand_clustered) != n_clusters_:
            print(f"Corner case found, we were not able to cluster! Using the original reactions."
            f"Expected number of clusters: {n_clusters_}, but got {len(reactions_to_expand_clustered)} clustered reactions. "
            f"Original reactions to expand: {len(reactions_to_expand)}"
            )
            reactions_to_expand_clustered = reactions_to_expand
            reaction_costs_clustered = reaction_costs
            # change the number of clusters to be the same as the number of reactions
            n_clusters_ = len(reactions_to_expand_clustered)

        # assert that the lengths are equal to the number of clusters
        assert len(reactions_to_expand_clustered) == n_clusters_
        assert len(reaction_costs_clustered) == n_clusters_

        self.profiling["clustered_tree_width"].append(n_clusters_)
        self.profiling["tree_width"].append(len(disconnection_importance))

        verbose = False
        if verbose:
            print(f"Number of clusters: {n_clusters_}")
            print(f"Number of reactions: {len(reactions_to_expand)}")
            print(f"Number of clustered reactions: {len(reactions_to_expand_clustered)}")
            print(f"Disconnection importance: {disconnection_importance}")
            print(f"Reactions: {[reaction.reaction_smiles() for reaction in reactions_to_expand]}")
            print(f"Clustered reactions: {[reaction.reaction_smiles() for reaction in reactions_to_expand_clustered]}")


        # append the new nodes to the tree
        for cost, rxn in zip(reaction_costs_clustered, reactions_to_expand_clustered):
            new_nodes = node.add_stub(cost, rxn)
            self._mol_nodes.extend(new_nodes)

    def _filter_reaction(self, reaction: RetroReaction) -> bool:
        if not self.config.filter_policy.selection:
            return False
        try:
            self.config.filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _select(self) -> Optional[MoleculeNode]:
        scores = np.asarray(
            [
                node.target_value if node.expandable else np.inf
                for node in self._mol_nodes
            ]
        )

        # no expandable nodes left
        if scores.min() == np.inf:
            return None

        return self._mol_nodes[int(np.argmin(scores))]

    @staticmethod
    def _update(node: MoleculeNode) -> None:
        v_delta = node.close()
        if node.parent and np.isfinite(v_delta):
            node.parent.update(v_delta, from_mol=node.mol)


    def _calculate_reaction_cluster(self, reaction_fingerprints, verbose = True) -> np.ndarray:
        """
        Calculate the importance of reactions using Affinity Propagation clustering.
        Parameters:
        reaction_fingerprints (array-like): A matrix of reaction fingerprints where each row represents a reaction.
        Returns:
        tuple: A tuple containing:
            - important_reactions (numpy.ndarray): An array indicating the important reactions (1 for important, 0 otherwise).
            - cluster_centers_indices (numpy.ndarray): Indices of cluster centers.
            - labels (numpy.ndarray): Cluster labels for each reaction.
            - n_clusters_ (int): The number of clusters found by the algorithm.
        """
        

        af = AffinityPropagation(damping=0.9, max_iter=500, convergence_iter=30, preference=-50, verbose=verbose).fit(reaction_fingerprints)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        n_clusters_ = len(cluster_centers_indices)
        
        important_reactions = np.zeros(len(reaction_fingerprints))
        
        # go through the labels and set the first occurrence of each label to 1
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            important_reactions[label_indices[0]] = 1
        
        return important_reactions, cluster_centers_indices, labels, n_clusters_
