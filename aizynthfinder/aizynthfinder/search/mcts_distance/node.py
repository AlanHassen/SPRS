""" Module containing a class that represents a node in the search tree.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import jaccard

from aizynthfinder.chem import TreeMolecule, deserialize_action, serialize_action
from aizynthfinder.search.mcts.state import MctsState
from aizynthfinder.search.mcts.utils import ReactionTreeFromSuperNode, route_to_node
from aizynthfinder.utils.exceptions import (
    NodeUnexpectedBehaviourException,
    RejectionException,
)
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import (
        MoleculeDeserializer,
        MoleculeSerializer,
        RetroReaction,
    )
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.reactiontree import ReactionTree
    from aizynthfinder.search.mcts.search import MctsSearchTree
    from aizynthfinder.utils.type_utils import List, Optional, StrDict, Tuple


class MctsNode:
    """
    A node in the search tree.

    The children are instantiated lazily for efficiency: only when
    a child is selected the reaction to create that child is applied.

    Properties of an instantiated children to a node can be access with:

    .. code-block::

        children_attr = node[child]

    the return value is a dictionary with keys "action", "value", "prior"
    and "visitations".

    :ivar is_expanded: if the node has had children added to it
    :ivar is_expandable: if the node is expandable
    :ivar tree: the tree owning this node

    :param state: the state of the node
    :param owner: the tree that owns this node
    :param config: settings of the tree search algorithm
    :param parent: the parent node, defaults to None
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[MctsNode] = None,
    ):
        self._state = state
        self._config = config
        self._expansion_policy = config.expansion_policy
        self._filter_policy = config.filter_policy
        self.tree = owner
        self.is_expanded: bool = False
        self.is_expandable: bool = not self.state.is_terminal
        self._parent = parent

        if owner is None:
            self.created_at_iteration: Optional[int] = None
        else:
            self.created_at_iteration = self.tree.profiling["iterations"]

        self._children_values: List[float] = []
        self._children_priors: List[float] = []
        self._children_visitations: List[int] = []
        self._children_actions: List[RetroReaction] = []
        self._children: List[Optional[MctsNode]] = []

        self._internal_children_importance = []

        self.blacklist = set(mol.inchi_key for mol in state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

        if self._algo_config["mcts_grouping"]:
            self._degeneracy_check = self._algo_config["mcts_grouping"].lower()
        else:
            self._degeneracy_check = "none"
        self._logger = logger()

    def __getitem__(self, node: "MctsNode") -> StrDict:
        idx = self._children.index(node)
        return {
            "action": self._children_actions[idx],
            "value": self._children_values[idx],
            "prior": self._children_priors[idx],
            "visitations": self._children_visitations[idx],
        }

    @classmethod
    def create_root(
        cls, smiles: str, tree: MctsSearchTree, config: Configuration
    ) -> "MctsNode":
        """
        Create a root node for a tree using a SMILES.

        :param smiles: the SMILES representation of the root state
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :return: the created node
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = MctsState(mols=[mol], config=config)
        return MctsNode(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        tree: MctsSearchTree,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: Optional["MctsNode"] = None,
    ) -> "MctsNode":
        """
        Create a new node from a dictionary, i.e. deserialization

        :param dict_: the serialized node
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :param molecules: the deserialized molecules
        :param parent: the parent node
        :return: a deserialized node
        """
        # pylint: disable=protected-access
        state = MctsState.from_dict(dict_["state"], config, molecules)
        node = MctsNode(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children_actions = [
            deserialize_action(action_dict, molecules)
            for action_dict in dict_["children_actions"]
        ]
        node._children = [
            MctsNode.from_dict(child, tree, config, molecules, parent=node)
            if child
            else None
            for child in dict_["children"]
        ]
        return node

    @property
    def children(self) -> List["MctsNode"]:
        """
        Returns all of the instantiated children

        :return: the children
        """
        return [child for child in self._children if child]

    @property
    def is_solved(self) -> bool:
        """Return if the state is solved"""
        return self.state.is_solved

    @property
    def parent(self) -> Optional["MctsNode"]:
        """Return the parent of the node"""
        return self._parent

    @property
    def state(self) -> MctsState:
        """Return the underlying state of the node"""
        return self._state

    @property
    def _algo_config(self) -> StrDict:
        """Just a convinient, shorter name of this"""
        return self._config.search.algorithm_config

    def actions_to(self) -> List[RetroReaction]:
        """
        Returns the actions leading to this node

        :return: the list of actions
        """
        return self.path_to()[0]

    def backpropagate(self, child: "MctsNode", value_estimate: float) -> None:
        """
        Update the number of visitations of a particular child and its value.

        :param child: the child node
        :param value_estimate: the value to add to the child value
        """
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        self._children_values[idx] += value_estimate

    def children_view(self) -> StrDict:
        """
        Creates a view of the children attributes. Each of the
        list returned is a new list, although the actual children
        are not copied.

        The return dictionary will have keys "actions", "values",
        "priors", "visitations" and "objects".

        :return: the view
        """
        return {
            "actions": list(self._children_actions),
            "values": list(self._children_values),
            "priors": list(self._children_priors),
            "visitations": list(self._children_visitations),
            "objects": list(self._children),
        }

    def expand(self) -> None:
        """
        Expand the node.

        Expansion is the process of creating the children of the node,
        without instantiating a child object. The actions and priors are
        taken from the policy network.

        If immediate instantiation is marked for some policies, however, the
        children nodes will be instantiated.
        """
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            self._logger.debug(msg)
            raise NodeUnexpectedBehaviourException(msg)

        if self.is_expanded or not self.is_expandable:
            return

        self.is_expanded = True

        cache_molecules = []
        if self.parent:
            for child in self.parent.children:
                if child is not self:
                    cache_molecules.extend(child.state.expandable_mols)

        # Calculate the possible actions, fill the child_info lists
        # Actions by default only assumes 1 set of reactants
        (
            self._children_actions,
            self._children_priors,
        ) = self._expansion_policy(self.state.expandable_mols, cache_molecules)
        nactions = len(self._children_actions)
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        if self._algo_config["use_prior"]:
            self._children_values = list(self._children_priors)
        else:
            self._children_values = [self._algo_config["default_prior"]] * nactions

        if nactions == 0:  # Reverse the expansion if it did not produce any children
            self.is_expandable = False
            self.is_expanded = False

        if self.tree:
            self.tree.profiling["expansion_calls"] += 1

        if not self._algo_config["immediate_instantiation"]:
            return
        
        # Instantiate all children actions created by the marked policy,
        # a new list of actions will be iterated over, because it can grow due
        # to instantiation
        for child_idx, action in enumerate(self._children_actions[:nactions]):
            policy_name = action.metadata.get("policy_name")
            if (
                policy_name
                and policy_name in self._algo_config["immediate_instantiation"]
            ):
                self._instantiate_child(child_idx)

    def is_terminal(self) -> bool:
        """
        Node is terminal if its unexpandable, or the internal state is terminal (solved)

        :return: the terminal attribute of the node
        """
        return not self.is_expandable or self.state.is_terminal

    def path_to(self) -> Tuple[List[RetroReaction], List[MctsNode]]:
        """
        Return the path to this node, which is a list of actions and a list of node

        :return: the actions and nodes
        """
        return route_to_node(self)

    def promising_child(self) -> Optional["MctsNode"]:
        """
        Return the child with the currently highest Q+U

        The selected child will be instantiated if it has not been already.

        If no actions could be found that were applicable, the method will
        return None.

        :return: the child
        """

        def _score_and_select():
            q = self._children_q()
            u = self._children_u()
            reaction_importance = self._children_reaction_importance()
            if len(u) != len(reaction_importance):
                print("Careful, it seems that you didnt instantiate the policy immediately")
                print(f"U: {u}")
                print(f"Reaction importance: {reaction_importance}")
            scores = q + u * reaction_importance
            #print(f"Q: {q}, U: {u}, Importance: {reaction_importance}")
            #print(f"Scores: {scores}")
            indices = np.where(scores == scores.max())[0]
            index = np.random.choice(indices)

            return self._select_child(index)

        child = None
        while child is None and max(self._children_values) > 0:
            child = _score_and_select()

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            self.is_expanded = False
            self.is_expandable = False

        return child

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the node object to a dictionary

        :param molecule_store: the serialized molecules
        :return: the serialized node
        """
        return {
            "state": self.state.serialize(molecule_store),
            "children_values": [float(value) for value in self._children_values],
            "children_priors": [float(value) for value in self._children_priors],
            "children_visitations": self._children_visitations,
            "children_actions": [
                serialize_action(action, molecule_store)
                for action in self._children_actions
            ],
            "children": [
                child.serialize(molecule_store) if child else None
                for child in self._children
            ],
            "is_expanded": self.is_expanded,
            "is_expandable": self.is_expandable,
        }

    def to_reaction_tree(self) -> ReactionTree:
        """
        Return reaction tree from the path of actions and nodes leading to this node

        :return: the constructed tree
        """
        return ReactionTreeFromSuperNode(self).tree

    def _check_child_reaction(self, reaction: RetroReaction) -> bool:
        if not reaction.reactants:
            self._logger.debug(f"{reaction} did not produce any reactants")
            return False

        # fmt: off
        reactants0 = reaction.reactants[0]
        if len(reaction.reactants) == 1 and len(reactants0) == 1 and reaction.mol == reactants0[0]:
            return False
        # fmt: on

        return True

    def _children_q(self) -> np.ndarray:
        return np.array(self._children_values) / np.array(self._children_visitations)

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._algo_config["C"] * np.sqrt(2 * total_visits / child_visits)

    def _children_reaction_importance(self) -> np.ndarray:
        
        # calculate the importance only once if the list is empty
        if len(self._internal_children_importance) == 0:
        
            reaction_fingerprints = []
            # loop through the reactions and get the ones that we should expand
            for reaction, value in zip(self._children_actions, self._children_values):
                if value == -1e6:
                    # bad disconnection anyways...
                    reaction_fingerprint = False
                else: 
                    try:
                        reaction_fingerprint = reaction.fingerprint(2,256)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        reaction_fingerprint = False
                reaction_fingerprints.append(reaction_fingerprint)

            minimum_distance = self._calculate_reaction_distance_against_already_known(reaction_fingerprints)

            if self.tree:
                tree_width = len(minimum_distance)
                self.tree.profiling["tree_width"].append(tree_width)

                # count the number of elements above 0.0
                clustered_tree_width = len([x for x in minimum_distance if x > 0.0])
                self.tree.profiling["clustered_tree_width"].append(clustered_tree_width)



            self._internal_children_importance = np.array(minimum_distance)
        return self._internal_children_importance

    def _calculate_reaction_distance_against_already_known(self, fingerprints):
        """
        Calculate the Tanimoto distance against the already known fingerprints 
        in a left-first, sweeping manner.
        """
        min_distances = []
        for i, fp in enumerate(fingerprints):
            # Skip the distance calculation if the fingerprint is False, ie the molecule is not valid
            if fp is False:
                min_distances.append(0.0)  # Assign a default value, e.g., 0.0 --> Careful, a 0.0 distance is bad!
                print(f"Reaction {i} is not valid")
                continue

            distances = []
            # Calculate distance only up to the current fingerprint, avoiding self-comparison
            for j, fp2 in enumerate(fingerprints[:i]):
                if fp2 is False:
                    continue
                tanimoto_distance = jaccard(fp, fp2)  # Assuming jaccard function returns the Tanimoto distance
                distances.append(tanimoto_distance)

            # Find the minimum distance (closest match), default to 1.0 if no comparisons as the molecule is new
            min_distance = min(distances) if distances else 1.0
            min_distances.append(min_distance)
        
        return min_distances

    def _create_children_nodes(
        self, states: List[MctsState], child_idx: int
    ) -> List["MctsNode"]:
        new_nodes = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            if self._generated_degeneracy(state, first_child_idx):
                # Only need to disable first new child,
                # if the action generated more states, we will just not generate
                # a child for that state
                if state_index == 0:
                    self._children_values[child_idx] = -1e6
                continue

            # If there's more than one outcome, the lists need be expanded
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            if self._filter_child_reaction(self._children_actions[child_idx]):
                self._children_values[child_idx] = -1e6
            else:
                new_node = MctsNode(
                    state=state, owner=self.tree, config=self._config, parent=self
                )
                self._children[child_idx] = new_node
                new_nodes.append(new_node)
        return new_nodes

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        new_action = self._children_actions[old_index].copy(index=action_index)
        self._children_actions.append(new_action)
        self._children_priors.append(self._children_priors[old_index])
        self._children_values.append(self._children_values[old_index])
        self._children_visitations.append(self._children_visitations[old_index])
        self._children.append(None)
        return len(self._children) - 1

    def _filter_child_reaction(self, reaction: RetroReaction) -> bool:
        if self._regenerated_blacklisted(reaction):
            self._logger.debug(
                f"Reaction {reaction.reaction_smiles()} "
                f"was rejected because it re-generated molecule not in stock"
            )
            return True

        if not self._filter_policy.selection:
            return False
        try:
            self._filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _generated_degeneracy(self, new_state: MctsState, child_idx: int) -> bool:
        """
        Check if a new MCTS state is equal to another MCTS state of a children
        node.

        The check can be "partial" in which the equality is based only on the expandable molecules,
        or "full" in which the equality is based on all molecules in the state.

        The comparison will not be made on unexpanded children nodes
        or terminal children nodes.

        The metadata of the degenerate action will be added to the metadata
        of the previously created equal state.
        """

        def equal_states(query_state):
            if self._degeneracy_check == "partial":
                return query_state.expandables_hash == new_state.expandables_hash
            return query_state == new_state

        if self._degeneracy_check not in ["partial", "full"]:
            return False
        previous_action = None
        for child, action in zip(self._children, self._children_actions):
            if (
                child is not None
                and not child.is_terminal()
                and equal_states(child.state)
            ):
                previous_action = action
                break

        if previous_action is None:
            return False

        # No need to copy the metadata because it will be the same
        if previous_action is self._children_actions[child_idx]:
            return True

        metadata_copy = dict(self._children_actions[child_idx].metadata)
        if "additional_actions" not in previous_action.metadata:
            previous_action.metadata["additional_actions"] = []
        previous_action.metadata["additional_actions"].append(metadata_copy)
        return True

    def _instantiate_child(self, child_idx: int) -> List["MctsNode"]:
        """
        Instantiate the children node

        The algorithm is:
        * Apply the reaction associated with the child
        * If the application of the action failed, set value to -1e6 and return None
        * Create a new state array, one new state for each of the reaction outcomes
        * Create new child nodes
            - If a filter policy is available and the reaction outcome is unlikely
              set value of child to -1e6
         * Return all new nodes
        """
        if self._children[child_idx] is not None:
            raise NodeUnexpectedBehaviourException("Node already instantiated")

        reaction = self._children_actions[child_idx]
        if reaction.unqueried:
            if self.tree:
                self.tree.profiling["reactants_generations"] += 1
            _ = reaction.reactants

        if not self._check_child_reaction(reaction):
            self._children_values[child_idx] = -1e6
            return []

        keep_mols = [mol for mol in self.state.mols if mol is not reaction.mol]
        new_states = [
            MctsState(keep_mols + list(reactants), self._config)
            for reactants in reaction.reactants
        ]
        return self._create_children_nodes(new_states, child_idx)

    def _regenerated_blacklisted(self, reaction: RetroReaction) -> bool:
        if not self._algo_config["prune_cycles_in_search"]:
            return False
        for reactants in reaction.reactants:
            for mol in reactants:
                if mol.inchi_key in self.blacklist:
                    return True
        return False

    def _select_child(self, child_idx: int) -> Optional["MctsNode"]:
        """
        Selecting a child node implies instantiating the children nodes

        If the child has already been instantiated, return immediately
        Otherwise, select a random node of the feasible ones to return
        """
        if self._children[child_idx]:
            return self._children[child_idx]

        new_nodes = self._instantiate_child(child_idx)
        if new_nodes:
            return random.choice(new_nodes)
        return None