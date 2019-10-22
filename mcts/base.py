import random
from abc import abstractmethod, ABC
from enum import Enum
from typing import Dict, Set, Iterator, TypeVar, Generic, NamedTuple, Tuple, List, Optional

from mühle import Action, GameState, PlayerID

T = TypeVar('T')


class SelectionPolicy(ABC):
    @abstractmethod
    def select_from(self, node: 'MCTS_Node') -> Optional['MCTS_Node']:
        ...


class ExpansionPolicy(ABC):
    @abstractmethod
    def expand(self, node: 'MCTS_Node') -> Iterator['MCTS_Node']:
        ...


class SimulationPolicy(ABC):
    @abstractmethod
    def simulate(self, node: 'MCTS_Node'):
        ...


class BackpropagationPolicy(ABC):
    @abstractmethod
    def backpropagate(self, node: 'MCTS_Node'):
        ...


class MCTS_Node:
    def __init__(self, state: GameState, parent: Optional['MCTS_Node'] = None) -> None:
        self.state = state
        state.freeze()
        self.parent = parent
        self.remaining_actions: List[Action] = list(state.calc_available_actions())
        self.children: Dict[Action, MCTS_Node] = dict()

        self.visited: int = 0
        self.simulations: int = 0
        self.reward_total: float = 0.0


class MCTS:
    def __init__(
            self,
            root: GameState,
            selection: SelectionPolicy,
            expansion: ExpansionPolicy,
            simulation: SimulationPolicy,
            propagation: BackpropagationPolicy
    ):
        self.node: MCTS_Node = MCTS_Node(root)
        self.selection = selection
        self.expansion = expansion
        self.simulation = simulation
        self.propagation = propagation

    def run_one_iteration(self):
        node = self.selection.select_from(self.node)
        if node:
            for child in self.expansion.expand(node):
                self.simulation.simulate(child)
                self.propagation.backpropagate(child)
            return True
        return False

    def calc_action_rewards(self) -> List[Tuple[float, Action, MCTS_Node]]:
        assert self.node.children
        return list((c.reward_total / c.simulations, a, c) for a, c in self.node.children.items())

    def calc_best_action(self) -> Tuple[float, Action, MCTS_Node]:
        children = self.calc_action_rewards()
        children.sort(key=lambda e: e[0], reverse=True)
        return children[0]

    def select_best(self) -> Tuple[Action, MCTS_Node]:
        reward, action, child = self.calc_best_action()

        # Set as current new node
        self.node = child
        self.node.parent = None  # Stop backpropagation to old roots

        return action, child

    def select_other(self, action: Action, game: GameState):
        # Try to use already pre-computed nodes...
        if action in self.node.children:
            self.node = self.node.children[action]
            self.node.parent = None
        else:
            self.node = MCTS_Node(game)
