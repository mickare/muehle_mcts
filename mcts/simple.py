import math
import random
from typing import Iterator, List, Tuple, Optional, NamedTuple, TypeVar

from mcts.base import SimulationPolicy, MCTS_Node, SelectionPolicy, BackpropagationPolicy, ExpansionPolicy, MCTS
from mühle import Action, GameState, PlayerID

T = TypeVar('T')


def pop_random_from_list(data: List[T]) -> T:
    index = random.randint(0, len(data) - 1)
    result = data[index]
    del data[index]
    return result


class SimulationResult(NamedTuple):
    steps: int
    winner: Optional[PlayerID]
    pieces_left: Tuple[int, int]


class SimpleSimulationPolicy(SimulationPolicy):
    def __init__(self, player: PlayerID, max_depth=50, branching=5, branching_depth=1):
        self.player = player
        self.max_depth = max_depth
        self.branching = branching
        self.branching_depth = branching_depth

    def simulate(self, node: MCTS_Node):
        state = node.state.clone()
        results = list(self._iter_simulate(state, list(state.calc_available_actions())))

        other_player = state.other(self.player).playerID

        reward = 0
        pieces_player = 0
        pieces_other = 0

        for result in results:
            pieces_player = result.pieces_left[self.player]
            pieces_other = result.pieces_left[other_player]
            ratio = (pieces_player - pieces_other) / (pieces_player + pieces_other)
            reward += 2 * ratio
            """
            if result.winner is None:  # Max depth reached
                reward += ratio
            elif result.winner == self.player:  # We won the game
                reward += 1.0
            else:  # We lost the game
                reward -= 1.0
            """
        node.reward_total += reward / len(results)
        node.simulations += 1

    def take_action(self, actions: List[Action]) -> Action:
        assert actions
        actions_with_attack = list((n, a) for n, a in enumerate(actions) if a.attacks)
        if actions_with_attack:
            index, action = pop_random_from_list(actions_with_attack)
            del actions[index]
            return action
        else:
            return pop_random_from_list(actions)

    def _iter_simulate(self, state: GameState, actions: List[Action], steps: int = 0):
        if steps >= self.max_depth:
            yield SimulationResult(steps, winner=None, pieces_left=state.board.count_total())
        elif not actions:
            yield SimulationResult(steps, winner=state.other(state.player_turn).playerID,
                                   pieces_left=state.board.count_total())
        else:
            if self.branching > 1 and steps < self.branching_depth:
                for n in range(0, min(self.branching, len(actions))):
                    action = self.take_action(actions)
                    newstate = state.clone()
                    newstate.execute(action)
                    yield from self._iter_simulate(newstate, list(newstate.calc_available_actions()), steps + 1)
            else:
                action = self.take_action(actions)
                state.execute(action)
                yield from self._iter_simulate(state, list(state.calc_available_actions()), steps + 1)


# UCB1
def UCB1(node: MCTS_Node, ln_simulations_total):
    #print(node.reward_total / node.simulations, math.sqrt(2 * ln_simulations_total / node.simulations))
    return node.reward_total / node.simulations + math.sqrt(2 * ln_simulations_total / node.simulations)


class SimpleSelectionPolicy(SelectionPolicy):
    def select_from(self, node: MCTS_Node) -> Optional[MCTS_Node]:
        if node.remaining_actions:  # Not fully expanded
            return node
        else:
            if node.children:
                ln_simulations_total = math.log(node.simulations)
                children = list((UCB1(child, ln_simulations_total), child) for action, child in node.children.items())
                children.sort(key=lambda e: e[0], reverse=True)
                ucb1_result, best = children[0]
                return self.select_from(best)
            else:
                return None  # Terminal node


class SimpleExpansionPolicy(ExpansionPolicy):
    def __init__(self, branching=3):
        assert branching > 0
        self.branching = branching

    def expand(self, node: MCTS_Node) -> Iterator[MCTS_Node]:
        assert node.remaining_actions
        for _ in range(0, min(self.branching, len(node.remaining_actions))):
            action = pop_random_from_list(node.remaining_actions)
            newstate = node.state.clone()
            newstate.execute(action)
            child = MCTS_Node(newstate, parent=node)
            node.children[action] = child
            yield child


class SimpleBackpropagationPolicy(BackpropagationPolicy):
    def backpropagate(self, node: MCTS_Node):
        if node.parent:
            parent = node.parent
            parent.simulations += node.simulations
            parent.reward_total += node.reward_total
            parent.visited += 1
            self.backpropagate(parent)


def init_mcts(game: GameState, playerID: PlayerID) -> MCTS:
    return MCTS(
        game,
        SimpleSelectionPolicy(),
        SimpleExpansionPolicy(),
        SimpleSimulationPolicy(playerID),
        SimpleBackpropagationPolicy()
    )
