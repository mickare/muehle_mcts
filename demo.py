import time
from typing import Tuple, List

from mcts.base import MCTS
from mcts.simple import init_mcts
from mühle import GameState, PlayerID


def other(playerID: PlayerID):
    return (playerID + 1) % 2


def min_max_mcts(mcts: MCTS):
    rewards = mcts.calc_action_rewards()
    return min(rewards, key=lambda e: e[0]), max(rewards, key=lambda e: e[0])


def run_mcts(mcts: MCTS, timeout=1.0) -> int:
    start = time.time()
    end_timeout = start + timeout

    i = 0
    while True:
        if i > 10 and time.time() > end_timeout:
            break

        i += 1
        if not mcts.run_one_iteration():
            break

        #if len(mcts.node.children) > 100:
        #    rew_min, rew_max = min_max_mcts(mcts)
        #    if rew_max > rew_min or rew_min > 0:
        #        break

    return i


if __name__ == "__main__":

    game = GameState(sides=4)

    players: List[MCTS] = [
        init_mcts(game, 0),
        init_mcts(game, 1)
    ]

    n = 0
    while True:
        player = n % 2

        print("Player %s turn..." % player)

        mcts_current: MCTS = players[player]
        print("MCTS iterations: %s" % run_mcts(mcts_current, timeout=4))
        action, node = mcts_current.select_best()
        print(action)

        mcts_other: MCTS = players[other(player)]
        mcts_other.select_other(action, node.state)
        if not mcts_other.node.children and not mcts_other.node.remaining_actions:
            print("Player %s wins!" % player)
            break

        print(node.state.pretty_print())
        print("**********************\n\n")

        n += 1
