import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Tuple, NewType, List, Optional, Dict, Iterator, TypeVar, Generic, Type, Union

PlayerID = int


class Player:
    def __init__(self, playerID: PlayerID):
        self.playerID = playerID


class Piece:
    def __init__(self, playerID: PlayerID):
        self.playerID = playerID


P = TypeVar('P', bound='Position')


class Position:
    def __init__(self, ring: int, index: int):
        self.ring = ring
        self.index = index

    def __hash__(self):
        return hash((self.ring, self.index))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.ring == other.ring and self.index == other.index
        return False

    def __repr__(self):
        return "Pos(%s)" % self.__str__()

    def __str__(self):
        return "%s:%s" % (self.ring, self.index)


class BoardDesign(ABC):

    @abstractmethod
    def distance_between(self, a: Position, b: Position) -> Tuple[int, int]:
        ...

    @abstractmethod
    def is_linked_to(self, a: Position, b: Position) -> bool:
        ...

    @abstractmethod
    def get_third_in_line(self, first: Position, second: Position) -> Position:
        ...

    @abstractmethod
    def neighbors_of(self, pos: Position) -> Iterator[Position]:
        ...

    @abstractmethod
    def iter_fields(self) -> Iterator[Position]:
        ...

    @abstractmethod
    def iter_lines(self) -> Iterator[Tuple[Position, Position, Position]]:
        ...

    @abstractmethod
    def iter_lines_of(self, pos: Position) -> Iterator[Tuple[Position, Position]]:
        ...

    @abstractmethod
    def pretty_print(self, board: 'GameBoard'):
        ...


class DefaultBoardDesign(BoardDesign):
    def __init__(self, sides=4, extended=False):
        assert sides >= 3
        self.sides = sides
        self.ring_size = 2 * sides
        self.extended = extended

    def distance_between(self, a: Position, b: Position) -> Tuple[int, int]:
        return abs(a.ring - b.ring), min(abs(a.index - b.index), self.ring_size - abs(a.index - b.index))

    def is_linked_to(self, a: Position, b: Position) -> bool:
        if a.ring == b.ring:
            return abs(a.index - b.index) % (self.ring_size - 2) == 1
        elif a.index == b.index:
            if self.extended or a.index % 2 == 1:
                return abs(a.ring - b.ring) == 1
        return False

    def neighbors_of(self, pos: Position) -> Iterator[Position]:
        yield Position(pos.ring, (pos.index - 1) % self.ring_size)
        yield Position(pos.ring, (pos.index + 1) % self.ring_size)
        if self.extended or pos.index % 2 == 1:
            if pos.ring > 0:
                yield Position(pos.ring - 1, pos.index)
            if pos.ring < 2:
                yield Position(pos.ring + 1, pos.index)

    def get_third_in_line(self, first: Position, second: Position) -> Position:
        if first.ring == second.ring and first.index != second.index:
            a = min(first.index, second.index)
            b = max(first.index, second.index)
            if b - a > 2:  # Special condition at ring end/start
                a, b = (b, a + self.ring_size)

            if b - a == 1:
                if a % 2 == 0:
                    return Position(first.ring, (a + 2) % self.ring_size)
                else:
                    return Position(first.ring, (a - 1) % self.ring_size)
            elif b - a == 2:
                if a % 2 == 0:
                    return Position(first.ring, (a + 1) % self.ring_size)
                else:
                    raise ValueError("Not in a valid line (two odd points cannot be on a line)")
            else:
                raise ValueError("Not in a valid line (distance more than 2)")

        if first.ring != second.ring and first.index == second.index:
            if self.extended or first.index % 2 == 1:
                assert first.ring != second.ring
                a = min(first.ring, second.ring)
                b = max(first.ring, second.ring)
                if a == 0:
                    if b == 1:
                        return Position(2, first.index)
                    else:
                        return Position(1, first.index)
                else:
                    return Position(0, first.index)

        raise ValueError("Not in a valid line")

    def iter_fields(self) -> Iterator[Position]:
        for ring in range(0, 3):
            for index in range(0, self.ring_size):
                yield Position(ring, index)

    def iter_lines(self) -> Iterator[Tuple[Position, Position, Position]]:
        for ring in range(0, 3):
            for side in range(0, self.sides):
                i = side * 2
                # Ring-parallel
                yield Position(ring, i), Position(ring, i + 1), Position(ring, (i + 2) % self.ring_size)
        for index in range(0, self.ring_size):
            if self.extended or index % 2 == 1:
                yield Position(0, index), Position(1, index), Position(2, index)

    def iter_lines_of(self, pos: Position) -> Iterator[Tuple[Position, Position]]:
        if self.extended or pos.index % 2 == 1:
            # Between rings
            yield Position((pos.ring + 1) % 3, pos.index), Position((pos.ring + 2) % 3, pos.index)
        if pos.index % 2 == 1:
            # From an odd position, parallel to a ring
            yield Position(pos.ring, pos.index - 1), Position(pos.ring, (pos.index + 1) % self.ring_size),
        else:
            # From an even position, parallel to a ring
            yield Position(pos.ring, (pos.index - 1) % self.ring_size), Position(pos.ring,
                                                                                 (pos.index - 2) % self.ring_size),
            yield Position(pos.ring, (pos.index + 1) % self.ring_size), Position(pos.ring,
                                                                                 (pos.index + 2) % self.ring_size),

    def pretty_print(self, board: 'GameBoard'):
        data = [[" "] * self.ring_size,[" "] * self.ring_size,[" "] * self.ring_size]
        for ring in range(0, 3):
            for index in range(0, self.ring_size):
                piece = board.get(Position(ring, index))
                if piece:
                    data[ring][index] = str(piece.playerID)
        ring_del = "\n  " + "   ".join([" ", "|"] * self.sides) + "\n"
        return ring_del.join("- " + " - ".join(ring) for ring in data)


class GameBoard:
    def __init__(self, design: BoardDesign):
        self.design = design
        self.fields: Dict[Position, Piece] = dict()
        self._count: List[int] = [0] * 2

    def clone(self) -> 'GameBoard':
        board = GameBoard(design=self.design)
        board.fields = dict(self.fields)
        board._count = list(self._count)
        return board

    def count(self, playerID: PlayerID) -> int:
        # return sum(1 for p in self.fields.values() if p.playerID == playerID)
        return self._count[playerID]

    def count_total(self) -> Tuple[int, int]:
        return (self._count[0], self._count[1])

    def is_empty(self, pos: Position):
        return pos not in self.fields

    def iter_empty(self) -> Iterator[Position]:
        yield from (pos for pos in self.design.iter_fields() if self.is_empty(pos))

    def iter_pieces(self, playerID: Optional[PlayerID] = None) -> Iterator[Tuple[Position, Piece]]:
        if playerID is None:
            yield from (e for pos, e in self.fields.items())
        else:
            yield from ((pos, e) for pos, e in self.fields.items() if e.playerID == playerID)

    def iter_pieces_inside_mill(self, playerID: PlayerID) -> Iterator[Tuple[Position, Piece]]:
        all_mills_pos = set(pos for mill in self.iter_mills(playerID) for pos in mill)
        for pos, piece in self.fields.items():
            if piece.playerID == playerID and pos in all_mills_pos:
                yield pos, piece

    def iter_pieces_outside_mill(self, playerID: PlayerID) -> Iterator[Tuple[Position, Piece]]:
        all_mills_pos = set(pos for mill in self.iter_mills(playerID) for pos in mill)
        for pos, piece in self.fields.items():
            if piece.playerID == playerID and pos not in all_mills_pos:
                yield pos, piece

    def is_inside_a_mill(self, playerID: PlayerID, pos: Position):
        piece = self.fields.get(pos, None)
        if piece and piece.playerID == playerID:
            for line in self.design.iter_lines_of(pos):
                pieces = (self.fields.get(p, None) for p in line)
                if all((p.playerID == playerID if p else False) for p in pieces):
                    return True
        return False

    def iter_ready_mills(self, playerID: PlayerID, pos: Position) -> Iterator[Tuple[Position, Position]]:
        if pos in self.fields:  # Not empty, so no mill is ready for pos
            return
        for second_pos, third_pos in self.design.iter_lines_of(pos):
            second = self.fields.get(second_pos, None)
            third = self.fields.get(third_pos, None)
            if second and third and second.playerID == playerID and third.playerID == playerID:
                yield second_pos, third_pos

    def iter_mills(self, playerID: PlayerID) -> Iterator[Tuple[Position, Position, Position]]:
        for line in self.design.iter_lines():
            pieces = (self.fields.get(p, None) for p in line)
            if all((p.playerID == playerID if p else False) for p in pieces):
                yield line

    def place(self, pos: Position, piece: Piece):
        assert pos not in self.fields
        self.fields[pos] = piece
        self._count[piece.playerID] += 1

    def get(self, pos: Position) -> Optional[Piece]:
        return self.fields.get(pos, None)

    def move(self, start: Position, end: Position):
        assert self.design.is_linked_to(start, end)
        self.jump(start, end)

    def jump(self, start: Position, end: Position):
        assert end not in self.fields
        piece = self.fields.pop(start)
        assert piece
        self.fields[end] = piece

    def remove(self, pos: Position) -> Optional[Piece]:
        piece = self.fields.pop(pos, None)
        if piece:
            self._count[piece.playerID] -= 1
        return piece

    def iter_available_moves(self, pos: Position) -> Iterator[Position]:
        for npos in self.design.neighbors_of(pos):
            if npos not in self.fields:
                yield npos


class ActionType(Enum):
    PLACE = 1
    MOVE = 2
    JUMP = 3


class Action(ABC):
    def __init__(self, actiontype: ActionType, playerID: PlayerID, attacks: Optional[List[Position]] = None):
        self.actiontype = actiontype
        self.playerID = playerID
        self.attacks = attacks

    @abstractmethod
    def __repr__(self):
        ...

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.actiontype == other.actiontype and self.playerID == other.playerID and self.attacks == other.attacks
        return False

    def __hash__(self):
        attacks_hash = 0
        if self.attacks:
            attacks_hash = hash((*self.attacks,))
        return hash((self.actiontype, self.playerID, attacks_hash))


class PlaceAction(Action):
    def __init__(self, playerID: PlayerID, pos: Position, **kwargs):
        super(PlaceAction, self).__init__(ActionType.PLACE, playerID, **kwargs)
        self.pos = pos

    def __repr__(self):
        if self.attacks:
            return "Place(%s, %s, attacks=%s)" % (self.playerID, self.pos, self.attacks)
        return "Place(%s, %s)" % (self.playerID, self.pos)

    def __eq__(self, other):
        if isinstance(other, PlaceAction):
            return super(PlaceAction, self).__eq__(other) and self.pos == other.pos
        return False

    def __hash__(self):
        return hash((super(PlaceAction, self).__hash__(), self.pos))


class MoveAction(Action):
    def __init__(self, playerID: PlayerID, start: Position, end: Position, **kwargs):
        super(MoveAction, self).__init__(ActionType.MOVE, playerID, **kwargs)
        self.start = start
        self.end = end

    def __repr__(self):
        if self.attacks:
            return "Move(%s, from=%s, to=%s, attacks=%s)" % (self.playerID, self.start, self.end, self.attacks)
        return "Move(%s, from=%s, to=%s)" % (self.playerID, self.start, self.end)

    def __eq__(self, other):
        if isinstance(other, MoveAction):
            return super(MoveAction, self).__eq__(other) and self.start == other.start and self.end == other.end
        return False

    def __hash__(self):
        return hash((super(MoveAction, self).__hash__(), self.start, self.end))


class JumpAction(Action):
    def __init__(self, playerID: PlayerID, start: Position, end: Position, **kwargs):
        super(JumpAction, self).__init__(ActionType.JUMP, playerID, **kwargs)
        self.start = start
        self.end = end

    def __repr__(self):
        if self.attacks:
            return "Jump(%s, from=%s, to=%s, attacks=%s)" % (self.playerID, self.start, self.end, self.attacks)
        return "Jump(%s, from=%s, to=%s)" % (self.playerID, self.start, self.end)

    def __eq__(self, other):
        if isinstance(other, JumpAction):
            return super(JumpAction, self).__eq__(other) and self.start == other.start and self.end == other.end
        return False

    def __hash__(self):
        return hash((super(JumpAction, self).__hash__(), self.start, self.end))


class GameState:
    def __init__(self, sides: int = 4):
        self.players = (Player(0), Player(1))
        self.placed_items = defaultdict(lambda: 0)
        self.player_turn = self.players[0]
        self.board = GameBoard(DefaultBoardDesign(sides=sides))
        self.max_placed_items = int(sides / 4 * 9)
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def clone(self) -> 'GameState':
        game = GameState()
        game.players = self.players
        game.placed_items = defaultdict(lambda: 0, self.placed_items)
        game.player_turn = self.player_turn
        game.board = self.board.clone()
        game.max_placed_items = self.max_placed_items
        return game

    def other(self, player: Union[Player, PlayerID]) -> Player:
        if isinstance(player, Player):
            return self.players[(player.playerID + 1) % 2]
        elif isinstance(player, PlayerID):
            return self.players[(player + 1) % 2]
        else:
            raise ValueError("Invalid player format!")

    def calc_available_attacks(self, player: Player) -> List[Tuple[Position, Piece]]:
        otherID = self.other(player).playerID
        outside = list(self.board.iter_pieces_outside_mill(otherID))
        if outside:
            return outside
        inside = list(self.board.iter_pieces_inside_mill(otherID))
        if inside:
            return inside
        return []

    def calc_available_actions(self, player: Player = None) -> Iterator[Action]:
        player = self.player_turn if player is None else player
        playerID = player.playerID
        player_board_count = self.board.count(playerID)

        available_attacks = self.calc_available_attacks(player)

        def enhance_with_attacks(pos: Position, action_cls: Type[Action], *args, **kwargs):
            mills = list(self.board.iter_ready_mills(player.playerID, pos))
            if mills:
                for comb in itertools.combinations(available_attacks, len(mills)):
                    kwargs['attacks'] = list(p for p, e in comb)
                    yield action_cls(*args, **kwargs)
            else:
                yield action_cls(*args, **kwargs)

        # Check if in placing game phase
        if self.placed_items[playerID] < self.max_placed_items:
            for empty_pos in self.board.iter_empty():
                yield from enhance_with_attacks(empty_pos, PlaceAction, playerID, empty_pos)

        # Normal move phase
        elif player_board_count > 3:
            for pos, piece in self.board.iter_pieces(playerID):
                for end_pos in self.board.iter_available_moves(pos):
                    yield from enhance_with_attacks(end_pos, MoveAction, playerID, pos, end_pos)

        # Check if in jumping game phase
        elif player_board_count == 3:
            for pos, piece in self.board.iter_pieces(playerID):
                for empty_pos in self.board.iter_empty():
                    yield from enhance_with_attacks(empty_pos, JumpAction, playerID, pos, empty_pos)

    def execute(self, action: Action) -> 'GameState':
        assert not self.frozen

        playerID = action.playerID
        assert playerID == self.player_turn.playerID

        if isinstance(action, PlaceAction):
            assert self.placed_items[playerID] < self.max_placed_items
            self.board.place(action.pos, Piece(playerID))
            self.placed_items[playerID] += 1

        elif isinstance(action, MoveAction):
            piece = self.board.get(action.start)
            assert piece
            assert piece.playerID == playerID
            self.board.move(action.start, action.end)

        elif isinstance(action, JumpAction):
            piece = self.board.get(action.start)
            assert piece
            assert piece.playerID == playerID
            self.board.jump(action.start, action.end)

        else:
            raise ValueError("Invalid action type")

        if action.attacks:
            for attack_pos in action.attacks:
                attacked = self.board.remove(attack_pos)
                assert attacked.playerID != playerID

        self.player_turn = self.other(self.player_turn)

    def pretty_print(self):
        return self.board.design.pretty_print(self.board)


def main():
    g = GameState(sides=4)
    while True:
        actions = list(g.calc_available_actions())
        if not actions:
            print("Game end! Winner: %s" % g.other(g.player_turn))
            break
        action = random.choice(actions)
        print(action)
        g = g.clone()
        g.execute(action)
        print(g.board.design.pretty_print(g.board))
        print("**********************\n\n")


if __name__ == "__main__":
    main()
