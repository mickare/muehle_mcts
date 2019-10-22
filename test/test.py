import unittest
from typing import Callable

from mühle import DefaultBoardDesign, Position


class TestDefaultBoardDesign(unittest.TestCase):

    def test_neighbor_positions(self):
        for sides in range(3, 5):
            b = DefaultBoardDesign(sides=sides)
            self.assertEqual(
                set(b.neighbors_of(Position(1, 1))),
                {Position(0, 1), Position(2, 1), Position(1, 0), Position(1, 2)}
            )
            self.assertEqual(
                set(b.neighbors_of(Position(1, 0))),
                {Position(1, b.ring_size - 1), Position(1, 1)}
            )

    def test_get_third_in_line(self):
        for extended in (False, True):
            for sides in range(3, 5):
                b = DefaultBoardDesign(sides=sides, extended=extended)
                # Check line getter on rings
                for ring in range(0, 2):
                    for n in range(0, b.sides):
                        i = 2 * n
                        self.assertEqual(
                            b.get_third_in_line(Position(ring, i+0), Position(ring, i+1)),
                            Position(ring, (i+2)%b.ring_size)
                        )
                        self.assertEqual(
                            b.get_third_in_line(Position(ring, i+0), Position(ring, (i+2)%b.ring_size)),
                            Position(ring, (i+1)%b.ring_size)
                        )
                        self.assertEqual(
                            b.get_third_in_line(Position(ring, i+1), Position(ring, (i+2)%b.ring_size)),
                            Position(ring, (i+0)%b.ring_size)
                        )
                # Check line getter between rings
                for index in range(0, b.ring_size):
                    if b.extended or index % 2 == 1:
                        self.assertEqual(
                            b.get_third_in_line(Position(0, index), Position(1, index)),
                            Position(2, index)
                        )
                        self.assertEqual(
                            b.get_third_in_line(Position(0, index), Position(2, index)),
                            Position(1, index)
                        )
                        self.assertEqual(
                            b.get_third_in_line(Position(1, index), Position(2, index)),
                            Position(0, index)
                        )

    def test_is_linked_to(self):

        for extended in (False, True):
            for sides in range(3, 5):
                b = DefaultBoardDesign(sides=sides, extended=extended)

                def flip(check, x, y):
                    check(b.is_linked_to(x, y))
                    check(b.is_linked_to(y, x))

                # On ring
                flip(self.assertTrue, Position(0,0), Position(0, 1))
                flip(self.assertTrue, Position(0,1), Position(0, 2))
                flip(self.assertTrue, Position(0,b.ring_size-1), Position(0, 0))

                flip(self.assertFalse, Position(0,0), Position(0, 2))
                flip(self.assertFalse, Position(0,1), Position(0, 3))
                flip(self.assertFalse, Position(0,b.ring_size-1), Position(0, 1))
                flip(self.assertFalse, Position(0,b.ring_size-2), Position(0, 0))

                # Between rings
                flip(self.assertTrue, Position(0,1), Position(1, 1))
                flip(self.assertTrue, Position(1,1), Position(2, 1))
                if not extended:
                    flip(self.assertFalse, Position(0,1), Position(2, 1))
                flip(self.assertFalse, Position(0,1), Position(1, 3))

                if extended:
                    flip(self.assertTrue, Position(0,0), Position(1, 0))
                    flip(self.assertTrue, Position(1,0), Position(2, 0))
                    flip(self.assertFalse, Position(0,0), Position(2, 0))
                    flip(self.assertFalse, Position(0,0), Position(1, 2))
