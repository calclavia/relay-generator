from world import *
from util import Direction
from search import *
import unittest

class Problem:
    """
    Defines the search problem.
    """
    def __init__(self, world):
        self.world = world

    def get_start(self):
        """
        Gets the starting state of the problem
        """
        if len(self.world.directions) > 0:
            for index, value in np.ndenumerate(self.world.blocks):
                if value == BlockType.start.value:
                    return (index, 0)

        return None

    def get_next(self, state):
        """
        Gets the next states the currrent state can reach
        Return: The successor state, action to reach successor, cost
        """
        pos, dir_index = state

        if dir_index < len(self.world.directions):
            dx, dy = self.world.directions[dir_index].value[1]

            i = 1
            current = (pos[0] + dx, pos[1] + dy)

            while self.world.in_bounds(current) and\
                  self.world.blocks[current] != BlockType.solid.value:
                yield ((current, dir_index + 1), i, 1)
                current = (current[0] + dx, current[1] + dy)
                i += 1

    def is_goal(self, state):
        """
        Checks if a given state is a goal state
        """
        return self.world.blocks[state[0]] == BlockType.end.value

class TestProblem(unittest.TestCase):
    def test_get_start(self):
        world = World((3, 3))
        world.blocks[1, 2] = BlockType.start.value
        world.directions = [Direction.up, Direction.down]
        self.assertEqual(Problem(world).get_start(), ((1, 2), 0))

    def test_is_goal(self):
        world = World((3, 3))
        world.blocks[1, 2] = BlockType.start.value
        world.blocks[0, 1] = BlockType.end.value
        world.directions = [Direction.up, Direction.down]
        self.assertEqual(Problem(world).is_goal(((0, 0), 0)), False)
        self.assertEqual(Problem(world).is_goal(((0, 2), 0)), False)
        self.assertEqual(Problem(world).is_goal(((0, 1), 0)), True)

    def test_get_next_1(self):
        world = World((3, 3))
        world.blocks[0, 0] = BlockType.start.value
        world.blocks[1, 2] = BlockType.end.value
        world.directions = [Direction.up, Direction.right]

        p = Problem(world)
        s = p.get_start()
        successors = list(p.get_next(s))
        self.assertEqual(len(successors), 2)
        self.assertEqual(successors[0], (((0, 1), 1), 1, 1))
        self.assertEqual(successors[1], (((0, 2), 1), 2, 1))

        successors = list(p.get_next(successors[1][0]))
        self.assertEqual(successors[0], (((1, 2), 2), 1, 1))
        self.assertEqual(successors[1], (((2, 2), 2), 2, 1))

    def test_get_next_2(self):
        world = World((3, 3))
        world.blocks[0, 0] = BlockType.start.value
        world.blocks[0, 2] = BlockType.solid.value
        world.directions = [Direction.up, Direction.right]
        p = Problem(world)
        s = p.get_start()
        successors = list(p.get_next(s))
        self.assertEqual(len(successors), 1)
        self.assertEqual(successors[0], (((0, 1), 1), 1, 1))

    def test_solve(self):
        world = World((3, 3))
        world.blocks[0, 0] = BlockType.start.value
        world.blocks[2, 2] = BlockType.end.value
        world.directions = [Direction.up, Direction.right]
        p = Problem(world)
        self.assertEqual(search(p), [2, 2])

    def test_solve_no_solution(self):
        world = World((3, 3))
        world.blocks[0, 0] = BlockType.start.value
        world.blocks[2, 2] = BlockType.end.value
        world.directions = [Direction.up, Direction.down]
        p = Problem(world)
        self.assertEqual(search(p), None)

if __name__ == '__main__':
    unittest.main()
