import numpy as np
from enum import Enum

num_directions = 4

class Direction(Enum):
    right = (0, (1, 0))
    left = (1, (-1, 0))
    up = (2, (0, 1))
    down = (3, (0, -1))

DirectionMap = {
    0: Direction.right,
    1: Direction.left,
    2: Direction.up,
    3: Direction.down,
}

RotMap = np.array([
    [2, 3],
    [3, 2],
    [1, 0],
    [0, 1],
])

# Movement directions.
num_move_dirs = 3

class MoveDirection(Enum):
    left = 0
    forward = 1
    right = 2

# Types of blocks available
num_block_type = 4

class BlockType(Enum):
    solid = 0
    empty = 1
    start = 2
    end = 3
