import numpy as np
from enum import Enum

class World:
    """
    Represents the starting state of a generated world.
    """
    def __init__(self, dim):
        # Blocks in the world as a matrix
        self.blocks = np.zeros(dim, dtype=np.int)
        # A list of directions for the emitter
        self.directions = []

    def in_bounds(self, pos):
        return pos[0] >= 0 and pos[1] >= 0 and\
               pos[0] < self.blocks.shape[0] and\
               pos[1] < self.blocks.shape[1]

class BlockType(Enum):
    empty = 0
    solid = 1
    start = 2
    end = 3
