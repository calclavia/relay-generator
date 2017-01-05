import numpy as np
from enum import Enum

from .util import *

class World:
    """
    Represents the starting state of a generated world.
    """
    def __init__(self, dim):
        # Blocks in the world as a matrix
        self.blocks = np.empty(dim, dtype=np.int)
        self.blocks.fill(BlockType.solid.value)

    def in_bounds(self, pos):
        return pos[0] >= 0 and pos[1] >= 0 and\
               pos[0] < self.blocks.shape[0] and\
               pos[1] < self.blocks.shape[1]

    def get_type(self, block_type):
        """
        Gets the starting state of the problem
        """
        for index, value in np.ndenumerate(self.blocks):
            if value == block_type.value:
                return (index, 0)

        return None

    def count_type(self, block_type):
        """
        Gets the starting state of the problem
        """
        i = 0
        for index, value in np.ndenumerate(self.blocks):
            if value == block_type.value:
                i += 1

        return i
