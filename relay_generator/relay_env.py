import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .problem import *
from .search import *
from .util import *

num_block_type = 3
num_directions = 4

class RelayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dim=(9, 9)):
        self.dim = dim
        self.size = dim[0] * dim[1]

        # Observe the world
        self.observation_space = spaces.Tuple(
            tuple(spaces.Discrete(num_block_type) for _ in range(self.size))
        )

        # Actions allow the world to be populated.
        self.action_space = spaces.Discrete(num_directions)

    def _step(self, action):
        # Apply action
        d = DirectionMap[action].value[1]
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])

        done = False
        reward = 0

        if not self.world.in_bounds(self.pos):
            # We went out of the map
            done = True
        elif self.world.blocks[self.pos] == BlockType.empty.value:
            # We went back to a previous position
            done = True
        elif self.world.blocks[self.pos] == BlockType.start.value:
            # We've came back!
            done = True
            reward += 10
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            reward += 1

        observation = self.world.blocks.flatten()
        return observation, reward, done, {}

    def _reset(self):
        self.block_counter = {
            BlockType.start: 0,
            BlockType.empty: 0,
            BlockType.solid: 0,
        }

        self.world = World(self.dim)
        # Generate random starting position
        self.pos = (np.random.randint(self.dim[0]), np.random.randint(self.dim[1]))
        self.world.blocks[self.pos] = BlockType.start.value
        return self.world.blocks.flatten()

    def _render(self, mode='human', close=False):
        pass
