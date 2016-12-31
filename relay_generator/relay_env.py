import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .problem import *
from .search import *
from .util import *

num_block_type = 4
num_directions = 4


class RelayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dim=(9, 9)):
        self.dim = dim
        self.size = dim[0] * dim[1]

        # Observe the world
        self.observation_space = spaces.Tuple(
            tuple(spaces.Discrete(num_block_type) for _ in range(self.size)))

        # Actions allow the world to be populated.
        self.action_space = spaces.Discrete(num_block_type)

    def _step(self, action):
        # Apply action
        action += 1

        self.world.blocks[np.unravel_index(self.actions, self.dim)] = action
        self.actions += 1

        observation = self.world.blocks.flatten()
        done = self.actions >= self.size
        reward = 0

        self.block_counter[BlockType(action)] += 1

        if action == BlockType.start.value:
            if self.block_counter[BlockType.start] == 1:
                reward += 1
            else:
                reward -= 1

        if action == BlockType.end.value:
            if self.block_counter[BlockType.end] == 1:
                reward += 1
            else:
                reward -= 1

        # Reward if there exists a solution to this level
        if done:
            reward -= abs(self.block_counter[BlockType.solid] - 0.6 * self.size) / self.size

            # Generating a map with a solution is worth a lot
            if self.block_counter[BlockType.start] == 1 and\
               self.block_counter[BlockType.end] == 1 and\
               search(Problem(self.world)) != None:
                reward += 50

        return observation, reward, done, {}

    def _reset(self):
        self.actions = 0

        self.block_counter = {
            BlockType.start: 0,
            BlockType.end: 0,
            BlockType.empty: 0,
            BlockType.solid: 0,
        }

        self.world = World(self.dim)
        # Generate random directions
        self.world.directions = [Direction.up, Direction.right]
        # [DirectionMap[np.random.randint(4)] for i in range(4)]
        return self.world.blocks.flatten()

    def _render(self, mode='human', close=False):
        pass
