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
            tuple(spaces.Discrete(num_block_type) for _ in range(self.size)) +
            (spaces.Discrete(dim[0]), spaces.Discrete(dim[1])) +
            (spaces.Box(4, 10, shape=(1)),)
        )

        # Actions allow the world to be populated.
        self.action_space = spaces.Discrete(num_directions)

    def _step(self, action):
        # Apply action
        direction = DirectionMap[action].value[1]
        self.pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])

        done = False
        reward = 0

        if not self.world.in_bounds(self.pos):
            # We went out of the map
            done = True
            #reward -= self.size
        elif self.pos in self.visited:
            # We went back to a previous position
            done = True
            #reward -= self.size
        elif self.world.blocks[self.pos] == BlockType.start.value:
            # We've came back!
            done = True
            reward += self.size
            # Additional rewards
            reward += self.size / (self.difficulty - self.turns + 1)
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            if direction != self.prev_dir:
                self.turns += 1
                self.prev_dir = direction
            reward += 1

        self.visited.append(self.pos)
        self.actions += 1
        return self.build_observation(), reward, done, {}

    def _reset(self):
        # Number of actions performed
        self.actions = 0
        # Number of turns made
        self.turns = 0
        self.prev_dir = None
        self.visited = []

        self.world = World(self.dim)
        # Generate random starting position
        self.pos = (np.random.randint(self.dim[0]), np.random.randint(self.dim[1]))
        self.world.blocks[self.pos] = BlockType.start.value
        # Generate random difficulty
        self.difficulty = np.random.uniform(4, 10)
        self.visited.append(self.pos)
        return self.build_observation()

    def build_observation(self):
        blocks = np.array(self.world.blocks.flatten(), dtype='float')
        return np.concatenate((blocks, np.array(self.pos), [self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
