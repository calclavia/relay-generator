import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .problem import *
from .search import *
from .util import *

class RelayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dim=(9, 9)):
        self.dim = dim
        self.size = dim[0] * dim[1]

        # Observe the world
        self.observation_space = spaces.Tuple((
            spaces.Box(0, num_block_type, shape=dim),
            spaces.Box(np.array([0, 0]), np.array(dim)),
            spaces.Box(4, 10, shape=(1))
        ))

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
            reward -= self.size
        elif self.pos == self.prev_pos or\
             self.world.blocks[self.pos] == BlockType.empty.value:
            # We went back to a previous/solid position
            done = True
            reward -= self.size
        elif self.world.blocks[self.pos] == BlockType.start.value:
            # We've came back!
            done = True
            reward += self.size
            # Additional rewards
            # reward += 1 / (self.difficulty - self.turns + 1)
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            if direction != self.prev_dir:
                self.turns += 1
                self.prev_dir = direction

                if self.difficulty <= self.turns:
                    reward += 1
                else:
                    reward -= 1

        self.prev_pos = self.pos
        return self.build_observation(), reward, done, {}

    def _reset(self):
        # Number of turns made
        self.turns = 0
        # The last direction made
        self.prev_dir = None

        self.world = World(self.dim)
        # Generate random starting position
        # TODO: Turn this into numpy int array?
        self.prev_pos = self.pos = (
            np.random.randint(self.dim[0]),
            np.random.randint(self.dim[1])
        )
        self.world.blocks[self.pos] = BlockType.start.value
        # Generate random difficulty
        self.difficulty = np.random.uniform(4, 10)
        return self.build_observation()

    def build_observation(self):
        return (self.world.blocks, np.array(self.pos), np.array([self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
