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
        self.max_turns = 10

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
        direction = DirectionMap[action]
        dx, dy = direction.value[1]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        done = False
        reward = 0

        if not self.world.in_bounds(self.pos):
            # We went out of the map
            done = True
            reward -= 1
        elif (self.actions == 1 and self.world.blocks[self.pos] == BlockType.start.value) or\
             self.world.blocks[self.pos] == BlockType.empty.value:
            # We went back to a previous/solid position
            done = True
            reward -= 1
        elif self.world.blocks[self.pos] == BlockType.start.value:
            # We've came back!
            done = True
            reward += 1

            # Award for clustering
            average_cluster = 0
            num_empty_blocks = 0

            for index, block_val in np.ndenumerate(self.world.blocks):
                if block_val == BlockType.empty.value:
                    num_empty_blocks += 1

                    for d in DirectionMap.values():
                        ddx, ddy = d.value[1]
                        neighbor_pos = (index[0] + ddx, index[1] + ddy)
                        if self.world.in_bounds(neighbor_pos) and\
                           self.world.blocks[neighbor_pos] == BlockType.empty.value:
                            average_cluster += 1

            average_cluster /= num_empty_blocks
            reward += average_cluster - 1

            # Reward for turning based on difficulty
            reward += 1 / (self.difficulty - self.turns + 1)
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            if direction != self.prev_dir:
                self.turns += 1
                self.prev_dir = direction
                # reward += (1 if self.difficulty <= self.turns else -1) / (self.max_turns)

        self.actions += 1
        return self.build_observation(), reward, done, {}

    def _reset(self):
        self.actions = 0
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
        self.difficulty = np.random.uniform(4, self.max_turns)
        return self.build_observation()

    def build_observation(self):
        return (self.world.blocks, np.array(self.pos), np.array([self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
