import numpy as np
import gym
from math import *
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
        self.max_difficulty = 100
        self.max_blocks_per_turn = max(*dim)
        self.target_difficulty = None

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
            reward -= 2
        elif self.world.blocks[self.pos] != BlockType.solid.value:
            # We went back to a non-solid position
            done = True
            reward -= 1
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            if direction != self.prev_dir:
                # Direction change happened
                if self.turns < self.target_turns:
                    reward += 1 / self.target_turns
                else:
                    self.world.blocks[self.pos] = BlockType.end.value
                    done = True
                    reward += 1

                self.blocks_in_dir = 0
                self.turns += 1
                self.prev_dir = direction

            m = 1 if self.blocks_in_dir <= self.target_blocks_per_turn else -1
            reward += m / (self.target_blocks_per_turn * self.target_turns)
            self.blocks_in_dir += 1
            """
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
            """

        return self.build_observation(), reward, done, {}

    def _reset(self):
        # Number of blocks in the same direction
        self.blocks_in_dir = 0
        # Number of turns made
        self.turns = 0
        # The last direction made
        self.prev_dir = None

        # Generate random difficulty
        if self.target_difficulty is None:
            self.difficulty = np.random.uniform(4, self.max_difficulty)
        else:
            self.difficulty = self.target_difficulty

        self.target_turns = round(log(0.5 * self.difficulty + 1, 1.4) + 1)
        self.target_blocks_per_turn = round(1 / (0.005 * self.difficulty + 0.2 - 0.01 * self.max_blocks_per_turn) + 1)

        self.world = World(self.dim)
        # Generate random starting position
        # TODO: Turn this into numpy int array?
        self.prev_pos = self.pos = (
            np.random.randint(self.dim[0]),
            np.random.randint(self.dim[1])
        )
        self.world.blocks[self.pos] = BlockType.start.value
        return self.build_observation()

    def build_observation(self):
        return (self.world.blocks, np.array(self.pos), np.array([self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
