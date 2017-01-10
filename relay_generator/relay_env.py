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

    def __init__(self, dim=(9, 16)):
        self.dim = dim
        self.size = dim[0] * dim[1]
        self.max_blocks_per_turn = max(*dim)
        self.target_difficulty = None
        self.target_pos = None

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
        prev = self.pos
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        done = False
        reward = 0

        if not self.world.in_bounds(self.pos):
            # We went out of the map. Revert.
            done = True
            reward -= 5
        elif self.world.blocks[self.pos] != BlockType.solid.value:
            # We went back to a non-solid position
            done = True
            reward -= 4
        else:
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value
            if direction != self.prev_dir:
                # Direction change happened
                if self.turns < self.target_turns:
                    reward += 1 / self.target_turns
                else:
                    # We transform the action to marking this as an end block
                    self.world.blocks[prev] = BlockType.end.value
                    done = True
                    reward += 1

                self.blocks_in_dir = 0
                self.turns += 1
                self.prev_dir = direction

            if not done:
                # Award for keeping block in direction
                r = 1 if self.blocks_in_dir <= self.target_blocks_per_turn else -1
                reward += r / (self.target_blocks_per_turn * self.target_turns)

                # Award for clustering non-solid blocks together
                num_neighbors = 0
                # There must be an adjacent block. Don't count that one.
                cluster = -1

                for d in DirectionMap.values():
                    ddx, ddy = d.value[1]
                    neighbor_pos = (self.pos[0] + ddx, self.pos[1] + ddy)
                    if self.world.in_bounds(neighbor_pos):
                        if self.world.blocks[neighbor_pos] != BlockType.solid.value:
                            cluster += 1
                        num_neighbors += 1

                reward += cluster / \
                    (num_neighbors * self.size * 0.03) * self.difficulty

                self.blocks_in_dir += 1

                # Reward for more center blocks
                reward -= abs(self.pos[0] - self.center_pos[0]) / self.dim[0]
                reward -= abs(self.pos[1] - self.center_pos[1]) / self.dim[1]

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
            self.difficulty = np.random.uniform(0, 1)
        else:
            self.difficulty = self.target_difficulty

        self.target_turns = 20 * self.difficulty + 3
        self.target_blocks_per_turn = self.max_blocks_per_turn * \
            (1 - self.difficulty) + 1

        self.world = World(self.dim)
        self.center_pos = (self.dim[0] // 2, self.dim[1] // 2)

        if self.target_pos is None:
            # Generate random starting position
            self.pos = (
                np.random.randint(self.dim[0]),
                np.random.randint(self.dim[1])
            )
        else:
            self.pos = self.target_pos

        self.world.blocks[self.pos] = BlockType.start.value
        return self.build_observation()

    def build_observation(self):
        return (self.world.blocks, np.array(self.pos), np.array([self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
