import numpy as np
import gym
from math import *
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .problem import *
from .search import *
from .util import *
import random

def interest_curve(x):
    """
    Models the interest curve.
    Paramters:
        x - A number between 0 to 1 representing progression.

    Returns a value from 0 to 1, where 1 indicates highest interest/intensity
    """
    assert 0 <= x and x <= 1
    res = 0.8 * (- 1 / (x + 1) + 1) * (x + exp(0.05 * x) * sin(30 * x)) + 0.2
    assert 0 <= res and res <= 1
    return res

class RelayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dim=(16, 9)):
        self.dim = dim
        self.size = dim[0] * dim[1]
        self.max_blocks_per_turn = min(*dim)
        self.target_difficulty = None
        self.target_pos = None

        # Observe the world
        # TODO: Provide current direction as discrete variable.
        self.observation_space = spaces.Tuple((
            spaces.Box(0, num_block_type, shape=dim),
            spaces.Box(np.array([0, 0]), np.array(dim)),
            spaces.Discrete(num_directions + 1),
            spaces.Box(0, 1, shape=(1))
        ))

        # Actions allow the world to be populated.
        self.action_space = spaces.Discrete(num_directions)

    def _step(self, action):
        """
        Good levels (in order of priority):
        turns ~= target turns
        empty blocks between turns follow interest/intensity curve
        non-solid blocks are near each other TODO: Not always optimal
        empty blocks are near the center of map

        An episode consists of starting at a random position and
        performing a random walk to create a solution.
        """
        # Retrieve action
        direction = DirectionMap[action]
        dx, dy = direction.value[1]

        # Apply action
        prev = self.pos
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        done = False
        reward = 0

        # Invalid moves will cause episode to finish
        if not self.world.in_bounds(self.pos):
            # We went out of the map.
            if self.world.blocks[prev] == BlockType.empty.value:
                # Previous block is empty. We just end the episode here.
                self.world.blocks[prev] = BlockType.end.value
                done = True
            else:
                valid_pos = []
                # Pick a random direction that is valid
                for d in DirectionMap.values():
                    ddx, ddy = d.value[1]
                    neighbor_pos = (prev[0] + ddx, prev[1] + ddy)
                    if self.world.in_bounds(neighbor_pos) and self.world.blocks[neighbor_pos] == BlockType.solid.value:
                        valid_pos.append[neighbor_pos]

                self.pos = random.choice(valid_pos)
                direction = d

        elif self.world.blocks[self.pos] != BlockType.solid.value:
            # We went back to a non-solid position.
            # Previous block MUST be empty. We just end the episode here.
            self.world.blocks[prev] = BlockType.end.value
            done = True

        if not done:
            # This is a valid move
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value

            if direction != self.prev_dir:
                # Direction changed. Give turn reward.
                turn_reward = 1 if self.turns < self.target_turns else -1
                reward += turn_reward / self.target_turns

                # Reset
                self.blocks_in_dir = 0
                self.turns += 1
                # Model number of blocks required in this transition using intensity curve.
                self.target_blocks_per_turn = self.max_blocks_per_turn * (1 - interest_curve(min(self.turns / self.target_turns, 1))) + 1
                self.prev_dir = direction

            # Award for keeping block in direction (+1 total)
            dir_reward = 1 if self.blocks_in_dir <= self.target_blocks_per_turn else -1
            reward += dir_reward / (self.target_blocks_per_turn * self.target_turns)

            # Award for clustering non-solid blocks together (+1 total)
            # There must be an adjacent block. Don't count that one.
            """
            num_clusters = 0

            for d in DirectionMap.values():
                ddx, ddy = d.value[1]
                neighbor_pos = (self.pos[0] + ddx, self.pos[1] + ddy)
                if self.world.in_bounds(neighbor_pos):
                    if self.world.blocks[neighbor_pos] != BlockType.solid.value:
                        num_clusters += 1

            cluster_reward = 1 if num_clusters > 1 else -1
            reward += cluster_reward / self.size

            self.blocks_in_dir += 1
            """

            # TODO: Reward for more center blocks (+1 total)
            # Mahattan distance
            # dist_to_center = abs(self.pos[0] - self.center_pos[0]) + abs(self.pos[1] - self.center_pos[1])
            # reward += dist_to_center /

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

        # TODO: Logarithmic difficulty relation that approaches 1?
        self.target_turns = 20 * self.difficulty + 3
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
