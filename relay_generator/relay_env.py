import numpy as np
import gym
from math import *
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .util import *
import random

max_ep_reward = 3
# Move left, forward, right or end
num_actions = 4

punishment = 0.01

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

    def __init__(self, dim=(14, 9)):
        self.dim = dim
        self.size = dim[0] * dim[1]
        self.max_blocks_per_turn = max(dim)
        self.target_difficulty = None
        self.target_pos = None

        # Observe the world
        self.observation_space = spaces.Tuple((
            spaces.Box(0, num_block_type, shape=dim),
            spaces.Box(np.array([0, 0]), np.array(dim)),
            spaces.Discrete(num_directions),
            spaces.Box(0, 1, shape=(1))
        ))

        # Actions allow the world to be populated.
        self.action_space = spaces.Discrete(num_actions)

    def _step(self, action):
        """
        Good levels (in order of priority):
        turns ~= target turns
        empty blocks between turns follow interest/intensity curve
        non-solid blocks are near each other.

        An episode consists of starting at a random position and
        performing a random walk to create a solution.
        """

        def choose_random(from_pos):
            """
            Chooses a random position to move to from a given position
            """
            # Must be a starting block. We choose a random direction.
            valid_pos = []
            # Pick a random direction that is valid
            for d in DirectionMap.values():
                ddx, ddy = d.value[1]
                neighbor_pos = (from_pos[0] + ddx, from_pos[1] + ddy)
                if self.world.in_bounds(neighbor_pos):
                    # Any neighbor in the map must be solid
                    valid_pos.append((neighbor_pos, d))

            if len(valid_pos) == 0:
                return None

            return random.choice(valid_pos)

        done = False
        reward = 0

        if action == num_actions - 1:
            # This is the done action
            if self.world.blocks[self.pos] == BlockType.empty.value:
                self.world.blocks[self.pos] = BlockType.end.value
                return self.build_observation(), 0, True, {}
            else:
                # This is the start block. We can't call done here!
                # Random movement!
                prev = self.pos
                self.pos, direction = choose_random(prev)
                reward -= punishment
        else:
            # Retrieve direction from action
            # 0 = forward, 1 = turn left, 2 = turn right
            if action == 0:
                dir_index = self.prev_dir.value[0]
            else:
                dir_index = RotMap[self.prev_dir.value[0]][action - 1]

            direction = DirectionMap[dir_index]
            dx, dy = direction.value[1]

            # Apply action
            prev = self.pos
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)

        # Invalid moves will cause episode to finish
        if not self.world.in_bounds(self.pos):
            # We went out of the map.
            # Make a random move instead
            rmove = choose_random(prev)
            if rmove is None:
                done = True
            else:
                self.pos, direction = rmove
            reward -= punishment
        elif self.world.blocks[self.pos] != BlockType.solid.value:
            # We went back to a non-solid position.
            # Make a random move
            rmove = choose_random(prev)
            if rmove is None:
                done = True
            else:
                self.pos, direction = rmove
            reward -= punishment

        if not done:
            # This is a valid move
            # Empty this block
            self.world.blocks[self.pos] = BlockType.empty.value

            if direction != self.prev_dir:
                # Direction changed. Give turn reward. (+1 total)
                turn_reward = 1 if self.turns < self.target_turns else -1
                reward += turn_reward / self.target_turns

                # Reset
                self.blocks_in_dir = 0
                self.turns += 1
                # Model number of blocks required in this transition using
                # intensity curve.
                self.update_target_blocks_per_turn()
                self.prev_dir = direction

            # Award for keeping block in direction (+1 total)
            dir_reward = 1 if self.blocks_in_dir <= self.target_blocks_per_turn else -1
            total = self.target_blocks_per_turn * self.target_turns
            reward += dir_reward / total
            self.blocks_in_dir += 1

            # Punish for clustering blocks together (- < 1 total)
            # There must be an adjacent block. Don't count that one.
            num_clusters = 0

            for d in DirectionMap.values():
                ddx, ddy = d.value[1]
                neighbor_pos = (self.pos[0] + ddx, self.pos[1] + ddy)
                if self.world.in_bounds(neighbor_pos):
                    if self.world.blocks[neighbor_pos] != BlockType.solid.value:
                        num_clusters += 1

            cluster_reward = 1 if num_clusters > 1 else -1
            reward += (cluster_reward / self.size) * -1

        info = {
            # The actual direction the agent took
            'actual_dir': direction
        }

        return self.build_observation(), reward, done, info

    def update_target_blocks_per_turn(self):
        self.target_blocks_per_turn = self.max_blocks_per_turn * \
            (1 - interest_curve(min(self.turns / self.target_turns, 1))) + 1

    def _reset(self):
        # Number of blocks in the same direction
        self.blocks_in_dir = 0
        # Number of turns made
        self.turns = 0
        # The last direction made
        self.prev_dir = DirectionMap[int(random.random() * num_directions)]

        # Generate random difficulty
        if self.target_difficulty is None:
            self.difficulty = np.random.uniform(0, 1)
        else:
            self.difficulty = self.target_difficulty

        # Number of turns we want.
        self.target_turns = 50 * (- 1 / (self.difficulty + 1) + 1) + 1
        self.update_target_blocks_per_turn()

        self.world = World(self.dim)
        self.center_pos = (self.dim[0] // 2, self.dim[1] // 2)
        self.max_dist_to_center = self.center_pos[0] / 2 + self.center_pos[1] / 2

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
        dir_ord = self.prev_dir.value[0]
        return (self.world.blocks, np.array(self.pos), np.array([dir_ord]), np.array([self.difficulty]))

    def _render(self, mode='human', close=False):
        pass
