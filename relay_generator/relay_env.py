import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .world import *
from .problem import *
from .search import *

num_block_type = 4
num_directions = 4

class RelayEnv(gym.Env):

  def __init__(self, dim=(15, 15)):
      self.dim = dim
      self.size = dim[0] * dim[1]

      # Observe the world
      self.observation_space = spaces.Tuple(tuple(spaces.Discrete(num_block_type) for _ in range(self.size)))

      # Actions allow the world to be populated.
      self.action_space = spaces.Discrete(num_block_type)

  def _step(self, action):
      # Apply action
      self.world.blocks[np.unravel_index(self.actions, self.dim)] = action
      self.actions += 1

      observation = self.world.blocks.flatten()
      done = self.actions >= self.size
      reward = 0

      if done:
          # Reward if there's a start and goal
          if self.world.count_type(BlockType.start) == 1:
              reward += 1

          if self.world.count_type(BlockType.end) == 1:
              reward += 1

          # Reward if there exists a solution to this level
          if reward == 2 and search(Problem(self.world)) != None:
              reward += 1

          print(self.world.blocks)

      return observation, reward, done, {}

  def _reset(self):
      self.actions = 0
      self.world = World(self.dim)
      return self.world.blocks.flatten()

  def _render(self, mode='human', close=False):
      print(self.world.blocks)
