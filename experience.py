import numpy as np
from util import *
from collections import deque
from gym import spaces

class NullExperience:
    """
    Base class for experience
    """

    def __init__(self):
        self.clear()

    def observe(self, observation):
        pass

    def act(self, action):
        pass

    def reward(self, reward):
        pass

    def clear(self):
        pass


class Experience(NullExperience):
    """
    Represents an agent's stored memory
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__()

    def observe(self, observation):
        self.observations.append(observation)

    def act(self, action):
        self.actions.append(action)

    def reward(self, reward):
        self.rewards.append(reward)

    def clear(self):
        # Observations made
        self.observations = []
        # Actions taken
        self.actions = []
        # Rewards received
        self.rewards = []

    def get_state(self):
        """ Returns the current observation state """
        return self.observations[-1]

    def get_states(self):
        return self.observations


class TemporalExperience(Experience):

    def __init__(self, observation_space, action_space, time_steps):
        self.time_steps = time_steps
        super().__init__(observation_space, action_space)

    def observe(self, observation):
        super().observe(observation)

        # Fill temporal memory empty spaces with zeros
        while len(self.temporal_memory) < self.time_steps - 1:
            if isinstance(self.observation_space, spaces.Tuple):
                fill = tuple(np.zeros_like(o) for o in observation)
            else:
                fill = np.zeros_like(observation)

            self.temporal_memory.appendleft(fill)

        self.temporal_memory.append(observation)
        self.states.append(self.get_state())

    def clear(self):
        super().clear()

        self.states = []
        # Reset temporal memory
        self.temporal_memory = deque(maxlen=self.time_steps)

    def get_state(self):
        return list(self.temporal_memory)

    def get_states(self):
        return self.states
