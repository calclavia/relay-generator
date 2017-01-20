"""
Relay level generation server
"""
import tensorflow as tf
import gym
import threading
import os
from concurrent.futures import ThreadPoolExecutor

import relay_generator
from relay_generator.util import Direction
from rl import *
from models import *
from world_pb2 import WorldSet, Direction as ProtoDir

# Create relay environment
env_name = 'relay-generator-v0'
env = gym.make(env_name)

acceptance = 1.1
difficulty_steps = 100
random_steps = 100
max_tries = 1000


def track(env):
    """
    Wraps a Gym environment to keep track of the results of step calls visited.
    """
    step = env.step

    def step_override(*args, **kwargs):
        result = step(*args, **kwargs)
        # Disregard actions after done
        if not result[2]:
            actual_dir = env.direction.value[1]
            # Convert actual direction to action
            for d_obj in Direction:
                d = d_obj.value
                if d[1][0] == actual_dir[0] and\
                   d[1][1] == actual_dir[1]:
                   env.actions.append(d[0])
        env.total_reward += result[1]
        return result
    env.step = step_override

    reset = env.reset

    def reset_override(*args, **kwargs):
        env.total_reward = 0
        env.actions = []
        return reset(*args, **kwargs)
    env.reset = reset_override

    return env


def generate(pos):
    print('Generating pos: ', pos)

    for i in range(difficulty_steps):
        world_set = WorldSet()
        truncated_difficulty = float(i) / difficulty_steps

        for k in range(random_steps):
            world = world_set.worlds.add()
            difficulty = float(i * k) / (difficulty_steps * random_steps)

            # Set env
            env.target_difficulty = difficulty
            env.target_pos = pos

            # Keep generating until we have a valid map
            total_reward = 0
            j = 0
            while total_reward < acceptance and j < max_tries:
                agent.run(sess, env)
                total_reward = env.total_reward
                j += 1

            if total_reward < acceptance:
                raise 'Unable to generate valid solution'

            for a in env.actions:
                world.dirs.append(a)

        directory = 'out/gen/{}_{}/'.format(pos[0], pos[1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + '/{}'.format(truncated_difficulty), 'wb') as f:
            f.write(world_set.SerializeToString())
    return None

# Global cache
sess = tf.Session()

with tf.device("/cpu:0"):
    agent = A3CAgent(
        env.action_space.n,
        lambda: relay_dense(env.observation_space),
        preprocess=relay_preprocess,
        model_path='model'
    )

    agent.load(sess)

    env = track(gym.make(env_name))

    for r in range(env.dim[0]):
        for c in range(env.dim[1]):
            generate((r, c))
