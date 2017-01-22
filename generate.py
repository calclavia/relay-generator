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

acceptance = 1.2
difficulty_steps = 500
min_random_steps = 10
max_random_steps = 40
max_workers = 8

def track(env):
    """
    Wraps a Gym environment to keep track of the results of step calls visited.
    """
    step = env.step

    def step_override(*args, **kwargs):
        result = step(*args, **kwargs)
        # Disregard actions after done
        if not result[2]:
            info = result[3]
            env.actions.append(info['actual_dir'].value[0])
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
    env = track(gym.make(env_name))

    for i in range(difficulty_steps):
        world_set = WorldSet()
        truncated_difficulty = float(i) / difficulty_steps

        # Set env
        env.target_difficulty = truncated_difficulty
        env.target_pos = pos

        # Keep generating until we have sufficient maps
        k = 0
        # Number of random steps decreases proportional to difficulty
        random_steps = (1 - truncated_difficulty) * \
            (max_random_steps - min_random_steps) + min_random_steps

        # While we still need more worlds
        while k < random_steps:
            agent.run(sess, env)
            if env.total_reward >= acceptance:
                world = world_set.worlds.add()

                for a in env.actions:
                    world.dirs.append(a)
                k += 1

        directory = 'out/gen/{}_{}/'.format(pos[0], pos[1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + '/{0:.3f}'.format(truncated_difficulty), 'wb') as f:
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        threads = []

        for r in range(env.dim[0]):
            for c in range(env.dim[1]):
                threads.append(executor.submit(generate, (r, c)))

        for t in threads:
            t.result()
