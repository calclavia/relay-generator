"""
Relay level generation server
"""
import tensorflow as tf
import gym
from flask import Flask, jsonify

import relay_generator
from a3c import *

app = Flask(__name__)

# Create relay environment
env_name = 'relay-generator-v0'
env = gym.make(env_name)

# Agent action space size
num_actions = action_to_shape(env.action_space)

# Global cache
sess = tf.Session()

with tf.device("/cpu:0"):
    agent = A3CAgent(
        num_actions,
         lambda: relay_dense(env.observation_space),
        preprocess=relay_preprocess
    )

    agent.load(sess)

def track(env):
    """
    Wraps a Gym environment to keep track of the results of step calls visited.
    """
    step = env.step
    def step_override(*args, **kwargs):
        result = step(*args, **kwargs)
        env.step_cache.append(result)
        env.total_reward += result[1]
        return result
    env.step = step_override

    reset = env.reset
    def reset_override(*args, **kwargs):
        env.total_reward = 0
        env.step_cache = []
        return reset(*args, **kwargs)
    env.reset = reset_override

    return env

@app.route('/')
def generate():
    env = track(gym.make(env_name))
    agent.run_sess(sess, env)
    final_state = env.step_cache[-1][0][0].tolist()
    results = {
        'blocks': final_state,
        'rewards': env.total_reward
    }
    return jsonify(results)
