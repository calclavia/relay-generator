"""
Relay level generation server
"""
import tensorflow as tf
import gym
from flask import Flask, jsonify

import relay_generator
from a3c import A3CAgent
from util import track
from models import *

app = Flask(__name__)

# Create relay environment
env_name = 'relay-generator-v0'
env = gym.make(env_name)

# Global cache
sess = tf.Session()

with tf.device("/cpu:0"):
    agent = A3CAgent(
        env.action_space.n,
        lambda: relay_dense(env.observation_space),
        preprocess=relay_preprocess
    )

    agent.load(sess)

@app.route('/')
def generate():
    env = track(gym.make(env_name))
    # Keep generating until we have a valid map
    total_reward = 0
    while total_reward < 1:
        agent.run_sess(sess, env)
        total_reward = env.total_reward

    final_state = env.step_cache[-1][0][0].tolist()
    results = {
        'blocks': final_state,
        'rewards': env.total_reward
    }
    return jsonify(results)
