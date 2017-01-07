"""
Relay level generation server
"""
import tensorflow as tf
import gym
from flask import Flask, jsonify, request

import relay_generator
from a3c import A3CAgent
from util import track
from models import *

app = Flask(__name__)

# Create relay environment
env_name = 'relay-generator-v0'
env = gym.make(env_name)

acceptance = 4

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
    # Parse arguments
    difficulty = float(request.args.get('difficulty'))

    if request.args.get('pos_x') and request.args.get('pos_y'):
        px = int(request.args.get('pos_x'))
        py = int(request.args.get('pos_y'))
        pos = (px, py)
    else:
        pos = None

    env.target_difficulty = difficulty
    env.target_pos = pos

    # Keep generating until we have a valid map
    total_reward = 0
    while total_reward < acceptance:
        agent.run_sess(sess, env)
        total_reward = env.total_reward

    final_state = env.step_cache[-1][0][0].tolist()
    print(env.step_cache[-1][0][0])
    results = {
        'rewards': env.total_reward,
        'difficulty': env.difficulty,
        'turns': env.target_turns,
        'blocks_per_turn': env.target_blocks_per_turn,
        'blocks': final_state
    }
    return jsonify(results)