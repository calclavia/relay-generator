"""
Relay level generation server
"""
import tensorflow as tf
import gym
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import relay_generator
from a3c import A3CAgent
from util import track
from models import *

app = Flask(__name__)
CORS(app)

# Create relay environment
env_name = 'relay-generator-v0'
env = gym.make(env_name)

acceptance = 1

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
    i = 0
    while total_reward < acceptance and i < 100:
        agent.run(sess, env)
        total_reward = env.total_reward
        i += 1

    if total_reward < acceptance:
        raise 'Unable to generate valid solution'

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

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(debug=True)
