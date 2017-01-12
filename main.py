"""
Runner file used to train the neural network.
"""
import tensorflow as tf
import gym

from a3c import *
from optparse import OptionParser

import relay_generator

parser = OptionParser()
parser.add_option("-r", "--run",  help="Run only?")
parser.add_option("-m", "--model",  help="Path to save model")
(options, args) = parser.parse_args()

run = True if options.run is not None else False

env_name = 'relay-generator-v0'
env = gym.make(env_name)

# Directories
output_path = './out'
summary_path = output_path + '/summary'
model_path = str(
    options.model) if options.model is not None else output_path + '/model'

# Make directories for outputs
for path in [summary_path, model_path]:
    if not os.path.exists(path):
        os.makedirs(path)

with tf.device("/cpu:0"):
    agent = A3CAgent(
        env.action_space.n,
        lambda: relay_dense(env.observation_space),
        model_path=model_path,
        preprocess=relay_preprocess,
        entropy_factor=0.05
    )

    if run:
        print('Running')

        with tf.Session() as sess:
            agent.load(sess)
            env = track(env)
            agent.run(sess, env)
            print('Final state', env.step_cache[-1][0][0])
            print('Difficulty', env.difficulty)
            print('Reward', env.total_reward)
    else:
        agent.train(
            env_name,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4)
        )
