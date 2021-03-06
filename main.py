"""
Runner file used to train the neural network.
"""
import tensorflow as tf
from keras import backend as K
import gym

from models import *
from rl import *
from optparse import OptionParser
import os
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
model_path = str(options.model) if options.model is not None else output_path + '/model'

# Make directories for outputs
for path in [summary_path, model_path]:
    if not os.path.exists(path):
        os.makedirs(path)

with tf.Session() as sess:
    K.set_session(sess)

    agent = A3CAgent(
        lambda: relay_dense(env.observation_space, env.action_space.n),
        model_path=model_path,
        preprocess=relay_preprocess
    )

    agent.compile(sess)

    try:
        agent.load(sess)
        print('Loading last saved session')
    except Exception as e:
        print('Starting new session')
        print(e)
        
    if run:
        env = track(env)
        agent.run(sess, env)
        print('Final state', env.step_cache[-1][0][0])
        print('Difficulty', env.difficulty)
        print('Reward', env.total_reward)
    else:
        print('Training')
        agent.train(sess, lambda: gym.make(env_name)).join()
