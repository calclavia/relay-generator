import tensorflow as tf
import gym

from a3c import *
from optparse import OptionParser

import relay_generator

units = 256
layers = 8

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
(options, args) = parser.parse_args()

env = gym.make(options.env)
# Observation space size
state_shape = space_to_shape(env.observation_space)
# Agent action space size
num_actions = action_to_shape(env.action_space)

tf.reset_default_graph()

# Directories
output_path = './out'
model_path = output_path + '/model'
state_path = output_path + '/states'
summary_path = output_path + '/tb'

# Make directories for outputs
for path in [model_path, state_path, summary_path]:
    if not os.path.exists(path):
        os.makedirs(path)

model_builder = lambda: dense(state_shape, units, layers)

"""
def state_saver(agent):
    if agent.episode_count % 100 == 0:
        state = agent.run_episode(0, sess, False).reshape(9, 9)
        with open(state_path + '/state-' + str(agent.episode_count), 'w') as f:
            f.write(np.array2string(state, separator=', '))
"""
coord = A3CCoordinator(num_actions, model_builder)
cbs = [summary_writer(summary_path)]#, saver(model_path, coord.saver)]
coord.train(options.env, callbacks=cbs)
