import tensorflow as tf
import gym

from a3c import *
from optparse import OptionParser

import relay_generator

units = 100
layers = 5

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
(options, args) = parser.parse_args()

env = gym.make(options.env)
# Observation space size
state_shape = space_to_shape(env.observation_space)
# Agent action space size
num_actions = action_to_shape(env.action_space)

# Directories
output_path = './out'
summary_path = output_path + '/summary'
model_path = output_path + '/model'

# Make directories for outputs
for path in [summary_path, model_path]:
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
with tf.device("/cpu:0"):
    # TODO: Output
    coord = A3CCoordinator(num_actions, model_builder)
    coord.train(options.env)
