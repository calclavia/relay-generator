import tensorflow as tf
import gym

from a3c import *
from optparse import OptionParser

import relay_generator

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
parser.add_option("-r", "--run",  help="Run only?")
parser.add_option("-s", "--size",  help="Number of hidden units")
parser.add_option("-l", "--layers",  help="Number of layers")
(options, args) = parser.parse_args()

run = True if options.run is not None else False
units = int(options.size) if options.size is not None else 128
layers = int(options.layers) if options.layers is not None else 5

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

model_builder = lambda: dense_1(state_shape) # dense(state_shape, units, layers, dropout=0.25)

with tf.device("/cpu:0"):
    # TODO: Output
    coord = A3CCoordinator(num_actions, model_builder)

    if run:
        coord.run(options.env)
    else:
        coord.train(options.env)
