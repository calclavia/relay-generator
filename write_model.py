"""
Outputs a JSON of the Keras model
"""
# TODO: Use constants

import gym
from keras.models import load_model

from models import *
import relay_generator

env_name = 'relay-generator-v0'
env = gym.make(env_name)
model = relay_dense(env.observation_space, env.action_space.n)

with open('out/model.json', 'w') as f:
    f.write(model.to_json())

print('Model JSON output at out/')
