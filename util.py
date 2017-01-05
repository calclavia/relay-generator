from keras import backend as K
import time
import numpy as np
import tensorflow as tf
from gym import spaces

def flatten_space(s):
    return flatten_space(s.spaces) if isinstance(s, spaces.Tuple) else s

def action_to_shape(space):
    return space.n if isinstance(space, spaces.Discrete) else space.shape


def one_hot(index, size):
    return [1 if index == i else 0 for i in range(size)]


def discount(rewards, discount, current=0):
    """ Takes an array of rewards and compute array of discounted reward """
    discounted_r = np.zeros_like(rewards)

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

def make_summary(data, prefix=''):
    if prefix != '':
        prefix += '/'

    summary = tf.Summary()
    for name, value in data.items():
        summary.value.add(tag=prefix + name, simple_value=float(value))

    return summary


def save_worker(sess, coord, agent):
    while not coord.should_stop():
        time.sleep(30)
        agent.save(sess)

def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
