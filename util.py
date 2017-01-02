from keras import backend as K
import numpy as np
import tensorflow as tf
from gym import spaces

def discount(rewards, discount):
    """ Takes an array of rewards and compute array of discounted reward """
    discounted_r = np.zeros_like(rewards)
    current = 0

    for t in reversed(range(len(rewards))):
        current = current * discount + rewards[t]
        discounted_r[t] = current

    return discounted_r

def space_to_shape(space):
    if isinstance(space, spaces.Discrete):
        # One hot vectors of states
        return (space.n,)

    if isinstance(space, spaces.Tuple):
        return (len(space.spaces),)

    return space.shape

def action_to_shape(space):
    return space.n if isinstance(space, spaces.Discrete) else space.shape

def one_hot(index, size):
    return [1 if index == i else 0 for i in range(size)]

def build_summaries(names):
    """ Set up some episode summary ops to visualize on tensorboard. """
    var_cache = {}
    new_val_cache = {}
    update_op_cache = {}

    for name in names:
        var_cache[name] = var = tf.Variable(0.)
        tf.summary.scalar(name, var)
        new_val_cache[name] = new_val = tf.placeholder("float")
        update_op_cache[name] = var.assign(new_val)

    def update_summary(sess, update_dict):
        ops = [ update_op_cache[name] for name in update_dict.keys() ]
        feed = { new_val_cache[name]: val for name, val in update_dict.items() }
        sess.run(ops, feed)

    summary_op = tf.merge_all_summaries()
    return update_summary, summary_op

def saver(model_path, saver):
    def save(agent):
        if agent.name == 'worker_1' and agent.episode_count % 100 == 0:
            saver.save(sess, model_path + '/model-' + str(agent.episode_count) + '.cptk')
    return save

def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
