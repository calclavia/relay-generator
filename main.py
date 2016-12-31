import threading
import multiprocessing
import tensorflow as tf
import gym

from a3c import *
from optparse import OptionParser

import relay_generator


max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
units = 256
layers = 8
output_path = './out'
model_path = output_path + '/model'
state_path = output_path + '/states'
summary_path = output_path + '/tb'

# Set workers ot number of available CPU threads
num_workers = multiprocessing.cpu_count()

parser = OptionParser()
parser.add_option("-e", "--env",  help="Gym Environment")
(options, args) = parser.parse_args()

env = gym.make(options.env)
# Observation space size
state_shape = space_to_shape(env.observation_space)
# Agent action space size
a_size = action_to_shape(env.action_space)

tf.reset_default_graph()

for path in [model_path, state_path, summary_path]:
    if not os.path.exists(path):
        os.makedirs(path)

with tf.device("/cpu:0"), tf.Session() as sess:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # Generate global network
    global_network = AC_Network(dense(state_shape, units, layers), a_size, 'global', optimizer)
    saver = tf.train.Saver(max_to_keep=5)
    workers = []

    def saver_fn(agent):
        if agent.episode_count % 100 == 0:
            saver.save(sess, model_path + '/model-' + str(agent.episode_count) + '.cptk')

    # Create worker classes
    for i in range(num_workers):
        env = gym.make(options.env)
        name = 'worker_' + str(i)
        model = AC_Network(dense(state_shape, units, layers), a_size, name, optimizer)
        writer_fn = summary_writer('', tf.summary.FileWriter(summary_path + '/' + name))

        def state_saver(agent):
            if agent.episode_count % 100 == 0:
                state = agent.run_episode(0, sess, False).reshape(9, 9)
                with open(state_path + '/state-' + str(agent.episode_count), 'w') as f:
                    f.write(np.array2string(state, separator=', '))

        cbs = [writer_fn]

        if i == 0:
            cbs += [writer_fn, saver_fn, state_saver]

        workers.append(A3CAgent(env, name, model, gamma, cbs))

    coord = tf.train.Coordinator()

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.run(max_episode_length, sess, coord)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
