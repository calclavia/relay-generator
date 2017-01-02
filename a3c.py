import numpy as np
import tensorflow as tf
import os
import gym
import threading
import multiprocessing
from gym import spaces

from keras import backend as K
from keras.layers import Dense
from models import *
from util import *

class AC_Network():
    def __init__(self, model_builder, num_actions, scope):
        self.scope = scope
        self.num_actions = num_actions

        with tf.variable_scope(self.scope):
            self.inputs, x = model_builder()

            #Output layers for policy and value estimations
            self.policy = Dense(num_actions, activation='softmax', name='policy_output')(x)
            self.value = Dense(1, activation='linear', name='value_output')(x)

    def compile(self, optimizer):
        # Only the worker network need ops for loss functions and gradient updating.
        with tf.variable_scope(self.scope):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_hot = tf.one_hot(self.actions, self.num_actions)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            responsible_outputs = tf.reduce_sum(self.policy * actions_hot, [1])

             # Value loss (Mean squared error)
            self.value_loss = tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            # Entropy regularization
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            # Policy loss
            self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * self.advantages)
            # Learning rate for Critic is half of Actor's, so multiply by 0.5
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            # Clip norm of gradients
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.train = optimizer.apply_gradients(zip(grads, global_vars))

class A3CAgent:
    def __init__(self, env, name, model, discount, callbacks):
        # Name of the worker
        self.name = name
        # Discount factor
        self.discount = discount
        # Graph the metrics
        self.metrics = {
            'value_loss': [],
            'policy_loss': [],
            'grad_norm': [],
            'var_norm': [],
            'rewards': [],
            'lengths': [],
            'mean_values': []
        }

        self.callbacks = callbacks

        # Local copy of the model
        self.model = model
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = env
        self.max_buffer = 30

    def train(self,
            sess,
            observations,
            actions,
            rewards,
            next_observations,
            values,
            bootstrap_value):

        # Here we take the rewards and values from the exp, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        rewards_plus = rewards + [bootstrap_value]
        discounted_rewards = discount(rewards_plus, self.discount)[:-1]
        value_plus = np.array(values + [bootstrap_value])
        advantages = rewards + self.discount * value_plus[1:] - value_plus[:-1]
        advantages = discount(advantages, self.discount)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                self.model.value_loss,
                self.model.policy_loss,
                self.model.entropy,
                self.model.grad_norms,
                self.model.var_norms,
                self.model.train
            ],
            {
                self.model.target_v: discounted_rewards,
                self.model.inputs: np.vstack(observations),
                self.model.actions: actions,
                self.model.advantages: advantages,
                K.learning_phase(): 1
            }
        )

        N = len(observations)

        # Record metrics
        self.metrics['value_loss'].append(v_l / N)
        self.metrics['policy_loss'].append(p_l / N)
        self.metrics['grad_norm'].append(g_n / N)
        self.metrics['var_norm'].append(v_n / N)

    def preprocess(self, observation):
        """
        Preprocesses the input observation before recording it into experience
        """
        if isinstance(self.env.observation_space, spaces.Discrete):
            return one_hot(observation, self.env.observation_space.n)
        return observation

    def run(self, sess, coord, learn=True):
        self.episode_count = 0

        print("Starting worker " + str(self.name))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                self.run_episode(sess, learn)
                self.episode_count += 1

                # Execute callbacks
                for cb in self.callbacks:
                    cb(self)

    def run_episode(self, sess, learn=True):
        # Sync local network with global network
        sess.run(self.update_local_ops)

        # Buffer the data obtained during the episode
        values = []
        states = []
        next_states = []
        actions = []
        rewards = []

        total_reward = 0
        total_value = 0
        episode_step_count = 0
        done = False

        observation = self.preprocess(self.env.reset())

        while not done:
            # Take an action using probabilities from policy network output.
            a_dist, v = sess.run([self.model.policy, self.model.value], {
                self.model.inputs: [observation],
                K.learning_phase(): 0
            })

            action = np.random.choice(a_dist[0], p=a_dist[0])
            action = np.argmax(a_dist == action)
            value = v[0, 0]

            next_observation, reward, done, info = self.env.step(action)
            next_observation = self.preprocess(next_observation)

            # Bookkeeping
            states.append(observation)
            next_states.append(next_observation)
            rewards.append(reward)
            actions.append(action)
            values.append(value)

            total_value += value
            total_reward += reward
            observation = next_observation
            episode_step_count += 1

            # If the episode hasn't ended, but the experience buffer is
            # full, then we make an update step using that experience.
            if learn and len(states) == self.max_buffer and not done:
                # Since we don't know what the true final return is,
                # we "bootstrap" from our current value estimation.
                v1 = sess.run(self.model.value, {
                    self.model.inputs: [observation],
                    K.learning_phase(): 0
                })[0,0]

                self.train(
                    sess,
                    states,
                    actions,
                    rewards,
                    next_states,
                    values,
                    v1
                )

                values = []
                states = []
                next_states = []
                actions = []
                rewards = []

                sess.run(self.update_local_ops)

        if learn:
            # Train the network using the experience buffer at the end of the episode.
            self.train(
                sess,
                states,
                actions,
                rewards,
                next_states,
                values,
                0.0
            )

        # Record metrics
        self.metrics['rewards'].append(total_reward)
        self.metrics['lengths'].append(episode_step_count)
        self.metrics['mean_values'].append(total_value / episode_step_count)
        return observation

class A3CCoordinator:
    def __init__(self, num_actions, model_builder):
        self.num_actions = num_actions
        self.model_builder = model_builder
        # Generate global network
        self.model = AC_Network(model_builder, num_actions, 'global')
        self.saver = tf.train.Saver(max_to_keep=5)

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state(model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self,
              env_name,
              discount=.99,
              callbacks=[],
              num_workers=multiprocessing.cpu_count(),
              optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):

        with tf.Session() as sess:
            workers = []
            # Create worker classes
            for i in range(num_workers):
                env = gym.make(env_name)
                name = 'worker_' + str(i)
                model = AC_Network(self.model_builder, self.num_actions, name)
                model.compile(optimizer)
                workers.append(A3CAgent(env, name, model, discount, callbacks))

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                t = threading.Thread(target=lambda: worker.run(sess, coord))
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)
