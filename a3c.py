import numpy as np
import tensorflow as tf
import os
import gym
import threading
import multiprocessing
import time
from gym import spaces

from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from models import *
from util import *
from experience import *

class AC_Network():
    def __init__(self, model_builder, num_actions, scope, beta=1e-2):
        self.scope = scope
        self.num_actions = num_actions
        # Entropy weight
        self.beta = beta

        with tf.variable_scope(self.scope):
            self.inputs, x = model_builder()
            #Output layers for policy and value estimations
            self.policy = Dense(num_actions, activation='softmax', name='policy_output')(x)
            self.value = Dense(1, activation='linear', name='value_output')(x)
            self.model = Model(self.inputs, [self.policy, self.value])

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
            self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-9))
            # Policy loss
            self.policy_loss = -tf.reduce_sum(tf.log(responsible_outputs) * self.advantages)
            # Learning rate for Critic is half of Actor's, so multiply by 0.5
            # TODO: In very sprase rewards, entropy loss causes NAN..., even when BETA = 0
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.beta * self.entropy

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            # Clip norm of gradients
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.train = optimizer.apply_gradients(zip(grads, global_vars))

def a3c_worker(sess, coord, writer, env_name, num, model, sync,
               gamma, time_steps, max_buffer=32, stagger=1):
    # Thread setup
    env = gym.make(env_name)

    episode_count = 0

    # Stagger threads to decorrelate experience
    time.sleep(stagger * num)

    # Reset per-episode vars
    state = preprocess(env, env.reset())
    terminal = False
    total_reward = 0
    episode_step_count = 0

    print("Running worker " + str(num))

    with sess.as_default(), sess.graph.as_default():
        while not coord.should_stop():
            # Sync local network with global network
            sess.run(sync)

            # Run a training batch
            t = 0
            t_start = t

            exp = TemporalExperience(env.observation_space, env.action_space, time_steps)
            values = []

            while not (terminal or ((t - t_start) == max_buffer)):
                exp.observe(state)

                # Perform action according to policy pi(a_t | s_t)
                probs, value = sess.run([model.policy, model.value], {
                    model.inputs: [exp.get_state()],
                    K.learning_phase(): 0
                })

                # Remove batch dimension
                probs = probs[0]
                value = value[0]
                # Sample an action from an action probability distribution output
                action = np.random.choice(len(probs), p=probs)

                exp.act(action)
                values.append(value)

                next_state, reward, terminal, info = env.step(action)
                next_state = preprocess(env, next_state)

                # TODO: Reward clipping? Experiment >2 tasks
                # r_t = np.clip(r_t, -1, 1)
                exp.reward(reward)

                total_reward += reward
                episode_step_count += 1
                t += 1

                state = next_state

            if terminal:
                reward = 0
            else:
                # Bootstrap from last state
                reward = sess.run(model.value, {
                    model.inputs: [exp.get_state()],
                    K.learning_phase(): 0
                })[0][0]

            # Here we take the rewards and values from the exp, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            discounted_rewards = discount(exp.rewards, gamma, reward)
            value_plus = np.array(values + [reward])
            advantages = discount(exp.rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)

            # Train network
            v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                    model.value_loss,
                    model.policy_loss,
                    model.entropy,
                    model.grad_norms,
                    model.var_norms,
                    model.train
                ],
                {
                    model.target_v: discounted_rewards,
                    model.inputs: exp.get_states(),
                    model.actions: exp.actions,
                    model.advantages: advantages,
                    K.learning_phase(): 1
                }
            )

            if terminal:
                # Record metrics
                writer.add_summary(
                    make_summary({
                        'rewards': total_reward,
                        'lengths': episode_step_count,
                        'value_loss': v_l,
                        'policy_loss': p_l,
                        'entropy_loss': e_l,
                        'grad_norm': g_n,
                        'value_norm': v_n,
                        'mean_values': np.mean(values)
                    }),
                    episode_count
                )

                if episode_count % 10 == 0:
                    writer.flush()

                # Reset per-episode vars
                episode_count += 1
                state = preprocess(env, env.reset())
                terminal = False
                total_reward = 0
                total_value = 0
                episode_step_count = 0

class A3CCoordinator:
    def __init__(self, num_actions, model_builder, time_steps=1, model_path='out/model'):
        self.num_actions = num_actions
        self.model_builder = model_builder
        self.time_steps = time_steps
        self.model_path = model_path
        # Generate global network
        self.model = AC_Network(model_builder, num_actions, 'global')
        self.saver = tf.train.Saver(max_to_keep=5)

        print(self.model.model.summary())

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save(self, sess):
        self.saver.save(sess, self.model_path + '/model.cptk')

    def train(self,
              env_name,
              discount=.99,
              summary_path='out/summary/',
              num_workers=multiprocessing.cpu_count(),
              optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)):
        print('Training')

        with tf.Session() as sess:
            workers = []

            # Create worker classes
            for i in range(num_workers):
                name = 'worker_' + str(i)
                model = AC_Network(self.model_builder, self.num_actions, name)
                model.compile(optimizer)
                sync = update_target_graph('global', name)
                workers.append((name, model, sync))

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            worker_threads = []

            for i, (name, model, sync) in enumerate(workers):
                writer = tf.summary.FileWriter(summary_path + name, sess.graph)
                t = threading.Thread(target=a3c_worker, args=(sess, coord, writer, env_name, i, model, sync, discount, self.time_steps))
                t.start()
                worker_threads.append(t)

            t = threading.Thread(target=save_worker, args=(sess, coord, self))
            t.start()
            worker_threads.append(t)

            coord.join(worker_threads)

    def run(self, env_name):
        print('Running')
        with tf.Session() as sess:
            self.load(sess)
            env = gym.make(env_name)

            state = preprocess(env, env.reset())
            total_reward = 0
            terminal = False

            while not terminal:
                # Perform action according to policy pi(a_t | s_t)
                probs, value = sess.run([self.model.policy, self.model.value], {
                    self.model.inputs: [state],
                    K.learning_phase(): 0
                })

                # Remove batch dimension
                probs = probs[0]
                value = value[0]
                # Sample an action from an action probability distribution output
                action = np.random.choice(len(probs), p=probs)
                # TODO: Don't hardcode this
                print('State', state[:-3].reshape(9,9))
                print('Action', action)
                state, reward, terminal, info = env.step(action)
                print('Reward', reward)
                total_reward += reward

            print('Total Reward:', total_reward)
