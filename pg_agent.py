# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning PA - Spring 2018

Details:
    File name:          my_agent.py
    Author(s):          TODO: fill in your own name(s) and student ID(s)
    Date created:       28 March 2018
    Date last modified: TODO: fill in
    Python Version:     3.4

Description:
    TODO: briefly explain which algorithm you have implemented and what this
    agent actually does

Related files:
    base_agent.py
"""

from base_agent import BaseAgent

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import time


RENDER_REWARD_MIN = 200
RENDER_ENV = False


class PGAgent(BaseAgent):
    """ TODO: add description for this class
    """

    def __init__(self, *args, gamma=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrapper._env.seed(kwargs['seed'])
        self.n_x = self._wrapper._env.observation_space.shape[0]
        self.n_y = self._wrapper._env.action_space.n
        self.lr = gamma
        self.gamma = 0.99
        self.rewards = []
        self.episode_state, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.sess = tf.Session()
        
        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_state.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)
        

    def initialise_episode(self):
        ##self._total_reward = 0
        return self._wrapper.reset()

    def select_action(self, state):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (1, num_features)
        state = state[np.newaxis, :]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: state})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def train(self):

        global RENDER_ENV
        state = self.initialise_episode()
        tic = time.clock()
        while True:
            
            if RENDER_ENV: self._wrapper.render()
            action = self.select_action(state)
            state_, reward, done, info = self._wrapper.step(action)
            self.store_transition(state,action,reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 120:
                done = True

            if(sum(self.episode_rewards) < -250):
                done = True

            if done:
                episode_rewards_sum = sum(self.episode_rewards)
                self.rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(self.rewards)
                print("==========================================")
                print('Max Reward So Far: ',max_reward_so_far)
                print("Seconds: ", elapsed_sec)
                
                # Discount and normalize episode reward
                reward = self.discount_and_norm_rewards()

                # Train on episode
                self.sess.run(self.train_op, feed_dict={
                    self.X: np.vstack(self.episode_state), # shape [ examples, number of inputs]
                    self.Y: np.array(self.episode_actions), # shape [actions, ]
                    self.reward: reward,
                })

                # Reset the episode data
                self.episode_state, self.episode_actions, self.episode_rewards  = [], [], []

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
                else: RENDER_ENV = False

                break
            #update state
            state = state_

        return episode_rewards_sum
    
    def discount_and_norm_rewards(self):
        reward = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            reward[t] = cumulative

        reward -= np.mean(reward)
        reward /= np.std(reward)
        return reward


    def build_network(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [None, self.n_x], name="X")
            self.Y = tf.placeholder(tf.int32, [None, ], name="Y")
            self.reward = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        A1 = tf.layers.dense(
            inputs=self.X,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        A2 = tf.layers.dense(
            inputs=A1,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        # fc3
        Z3 = tf.layers.dense(
            inputs=A2,
            units=self.n_y,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )

        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(Z3, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
            loss = tf.reduce_mean(neg_log_prob * self.reward)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    # TODO: implement all other functions and methods needed for your agent