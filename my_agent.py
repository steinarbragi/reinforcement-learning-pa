# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning PA - Spring 2018

Details:
    File name:          my_agent.py
    Author(s):          Sevak Mardirosian - s2077086 & Steinar Bragi Sigurðarson - s2017873
    Date created:       28 March 2018
    Date last modified: 21 May 2018
    Python Version:     3.4

Description:
    A Deep Policy Gradient agent with a 3 layer neural network. 
    It uses reward-guided softmax cross-entropy loss function, and the Adam Optimizer.
    The rewards are normalized and discounted for increased stability.
    The Neural Network structure, as well as using Adam in combination with a reward-guided loss function was inspired by Gabriel Garza's blog post https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6.

Related files:
    base_agent.py
"""

from base_agent import BaseAgent

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import time

class MyAgent(BaseAgent):
    """ This class provides functions for a Deep Policy Gradient Agent, 
    and can be used with OpenAI gym environment wrappers.
    """

    def __init__(self, *args, gamma=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrapper._env.seed(kwargs['seed'])
        self.n_x = self._wrapper._env.observation_space.shape[0]
        self.n_y = self._wrapper._env.action_space.n
        self.lr = gamma
        self.discount = 0.99
        self.rewards = []
        self.episode_state, self.episode_actions, self.episode_rewards = [], [], []

        self.init_network()

        #TensorFlow session and logging
        self.sess = tf.Session()
        tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def store_step(self, s, a, r):
        """
        Store state, rewards and actions for current timestep in episode history
        """
        self.episode_state.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)
        

    def initialise_episode(self):
        return self._wrapper.reset()

    def select_action(self, state):
        """
            Choose action based on our observation state.
            First, reshape the observation state and
            do forward propagation to get action probabilities,
            then select an action from a biased sample
        """
        state = state[np.newaxis, :]
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: state})
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def train(self):

        """
            This function is called at the beginning of each episode.
            The main loop runs for each timestep, continuously using
            the current observation state to select actions and 
            store the whole state,action, reward history in memory.
            When the episode is done, train the network and reset the history.
        """

        global RENDER_ENV
        state = self.initialise_episode()
        tic = time.clock()
        while True:
            
            if RENDER_ENV: self._wrapper.render()
            action = self.select_action(state)
            state_, reward, done, info = self._wrapper.step(action)
            self.store_step(state,action,reward)

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
                print('Reward: ', episode_rewards_sum)
                
                
                # Discount and normalize episode reward
                normalized_discounted_rewards = self.normalize_discounted()
                
                # Train from episode history
                self.sess.run(self.train_op, feed_dict={
                    self.X: np.vstack(self.episode_state), # shape [ examples, number of inputs]
                    self.Y: np.array(self.episode_actions), # shape [actions, ]
                    self.reward: normalized_discounted_rewards,
                })

                #Reset episode history when done
                self.episode_state, self.episode_actions, self.episode_rewards  = [], [], []

                break
            #update state
            state = state_

        return episode_rewards_sum
    
    def normalize_discounted(self):
        reward = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.discount + self.episode_rewards[t]
            reward[t] = cumulative

        reward -= np.mean(reward)
        reward /= np.std(reward)
        return reward


    def init_network(self):
        '''
        Initialize the TensorFlow Neural Network.
        Training inputs contain observations (X), actions(Y) and reward
        A1 and A2 are our two hidden layers.
        The output layer Z3 represents action probabilities.
        '''
        tf.reset_default_graph()
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
            name='fc1',
            reuse=tf.AUTO_REUSE
        )
        # fc2
        A2 = tf.layers.dense(
            inputs=A1,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2',
            reuse=tf.AUTO_REUSE
        )
        # fc3
        Z3 = tf.layers.dense(
            inputs=A2,
            units=self.n_y,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3',
            reuse=tf.AUTO_REUSE
        )
        #Softmax output
        self.outputs_softmax = tf.nn.softmax(Z3, name='A3')
        with tf.name_scope('loss'):
            #Get negative log probabilities from sparse softmax crossentropy
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
            #Calculate reward-guided-loss
            loss = tf.reduce_mean(neg_log_prob * self.reward)
        with tf.name_scope('train'):
            #train the model with reward-guided loss to encourage good actions
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
