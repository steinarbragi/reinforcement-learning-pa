# -*- coding: utf-8 -*-
""" Lunar Lander environment wrapper for Reinforcement Learning PA - Spring 2018

Details:
    File name:          lunarlander_wrapper.py
    Author(s):          Sevak Mardirosian - s2077086 & Steinar Bragi Sigurðarson - s2017873
    Date created:       28 March 2018
    Date last modified: 21 May 2018
    Python Version:     3.4

Description:
    Implementation of a wrapper for the Lunar Lander environment as presented in
    https://gym.openai.com/envs/LunarLander-v2/

Related files:
    wrapper.py
"""

from wrapper import Wrapper
import math
import pandas as pd


class LunarLanderWrapper(Wrapper):
    """ 
    A simple wrapper for the Lunar Lander Environment.
    It works with the Qlearning agent, as well as the Policy Gradient Agent.
    The constructor initialized bins to be used with Q-Learning. 
    """

    # Discrete action space: 0: nothing, 1: main, 2:left, 3: right
    _actions = [0,1,2,3]

    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)  # Don't change environment name
        self.init_bins() #initialize bins for Q-Learner
        self._penalty = -100

    def init_bins(self):
        '''
        This function models the LunarLander Environment using bins and value limits.
        To be used with the Q-Learning Agent
        '''
        pos_x_lim = 10
        pos_y_lim = 10
        vel_x_lim = 10
        vel_y_lim = 10
        ang_lim = 12 * 2 * math.pi / 360  # maximum angle (in radians)
        ang_velo_lim = 2
        
        n_pos_x_bins = 20
        n_pos_y_bins = 20
        n_vel_x_bins = 20
        n_vel_y_bins = 20
        n_ang_bins = 50
        n_ang_velo_bins = 40
        n_left_leg_bins = 2
        n_right_leg_bins = 2
        
        pos_x_bins = pd.cut([-pos_x_lim, pos_x_lim], bins=n_pos_x_bins, retbins=True)[1][1:-1]
        pos_y_bins = pd.cut([-pos_y_lim, pos_y_lim], bins=n_pos_y_bins, retbins=True)[1][1:-1]
        
        vel_x_bins = pd.cut([-vel_x_lim, vel_x_lim], bins=n_vel_x_bins, retbins=True)[1][1:-1]
        vel_y_bins = pd.cut([-vel_y_lim, vel_y_lim], bins=n_vel_y_bins, retbins=True)[1][1:-1]
        
        ang_bins = pd.cut([-ang_lim, ang_lim], bins=n_ang_bins, retbins=True)[1][1:-1]
        ang_velo_bins = pd.cut([-ang_velo_lim, ang_velo_lim], bins=n_ang_velo_bins, retbins=True)[1][1:-1]

        left_leg_bins = pd.cut([0, 1], bins=n_left_leg_bins, retbins=True)[1][1:-1]
        right_leg_bins = pd.cut([0, 1], bins=n_right_leg_bins, retbins=True)[1][1:-1]

        
        self._bins = [
            pos_x_bins,
            pos_y_bins,
            vel_x_bins,
            vel_y_bins,
            ang_bins,
            ang_velo_bins,
            left_leg_bins,
            right_leg_bins
        ]

    def get_bins(self):
        return self._bins


    def solved(self, rewards):
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r >= 200) >= 10):
            return True
        return False

    def episode_over(self):
        return True if self._number_of_steps >= 5000 else False
        #return self._env.unwrapped.game_over

    def render(self):
        self._env.render()

    def penalty(self):
        return self._penalty
