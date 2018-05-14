# -*- coding: utf-8 -*-
""" Lunar Lander environment wrapper for Reinforcement Learning PA - Spring 2018

Details:
    File name:          lunarlander_wrapper.py
    Author(s):          TODO: fill in your own name(s) and student ID(s)
    Date created:       28 March 2018
    Date last modified: TODO: fill in
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
    """ TODO: Add a description for your wrapper
    """

    # Discrete action space: 0: nothing, 1: main, 2:left, 3: right
    _actions = [0,1,2,3]

    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)  # Don't change environment name
        self._penalty = 0

    def solved(self, rewards):
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r >= 200) >= 10):
            return True
        return False

    def episode_over(self):
        return True if self._number_of_steps > 5000 else False
        #return self._env.unwrapped.game_over

    def render(self):
        self._env.render()

    #def penalty(self):
    #    return self._penalty

    # TODO: implement all other functions and methods needed for your wrapper
