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


class LunarLanderWrapper(Wrapper):
    """ TODO: Add a description for your wrapper
    """

    _actions = []   # TODO: define list of actions (HINT: check LunarLander-v2 source code to figure out what those actions are)
    _penalty = []
    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)  # Don't change environment name
        actions = [0, 1, 2]   # left (0), right (1), bottom (2)
        _penalty = 0


    def solved(self, reward):
        return True if reward >= 200 else False

    def episode_over(self):
        #I guess it should return true if module has landed, crashed or gone out of frame?
        pass
        #return True if

    def penalty(self):
        return self._penalty

    # TODO: implement all other functions and methods needed for your wrapper
