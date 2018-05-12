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
    # Action is two floats [main engine, left-right engines].
    # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
    # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

    _actions = [0.0,0.0]
    _penalty = []
    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)  # Don't change environment name
        self._bins = []



    def get_bins(self):
        return self._bins


    def solved(self, rewards):
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r >= 200) >= 10):
            return True
        return False

    def episode_over(self):
        #I guess it should return true if module has landed, crashed or gone out of frame?
        pass
        #return True if

    def penalty(self):
        return self._penalty

    # TODO: implement all other functions and methods needed for your wrapper
