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


class MyAgent(BaseAgent):
    """ TODO: add description for this class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: implement the rest of your initialisation

    def initialise_episode(self):
        # TODO: implement your own method
        pass

    def select_action(self, *args):
        # TODO: implement your own function
        pass

    def train(self):
        # TODO: implement your own function
        pass
        #return reward

    # TODO: implement all other functions and methods needed for your agent