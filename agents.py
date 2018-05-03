import numpy as np


class BaseAgent(object):
    """ Basic agent with some basic functions implemented, such as
    and epsilon-greedy action selection.

    self._wrapper   A subclass of the Wrapper class, which translates the
                    environment to an interface for generic Reinforcement
                    Learning Agents
    self._total_reward  Total reward for one training episode

    Also has some basic algorithm parameters:
    self._epsilon   Value in [0, 1] for creating randomness in the greedy method
    self._alpha     Step size parameter

    """

    def __init__(self, wrapper, epsilon=0.1, alpha=0.5, seed=42):
        self._wrapper = wrapper  # environment wrapper that provides extra info
        self._epsilon = epsilon  # randomness
        self._alpha = alpha      # step size parameter
        self._total_reward = 0
        np.random.seed(seed)

    def initialise_episode(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    def select_action(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

    def epsilon_greedy(self, q):
        """Select action in an epsilon-greedy fashion, based on action values q.
        Return corresponding action id

        :param q:   Array of length |actions|, containing the action values
        """
        # Select an action greedily
        if np.random.random_sample() > self._epsilon:
            # which actions have the highest expectation?
            max_exp = max(q)
            max_exp_action_idx = [i for i in range(len(q))
                                  if q[i] == max_exp]
            if not max_exp_action_idx:
                print(q)
            return int(np.random.choice(max_exp_action_idx, 1)[0])
        # Or select an action randomly
        return np.random.choice(len(q))

    def learn(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")


class MyAgent(BaseAgent):
    """ TODO: add description for this class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actions = self._wrapper.actions()
        self._action_value = dict()
        # TODO: implement the rest of your initialisation

    def initialise_episode(self):
        self._total_reward = 0
        return self._wrapper.reset()

    def select_action(self, state):
        q = [self.get_action_value(state, a) for a in self._actions]
        a_id = self.epsilon_greedy(q)
        return self._actions[a_id]

    def get_action_value(self, state, action):
        return self._q.get((state, action), 0.0)

    def train(self):
        # TODO: implement your own function
        return self._total_reward

    # TODO: implement all other functions and methods needed for your agent