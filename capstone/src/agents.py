import random
import numpy as np

class RandomAgent(object):
    """ This agent selects a random action independent of the current state """

    def __init__(self, name, state_dim, action_dim, seed=16):
        """ Accepts a unique agent name, number of variables in the state,
            number of actions and initialize the agent """
        self.name       = name
        self.state_dim  = state_dim
        self.action_dim = action_dim
        random.seed(seed)
        ## Only for determining the mean and standard deviation of state variables.
        self.state_sum    = np.zeros(self.state_dim)
        self.statesqr_sum = np.zeros(self.state_dim)
        self.observation_count = 0
        

    def decide(self, curstate):
        """ Accepts current state as input and returns action to take """
        return random.randint(0, self.action_dim-1)

    def observe(self, prevstate, action, reward, curstate, done):
        """ Accepts an observation (s,a,r,s',done) as input, uses it to compute the
            mean and std of state variables """
        self.state_sum      += prevstate
        self.statesqr_sum   += (prevstate**2)
        self.observation_count += 1

    def learn(self):
        return 0.0

    def describe_state_variables(self):
        mean    = self.state_sum / float(self.observation_count)
        sqrmean = self.statesqr_sum / float(self.observation_count)
        std     = np.sqrt(sqrmean - (mean**2))
        return mean, std

class RandomSearchAgent(object):
    """ This agent samples action-vectors at random when episode reward is less than that for
        previous values of action-vectors """

    def __init__(self, name, state_dim, action_dim, seed=16, stop_search_reward=200):
        """ Accepts a unique agent name, number of variables in the state,
            number of actions and initialize the agent """
        self.name       = name
        self.state_dim  = state_dim
        self.action_dim = action_dim
        #random.seed(seed)
        np.random.seed(seed)
        self.action_vectors = np.random.rand(self.action_dim, self.state_dim)*2 - 1.0
        self.best_action_vectors = None
        self.best_reward = -99999.0
        self.episode_reward = 0.0
        self.stop_search_reward = stop_search_reward
        self.do_search = True

    def decide(self, curstate):
        """ Accepts current state as input and returns action to take """
        return np.argmax(self._get_action_scores(curstate))

    def _get_action_scores(self, curstate):
        return np.dot(self.action_vectors, curstate)

    def observe(self, prevstate, action, reward, curstate, done):
        """ Accepts an observation (s,a,r,s',done) as input, accumulates episode reward and samples new action-vectors
            if current episode reward is lower than that for previous action-vectors """
        self.episode_reward += reward
        if done:
            if self.do_search and self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                self.best_action_vectors = self.action_vectors
                if self.best_reward >= self.stop_search_reward:
                    self.do_search = False
                    print 'Solved an episode. Stopped searching.'
                else:
                    self.action_vectors = np.random.rand(self.action_dim, self.state_dim)*2 - 1.0
            self.episode_reward = 0.0

    def learn(self):
        return 0.0
