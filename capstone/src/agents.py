import os.path
import pickle
import random
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from memory import Memory


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
        

    def decide(self, curstate, testmode=False):
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

    def decide(self, curstate, testmode=False):
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


class DQNAgent(object):
    """ This agent uses DQN for making action decisions with 1-epsilon probability """

    def __init__(self, name, state_dim, action_dim, epsdecay=0.995,
                 buffersize=500000, samplesize=32, minsamples=10000,
                 gamma=0.99, state_norm_file='../params/state-stats.pkl', update_target_freq=600,
                 nnparams = {  # Basic DQN setting
                     'hidden_layers'  : [ (40, 'relu'), (40, 'relu') ],
                     'loss'           : 'mse',
                     'optimizer'      : Adam(lr=0.00025),
                     'target_network' : False }):
        """ Accepts a unique agent name, number of variables in the state,
            number of actions and parameters of DQN then initialize the agent"""
        self.name       = name
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.memory     = Memory(maxsize=buffersize)
        self.eps        = 1.0
        self.minsamples = minsamples
        self.samplesize = samplesize
        self.epsdecay   = epsdecay
        self.gamma      = gamma
        self.nnparams   = nnparams
        self._create_nns_()
        self._load_state_normalizer_(state_norm_file)
        self.update_target_freq = update_target_freq
        self.started_learning = False
        self.steps = 0

    def _load_state_normalizer_(self, state_norm_file):
        self.mean = np.zeros(self.state_dim)
        self.std  = np.ones(self.state_dim)
        if os.path.isfile(state_norm_file):
            self.mean, self.std = pickle.load( open( state_norm_file, 'rb') )
            print 'Loaded mean and std of state space from %s' % state_norm_file
        else:
            print 'Warning : Not using state space normalization'

    def _preprocess_state_(self, instate):
        return ((instate - self.mean)/self.std)
        
    def _create_nns_(self):
        self.use_target_network = self.nnparams['target_network']
        self.model        = self._create_model_()
        if self.use_target_network:
            self.target_model = self._create_model_()

    def _create_model_(self):
        model = Sequential()
        layeridx = 0
        for layer_params in self.nnparams['hidden_layers']:
            units, activation_name = layer_params[0], layer_params[1]
            if layeridx == 0:
                model.add(Dense(units, input_dim=self.state_dim))
            else:
                model.add(Dense(units))
            model.add(Activation(activation_name))
        model.add(Dense(self.action_dim))
        model.add(Activation('linear'))
        model.compile(loss=self.nnparams['loss'], optimizer=self.nnparams['optimizer'])
        return model

    def _update_target_model_(self):
        if self.use_target_network:
            self.target_model.set_weights(self.model.get_weights())

    def decide(self, curstate, testmode=False):
        """ Accepts current state as input and returns action to take """
        # Do not do eps greedy policy for test trials
        if not testmode:
            if (random.random() <= self.eps) or (not self.started_learning):
                return random.randint(0, self.action_dim-1)
        # convert state to a matrix with one row
        s = np.array([self._preprocess_state_(curstate)])
        return np.argmax(self.model.predict(s)[0])

    def observe(self, prevstate, action, reward, curstate, done):
        """ Accepts an observation (s,a,r,s',done) as input, store them in memory buffer for
            experience replay """
        prevstate_normalized = self._preprocess_state_(prevstate)
        curstate_normalized  = self._preprocess_state_(curstate)
        self.memory.save(prevstate_normalized, action, reward, curstate_normalized, done)
        if done:
            # Finished episode, so time to decay epsilon
            self.eps *= self.epsdecay
        if self.steps % self.update_target_freq == 0 and self.use_target_network:
            self._update_target_model_()
        self.steps += 1

    def learn(self):
        # Do not learn if number of observations in buffer is low
        if self.memory.getsize() <= self.minsamples:
            return 0.0
        if not self.started_learning:
            self.started_learning = True
        X, y = self._compute_training_batch_()
        history = self.model.fit(X, y, batch_size=self.samplesize, nb_epoch=1, verbose=False)
        return history.history['loss'][-1]

    def _compute_training_batch_(self):
        s, a, r, s1, done = self.memory.sample(self.samplesize)
        s  = np.array(s)
        s1 = np.array(s1)
        q  = self.model.predict(s)
        q1 = self.target_model.predict(s1) if self.use_target_network else self.model.predict(s1)
        X = s
        y = np.zeros((self.samplesize, self.action_dim))
        for idx in xrange(self.samplesize):
            reward = r[idx]
            action = a[idx]
            target = q[idx]
            # We can improve only the target for the action
            # in the observation <s,a,r,s'>
            target_for_action = reward # correct if state is final.
            if not done[idx]:
                # if not add to it the discounted future rewards per current policy
                target_for_action += ( self.gamma*max(q1[idx]) )
            target[action] = target_for_action
            y[idx, :] = target
        return X, y
